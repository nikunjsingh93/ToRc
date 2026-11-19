//! parity - the native kernels against each other and against the types' definitions.
//!
//! Two properties, per weight type the kernels take:
//! - consistency: every path for a type (one vector or many, rows a pass, SMMLA, VNNI) gives the same bits for the
//!   same row and vector. The engine depends on this: a token's logits are the same whichever block it is read in.
//! - correctness: each product is within float rounding of an f64 product of the weights dequantised by ggml's
//!   definitions (quant::dequant) and the Q8_K activations (either of the block's two encodings: 256-blocks, d * qs,
//!   or 32-blocks, d32 * qs32).
//!
//! The Q8_K quantiser's native path against its scalar definition, bit for bit. The DeltaNet steps: tests/gdn32.rs.

use kernels::{Q8, Q8s, QK};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// roughly normal, from 4 uniforms
    fn normal(&mut self) -> f32 {
        (0..4).map(|_| (self.next() >> 40) as f32 / (1u64 << 24) as f32).sum::<f32>() - 2.0
    }
}

/// Activations with the awkward cases: outliers, an all-zero block, tiny values, exact half-steps.
fn activations(rng: &mut Rng, nb: usize) -> Vec<f32> {
    let mut x: Vec<f32> = (0..nb * QK).map(|_| rng.normal()).collect();
    if nb > 1 {
        x[QK..2 * QK].fill(0.0);
    }
    if nb > 2 {
        x[2 * QK + 5] = 300.0;
        x[2 * QK + 40] = -1e-30;
    }
    if nb > 3 {
        // values on exact half-steps of the scale (ties to even)
        for i in 0..QK {
            x[3 * QK + i] = (i as f32 % 64.0) - 31.5;
        }
        x[3 * QK] = 127.0;
    }
    x
}

fn quantized(rng: &mut Rng, nv: usize, nb: usize) -> Q8 {
    let mut xq = Q8::new(nv * nb);
    for v in 0..nv {
        kernels::q8k_quantize(&activations(rng, nb), xq.sub_mut(v * nb..(v + 1) * nb));
    }
    xq
}

/// (ggml type, block bytes, numbers a block, offsets of its fp16 fields): the types the row kernels take.
const TYPES: &[(u32, usize, usize, &[usize])] = &[
    (2, 18, 32, &[0]),       // Q4_0
    (3, 20, 32, &[0, 2]),    // Q4_1
    (6, 22, 32, &[0]),       // Q5_0
    (7, 24, 32, &[0, 2]),    // Q5_1
    (8, 34, 32, &[0]),       // Q8_0
    (11, 110, 256, &[108]),  // Q3_K
    (12, 144, 256, &[0, 2]), // Q4_K
    (13, 176, 256, &[0, 2]), // Q5_K
    (14, 210, 256, &[208]),  // Q6_K
    #[cfg(target_arch = "aarch64")]
    (18, 98, 256, &[0]), // IQ3_XXS (NEON only so far)
    (20, 18, 32, &[0]),      // IQ4_NL
    (21, 110, 256, &[0]),    // IQ3_S
    #[cfg(target_arch = "aarch64")]
    (22, 82, 256, &[0]), // IQ2_S (NEON only so far)
    (23, 136, 256, &[0]),    // IQ4_XS
    #[cfg(target_arch = "aarch64")]
    (42, 18, 64, &[0]), // Q2_0 (NEON only so far)
    (39, 17, 32, &[]),       // MXFP4 (its scale is one E8M0 byte, set in weights)
];

fn geometry(ty: u32) -> (usize, usize, &'static [usize]) {
    let t = TYPES.iter().find(|t| t.0 == ty).unwrap();
    (t.1, t.2, t.3)
}

/// `rows` rows of nb x 256 numbers of type ty: random bytes, every fp16 field a finite number of either sign over a
/// wide range (2^-11 .. 2^8).
fn weights(rng: &mut Rng, ty: u32, rows: usize, nb: usize) -> (Vec<u8>, usize) {
    let (bs, qk, f16s) = geometry(ty);
    let rb = nb * QK / qk * bs;
    let mut w = vec![0u8; rows * rb];
    for b in w.iter_mut() {
        *b = rng.next() as u8;
    }
    for blk in w.chunks_exact_mut(bs) {
        for &o in f16s {
            let h = (rng.next() as u16 & 0x83ff) | ((((rng.next() % 20) + 4) as u16) << 10);
            blk[o..o + 2].copy_from_slice(&h.to_le_bytes());
        }
        if ty == 39 {
            // an E8M0 scale over the same range, 2^-11 .. 2^8
            blk[0] = (117 + rng.next() % 20) as u8;
        }
    }
    (w, rb)
}

/// The f64 products of every row with every vector under each activation encoding, and the sum of |w x| (the scale
/// of the rounding a product may carry): y[v * rows + r].
fn reference(ty: u32, w: &[u8], rb: usize, rows: usize, xq: Q8s, nv: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
    let nb = xq.len() / nv;
    let n = nb * QK;
    let mut wf = vec![0f32; n];
    let (mut out, mut mag) = (vec![[0f64; 2]; nv * rows], vec![0f64; nv * rows]);
    for r in 0..rows {
        assert!(kernels::dequant::dequantize(ty, &w[r * rb..(r + 1) * rb], &mut wf), "type {ty}");
        for v in 0..nv {
            let (mut a, mut b, mut m) = (0f64, 0f64, 0f64);
            for (i, &wi) in wf.iter().enumerate() {
                let (blk, g) = (&xq.k[v * nb + i / QK], &xq.g[v * nb + i / QK]);
                let j = i % QK;
                let x256 = blk.d as f64 * blk.qs[j] as f64;
                let x32 = g.d[j / 32] as f64 * g.qs[j] as f64;
                a += wi as f64 * x256;
                b += wi as f64 * x32;
                m += (wi as f64 * x256).abs().max((wi as f64 * x32).abs());
            }
            out[v * rows + r] = [a, b];
            mag[v * rows + r] = m;
        }
    }
    (out, mag)
}

fn assert_close(ty: u32, what: &str, got: &[f32], want: &[[f64; 2]], mag: &[f64]) {
    for (i, (&g, (w, &m))) in got.iter().zip(want.iter().zip(mag)).enumerate() {
        let tol = 1e-4 * m + 1e-30;
        let ok = w.iter().any(|&w| (g as f64 - w).abs() <= tol);
        assert!(ok, "type {ty} {what}: product {i} = {g}, want {} or {} (tolerance {tol})", w[0], w[1]);
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Rows against nv vectors by the type's multi-vector row kernel, y[v * rows + r].
#[cfg(target_arch = "aarch64")]
fn matmul(ty: u32, w: &[u8], rb: usize, rows: usize, xq: Q8s, nv: usize) -> Vec<f32> {
    use kernels::neon;
    let mut y = vec![0f32; nv * rows];
    let p = y.as_mut_ptr();
    // SAFETY: FEAT_DotProd checked by the caller; y holds nv x rows; the kernels assert the slice sizes
    unsafe {
        match ty {
            2 => neon::q4_0_matmul(w, rb, rows, xq.g, nv, p, rows),
            3 => neon::q4_1_matmul(w, rb, rows, xq.g, nv, p, rows),
            6 => neon::q5_0_matmul(w, rb, rows, xq.g, nv, p, rows),
            7 => neon::dm_matmul(true, w, rb, rows, xq.g, nv, p, rows),
            8 => neon::q8_0_matmul(w, rb, rows, xq.g, nv, p, rows),
            11 => neon::q3k_matmul(w, rb, rows, xq.k, nv, p, rows),
            12 => neon::q45k_matmul(false, w, rb, rows, xq.k, nv, p, rows),
            13 => neon::q45k_matmul(true, w, rb, rows, xq.k, nv, p, rows),
            14 => neon::q6k_matmul(w, rb, rows, xq.k, nv, p, rows),
            20 => neon::iq4nl_matmul(w, rb, rows, xq.g, nv, p, rows),
            39 => neon::mxfp4_matmul(w, rb, rows, xq.g, nv, p, rows),
            21 => neon::iq3s_matmul(w, rb, rows, xq.k, nv, p, rows),
            18 => neon::iq3xxs_matmul(w, rb, rows, xq.k, nv, p, rows),
            22 => neon::iq2s_matmul(w, rb, rows, xq.k, nv, p, rows),
            42 => neon::q2_0_matmul(w, rb, rows, xq.g, nv, p, rows),
            23 => neon::iq4xs_matmul(w, rb, rows, xq.k, nv, p, rows),
            _ => unreachable!(),
        }
    }
    y
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
fn matmul(ty: u32, w: &[u8], rb: usize, rows: usize, xq: Q8s, nv: usize) -> Vec<f32> {
    let mut y = vec![0f32; nv * rows];
    // SAFETY: AVX2, FMA and F16C are the build's target; y holds nv x rows; the kernels assert the slice sizes
    unsafe {
        if ty == 2 {
            kernels::x86::q4_0_dot_multi(w, rb, rows, xq.g, nv, y.as_mut_ptr(), rows);
        } else {
            kernels::x86mm::matmul(ty, w, rb, rows, xq, nv, y.as_mut_ptr(), rows);
        }
    }
    y
}

fn native() -> bool {
    #[cfg(target_arch = "aarch64")]
    return std::arch::is_aarch64_feature_detected!("dotprod");
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
    return true;
    #[allow(unreachable_code)]
    false
}

/// Every type: the multi-vector kernel over nv vectors gives each vector the bits of the same kernel over that
/// vector alone, and every product is within rounding of the f64 reference.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c")
))]
#[test]
fn multi_vector_consistent_and_correct() {
    if !native() {
        return;
    }
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for &(ty, ..) in TYPES {
        for (nb, rows) in [(1usize, 7usize), (3, 64), (8, 33)] {
            let (w, rb) = weights(&mut rng, ty, rows, nb);
            for nv in [1usize, 2, 3, 4, 5, 8, 9] {
                let xq = quantized(&mut rng, nv, nb);
                let all = matmul(ty, &w, rb, rows, xq.view(), nv);
                let (want, mag) = reference(ty, &w, rb, rows, xq.view(), nv);
                assert_close(ty, &format!("nb {nb} rows {rows} nv {nv}"), &all, &want, &mag);
                for v in 0..nv {
                    let one = matmul(ty, &w, rb, rows, xq.sub(v * nb..(v + 1) * nb), 1);
                    assert_eq!(
                        bits(&all[v * rows..(v + 1) * rows]),
                        bits(&one),
                        "type {ty} nb {nb} rows {rows}: vector {v} of {nv} differs from it alone"
                    );
                }
            }
        }
    }
}

/// The correctness check has teeth: Q4_0 products against the IQ4_NL reading of the same bytes (the same block
/// layout, other values) must fail it.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c")
))]
#[test]
fn reference_rejects_a_wrong_decoding() {
    if !native() {
        return;
    }
    let mut rng = Rng(83);
    let (w, rb) = weights(&mut rng, 2, 16, 2);
    let xq = quantized(&mut rng, 2, 2);
    let got = matmul(2, &w, rb, 16, xq.view(), 2);
    let (want, mag) = reference(20, &w, rb, 16, xq.view(), 2);
    let r = std::panic::catch_unwind(|| assert_close(2, "wrong decoding", &got, &want, &mag));
    assert!(r.is_err(), "Q4_0 products passed against IQ4_NL's values");
}

/// ARM: the single-vector row kernels give the multi-vector kernel's bits over one vector.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_single_vector_kernels_match() {
    if !native() {
        return;
    }
    use kernels::neon;
    let mut rng = Rng(43);
    for nb in [2usize, 8] {
        let xq = quantized(&mut rng, 1, nb);
        for rows in [1usize, 4, 7, 64, 641] {
            for ty in [2u32, 3, 8, 14, 20] {
                let (w, rb) = weights(&mut rng, ty, rows, nb);
                let want = bits(&matmul(ty, &w, rb, rows, xq.view(), 1));
                let mut got: Vec<(String, Vec<f32>)> = Vec::new();
                let mut run = |name: String, f: &dyn Fn(&mut [f32])| {
                    let mut y = vec![0f32; rows];
                    f(&mut y);
                    got.push((name, y));
                };
                // SAFETY: FEAT_DotProd checked; the kernels assert the sizes
                unsafe {
                    match ty {
                        2 => {
                            run("q4_0_dot_rows".into(), &|y| neon::q4_0_dot_rows(&w, rb, rows, &xq.g, y));
                            run("q4_0_dot_rows1".into(), &|y| neon::q4_0_dot_rows1(&w, rb, rows, &xq.g, y));
                            for (r, pf) in [(1usize, 0usize), (2, 0), (2, 2), (4, 1), (4, 2)] {
                                run(format!("q4_0_dot_rows_var r {r} pf {pf}"), &|y| {
                                    neon::q4_0_dot_rows_var(&w, rb, rows, &xq.g, y, r, pf)
                                });
                            }
                        }
                        3 => run("q4_1_dot_rows".into(), &|y| neon::q4_1_dot_rows(&w, rb, rows, &xq.g, y)),
                        8 => run("q8_0_dot_rows".into(), &|y| neon::q8_0_dot_rows(&w, rb, rows, &xq.g, y)),
                        14 => run("q6k_dot_rows".into(), &|y| neon::q6k_dot_rows(&w, rb, rows, &xq.k, y)),
                        _ => {
                            for r in [1usize, 2, 4] {
                                run(format!("iq4nl_dot_rows r {r}"), &|y| {
                                    neon::iq4nl_dot_rows(&w, rb, rows, &xq.g, y, r)
                                });
                            }
                        }
                    }
                }
                for (name, y) in got {
                    assert_eq!(bits(&y), want, "{name}: nb {nb} rows {rows}");
                }
            }
        }
    }
}

/// ARM with I8MM: the SMMLA Q4_0 rows over many vectors give each vector the multi-vector kernel's bits.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_mm_q4_0_matches() {
    if !native() || !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    let mut rng = Rng(59);
    for (nb, rows) in [(2usize, 64usize), (8, 130)] {
        if nb > kernels::neon_mm::MAX_NB {
            continue;
        }
        let (w, rb) = weights(&mut rng, 2, rows, nb);
        for nv in [4usize, 5, 8, 15, 16] {
            let xq = quantized(&mut rng, nv, nb);
            let want = bits(&matmul(2, &w, rb, rows, xq.view(), nv));
            let mut y = vec![0f32; nv * rows];
            // SAFETY: I8MM and DotProd checked; y holds nv x rows
            unsafe { kernels::neon_mm::q4_0_rows_multi(&w, rb, rows, &xq.g, nv, y.as_mut_ptr(), rows) };
            assert_eq!(bits(&y), want, "nb {nb} rows {rows} nv {nv}");
        }
    }
}

/// ARM with I8MM: the SMMLA Q4_K / Q5_K rows (gemm_q45k) over many vectors give each vector the multi-vector
/// kernel's bits.
#[cfg(target_arch = "aarch64")]
#[test]
fn gemm_q45k_matches() {
    if !native() || !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    let mut rng = Rng(61);
    for ty in [12u32, 13] {
        for (nb, rows) in [(1usize, 4usize), (3, 64), (8, 36)] {
            let (w, rb) = weights(&mut rng, ty, rows, nb);
            for nv in [1usize, 2, 3, 4, 5, 8, 16, 17] {
                let xq = quantized(&mut rng, nv, nb);
                let want = bits(&matmul(ty, &w, rb, rows, xq.view(), nv));
                let mut pk = vec![kernels::neon::PackedK::default(); kernels::neon::pack_k_len(nv, nb)];
                kernels::neon::pack_k(&xq.k, nv, &mut pk);
                let mut y = vec![0f32; nv * rows];
                // SAFETY: I8MM checked; rows a multiple of 4; y holds nv x rows
                unsafe { kernels::neon::gemm_q45k(ty == 13, &w, rb, rows, &pk, nv, nb, y.as_mut_ptr(), rows) };
                assert_eq!(bits(&y), want, "type {ty} nb {nb} rows {rows} nv {nv}");
            }
        }
    }
}

/// q45k_lay4 + gemm_q45k_lay: the same bits as the reference (and so as gemm_q45k), 4-row and 8-row passes,
/// panels, odd vector counts.
#[cfg(target_arch = "aarch64")]
#[test]
fn gemm_q45k_lay_matches() {
    if !native() || !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    let mut rng = Rng(67);
    for ty in [12u32, 13] {
        let five = ty == 13;
        for (nb, rows) in [(1usize, 4usize), (3, 64), (8, 36)] {
            let (w, rb) = weights(&mut rng, ty, rows, nb);
            let quad = nb * kernels::neon::q45k_lay_bytes(five);
            let mut lay = vec![0u8; rows / 4 * quad];
            for r in (0..rows).step_by(4) {
                kernels::neon::q45k_lay4(five, &w[r * rb..], rb, nb, &mut lay[r / 4 * quad..(r / 4 + 1) * quad]);
            }
            for nv in [1usize, 2, 3, 4, 5, 8, 9, 16, 17, 40] {
                let xq = quantized(&mut rng, nv, nb);
                let want = bits(&matmul(ty, &w, rb, rows, xq.view(), nv));
                let mut pk = vec![kernels::neon::PackedK::default(); kernels::neon::pack_k_len(nv, nb)];
                kernels::neon::pack_k(&xq.k, nv, &mut pk);
                let mut y = vec![0f32; nv * rows];
                // SAFETY: I8MM checked; rows a multiple of 4; lay holds rows / 4 quads; y holds nv x rows
                unsafe { kernels::neon::gemm_q45k_lay(five, &lay, rows, &pk, nv, nb, y.as_mut_ptr(), rows) };
                assert_eq!(bits(&y), want, "type {ty} nb {nb} rows {rows} nv {nv}");
                if nv <= 3 {
                    let mut y1 = vec![0f32; nv * rows];
                    // SAFETY: DotProd (native); as above
                    unsafe { kernels::neon::q45k_lay_matmul(five, &lay, rows, &xq.k, nv, y1.as_mut_ptr(), rows) };
                    assert_eq!(bits(&y1), want, "lay_matmul type {ty} nb {nb} rows {rows} nv {nv}");
                }
            }
        }
    }
}

/// q6_lay4 + gemm_q6_lay / q6_lay_matmul: the same bits as q6k_matmul (and so as gemm_q6k), 4-row and 8-row
/// passes, panels, odd vector counts.
#[cfg(target_arch = "aarch64")]
#[test]
fn gemm_q6_lay_matches() {
    if !native() || !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    use kernels::neon;
    let mut rng = Rng(71);
    for (nb, rows) in [(1usize, 4usize), (3, 64), (8, 36)] {
        let (w, rb) = weights(&mut rng, 14, rows, nb);
        let quad = nb * neon::Q6_LAY_BYTES;
        let mut lay = vec![0u8; rows / 4 * quad];
        for r in (0..rows).step_by(4) {
            neon::q6_lay4(&w[r * rb..], rb, nb, &mut lay[r / 4 * quad..(r / 4 + 1) * quad]);
        }
        for nv in [1usize, 2, 3, 4, 5, 8, 9, 16, 17, 40] {
            let xq = quantized(&mut rng, nv, nb);
            let want = bits(&matmul(14, &w, rb, rows, xq.view(), nv));
            let mut pk = vec![neon::PackedQ6::default(); neon::pack_q6_len(nv, nb)];
            neon::pack_q6(&xq.k, nv, &mut pk);
            let mut y = vec![0f32; nv * rows];
            // SAFETY: I8MM checked; rows a multiple of 4; lay holds rows / 4 quads; y holds nv x rows
            unsafe { neon::gemm_q6_lay(&lay, rows, &pk, nv, nb, y.as_mut_ptr(), rows) };
            assert_eq!(bits(&y), want, "q6 lay nb {nb} rows {rows} nv {nv}");
            let mut y6 = vec![0f32; nv * rows];
            // SAFETY: as above, the source rows
            unsafe { neon::gemm_q6k(&w, rb, rows, &pk, nv, nb, y6.as_mut_ptr(), rows) };
            assert_eq!(bits(&y6), want, "gemm_q6k nb {nb} rows {rows} nv {nv}");
            if nv <= 3 {
                let mut y1 = vec![0f32; nv * rows];
                // SAFETY: DotProd (native); as above
                unsafe { neon::q6_lay_matmul(&lay, rows, &xq.k, nv, y1.as_mut_ptr(), rows) };
                assert_eq!(bits(&y1), want, "q6 lay_matmul nb {nb} rows {rows} nv {nv}");
            }
        }
    }
}

/// ARM: Q8_0 rows laid out by q8_0_lay4 under gemm_x4::<Q8x4> (from the packed vectors) and x4_matmul::<Q8x4> (1-3
/// vectors) give q8_0_matmul' bits over the source, as do the generic passes over Q4_0's repack (Q4x4).
#[cfg(target_arch = "aarch64")]
#[test]
fn gemm_x4_matches() {
    if !native() || !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    use kernels::neon;
    let mut rng = Rng(73);
    for (nb, rows) in [(1usize, 4usize), (3, 64), (8, 36)] {
        for ty in [8u32, 2] {
            let (w, rb) = weights(&mut rng, ty, rows, nb);
            let g = if ty == 8 { neon::Q8_LAY_GROUP } else { neon::Q4X4_GROUP };
            let quad = nb * 8 * g;
            let mut lay = vec![0u8; rows / 4 * quad];
            for r in (0..rows).step_by(4) {
                let o = &mut lay[r / 4 * quad..(r / 4 + 1) * quad];
                if ty == 8 {
                    neon::q8_0_lay4(&w[r * rb..], rb, nb, o)
                } else {
                    neon::q4_0_repack4(&w[r * rb..], rb, nb, o)
                }
            }
            for nv in [1usize, 2, 3, 4, 5, 8, 9, 16, 17, 40] {
                let xq = quantized(&mut rng, nv, nb);
                let want = bits(&matmul(ty, &w, rb, rows, xq.view(), nv));
                let mut pk = vec![neon::Packed::default(); neon::pack_len(nv, nb)];
                neon::mmla_pack(&xq.g, nv, &mut pk);
                let mut y = vec![0f32; nv * rows];
                // SAFETY: I8MM checked; rows a multiple of 4; lay holds rows / 4 quads; y holds nv x rows
                unsafe {
                    if ty == 8 {
                        neon::gemm_x4::<neon::Q8x4>(&lay, rows, &pk, nv, nb, y.as_mut_ptr(), rows)
                    } else {
                        neon::gemm_x4::<neon::Q4x4>(&lay, rows, &pk, nv, nb, y.as_mut_ptr(), rows)
                    }
                };
                assert_eq!(bits(&y), want, "gemm_x4 ty {ty} nb {nb} rows {rows} nv {nv}");
                if nv <= 3 {
                    let mut y1 = vec![0f32; nv * rows];
                    // SAFETY: DotProd (native); as above
                    unsafe {
                        if ty == 8 {
                            neon::x4_matmul::<neon::Q8x4>(&lay, rows, &xq.g, nv, y1.as_mut_ptr(), rows)
                        } else {
                            neon::x4_matmul::<neon::Q4x4>(&lay, rows, &xq.g, nv, y1.as_mut_ptr(), rows)
                        }
                    };
                    assert_eq!(bits(&y1), want, "x4_matmul ty {ty} nb {nb} rows {rows} nv {nv}");
                }
            }
        }
    }
}

/// An ignored benchmark: Q8_0 rows (512 x 2048) one thread, GMAC/s: gemm_mmla_packed::<Q80> and q8_0_matmul over the
/// source, gemm_x4::<Q8x4> and x4_matmul over q8_0_lay4's layout (the same size). cargo test --release -p torc-kernels
/// --test parity bench_q8 -- --ignored --nocapture
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore]
fn bench_q8() {
    if !native() || !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    use kernels::neon;
    let mut rng = Rng(11);
    let (rows, nb) = (512usize, 8usize);
    let time = |f: &mut dyn FnMut()| {
        f();
        let t = std::time::Instant::now();
        let mut n = 0u32;
        while t.elapsed().as_secs_f64() < 0.4 {
            f();
            n += 1;
        }
        t.elapsed().as_secs_f64() / n as f64
    };
    let (w, rb) = weights(&mut rng, 8, rows, nb);
    let quad = nb * 8 * neon::Q8_LAY_GROUP;
    let mut lay = vec![0u8; rows / 4 * quad];
    let lt = time(&mut || {
        for r in (0..rows).step_by(4) {
            neon::q8_0_lay4(&w[r * rb..], rb, nb, &mut lay[r / 4 * quad..(r / 4 + 1) * quad]);
        }
    });
    println!("Q8_0 layout {:.0} us per matrix", lt * 1e6);
    for nv in [1usize, 2, 3, 4, 8, 16, 32, 64, 128] {
        let xq = quantized(&mut rng, nv, nb);
        let mut y = vec![0f32; nv * rows];
        let macs = (rows * nb * QK * nv) as f64;
        let mut pk = vec![neon::Packed::default(); neon::pack_len(nv, nb)];
        neon::mmla_pack(&xq.g, nv, &mut pk);
        // SAFETY: I8MM checked; rows a multiple of 4; y holds nv x rows; lay holds rows / 4 quads
        let g = time(&mut || unsafe {
            neon::gemm_mmla_packed::<neon::Q80>(&w, rb, rows, &pk, nv, nb, y.as_mut_ptr(), rows)
        });
        let d = time(&mut || unsafe { neon::q8_0_matmul(&w, rb, rows, &xq.g, nv, y.as_mut_ptr(), rows) });
        let l = time(&mut || unsafe { neon::gemm_x4::<neon::Q8x4>(&lay, rows, &pk, nv, nb, y.as_mut_ptr(), rows) });
        let mut line = format!(
            "nv {nv:3}: Q8_0 gemm {:6.1} matmul {:6.1} lay {:6.1}",
            macs / g / 1e9,
            macs / d / 1e9,
            macs / l / 1e9
        );
        if nv <= 3 {
            // SAFETY: as above
            let s = time(&mut || unsafe { neon::x4_matmul::<neon::Q8x4>(&lay, rows, &xq.g, nv, y.as_mut_ptr(), rows) });
            line += &format!(" lay-matmul {:6.1}", macs / s / 1e9);
        }
        println!("{line}  (GMAC/s)");
    }
}

/// An ignored benchmark: a prompt block's dense product as the model runs it: one Q8_0 matrix of 8192 x 2048
/// (DeltaNet's qkv) against 512 packed vectors, tasks of R rows over T threads, the vectors whole or P pairs at a time
/// outside the rows (as TORC_VPANEL). Aggregate GMAC/s: whether the per-core rate holds with every core busy.
/// cargo test --release -p torc-kernels --test parity bench_q8_mt -- --ignored --nocapture
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore]
fn bench_q8_mt() {
    if !native() || !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    use kernels::neon;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut rng = Rng(13);
    let (rows, nb, nv) = (8192usize, 8usize, 512usize);
    let (w, rb) = weights(&mut rng, 8, rows, nb);
    let xq = quantized(&mut rng, nv, nb);
    let mut pk = vec![neon::Packed::default(); neon::pack_len(nv, nb)];
    neon::mmla_pack(&xq.g, nv, &mut pk);
    let mut y = vec![0f32; nv * rows];
    let yp = y.as_mut_ptr() as usize;
    let macs = (rows * nb * QK * nv) as f64;
    #[cfg(windows)]
    unsafe extern "system" {
        fn SetThreadAffinityMask(h: *mut core::ffi::c_void, mask: usize) -> usize;
        fn GetCurrentThread() -> *mut core::ffi::c_void;
    }
    // (threads, first core): pinned one a core from that core on (Windows)
    for (threads, c0) in [(1usize, 0usize), (1, 4), (4, 0), (4, 4), (2, 3), (8, 0)] {
        for (r, p) in [(16usize, 0usize), (128, 16)] {
            let run = || {
                let next = AtomicUsize::new(0);
                std::thread::scope(|s| {
                    let (w, pk) = (&w[..], &pk[..]);
                    for th in 0..threads {
                        #[cfg(not(windows))]
                        let _ = th;
                        let next = &next;
                        s.spawn(move || {
                            loop {
                                #[cfg(windows)]
                                // SAFETY: the calling thread's own affinity
                                unsafe {
                                    SetThreadAffinityMask(GetCurrentThread(), 1usize << (c0 + th))
                                };
                                let r0 = next.fetch_add(r, Ordering::Relaxed);
                                if r0 >= rows {
                                    break;
                                }
                                let n = r.min(rows - r0);
                                let np = nv / 2;
                                let pp = if p == 0 { np } else { p };
                                for p0 in (0..np).step_by(pp) {
                                    let nv1 = (2 * (p0 + pp)).min(nv) - 2 * p0;
                                    // SAFETY: I8MM checked; task rows r0..r0 + n, vectors 2p0.. disjoint per task
                                    unsafe {
                                        neon::gemm_mmla_packed::<neon::Q80>(
                                            &w[r0 * rb..(r0 + n) * rb],
                                            rb,
                                            n,
                                            &pk[p0 * nb * 8..],
                                            nv1,
                                            nb,
                                            (yp as *mut f32).add(2 * p0 * rows + r0),
                                            rows,
                                        )
                                    };
                                }
                            }
                        });
                    }
                });
            };
            run();
            let t = std::time::Instant::now();
            let mut k = 0;
            while t.elapsed().as_secs_f64() < 1.0 {
                run();
                k += 1;
            }
            let s = t.elapsed().as_secs_f64() / k as f64;
            println!(
                "threads {threads} from core {c0} rows/task {r:3} panel {p:2}: {:6.1} GMAC/s ({:5.1} a thread)",
                macs / s / 1e9,
                macs / s / 1e9 / threads as f64
            );
        }
    }
}

/// An ignored benchmark: the experts as a prompt block runs them: 256 laid-out Q4_K matrices of 512 x 2048 (~150 MB,
/// from RAM) or 4 of them over and over (in cache), nv vectors each, tasks of 16 rows over 8 pinned threads.
/// Aggregate GMAC/s: how much the experts lose to reading their weights from RAM, and to few vectors.
/// cargo test --release -p torc-kernels --test parity bench_q45k_mt -- --ignored --nocapture
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore]
fn bench_q45k_mt() {
    if !native() || !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    use kernels::neon;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(windows)]
    unsafe extern "system" {
        fn SetThreadAffinityMask(h: *mut core::ffi::c_void, mask: usize) -> usize;
        fn GetCurrentThread() -> *mut core::ffi::c_void;
    }
    let mut rng = Rng(17);
    let (rows, nb, nm) = (512usize, 8usize, 256usize);
    let (w, rb) = weights(&mut rng, 12, rows, nb);
    let quad = nb * neon::q45k_lay_bytes(false);
    let mut one = vec![0u8; rows / 4 * quad];
    for r in (0..rows).step_by(4) {
        neon::q45k_lay4(false, &w[r * rb..], rb, nb, &mut one[r / 4 * quad..(r / 4 + 1) * quad]);
    }
    // 256 copies (distinct memory, the same numbers)
    let lay: Vec<u8> = (0..nm).flat_map(|_| one.iter().copied()).collect();
    let mb = one.len();
    for nv in [4usize, 8, 16, 32] {
        let xq = quantized(&mut rng, nv, nb);
        let mut pk = vec![neon::PackedK::default(); neon::pack_k_len(nv, nb)];
        neon::pack_k(&xq.k, nv, &mut pk);
        let mut y = vec![0f32; nv * rows * 4];
        let yp = y.as_mut_ptr() as usize;
        for (label, mats) in [("RAM 256", nm), ("cache 4", 4usize)] {
            let tasks = nm * rows / 16;
            let run = || {
                let next = AtomicUsize::new(0);
                let (lay, pk) = (&lay[..], &pk[..]);
                std::thread::scope(|s| {
                    for th in 0..8usize {
                        #[cfg(not(windows))]
                        let _ = th;
                        let next = &next;
                        s.spawn(move || {
                            #[cfg(windows)]
                            // SAFETY: the calling thread's own affinity
                            unsafe {
                                SetThreadAffinityMask(GetCurrentThread(), 1usize << th)
                            };
                            loop {
                                let t = next.fetch_add(1, Ordering::Relaxed);
                                if t >= tasks {
                                    break;
                                }
                                let (m, r0) = ((t / (rows / 16)) % mats, t % (rows / 16) * 16);
                                let at = m * mb + r0 / 4 * quad;
                                // SAFETY: I8MM checked; 16 rows = 4 quads of matrix m; y rows (m % 4) x rows + r0..
                                unsafe {
                                    neon::gemm_q45k_lay(
                                        false,
                                        &lay[at..at + 4 * quad],
                                        16,
                                        pk,
                                        nv,
                                        nb,
                                        (yp as *mut f32).add((m % 4) * rows + r0),
                                        rows * 4,
                                    )
                                };
                            }
                        });
                    }
                });
            };
            run();
            let t = std::time::Instant::now();
            let mut k = 0;
            while t.elapsed().as_secs_f64() < 1.0 {
                run();
                k += 1;
            }
            let s = t.elapsed().as_secs_f64() / k as f64;
            let macs = (nm * rows * nb * QK * nv) as f64;
            println!(
                "nv {nv:2} {label}: {:6.1} GMAC/s, {:5.1} GB/s of weights",
                macs / s / 1e9,
                (nm * mb) as f64 / s / 1e9
            );
        }
    }
}

/// x86-64: the Q4_0 single-vector rows, and the 8-row VNNI passes where the CPU has them, give the multi-vector
/// kernels' bits.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
#[test]
fn x86_kernels_match() {
    let mut rng = Rng(61);
    let vnni = std::arch::is_x86_feature_detected!("avxvnni");
    for (nb, rows) in [(2usize, 64usize), (8, 40)] {
        let (w, rb) = weights(&mut rng, 2, rows, nb);
        let xq = quantized(&mut rng, 1, nb);
        let mut y = vec![0f32; rows];
        // SAFETY: the build's target features; y holds rows
        unsafe { kernels::x86dot::q4_0_rows(&w, rb, rows, &xq.g, &mut y) };
        assert_eq!(bits(&y), bits(&matmul(2, &w, rb, rows, xq.view(), 1)), "q4_0_rows nb {nb}");
        if !vnni {
            continue;
        }
        for (ty, name) in [
            (2u32, "q4_0_gemm8"),
            (8, "q8_0_gemm8"),
            (14, "q6k_gemm8"),
            (12, "q45k_gemm8 Q4_K"),
            (13, "q45k_gemm8 Q5_K"),
        ] {
            let (w, rb) = weights(&mut rng, ty, rows, nb);
            for nv in [1usize, 3, 4, 5, 8, 9, 17] {
                let xq = quantized(&mut rng, nv, nb);
                let mut y = vec![0f32; nv * rows];
                let r8 = rows / 8 * 8;
                // SAFETY: AVX-VNNI checked; y holds nv x rows; the kernels take whole groups of 8 rows
                unsafe {
                    match ty {
                        2 => kernels::x86::q4_0_gemm8(&w[..r8 * rb], rb, r8, &xq.g, nv, y.as_mut_ptr(), rows),
                        8 => kernels::x86::q8_0_gemm8(&w[..r8 * rb], rb, r8, &xq.g, nv, y.as_mut_ptr(), rows),
                        12 | 13 => {
                            kernels::x86::q45k_gemm8(ty == 13, &w[..r8 * rb], rb, r8, &xq.k, nv, y.as_mut_ptr(), rows)
                        }
                        _ => kernels::x86::q6k_gemm8(&w[..r8 * rb], rb, r8, &xq.k, nv, y.as_mut_ptr(), rows),
                    }
                }
                let want = matmul(ty, &w, rb, rows, xq.view(), nv);
                for v in 0..nv {
                    assert_eq!(
                        bits(&y[v * rows..v * rows + r8]),
                        bits(&want[v * rows..v * rows + r8]),
                        "{name} nb {nb} nv {nv} vector {v}"
                    );
                }
            }
        }
    }
}

/// x86-64: the packed GEMM (x86_gemm, VNNI in the form the CPU has: AVX-VNNI or AVX512-VNNI) gives the
/// multi-vector kernels' bits for Q4_0, Q8_0 and MXFP4.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
#[test]
fn x86_packed_gemm_matches() {
    if kernels::x86mm::vnni() == 0 {
        return;
    }
    let mut rng = Rng(67);
    for (nb, rows) in [(2usize, 64usize), (8, 40)] {
        for (ty, name) in [(2u32, "gemm_q4_0"), (8, "gemm_q8_0"), (39, "gemm_mxfp4")] {
            let (w, rb) = weights(&mut rng, ty, rows, nb);
            for nv in [1usize, 3, 4, 5, 8, 9, 17] {
                let xq = quantized(&mut rng, nv, nb);
                let mut y = vec![0f32; nv * rows];
                let r8 = rows / 8 * 8;
                // SAFETY: VNNI checked; y holds nv x rows; the kernel takes whole groups of 8 rows
                unsafe {
                    match ty {
                        2 => kernels::x86_gemm::gemm_q4_0(&w[..r8 * rb], rb, r8, &xq.g, nv, y.as_mut_ptr(), rows),
                        8 => kernels::x86_gemm::gemm_q8_0(&w[..r8 * rb], rb, r8, &xq.g, nv, y.as_mut_ptr(), rows),
                        _ => kernels::x86_gemm::gemm_mxfp4(&w[..r8 * rb], rb, r8, &xq.g, nv, y.as_mut_ptr(), rows),
                    }
                }
                let want = matmul(ty, &w, rb, rows, xq.view(), nv);
                for v in 0..nv {
                    assert_eq!(
                        bits(&y[v * rows..v * rows + r8]),
                        bits(&want[v * rows..v * rows + r8]),
                        "{name} nb {nb} nv {nv} vector {v}"
                    );
                }
            }
        }
    }
}

/// The Q8_K activation quantiser: the native path gives the scalar definition's bits.
#[test]
fn q8k_quantize_matches_scalar() {
    let mut rng = Rng(0x2545f4914f6cdd1d);
    for nb in [1, 4, 16, 40] {
        let x = activations(&mut rng, nb);
        let mut a = Q8::new(nb);
        let mut b = Q8::new(nb);
        kernels::q8k_quantize(&x, a.view_mut());
        kernels::q8k_quantize_scalar(&x, b.view_mut());
        for i in 0..nb {
            let (p, q, pg, qg) = (&a.k[i], &b.k[i], &a.g[i], &b.g[i]);
            assert_eq!(p.d.to_bits(), q.d.to_bits(), "nb {nb} block {i} d");
            assert_eq!(pg.d.map(f32::to_bits), qg.d.map(f32::to_bits), "nb {nb} block {i} d32");
            assert_eq!((p.qs, p.bsums, pg.qs, pg.sums), (q.qs, q.bsums, qg.qs, qg.sums), "nb {nb} block {i} ints");
        }
    }
}

/// The scalar Q4_0 dot (kernels::q4_0_dot_scalar) and the dispatching one agree, and both are within rounding of
/// the reference.
#[test]
fn q4_0_dot_scalar_matches() {
    let mut rng = Rng(71);
    for nb in [1usize, 2, 8, 20] {
        let xq = quantized(&mut rng, 1, nb);
        for _ in 0..32 {
            let (w, rb) = weights(&mut rng, 2, 1, nb);
            let a = kernels::q4_0_dot_scalar(&w, &xq.g);
            let b = kernels::q4_0_dot(&w, &xq.g);
            assert_eq!(a.to_bits(), b.to_bits(), "nb {nb}");
            let (want, mag) = reference(2, &w, rb, 1, xq.view(), 1);
            assert_close(2, "scalar", &[a], &want, &mag);
        }
    }
}

/// An ignored benchmark: one thread, the same shape (512 rows of 2048, nv vectors), GMAC/s of gemm_q45k (Q4_K /
/// Q5_K, each 4-row block unpacked once per call) against gemm_q4x4 (Q4_0 repacked once ahead of time, the layout
/// with nothing left to unpack but a mask and a shift): how much a repacked K-quant layout could gain at most.
/// cargo test --release -p torc-kernels --test parity bench_q45k -- --ignored --nocapture
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore]
fn bench_q45k() {
    if !native() || !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    let mut rng = Rng(7);
    let (rows, nb) = (512usize, 8usize);
    let time = |f: &mut dyn FnMut()| {
        f();
        let t = std::time::Instant::now();
        let mut n = 0u32;
        while t.elapsed().as_secs_f64() < 0.4 {
            f();
            n += 1;
        }
        t.elapsed().as_secs_f64() / n as f64
    };
    for nv in [1usize, 2, 3, 4, 8, 16, 32, 64] {
        let xq = quantized(&mut rng, nv, nb);
        let mut y = vec![0f32; nv * rows];
        let macs = (rows * nb * QK * nv) as f64;
        let mut line = format!("nv {nv:3}:");
        for ty in [12u32, 13] {
            let (w, rb) = weights(&mut rng, ty, rows, nb);
            let mut pk = vec![kernels::neon::PackedK::default(); kernels::neon::pack_k_len(nv, nb)];
            kernels::neon::pack_k(&xq.k, nv, &mut pk);
            // SAFETY: I8MM checked; rows a multiple of 4; y holds nv x rows
            let s = time(&mut || unsafe {
                kernels::neon::gemm_q45k(ty == 13, &w, rb, rows, &pk, nv, nb, y.as_mut_ptr(), rows)
            });
            line += &format!("  {} {:6.1}", if ty == 12 { "Q4_K" } else { "Q5_K" }, macs / s / 1e9);
            // SAFETY: DotProd (native); y holds nv x rows
            let s = time(&mut || unsafe {
                kernels::neon::q45k_matmul(ty == 13, &w, rb, rows, &xq.k, nv, y.as_mut_ptr(), rows)
            });
            line += &format!(" matmul {:6.1}", macs / s / 1e9);
            // the laid-out copy (q45k_lay4 once, its time apart) under gemm_q45k_lay
            let five = ty == 13;
            let quad = nb * kernels::neon::q45k_lay_bytes(five);
            let mut lay = vec![0u8; rows / 4 * quad];
            let lt = time(&mut || {
                for r in (0..rows).step_by(4) {
                    kernels::neon::q45k_lay4(five, &w[r * rb..], rb, nb, &mut lay[r / 4 * quad..(r / 4 + 1) * quad]);
                }
            });
            // SAFETY: as above; lay holds rows / 4 laid-out quads
            let s = time(&mut || unsafe {
                kernels::neon::gemm_q45k_lay(five, &lay, rows, &pk, nv, nb, y.as_mut_ptr(), rows)
            });
            line += &format!(" lay {:6.1} (layout {:5.0} us)", macs / s / 1e9, lt * 1e6);
            if nv <= 3 {
                // SAFETY: DotProd (native); as above
                let s = time(&mut || unsafe {
                    kernels::neon::q45k_lay_matmul(five, &lay, rows, &xq.k, nv, y.as_mut_ptr(), rows)
                });
                line += &format!(" lay-matmul {:6.1}", macs / s / 1e9);
            }
        }
        let (w, rb) = weights(&mut rng, 2, rows, nb);
        let quad = nb * 8 * kernels::neon::Q4X4_GROUP;
        let mut rp = vec![0u8; rows / 4 * quad];
        for r in (0..rows).step_by(4) {
            kernels::neon::q4_0_repack4(&w[r * rb..], rb, nb, &mut rp[r / 4 * quad..(r / 4 + 1) * quad]);
        }
        let mut pk = vec![kernels::neon::Packed::default(); kernels::neon::pack_len(nv, nb)];
        kernels::neon::mmla_pack(&xq.g, nv, &mut pk);
        // SAFETY: I8MM checked; rp holds rows / 4 repacked quads; y holds nv x rows
        let s = time(&mut || unsafe { kernels::neon::gemm_q4x4(&rp, rows, &pk, nv, nb, y.as_mut_ptr(), rows) });
        line += &format!("  Q4_0x4 {:6.1}  (GMAC/s)", macs / s / 1e9);
        println!("{line}");
    }
}

/// An ignored benchmark: Q6_K rows (512 x 2048) one thread, GMAC/s: gemm_q6k and q6k_matmul over the source,
/// gemm_q6_lay and q6_lay_matmul over q6_lay4's layout (the same size as the source). cargo test --release -p
/// torc-kernels --test parity bench_q6 -- --ignored --nocapture
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore]
fn bench_q6() {
    if !native() || !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    use kernels::neon;
    let mut rng = Rng(9);
    let (rows, nb) = (512usize, 8usize);
    let time = |f: &mut dyn FnMut()| {
        f();
        let t = std::time::Instant::now();
        let mut n = 0u32;
        while t.elapsed().as_secs_f64() < 0.4 {
            f();
            n += 1;
        }
        t.elapsed().as_secs_f64() / n as f64
    };
    let (w, rb) = weights(&mut rng, 14, rows, nb);
    let quad = nb * neon::Q6_LAY_BYTES;
    let mut lay = vec![0u8; rows / 4 * quad];
    let lt = time(&mut || {
        for r in (0..rows).step_by(4) {
            neon::q6_lay4(&w[r * rb..], rb, nb, &mut lay[r / 4 * quad..(r / 4 + 1) * quad]);
        }
    });
    println!("Q6_K layout {:.0} us per matrix", lt * 1e6);
    for nv in [1usize, 2, 3, 4, 8, 16, 32, 64] {
        let xq = quantized(&mut rng, nv, nb);
        let mut y = vec![0f32; nv * rows];
        let macs = (rows * nb * QK * nv) as f64;
        let mut pk = vec![neon::PackedQ6::default(); neon::pack_q6_len(nv, nb)];
        neon::pack_q6(&xq.k, nv, &mut pk);
        // SAFETY: I8MM checked; rows a multiple of 4; y holds nv x rows; lay holds rows / 4 quads
        let g = time(&mut || unsafe { neon::gemm_q6k(&w, rb, rows, &pk, nv, nb, y.as_mut_ptr(), rows) });
        let d = time(&mut || unsafe { neon::q6k_matmul(&w, rb, rows, &xq.k, nv, y.as_mut_ptr(), rows) });
        let l = time(&mut || unsafe { neon::gemm_q6_lay(&lay, rows, &pk, nv, nb, y.as_mut_ptr(), rows) });
        let mut line = format!(
            "nv {nv:3}: Q6_K gemm {:6.1} matmul {:6.1} lay {:6.1}",
            macs / g / 1e9,
            macs / d / 1e9,
            macs / l / 1e9
        );
        if nv <= 3 {
            // SAFETY: as above
            let s = time(&mut || unsafe { neon::q6_lay_matmul(&lay, rows, &xq.k, nv, y.as_mut_ptr(), rows) });
            line += &format!(" lay-matmul {:6.1}", macs / s / 1e9);
        }
        println!("{line}  (GMAC/s)");
    }
}

/// An ignored benchmark: one vector against many matrices that cannot stay in cache (256 Q4_K / Q5_K matrices of
/// 512 x 2048, ~150 / 180 MB, split over 8 threads): the decode case, where the weights stream from RAM.
/// q45k_matmul over the source against q45k_lay_matmul over the layout, GB/s of weights read and GMAC/s.
/// cargo test --release -p torc-kernels --test parity bench_q45k_ram -- --ignored --nocapture
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore]
fn bench_q45k_ram() {
    bench_q45k_ram_body();
}

/// An ignored benchmark: q45k_lay_matmul, 1 vector, Q4_K and Q5_K, T pinned threads (1, 2, 4, 6, 8) over the same work
/// from RAM (256 matrices) and from cache (4 matrices, reread): whether Q5_K's flat 4-to-8-thread rate is the memory
/// side (cache scales, RAM does not) or the cores (both flat). cargo test --release -p torc-kernels --test parity
/// bench_q45k_q5_scale -- --ignored --nocapture
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore]
fn bench_q45k_q5_scale() {
    if !native() || !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(windows)]
    unsafe extern "system" {
        fn SetThreadAffinityMask(h: *mut core::ffi::c_void, mask: usize) -> usize;
        fn GetCurrentThread() -> *mut core::ffi::c_void;
    }
    let (rows, nb, nm) = (512usize, 8usize, 256usize);
    // TORC_BENCH_OFF=o: thread th pinned to CPU (th + o) % 8 (which cores do the work)
    let off: usize = kernels::switches::num("TORC_BENCH_OFF").unwrap_or(0);
    #[cfg(not(windows))]
    let _ = off;
    let mut rng = Rng(23);
    let xq = quantized(&mut rng, 1, nb);
    for ty in [12u32, 13] {
        let five = ty == 13;
        let quad = nb * kernels::neon::q45k_lay_bytes(five);
        let (w, rb) = weights(&mut rng, ty, rows, nb);
        let mut one = vec![0u8; rows / 4 * quad];
        for r in (0..rows).step_by(4) {
            kernels::neon::q45k_lay4(five, &w[r * rb..], rb, nb, &mut one[r / 4 * quad..(r / 4 + 1) * quad]);
        }
        let lay: Vec<u8> = (0..nm).flat_map(|_| one.iter().copied()).collect();
        let mb = one.len();
        let mut line = format!("{}:", if five { "Q5_K" } else { "Q4_K" });
        for (label, mats) in [("RAM", nm), ("cache", 4usize)] {
            line += &format!("  {label}");
            for nt in [1usize, 2, 4, 6, 8] {
                // 16 passes over the work a spawn, so the threads' start-up is a small part of a run
                let (reps, per) = (16usize, nm * rows / 16);
                let tasks = reps * per;
                let run = || {
                    let next = AtomicUsize::new(0);
                    let lay = &lay[..];
                    std::thread::scope(|s| {
                        for th in 0..nt {
                            #[cfg(not(windows))]
                            let _ = th;
                            let (next, xq) = (&next, &xq);
                            s.spawn(move || {
                                #[cfg(windows)]
                                // SAFETY: the calling thread's own affinity
                                unsafe {
                                    SetThreadAffinityMask(GetCurrentThread(), 1usize << ((th + off) % 8))
                                };
                                let mut y = [0f32; 16];
                                loop {
                                    let t = next.fetch_add(1, Ordering::Relaxed);
                                    if t >= tasks {
                                        break;
                                    }
                                    let t = t % per;
                                    let (m, r0) = ((t / (rows / 16)) % mats, t % (rows / 16) * 16);
                                    let at = m * mb + r0 / 4 * quad;
                                    // SAFETY: DotProd (native); 16 rows = 4 quads of matrix m; y holds 16
                                    unsafe {
                                        kernels::neon::q45k_lay_matmul(
                                            five,
                                            &lay[at..at + 4 * quad],
                                            16,
                                            &xq.k,
                                            1,
                                            y.as_mut_ptr(),
                                            16,
                                        )
                                    };
                                }
                                std::hint::black_box(&y);
                            });
                        }
                    });
                };
                run();
                let t = std::time::Instant::now();
                let mut k = 0;
                while t.elapsed().as_secs_f64() < 0.8 {
                    run();
                    k += 1;
                }
                let s = t.elapsed().as_secs_f64() / k as f64;
                line += &format!(" {nt}:{:5.1}", (reps * nm * rows * nb * QK) as f64 / s / 1e9);
            }
        }
        println!("{line}  (GMAC/s by pinned threads)");
    }
}

#[cfg(target_arch = "aarch64")]
fn bench_q45k_ram_body() {
    if !native() || !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    let (rows, nb, nm) = (512usize, 8usize, 256usize);
    // TORC_BENCH_NT=t: threads (default 8; nm must divide by it)
    let nt: usize = kernels::switches::num("TORC_BENCH_NT").unwrap_or(8);
    assert!(nt > 0 && nm % nt == 0, "TORC_BENCH_NT={nt}: the {nm} matrices must split evenly over the threads");
    let mut rng = Rng(11);
    let xq = quantized(&mut rng, 1, nb);
    for ty in [12u32, 13] {
        let five = ty == 13;
        let quad = nb * kernels::neon::q45k_lay_bytes(five);
        let mats: Vec<(Vec<u8>, usize, Vec<u8>)> = (0..nm)
            .map(|_| {
                let (w, rb) = weights(&mut rng, ty, rows, nb);
                let mut lay = vec![0u8; rows / 4 * quad];
                for r in (0..rows).step_by(4) {
                    kernels::neon::q45k_lay4(five, &w[r * rb..], rb, nb, &mut lay[r / 4 * quad..(r / 4 + 1) * quad]);
                }
                (w, rb, lay)
            })
            .collect();
        let run = |lay: bool| {
            let t = std::time::Instant::now();
            let mut passes = 0;
            while t.elapsed().as_secs_f64() < 1.0 {
                std::thread::scope(|s| {
                    for c in mats.chunks(nm / nt) {
                        let xq = &xq;
                        s.spawn(move || {
                            let mut y = vec![0f32; rows];
                            for (w, rb, l) in c {
                                // SAFETY: DotProd (native); y holds rows
                                unsafe {
                                    if lay {
                                        kernels::neon::q45k_lay_matmul(five, l, rows, &xq.k, 1, y.as_mut_ptr(), rows)
                                    } else {
                                        kernels::neon::q45k_matmul(five, w, *rb, rows, &xq.k, 1, y.as_mut_ptr(), rows)
                                    }
                                }
                            }
                            std::hint::black_box(&y);
                        });
                    }
                });
                passes += 1;
            }
            t.elapsed().as_secs_f64() / passes as f64
        };
        let (src_b, lay_b) = (
            mats.iter().map(|m| m.0.len()).sum::<usize>() as f64,
            mats.iter().map(|m| m.2.len()).sum::<usize>() as f64,
        );
        let macs = (nm * rows * nb * QK) as f64;
        let (s0, s1) = (run(false), run(true));
        println!(
            "{} RAM, 1 vector, {} threads: source {:5.1} GB/s {:5.1} GMAC/s | layout {:5.1} GB/s {:5.1} GMAC/s ({:.0} vs {:.0} MB)",
            if five { "Q5_K" } else { "Q4_K" },
            nt,
            src_b / s0 / 1e9,
            macs / s0 / 1e9,
            lay_b / s1 / 1e9,
            macs / s1 / 1e9,
            src_b / 1e6,
            lay_b / 1e6
        );
    }
}

/// An ignored benchmark: x86-64 Q4_K / Q5_K rows (512 x 2048) one thread, GMAC/s: x86mm (a row's block for up to 4
/// vectors at a time) against q45k_gemm8 (8 rows in the lanes, AVX-VNNI). cargo test --release -p torc-kernels --test
/// parity bench_q45k_x86 -- --ignored --nocapture
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
#[test]
#[ignore]
fn bench_q45k_x86() {
    if !std::arch::is_x86_feature_detected!("avxvnni") {
        return;
    }
    let mut rng = Rng(5);
    let (rows, nb) = (512usize, 8usize);
    let time = |f: &mut dyn FnMut()| {
        f();
        let t = std::time::Instant::now();
        let mut n = 0u32;
        while t.elapsed().as_secs_f64() < 0.4 {
            f();
            n += 1;
        }
        t.elapsed().as_secs_f64() / n as f64
    };
    for nv in [1usize, 2, 4, 8, 16, 32, 64] {
        let xq = quantized(&mut rng, nv, nb);
        let mut y = vec![0f32; nv * rows];
        let macs = (rows * nb * QK * nv) as f64;
        let mut line = format!("nv {nv:3}:");
        for ty in [12u32, 13] {
            let (w, rb) = weights(&mut rng, ty, rows, nb);
            let d = time(&mut || {
                // SAFETY: the build's target features; y holds nv x rows
                unsafe { kernels::x86mm::matmul(ty, &w, rb, rows, xq.view(), nv, y.as_mut_ptr(), rows) }
            });
            // SAFETY: AVX-VNNI checked; rows a multiple of 8; y holds nv x rows
            let g = time(&mut || unsafe {
                kernels::x86::q45k_gemm8(ty == 13, &w, rb, rows, &xq.k, nv, y.as_mut_ptr(), rows)
            });
            line += &format!(
                "  {} matmul {:6.1} gemm8 {:6.1}",
                if ty == 12 { "Q4_K" } else { "Q5_K" },
                macs / d / 1e9,
                macs / g / 1e9
            );
        }
        println!("{line}  (GMAC/s)");
    }
}

/// IQ2_S's reference dequantiser against a line-by-line copy of ggml's dequantize_row_iq2_s (ggml-quants.c), which
/// walks the block with moving qs / signs pointers instead of the indexed helpers the kernels share.
#[test]
fn iq2s_dequant_matches_ggml_loop() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let (w, rb) = weights(&mut rng, 22, 5, 3);
    for r in 0..5 {
        let row = &w[r * rb..(r + 1) * rb];
        let mut got = vec![0f32; 3 * QK];
        assert!(kernels::dequant::dequantize(22, row, &mut got));
        let mut want = Vec::new();
        for blk in row.chunks_exact(82) {
            let d = kernels::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let (mut qs, mut signs) = (2usize, 2 + 32);
            let qh = 66;
            for ib32 in 0..8 {
                let sc = blk[74 + ib32];
                let db = [d * (0.5 + (sc & 15) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25];
                for l in 0..4 {
                    let dl = db[l / 2];
                    let idx = blk[qs + l] as usize | (((blk[qh + ib32] as usize) << (8 - 2 * l)) & 0x300);
                    let grid = kernels::iq2s_grid::IQ2S_GRID[idx].to_le_bytes();
                    for j in 0..8 {
                        want.push(dl * grid[j] as f32 * if blk[signs + l] & (1 << j) != 0 { -1.0 } else { 1.0 });
                    }
                }
                qs += 4;
                signs += 4;
            }
        }
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "row {r} weight {i}: {g} vs {w}");
        }
    }
}

/// Q2_0's reference dequantiser against ggml's dequantize_row_q2_0 as written there (levels 00=-1, 01=0, 10=+1, 11=+2).
#[test]
fn q2_0_dequant_matches_ggml_loop() {
    let mut rng = Rng(0x0fed_cba9_8765_4321);
    let (w, rb) = weights(&mut rng, 42, 5, 3);
    for r in 0..5 {
        let row = &w[r * rb..(r + 1) * rb];
        let mut got = vec![0f32; 3 * QK];
        assert!(kernels::dequant::dequantize(42, row, &mut got));
        let mut want = Vec::new();
        for blk in row.chunks_exact(18) {
            let d = kernels::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            for j in 0..64 {
                let q = (blk[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
                want.push((q as i32 - 1) as f32 * d);
            }
        }
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}

/// IQ3_XXS's reference dequantiser against ggml's dequantize_row_iq3_xxs as written there (moving qs pointer, the u32 of
/// signs and scale read with memcpy, the sign table indexed per run of 8).
#[test]
fn iq3xxs_dequant_matches_ggml_loop() {
    let mut rng = Rng(0x7777_1234_5678_9abc);
    let (w, rb) = weights(&mut rng, 18, 5, 3);
    for r in 0..5 {
        let row = &w[r * rb..(r + 1) * rb];
        let mut got = vec![0f32; 3 * QK];
        assert!(kernels::dequant::dequantize(18, row, &mut got));
        let mut want = Vec::new();
        for blk in row.chunks_exact(98) {
            let d = kernels::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let mut qs = 2usize;
            let sas = 2 + 64;
            for ib32 in 0..8 {
                let aux = u32::from_le_bytes(blk[sas + 4 * ib32..sas + 4 * ib32 + 4].try_into().unwrap());
                let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let signs = kernels::iq3xxs_grid::KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                    let g1 = kernels::iq3xxs_grid::IQ3XXS_GRID[blk[qs + 2 * l] as usize].to_le_bytes();
                    let g2 = kernels::iq3xxs_grid::IQ3XXS_GRID[blk[qs + 2 * l + 1] as usize].to_le_bytes();
                    let sg = |j: usize| if signs & (1 << j) != 0 { -1.0f32 } else { 1.0 };
                    let mut y = [0f32; 8];
                    for j in 0..4 {
                        y[j] = db * g1[j] as f32 * sg(j);
                        y[j + 4] = db * g2[j] as f32 * sg(j + 4);
                    }
                    want.extend_from_slice(&y);
                }
                qs += 8;
            }
        }
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "row {r} weight {i}: {g} vs {w}");
        }
    }
}
