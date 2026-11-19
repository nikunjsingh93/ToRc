//! neon.rs - the ARM64 kernels: SDOT (FEAT_DotProd) and SMMLA (FEAT_I8MM).
//!
//! Every kernel fixes its accumulator lane layout and the order of its final sum, so its result has the same bits
//! whatever the row grouping or the number of vectors (parity tests).

use crate::{BlockQ8K, Groups32, Q4_0_BLOCK, f16_to_f32};
use core::arch::aarch64::*;
use core::arch::asm;

/// SDOT: acc.4s += a.16b . b.16b (vdotq_s32 is still unstable in core::arch).
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    unsafe {
        asm!("sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
            options(pure, nomem, nostack));
    }
    acc
}

/// SDOT by element: acc.4s += a.16b . b.4b[I], i.e. each lane's 4 bytes of a against the 4 bytes in lane I of b.
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn sdot_lane<const I: i32>(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    unsafe {
        asm!("sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[{i}]", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
            i = const I, options(pure, nomem, nostack));
    }
    acc
}

/// A 4 x 4 transpose of 32-bit lanes: out[j] holds lane j of r0..r3.
#[inline]
unsafe fn tr4(r: [int8x16_t; 4]) -> [int8x16_t; 4] {
    unsafe {
        let (a, b, c, d) = (
            vreinterpretq_u32_s8(r[0]),
            vreinterpretq_u32_s8(r[1]),
            vreinterpretq_u32_s8(r[2]),
            vreinterpretq_u32_s8(r[3]),
        );
        let t0 = vtrn1q_u32(a, b);
        let t1 = vtrn2q_u32(a, b);
        let t2 = vtrn1q_u32(c, d);
        let t3 = vtrn2q_u32(c, d);
        let o0 = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
        let o1 = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
        let o2 = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
        let o3 = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
        [vreinterpretq_s8_u32(o0), vreinterpretq_s8_u32(o1), vreinterpretq_s8_u32(o2), vreinterpretq_s8_u32(o3)]
    }
}

/// Rows of Q4_0 against many vectors, 4 rows a register with one row in each lane (the same scheme as the x86
/// q4_0_gemm8). Each 32-weight sub-block of the 4 rows is unpacked (nibbles 0..15) and transposed once; then, for every
/// vector, SDOT by element builds each row's sub-block sum in its own lane (the sum of nibble x a, less 8 x the sum of
/// a, exactly as q4_0_dot does). The float steps for each (row, sub-block) are q4_0_dot's: d_w * g.d, fused onto that
/// lane's accumulator block after block, with the eight summed as ((0+4) + (2+6)) + ((1+5) + (3+7)). The result has the
/// same bits as q4_0_dot. rows must be a multiple of 4; the outputs go to y[v * y_stride + r].
///
/// # Safety
/// FEAT_DotProd; w holds rows * row_bytes and x holds nv * nb blocks; y is valid for writes at v * y_stride + r, and
/// nothing else touches those elements during the call.
#[target_feature(enable = "dotprod")]
pub unsafe fn q4_0_gemm4(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv;
    assert!(nv > 0 && x.len() == nv * nb && rows.is_multiple_of(4) && row_bytes >= nb * 8 * Q4_0_BLOCK);
    assert!(w.len() >= rows * row_bytes);
    unsafe {
        let m15 = vdupq_n_u8(15);
        let mut acc = vec![vdupq_n_f32(0.0); nv * 8];
        for r0 in (0..rows).step_by(4) {
            acc.fill(vdupq_n_f32(0.0));
            for b in 0..nb {
                for k in 0..8 {
                    let at = |i: usize| w.as_ptr().add((r0 + i) * row_bytes + (b * 8 + k) * Q4_0_BLOCK);
                    let d = |i: usize| f16_to_f32(u16::from_le_bytes([*at(i), *at(i).add(1)]));
                    let dv = [d(0), d(1), d(2), d(3)];
                    let dk = vld1q_f32(dv.as_ptr());
                    // row i: weights 0-15 (the low nibbles) and 16-31 (the high ones), as signed bytes 0..15
                    let lo = |i: usize| vreinterpretq_s8_u8(vandq_u8(vld1q_u8(at(i).add(2)), m15));
                    let hi = |i: usize| vreinterpretq_s8_u8(vshrq_n_u8::<4>(vld1q_u8(at(i).add(2))));
                    let tl = tr4([lo(0), lo(1), lo(2), lo(3)]);
                    let th = tr4([hi(0), hi(1), hi(2), hi(3)]);
                    for v in 0..nv {
                        let xb = &x[v * nb + b];
                        let a = xb.qs.as_ptr().add(32 * k);
                        let (a0, a1) = (vld1q_s8(a), vld1q_s8(a.add(16)));
                        let mut s = vdupq_n_s32(0);
                        s = sdot_lane::<0>(s, tl[0], a0);
                        s = sdot_lane::<1>(s, tl[1], a0);
                        s = sdot_lane::<2>(s, tl[2], a0);
                        s = sdot_lane::<3>(s, tl[3], a0);
                        s = sdot_lane::<0>(s, th[0], a1);
                        s = sdot_lane::<1>(s, th[1], a1);
                        s = sdot_lane::<2>(s, th[2], a1);
                        s = sdot_lane::<3>(s, th[3], a1);
                        // the nibbles less 8, as in q4_0_dot: the i16 pair sum, widened and multiplied by 8
                        let bias = (xb.sums[k] as i32) << 3;
                        s = vsubq_s32(s, vdupq_n_s32(bias));
                        let ac = acc.get_unchecked_mut(v * 8 + k);
                        *ac = vfmaq_f32(*ac, vmulq_f32(dk, vdupq_n_f32(xb.d[k])), vcvtq_f32_s32(s));
                    }
                }
            }
            for v in 0..nv {
                let a = &acc[v * 8..v * 8 + 8];
                let r = vaddq_f32(
                    vaddq_f32(vaddq_f32(a[0], a[4]), vaddq_f32(a[2], a[6])),
                    vaddq_f32(vaddq_f32(a[1], a[5]), vaddq_f32(a[3], a[7])),
                );
                let mut o = [0.0f32; 4];
                vst1q_f32(o.as_mut_ptr(), r);
                for (i, &val) in o.iter().enumerate() {
                    *y.add(v * y_stride + r0 + i) = val;
                }
            }
        }
    }
}

/// SMMLA (FEAT_I8MM): acc.4s += A . B^T, where A is the 2 x 8 bytes of a (its halves are rows 0 and 1) and B those of
/// b. The lanes are [a0.b0, a0.b1, a1.b0, a1.b1]: 32 products an instruction, against SDOT's 16.
#[inline]
#[target_feature(enable = "i8mm")]
pub(crate) unsafe fn smmla(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    unsafe {
        asm!("smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
            options(pure, nomem, nostack));
    }
    acc
}

/// FCVTL: 4 fp16 values (as raw bits) widened exactly to f32 (float16x4_t is still unstable in core::arch).
#[inline]
unsafe fn fcvtl(h: uint16x4_t) -> float32x4_t {
    let o: float32x4_t;
    unsafe {
        asm!("fcvtl {o:v}.4s, {h:v}.4h", o = out(vreg) o, h = in(vreg) h, options(pure, nomem, nostack));
    }
    o
}

/// The halves of two registers paired up for SMMLA: [a.lo, b.lo] and [a.hi, b.hi].
#[inline]
unsafe fn pair64(a: int8x16_t, b: int8x16_t) -> (int8x16_t, int8x16_t) {
    unsafe {
        let (a, b) = (vreinterpretq_u64_s8(a), vreinterpretq_u64_s8(b));
        (vreinterpretq_s8_u64(vzip1q_u64(a, b)), vreinterpretq_s8_u64(vzip2q_u64(a, b)))
    }
}

/// A weight format for gemm_mmla: one 32-weight block as two registers of signed bytes (weights 0-15 and 16-31) plus
/// its fp16 scale. BIAS means the bytes are the nibbles taken as unsigned, so 8 x the activations' sum must be taken
/// off.
pub trait MmlaBlock {
    const SIZE: usize;
    const BIAS: bool;
    /// # Safety
    /// p must be valid for SIZE bytes.
    unsafe fn load(p: *const u8) -> (int8x16_t, int8x16_t);
}

pub struct Q40;
impl MmlaBlock for Q40 {
    const SIZE: usize = Q4_0_BLOCK;
    const BIAS: bool = true;
    #[inline]
    unsafe fn load(p: *const u8) -> (int8x16_t, int8x16_t) {
        unsafe {
            let q = vld1q_u8(p.add(2));
            (vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(15))), vreinterpretq_s8_u8(vshrq_n_u8::<4>(q)))
        }
    }
}

pub struct Q80;
impl MmlaBlock for Q80 {
    const SIZE: usize = 34;
    const BIAS: bool = false;
    #[inline]
    unsafe fn load(p: *const u8) -> (int8x16_t, int8x16_t) {
        unsafe { (vld1q_s8(p.add(2) as *const i8), vld1q_s8(p.add(18) as *const i8)) }
    }
}

/// IQ4_NL: Q4_0's block layout, with the nibbles looked up in its 16 levels and no bias (ggml-org's Flash Next
/// experts).
pub struct IQ4NL;
impl MmlaBlock for IQ4NL {
    const SIZE: usize = Q4_0_BLOCK;
    const BIAS: bool = false;
    #[inline]
    unsafe fn load(p: *const u8) -> (int8x16_t, int8x16_t) {
        unsafe {
            let lv = vld1q_s8(IQ4NL_LEVELS.as_ptr());
            let q = vld1q_u8(p.add(2));
            (vqtbl1q_s8(lv, vandq_u8(q, vdupq_n_u8(15))), vqtbl1q_s8(lv, vshrq_n_u8::<4>(q)))
        }
    }
}

/// One pair of vectors' sub-block, packed once for all the rows (and for all the calls over the same vectors): the
/// activations as SMMLA's B operand (bytes 0-7, 8-15, 16-23 and 24-31 of each), the activations' sum (the i16 pair sum,
/// widened) and g.d, with the lanes [v0, v1, v0, v1].
#[derive(Clone, Copy)]
pub struct Packed {
    b: [int8x16_t; 4],
    sum: int32x4_t,
    d: float32x4_t,
}

impl Default for Packed {
    fn default() -> Packed {
        // SAFETY: plain vectors, for which all-zero bits are valid
        unsafe { core::mem::zeroed() }
    }
}

/// The number of packed entries for nv vectors of nb blocks.
pub fn pack_len(nv: usize, nb: usize) -> usize {
    nv.div_ceil(2) * nb * 8
}

/// Packs x (nv x nb blocks) for gemm_mmla_packed: entry (p * nb + b) * 8 + k holds vectors 2p and 2p + 1 (an odd last
/// vector is paired with itself). out holds pack_len(nv, nb) entries.
pub fn mmla_pack(x: &[Groups32], nv: usize, out: &mut [Packed]) {
    let nb = x.len() / nv;
    assert!(nv > 0 && x.len() == nv * nb && out.len() >= pack_len(nv, nb));
    unsafe {
        for p in 0..nv.div_ceil(2) {
            let (v0, v1) = (2 * p, (2 * p + 1).min(nv - 1));
            for b in 0..nb {
                let (x0, x1) = (&x[v0 * nb + b], &x[v1 * nb + b]);
                for k in 0..8 {
                    let (a0, a1) = (x0.qs.as_ptr().add(32 * k), x1.qs.as_ptr().add(32 * k));
                    let (b0, b1) = pair64(vld1q_s8(a0), vld1q_s8(a1));
                    let (b2, b3) = pair64(vld1q_s8(a0.add(16)), vld1q_s8(a1.add(16)));
                    let sum = |xb: &Groups32| xb.sums[k] as i32;
                    let (s0, s1) = (sum(x0), sum(x1));
                    let sv = [s0, s1, s0, s1];
                    let dv = [x0.d[k], x1.d[k], x0.d[k], x1.d[k]];
                    out[(p * nb + b) * 8 + k] =
                        Packed { b: [b0, b1, b2, b3], sum: vld1q_s32(sv.as_ptr()), d: vld1q_f32(dv.as_ptr()) };
                }
            }
        }
    }
}

/// Rows against many vectors by SMMLA: a tile of 2 rows x 2 vectors per instruction, and 4 rows (two tiles) a pass.
/// Each (row, vector, sub-block) integer sum is exact, as with SDOT. The float steps for each (row, vector, sub-block)
/// are q4_0_dot's (and q8_0_dotr's): d_w * g.d, fused onto its own accumulator block after block, with the eight summed
/// as ((0+4) + (2+6)) + ((1+5) + (3+7)). The result has the same bits as the single-vector products; only the lanes
/// hold (row, vector) pairs. An odd last vector's copy is dropped. rows must be a multiple of 4; the outputs go to
/// y[v * y_stride + r].
///
/// # Safety
/// FEAT_I8MM; w holds rows * row_bytes; pk is mmla_pack of nv vectors of nb blocks; y is valid for writes at v *
/// y_stride + r, and nothing else touches those elements during the call.
#[target_feature(enable = "i8mm")]
pub unsafe fn gemm_mmla_packed<B: MmlaBlock>(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    pk: &[Packed],
    nv: usize,
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    // 8 rows a pass from TORC_MMLA8_MIN vectors (default 16), so each packed entry serves four tiles instead of two.
    // Below that, 8 rows cost more in register pressure than they save
    static MIN8: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_MMLA8_MIN").unwrap_or(16));
    let r8 = if nv >= *MIN8 { rows / 8 * 8 } else { 0 };
    // TORC_VPANEL=P (default 0 = off): take the vectors P pairs at a time OUTSIDE the rows, so a panel's
    // packed vectors are read for every row of the call while they are still in L1, instead of once per 8 rows from L2
    // (a prompt block's 512 vectors pack to 3 MB). Each (row, vector) goes through the same steps in the same order, so
    // the bits are the same
    static VPANEL: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_VPANEL").unwrap_or(0));
    let np = nv.div_ceil(2);
    if *VPANEL > 0 && np > *VPANEL {
        for p0 in (0..np).step_by(*VPANEL) {
            let p1 = (p0 + *VPANEL).min(np);
            let nv1 = (2 * p1).min(nv) - 2 * p0;
            let pk1 = &pk[p0 * nb * 8..];
            unsafe {
                let y1 = y.add(2 * p0 * y_stride);
                if r8 > 0 {
                    gemm_mmla_tiles::<B, 4>(w, row_bytes, r8, pk1, nv1, nb, y1, y_stride);
                }
                if r8 < rows {
                    gemm_mmla_tiles::<B, 2>(
                        &w[r8 * row_bytes..],
                        row_bytes,
                        rows - r8,
                        pk1,
                        nv1,
                        nb,
                        y1.add(r8),
                        y_stride,
                    );
                }
            }
        }
        return;
    }
    unsafe {
        if r8 > 0 {
            gemm_mmla_tiles::<B, 4>(w, row_bytes, r8, pk, nv, nb, y, y_stride);
        }
        if r8 < rows {
            gemm_mmla_tiles::<B, 2>(&w[r8 * row_bytes..], row_bytes, rows - r8, pk, nv, nb, y.add(r8), y_stride);
        }
    }
}

/// gemm_mmla_packed's body: T tiles of 2 rows a pass (rows must be a multiple of 2T).
#[target_feature(enable = "i8mm")]
unsafe fn gemm_mmla_tiles<B: MmlaBlock, const T: usize>(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    pk: &[Packed],
    nv: usize,
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    assert!(nv > 0 && rows % (2 * T) == 0 && row_bytes >= nb * 8 * B::SIZE && pk.len() >= pack_len(nv, nb));
    assert!(w.len() >= rows * row_bytes);
    let np = nv.div_ceil(2);
    unsafe {
        // acc[(p * 8 + k) * T + t] is tile t (rows r0 + 2t and r0 + 2t + 1) of pair p at block position k. It lives on
        // the stack for up to 1024 vectors (there is one call per 16-row task, so no allocation each time), and on the
        // heap beyond that
        let mut stack = core::mem::MaybeUninit::<[float32x4_t; 1024]>::uninit();
        let mut heap = Vec::new();
        let acc: &mut [float32x4_t] = if np * 8 * T <= 1024 {
            crate::zeroed_prefix(&mut stack, np * 8 * T)
        } else {
            heap.resize(np * 8 * T, vdupq_n_f32(0.0));
            &mut heap
        };
        // the pairs go a panel at a time (TORC_MMLA_PANEL pairs, default 16; 0 takes them all at once), so a panel's
        // sums stay in L1 across all the blocks instead of streaming through it (with many vectors, as in a prompt's
        // dense products), at the cost of unpacking the rows' blocks again for each panel. The order of every sum is
        // unchanged, so the bits are the same
        static PANEL: std::sync::LazyLock<usize> =
            std::sync::LazyLock::new(|| crate::switches::num("TORC_MMLA_PANEL").unwrap_or(16));
        let panel = if *PANEL == 0 { np } else { (*PANEL).min(np) };
        for r0 in (0..rows).step_by(2 * T) {
            acc.fill(vdupq_n_f32(0.0));
            for p0 in (0..np).step_by(panel) {
                let p1 = (p0 + panel).min(np);
                for b in 0..nb {
                    for k in 0..8 {
                        let at = |i: usize| w.as_ptr().add((r0 + i) * row_bytes + (b * 8 + k) * B::SIZE);
                        // each tile's scales as [d_i d_i d_j d_j], widened 4 at a time (FCVTL, exact, as f16_to_f32)
                        let h = |i: usize| u16::from_le_bytes([*at(i), *at(i).add(1)]);
                        let mut dk = [vdupq_n_f32(0.0); T];
                        let mut a = [[vdupq_n_s8(0); 4]; T];
                        for q in 0..T / 2 {
                            let hv = [h(4 * q), h(4 * q + 1), h(4 * q + 2), h(4 * q + 3)];
                            let d = fcvtl(vld1_u16(hv.as_ptr()));
                            dk[2 * q] = vzip1q_f32(d, d);
                            dk[2 * q + 1] = vzip2q_f32(d, d);
                        }
                        for t in 0..T {
                            let (li, hi) = B::load(at(2 * t));
                            let (lj, hj) = B::load(at(2 * t + 1));
                            let (x0, x1) = pair64(li, lj);
                            let (x2, x3) = pair64(hi, hj);
                            a[t] = [x0, x1, x2, x3];
                        }
                        for p in p0..p1 {
                            let q = pk.get_unchecked((p * nb + b) * 8 + k);
                            let bias = vshlq_n_s32::<3>(q.sum);
                            let ac = acc.as_mut_ptr().add((p * 8 + k) * T);
                            for t in 0..T {
                                let mut s = smmla(vdupq_n_s32(0), a[t][0], q.b[0]);
                                s = smmla(s, a[t][1], q.b[1]);
                                s = smmla(s, a[t][2], q.b[2]);
                                s = smmla(s, a[t][3], q.b[3]);
                                if B::BIAS {
                                    s = vsubq_s32(s, bias);
                                }
                                *ac.add(t) = vfmaq_f32(*ac.add(t), vmulq_f32(dk[t], q.d), vcvtq_f32_s32(s));
                            }
                        }
                    }
                }
            }
            for p in 0..np {
                let (v0, v1) = (2 * p, 2 * p + 1);
                for t in 0..T {
                    let a = |k: usize| acc[(p * 8 + k) * T + t];
                    let r = vaddq_f32(
                        vaddq_f32(vaddq_f32(a(0), a(4)), vaddq_f32(a(2), a(6))),
                        vaddq_f32(vaddq_f32(a(1), a(5)), vaddq_f32(a(3), a(7))),
                    );
                    let mut o = [0.0f32; 4];
                    vst1q_f32(o.as_mut_ptr(), r);
                    let row = r0 + 2 * t;
                    *y.add(v0 * y_stride + row) = o[0];
                    *y.add(v0 * y_stride + row + 1) = o[2];
                    if v1 < nv {
                        *y.add(v1 * y_stride + row) = o[1];
                        *y.add(v1 * y_stride + row + 1) = o[3];
                    }
                }
            }
        }
    }
}

/// The bytes of one block position of 4 Q4_0 rows repacked for SMMLA (q4_0_repack4): the same as 4 blocks.
pub const Q4X4_GROUP: usize = 4 * Q4_0_BLOCK;

/// Repacks 4 Q4_0 rows (each nb * 8 blocks, row_bytes apart) for gemm_q4x4. Each block position holds the 4 fp16
/// scales, then, for rows (0, 1) and (2, 3), two 16-byte registers whose low and high nibbles are SMMLA's A operands as
/// they stand: register 1's low nibbles are [row i w0-7, row j w0-7] and its high ones [w8-15, w8-15], and register 2
/// is the same for w16-23 and w24-31. These are the same integers and scales as the source, only permuted, so any
/// product over it has the source's bits. out holds nb * 8 * Q4X4_GROUP bytes.
pub fn q4_0_repack4(src: &[u8], row_bytes: usize, nb: usize, out: &mut [u8]) {
    assert!(src.len() >= 3 * row_bytes + nb * 8 * Q4_0_BLOCK && out.len() >= nb * 8 * Q4X4_GROUP);
    for bk in 0..nb * 8 {
        let blk = |i: usize| &src[i * row_bytes + bk * Q4_0_BLOCK..i * row_bytes + (bk + 1) * Q4_0_BLOCK];
        // weight n (0..32) of row i, as its nibble value
        let wt = |i: usize, n: usize| if n < 16 { blk(i)[2 + n] & 15 } else { blk(i)[2 + n - 16] >> 4 };
        let o = &mut out[bk * Q4X4_GROUP..(bk + 1) * Q4X4_GROUP];
        for i in 0..4 {
            o[2 * i..2 * i + 2].copy_from_slice(&blk(i)[..2]);
        }
        for t in 0..2 {
            let (ri, rj) = (2 * t, 2 * t + 1);
            for half in 0..2 {
                let base = 8 + t * 32 + half * 16;
                for m in 0..8 {
                    let n = 16 * half + m;
                    o[base + m] = wt(ri, n) | (wt(ri, n + 8) << 4);
                    o[base + 8 + m] = wt(rj, n) | (wt(rj, n + 8) << 4);
                }
            }
        }
    }
}

/// gemm_mmla_packed::<Q40> over rows repacked by q4_0_repack4 (quads: rows / 4 groups of nb * 8 * Q4X4_GROUP bytes),
/// with no unpacking left beyond a mask and a shift. The same bits. As in gemm_mmla_packed, it takes two quads (8 rows)
/// a pass from TORC_MMLA8_MIN vectors (default 16) when the rows allow, and the pairs a panel at a time
/// (TORC_MMLA_PANEL).
///
/// # Safety
/// FEAT_I8MM; w holds rows / 4 repacked quads; pk is mmla_pack of nv vectors of nb blocks; y as for gemm_mmla_packed.
#[target_feature(enable = "i8mm")]
pub unsafe fn gemm_q4x4(w: &[u8], rows: usize, pk: &[Packed], nv: usize, nb: usize, y: *mut f32, y_stride: usize) {
    let quad = nb * 8 * Q4X4_GROUP;
    assert!(nv > 0 && rows.is_multiple_of(4) && pk.len() >= pack_len(nv, nb) && w.len() >= rows / 4 * quad);
    static MIN8: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_MMLA8_MIN").unwrap_or(16));
    let r8 = if nv >= *MIN8 { rows / 8 * 8 } else { 0 };
    unsafe {
        if r8 > 0 {
            gemm_q4x4_tiles::<2>(w, r8, pk, nv, nb, y, y_stride);
        }
        if r8 < rows {
            gemm_q4x4_tiles::<1>(&w[r8 / 4 * quad..], rows - r8, pk, nv, nb, y.add(r8), y_stride);
        }
    }
}

/// gemm_q4x4's body: Q quads (4Q rows, 2Q tiles) a pass (rows must be a multiple of 4Q).
#[target_feature(enable = "i8mm")]
unsafe fn gemm_q4x4_tiles<const Q: usize>(
    w: &[u8],
    rows: usize,
    pk: &[Packed],
    nv: usize,
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let quad = nb * 8 * Q4X4_GROUP;
    assert!(rows % (4 * Q) == 0 && w.len() >= rows / 4 * quad);
    let np = nv.div_ceil(2);
    unsafe {
        // acc[(p * 8 + k) * 2Q + t] is tile t of pair p at block position k; it lives on the stack for up to 1024
        // vectors
        let mut stack = core::mem::MaybeUninit::<[float32x4_t; 1024]>::uninit();
        let mut heap = Vec::new();
        let n = np * 8 * 2 * Q;
        let acc: &mut [float32x4_t] = if n <= 1024 {
            crate::zeroed_prefix(&mut stack, n)
        } else {
            heap.resize(n, vdupq_n_f32(0.0));
            &mut heap
        };
        static PANEL: std::sync::LazyLock<usize> =
            std::sync::LazyLock::new(|| crate::switches::num("TORC_MMLA_PANEL").unwrap_or(16));
        let panel = if *PANEL == 0 { np } else { (*PANEL).min(np) };
        let m15 = vdupq_n_u8(15);
        for r0 in (0..rows).step_by(4 * Q) {
            acc.fill(vdupq_n_f32(0.0));
            let base = w.as_ptr().add(r0 / 4 * quad);
            for p0 in (0..np).step_by(panel) {
                let p1 = (p0 + panel).min(np);
                for b in 0..nb {
                    for k in 0..8 {
                        // for each quad: its 4 scales as [d0 d0 d1 d1] and [d2 d2 d3 d3], and the 4 operands of each of
                        // its two tiles
                        let mut dk = [vdupq_n_f32(0.0); 4];
                        let mut a = [[vdupq_n_s8(0); 4]; 4];
                        for q in 0..Q {
                            let g = base.add(q * quad + (b * 8 + k) * Q4X4_GROUP);
                            let d = fcvtl(vld1_u16(g as *const u16));
                            dk[2 * q] = vzip1q_f32(d, d);
                            dk[2 * q + 1] = vzip2q_f32(d, d);
                            let lo = |r: uint8x16_t| vreinterpretq_s8_u8(vandq_u8(r, m15));
                            let hi = |r: uint8x16_t| vreinterpretq_s8_u8(vshrq_n_u8::<4>(r));
                            for t in 0..2 {
                                let (r1, r2) = (vld1q_u8(g.add(8 + 32 * t)), vld1q_u8(g.add(24 + 32 * t)));
                                a[2 * q + t] = [lo(r1), hi(r1), lo(r2), hi(r2)];
                            }
                        }
                        for p in p0..p1 {
                            let pq = pk.get_unchecked((p * nb + b) * 8 + k);
                            let bias = vshlq_n_s32::<3>(pq.sum);
                            let ac = acc.as_mut_ptr().add((p * 8 + k) * 2 * Q);
                            for t in 0..2 * Q {
                                let mut s = smmla(vdupq_n_s32(0), a[t][0], pq.b[0]);
                                s = smmla(s, a[t][1], pq.b[1]);
                                s = smmla(s, a[t][2], pq.b[2]);
                                s = smmla(s, a[t][3], pq.b[3]);
                                s = vsubq_s32(s, bias);
                                *ac.add(t) = vfmaq_f32(*ac.add(t), vmulq_f32(dk[t], pq.d), vcvtq_f32_s32(s));
                            }
                        }
                    }
                }
            }
            for p in 0..np {
                let (v0, v1) = (2 * p, 2 * p + 1);
                for t in 0..2 * Q {
                    let a = |k: usize| acc[(p * 8 + k) * 2 * Q + t];
                    let r = vaddq_f32(
                        vaddq_f32(vaddq_f32(a(0), a(4)), vaddq_f32(a(2), a(6))),
                        vaddq_f32(vaddq_f32(a(1), a(5)), vaddq_f32(a(3), a(7))),
                    );
                    let mut o = [0.0f32; 4];
                    vst1q_f32(o.as_mut_ptr(), r);
                    let row = r0 + 2 * t;
                    *y.add(v0 * y_stride + row) = o[0];
                    *y.add(v0 * y_stride + row + 1) = o[2];
                    if v1 < nv {
                        *y.add(v1 * y_stride + row) = o[1];
                        *y.add(v1 * y_stride + row + 1) = o[3];
                    }
                }
            }
        }
    }
}

/// Rows repacked by q4_0_repack4 against 1-3 vectors by SDOT, 4 rows a pass with one row per lane. For each block
/// position k, the 4 rows' scales are one 8-byte load; each tile register is dotted against the vector's 8 bytes twice
/// over ([x0-7, x0-7], with lanes 0-1 for row i and 2-3 for row j), and one pairwise add leaves the 4 rows' exact sums
/// in the 4 lanes. For each (row, vector) the float steps are q4_0_dot's: (d_w * g.d) fused onto the sub-block's own
/// accumulator block after block, then ((0+4) + (2+6)) + ((1+5) + (3+7)), so the bits are the same as q4_0_dot_rows.
/// The rows of a pass are read as one stream (the 4 interleaved) rather than four. The outputs go to y[v * y_stride +
/// r].
///
/// # Safety
/// FEAT_DotProd; w holds rows / 4 repacked quads of x.len() / nv blocks; rows is a multiple of 4; 1 <= nv <= 3; y is
/// valid for writes at v * y_stride + r, and nothing else touches those elements during the call.
#[target_feature(enable = "dotprod")]
pub unsafe fn q4x4_matmul(w: &[u8], rows: usize, x: &[Groups32], nv: usize, y: *mut f32, y_stride: usize) {
    let nb = x.len() / nv;
    let quad = nb * 8 * Q4X4_GROUP;
    assert!(rows.is_multiple_of(4) && (1..=3).contains(&nv) && x.len() == nv * nb && w.len() >= rows / 4 * quad);
    unsafe {
        match nv {
            1 => q4x4_pass::<1>(w, rows, x, nb, y, y_stride),
            2 => q4x4_pass::<2>(w, rows, x, nb, y, y_stride),
            _ => q4x4_pass::<3>(w, rows, x, nb, y, y_stride),
        }
    }
}

#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn q4x4_pass<const NV: usize>(w: &[u8], rows: usize, x: &[Groups32], nb: usize, y: *mut f32, y_stride: usize) {
    let quad = nb * 8 * Q4X4_GROUP;
    unsafe {
        let m15 = vdupq_n_u8(15);
        for r0 in (0..rows).step_by(4) {
            let base = w.as_ptr().add(r0 / 4 * quad);
            let mut acc = [[vdupq_n_f32(0.0); 8]; NV];
            for b in 0..nb {
                for k in 0..8 {
                    let g = base.add((b * 8 + k) * Q4X4_GROUP);
                    let dw = fcvtl(vld1_u16(g as *const u16));
                    let (r1, r2, r3, r4) =
                        (vld1q_u8(g.add(8)), vld1q_u8(g.add(24)), vld1q_u8(g.add(40)), vld1q_u8(g.add(56)));
                    let lo = |r: uint8x16_t| vreinterpretq_s8_u8(vandq_u8(r, m15));
                    let hi = |r: uint8x16_t| vreinterpretq_s8_u8(vshrq_n_u8::<4>(r));
                    let (a0, a1, a2, a3) = (lo(r1), hi(r1), lo(r2), hi(r2));
                    let (c0, c1, c2, c3) = (lo(r3), hi(r3), lo(r4), hi(r4));
                    for v in 0..NV {
                        let xb = x.get_unchecked(v * nb + b);
                        let a = xb.qs.as_ptr().add(32 * k);
                        let dup = |o: usize| vreinterpretq_s8_s64(vld1q_dup_s64(a.add(o) as *const i64));
                        let (x0, x1, x2, x3) = (dup(0), dup(8), dup(16), dup(24));
                        let s = sdot(sdot(sdot(sdot(vdupq_n_s32(0), a0, x0), a1, x1), a2, x2), a3, x3);
                        let t = sdot(sdot(sdot(sdot(vdupq_n_s32(0), c0, x0), c1, x1), c2, x2), c3, x3);
                        let bias = 8 * xb.sums[k] as i32;
                        let iv = vsubq_s32(vpaddq_s32(s, t), vdupq_n_s32(bias));
                        acc[v][k] = vfmaq_f32(acc[v][k], vmulq_n_f32(dw, xb.d[k]), vcvtq_f32_s32(iv));
                    }
                }
            }
            for (v, a) in acc.iter().enumerate() {
                let r = vaddq_f32(
                    vaddq_f32(vaddq_f32(a[0], a[4]), vaddq_f32(a[2], a[6])),
                    vaddq_f32(vaddq_f32(a[1], a[5]), vaddq_f32(a[3], a[7])),
                );
                vst1q_f32(y.add(v * y_stride + r0), r);
            }
        }
    }
}

pub use crate::lay::{Q8_LAY_GROUP, q8_0_lay4};

/// A 4-row layout for gemm_x4 / x4_matmul. Each block position (32 weights of 4 rows) takes GROUP bytes, the 4 fp16
/// scales first. ops gives tile t's (rows 2t and 2t + 1) four SMMLA A operands as signed bytes ([row i w0-7, row j
/// w0-7], [w8-15, w8-15], [w16-23, w16-23], [w24-31, w24-31]); BIAS means they are nibbles, so 8 x the activations' sum
/// must be taken off. This trait is only the per-quant part; the passes are shared.
pub trait X4 {
    const GROUP: usize;
    const BIAS: bool;
    /// # Safety
    /// g must be valid for GROUP bytes.
    unsafe fn ops(g: *const u8, t: usize) -> [int8x16_t; 4];
}

/// Q4_0 repacked by q4_0_repack4 (gemm_q4x4's layout).
pub struct Q4x4;
impl X4 for Q4x4 {
    const GROUP: usize = Q4X4_GROUP;
    const BIAS: bool = true;
    #[inline(always)]
    unsafe fn ops(g: *const u8, t: usize) -> [int8x16_t; 4] {
        unsafe {
            let (r1, r2) = (vld1q_u8(g.add(8 + 32 * t)), vld1q_u8(g.add(24 + 32 * t)));
            let m15 = vdupq_n_u8(15);
            let lo = |r: uint8x16_t| vreinterpretq_s8_u8(vandq_u8(r, m15));
            let hi = |r: uint8x16_t| vreinterpretq_s8_u8(vshrq_n_u8::<4>(r));
            [lo(r1), hi(r1), lo(r2), hi(r2)]
        }
    }
}

/// Q8_0 laid out by q8_0_lay4: the operands as they stand, in four loads.
pub struct Q8x4;
impl X4 for Q8x4 {
    const GROUP: usize = Q8_LAY_GROUP;
    const BIAS: bool = false;
    #[inline(always)]
    unsafe fn ops(g: *const u8, t: usize) -> [int8x16_t; 4] {
        unsafe {
            let p = g.add(8 + 64 * t) as *const i8;
            [vld1q_s8(p), vld1q_s8(p.add(16)), vld1q_s8(p.add(32)), vld1q_s8(p.add(48))]
        }
    }
}

/// gemm_q4x4 over any X4 layout (quads: rows / 4 groups of nb * 8 * B::GROUP bytes). For each (row, vector, sub-block)
/// it forms the exact integer sum and then applies gemm_mmla_packed's float steps, so the bits are the same as
/// gemm_mmla_packed over the source (Q8x4 as ::<Q80>, Q4x4 as ::<Q40>) and as the single-vector rows. Two quads a pass
/// from TORC_MMLA8_MIN vectors (default 16), and the pairs a panel at a time (TORC_MMLA_PANEL).
///
/// # Safety
/// FEAT_I8MM; w holds rows / 4 laid-out quads; pk is mmla_pack of nv vectors of nb blocks; y as for gemm_mmla_packed.
#[target_feature(enable = "i8mm")]
pub unsafe fn gemm_x4<B: X4>(w: &[u8], rows: usize, pk: &[Packed], nv: usize, nb: usize, y: *mut f32, y_stride: usize) {
    let quad = nb * 8 * B::GROUP;
    assert!(nv > 0 && rows.is_multiple_of(4) && pk.len() >= pack_len(nv, nb) && w.len() >= rows / 4 * quad);
    static MIN8: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_MMLA8_MIN").unwrap_or(16));
    let r8 = if nv >= *MIN8 { rows / 8 * 8 } else { 0 };
    unsafe {
        if r8 > 0 {
            gemm_x4_tiles::<B, 2>(w, r8, pk, nv, nb, y, y_stride);
        }
        if r8 < rows {
            gemm_x4_tiles::<B, 1>(&w[r8 / 4 * quad..], rows - r8, pk, nv, nb, y.add(r8), y_stride);
        }
    }
}

/// gemm_x4's body: Q quads (4Q rows, 2Q tiles) a pass (rows must be a multiple of 4Q).
#[target_feature(enable = "i8mm")]
unsafe fn gemm_x4_tiles<B: X4, const Q: usize>(
    w: &[u8],
    rows: usize,
    pk: &[Packed],
    nv: usize,
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let quad = nb * 8 * B::GROUP;
    assert!(rows % (4 * Q) == 0 && w.len() >= rows / 4 * quad);
    let np = nv.div_ceil(2);
    unsafe {
        // acc[(p * 8 + k) * 2Q + t] is tile t of pair p at block position k; it lives on the stack for up to 1024
        // vectors
        let mut stack = core::mem::MaybeUninit::<[float32x4_t; 1024]>::uninit();
        let mut heap = Vec::new();
        let n = np * 8 * 2 * Q;
        let acc: &mut [float32x4_t] = if n <= 1024 {
            crate::zeroed_prefix(&mut stack, n)
        } else {
            heap.resize(n, vdupq_n_f32(0.0));
            &mut heap
        };
        static PANEL: std::sync::LazyLock<usize> =
            std::sync::LazyLock::new(|| crate::switches::num("TORC_MMLA_PANEL").unwrap_or(16));
        let panel = if *PANEL == 0 { np } else { (*PANEL).min(np) };
        for r0 in (0..rows).step_by(4 * Q) {
            acc.fill(vdupq_n_f32(0.0));
            let base = w.as_ptr().add(r0 / 4 * quad);
            for p0 in (0..np).step_by(panel) {
                let p1 = (p0 + panel).min(np);
                for b in 0..nb {
                    for k in 0..8 {
                        // for each quad: its 4 scales as [d0 d0 d1 d1] and [d2 d2 d3 d3], and the 4 operands of each of
                        // its two tiles
                        let mut dk = [vdupq_n_f32(0.0); 4];
                        let mut a = [[vdupq_n_s8(0); 4]; 4];
                        for q in 0..Q {
                            let g = base.add(q * quad + (b * 8 + k) * B::GROUP);
                            let d = fcvtl(vld1_u16(g as *const u16));
                            dk[2 * q] = vzip1q_f32(d, d);
                            dk[2 * q + 1] = vzip2q_f32(d, d);
                            for t in 0..2 {
                                a[2 * q + t] = B::ops(g, t);
                            }
                        }
                        for p in p0..p1 {
                            let pq = pk.get_unchecked((p * nb + b) * 8 + k);
                            let ac = acc.as_mut_ptr().add((p * 8 + k) * 2 * Q);
                            for t in 0..2 * Q {
                                let mut s = smmla(vdupq_n_s32(0), a[t][0], pq.b[0]);
                                s = smmla(s, a[t][1], pq.b[1]);
                                s = smmla(s, a[t][2], pq.b[2]);
                                s = smmla(s, a[t][3], pq.b[3]);
                                if B::BIAS {
                                    s = vsubq_s32(s, vshlq_n_s32::<3>(pq.sum));
                                }
                                *ac.add(t) = vfmaq_f32(*ac.add(t), vmulq_f32(dk[t], pq.d), vcvtq_f32_s32(s));
                            }
                        }
                    }
                }
            }
            for p in 0..np {
                let (v0, v1) = (2 * p, 2 * p + 1);
                for t in 0..2 * Q {
                    let a = |k: usize| acc[(p * 8 + k) * 2 * Q + t];
                    let r = vaddq_f32(
                        vaddq_f32(vaddq_f32(a(0), a(4)), vaddq_f32(a(2), a(6))),
                        vaddq_f32(vaddq_f32(a(1), a(5)), vaddq_f32(a(3), a(7))),
                    );
                    let mut o = [0.0f32; 4];
                    vst1q_f32(o.as_mut_ptr(), r);
                    let row = r0 + 2 * t;
                    *y.add(v0 * y_stride + row) = o[0];
                    *y.add(v0 * y_stride + row + 1) = o[2];
                    if v1 < nv {
                        *y.add(v1 * y_stride + row) = o[1];
                        *y.add(v1 * y_stride + row + 1) = o[3];
                    }
                }
            }
        }
    }
}

/// q4x4_matmul over any X4 layout: 1-3 vectors by SDOT, 4 rows a pass with one row per lane. For each (row, vector) the
/// float steps are those of the single-vector rows (d_w * g.d fused onto the sub-block's own accumulator block after
/// block, then ((0+4) + (2+6)) + ((1+5) + (3+7))), so the bits are the same as q8_0_dot_rows / q4_0_dot_rows. The
/// outputs go to y[v * y_stride + r].
///
/// # Safety
/// FEAT_DotProd; w holds rows / 4 laid-out quads of x.len() / nv blocks; rows is a multiple of 4; 1 <= nv <= 3; y is
/// valid for writes at v * y_stride + r, and nothing else touches those elements during the call.
#[target_feature(enable = "dotprod")]
pub unsafe fn x4_matmul<B: X4>(w: &[u8], rows: usize, x: &[Groups32], nv: usize, y: *mut f32, y_stride: usize) {
    let nb = x.len() / nv;
    let quad = nb * 8 * B::GROUP;
    assert!(rows.is_multiple_of(4) && (1..=3).contains(&nv) && x.len() == nv * nb && w.len() >= rows / 4 * quad);
    unsafe {
        match nv {
            1 => x4_pass::<B, 1>(w, rows, x, nb, y, y_stride),
            2 => x4_pass::<B, 2>(w, rows, x, nb, y, y_stride),
            _ => x4_pass::<B, 3>(w, rows, x, nb, y, y_stride),
        }
    }
}

#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn x4_pass<B: X4, const NV: usize>(
    w: &[u8],
    rows: usize,
    x: &[Groups32],
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let quad = nb * 8 * B::GROUP;
    unsafe {
        for r0 in (0..rows).step_by(4) {
            let base = w.as_ptr().add(r0 / 4 * quad);
            let mut acc = [[vdupq_n_f32(0.0); 8]; NV];
            for b in 0..nb {
                for k in 0..8 {
                    let g = base.add((b * 8 + k) * B::GROUP);
                    let dw = fcvtl(vld1_u16(g as *const u16));
                    let [a0, a1, a2, a3] = B::ops(g, 0);
                    let [c0, c1, c2, c3] = B::ops(g, 1);
                    for v in 0..NV {
                        let xb = x.get_unchecked(v * nb + b);
                        let a = xb.qs.as_ptr().add(32 * k);
                        let dup = |o: usize| vreinterpretq_s8_s64(vld1q_dup_s64(a.add(o) as *const i64));
                        let (x0, x1, x2, x3) = (dup(0), dup(8), dup(16), dup(24));
                        let s = sdot(sdot(sdot(sdot(vdupq_n_s32(0), a0, x0), a1, x1), a2, x2), a3, x3);
                        let t = sdot(sdot(sdot(sdot(vdupq_n_s32(0), c0, x0), c1, x1), c2, x2), c3, x3);
                        let mut iv = vpaddq_s32(s, t);
                        if B::BIAS {
                            iv = vsubq_s32(iv, vdupq_n_s32(8 * xb.sums[k] as i32));
                        }
                        acc[v][k] = vfmaq_f32(acc[v][k], vmulq_n_f32(dw, xb.d[k]), vcvtq_f32_s32(iv));
                    }
                }
            }
            for (v, a) in acc.iter().enumerate() {
                let r = vaddq_f32(
                    vaddq_f32(vaddq_f32(a[0], a[4]), vaddq_f32(a[2], a[6])),
                    vaddq_f32(vaddq_f32(a[1], a[5]), vaddq_f32(a[3], a[7])),
                );
                vst1q_f32(y.add(v * y_stride + r0), r);
            }
        }
    }
}

/// gemm_mmla_packed with the packing done here (for one call over these vectors; for many calls, pack once instead).
///
/// # Safety
/// As for gemm_mmla_packed; x holds nv * nb blocks.
#[target_feature(enable = "i8mm")]
pub unsafe fn gemm_mmla<B: MmlaBlock>(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv;
    let mut pk = vec![Packed::default(); pack_len(nv, nb)];
    mmla_pack(x, nv, &mut pk);
    unsafe { gemm_mmla_packed::<B>(w, row_bytes, rows, &pk, nv, nb, y, y_stride) }
}

/// One row of Q4_0 against one vector.
///
/// # Safety
/// The CPU must have FEAT_DotProd, and w must hold x.len() * 8 Q4_0 blocks.
#[target_feature(enable = "dotprod")]
pub unsafe fn q4_0_dot(w: &[u8], x: &[Groups32]) -> f32 {
    let nb = x.len();
    assert!(w.len() >= nb * 8 * Q4_0_BLOCK);
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let m15 = vdupq_n_u8(15);
        for (b, xb) in x.iter().enumerate() {
            let wb = w.as_ptr().add(b * 8 * Q4_0_BLOCK);
            let mut s = [vdupq_n_s32(0); 8];
            for k in 0..8 {
                let q = vld1q_u8(wb.add(k * Q4_0_BLOCK + 2));
                let w0 = vreinterpretq_s8_u8(vandq_u8(q, m15));
                let w1 = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));
                let a = xb.qs.as_ptr().add(32 * k);
                let t = sdot(vdupq_n_s32(0), w0, vld1q_s8(a));
                s[k] = sdot(t, w1, vld1q_s8(a.add(16)));
            }
            let mut lo = vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
            let mut hi = vpaddq_s32(vpaddq_s32(s[4], s[5]), vpaddq_s32(s[6], s[7]));
            let bs = vld1q_s16(xb.sums.as_ptr());
            lo = vsubq_s32(lo, vshll_n_s16::<3>(vget_low_s16(bs)));
            hi = vsubq_s32(hi, vshll_high_n_s16::<3>(bs));
            let d = |k: usize| f16_to_f32(u16::from_le_bytes([*wb.add(k * Q4_0_BLOCK), *wb.add(k * Q4_0_BLOCK + 1)]));
            let h0 = [d(0), d(1), d(2), d(3)];
            let h1 = [d(4), d(5), d(6), d(7)];
            let sc0 = vmulq_f32(vld1q_f32(h0.as_ptr()), vld1q_f32(xb.d.as_ptr()));
            let sc1 = vmulq_f32(vld1q_f32(h1.as_ptr()), vld1q_f32(xb.d.as_ptr().add(4)));
            acc0 = vfmaq_f32(acc0, sc0, vcvtq_f32_s32(lo));
            acc1 = vfmaq_f32(acc1, sc1, vcvtq_f32_s32(hi));
        }
        let s = vaddq_f32(acc0, acc1);
        let t = vadd_f32(vget_low_f32(s), vget_high_f32(s));
        vget_lane_f32::<0>(t) + vget_lane_f32::<1>(t)
    }
}

/// Rows of Q4_0 against ONE vector (single-token decode), 2 rows a pass side by side. Each sub-block's activations are
/// loaded once for both rows, the bias (8 x the activations' sums) and g.d once per block, and each row's 8 fp16 scales
/// are put together in general registers and widened whole (FCVTL, exact, as f16_to_f32). Each row's float steps are
/// q4_0_dot's (d_w * g.d into lane k of acc0 / acc1, fused block after block, then ((0+4) + (2+6)) + ((1+5) + (3+7))),
/// so the bits are the same as q4_0_dot. The outputs go to y[r] for r < rows, and an odd last row goes through
/// q4_0_dot. Two rows and not four, and no software prefetch: 4 rows a pass read 4 streams per core, more than the
/// hardware prefetcher follows on long rows (q4_0_dot_rows_var has both options).
///
/// # Safety
/// FEAT_DotProd. (w holds rows * row_bytes, row_bytes >= x.len() * 8 * Q4_0_BLOCK and y >= rows: all asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q4_0_dot_rows(w: &[u8], row_bytes: usize, rows: usize, x: &[Groups32], y: &mut [f32]) {
    unsafe { q4_0_dot_rows_var(w, row_bytes, rows, x, y, 2, 0) }
}

/// q4_0_dot_rows with r rows a pass (2 or 4) and prefetch option pf (0: none, 1: each row's next-pass bytes into L1, 2:
/// into L2). For kernel benchmarks.
///
/// # Safety
/// As for q4_0_dot_rows.
#[doc(hidden)]
#[target_feature(enable = "dotprod")]
pub unsafe fn q4_0_dot_rows_var(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    y: &mut [f32],
    r: usize,
    pf: usize,
) {
    assert!(row_bytes >= x.len() * 8 * Q4_0_BLOCK && w.len() >= rows * row_bytes && y.len() >= rows);
    assert!(r == 1 || r == 2 || r == 4);
    let rr = rows / r * r;
    unsafe {
        for r0 in (0..rr).step_by(r) {
            let (wp, yp) = (w.as_ptr().add(r0 * row_bytes), y.as_mut_ptr().add(r0));
            match r {
                1 => q4_0_dotr::<1>(wp, row_bytes, x, yp, pf),
                2 => q4_0_dotr::<2>(wp, row_bytes, x, yp, pf),
                _ => q4_0_dotr::<4>(wp, row_bytes, x, yp, pf),
            }
        }
    }
    for r in rr..rows {
        y[r] = unsafe { q4_0_dot(&w[r * row_bytes..], x) };
    }
}

/// The 8 fp16 scales of one row's block of 256 (18 bytes apart), widened into lanes k = 0-3 and 4-7. Each group of 4 is
/// put together in a general register and moved over whole, because lane loads (LD1 {v.h}[i]) are slower than plain
/// ones.
#[inline]
pub(crate) unsafe fn q4_0_scales(p: *const u8) -> (float32x4_t, float32x4_t) {
    unsafe {
        let h = |k: usize| core::ptr::read_unaligned(p.add(k * Q4_0_BLOCK) as *const u16) as u64;
        let w4 = |k: usize| h(k) | h(k + 1) << 16 | h(k + 2) << 32 | h(k + 3) << 48;
        (fcvtl_x(w4(0)), fcvtl_x(w4(4)))
    }
}

/// FCVTL of 4 fp16 values (raw bits) packed in a general register.
#[inline]
unsafe fn fcvtl_x(h: u64) -> float32x4_t {
    let o: float32x4_t;
    unsafe {
        asm!("fmov {o:d}, {h}", "fcvtl {o:v}.4s, {o:v}.4h", h = in(reg) h, o = out(vreg) o,
            options(pure, nomem, nostack));
    }
    o
}

/// Rows of Q4_0 against ONE vector, a row at a time: for each block, the eight sub-blocks' sums first, then reduced 4
/// by 4, less the bias, and scaled. These are the same steps as q4_0_dot, so the bits are the same.
///
/// # Safety
/// FEAT_DotProd. (w holds rows * row_bytes, row_bytes >= x.len() * 8 * Q4_0_BLOCK and y >= rows: all asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q4_0_dot_rows1(w: &[u8], row_bytes: usize, rows: usize, x: &[Groups32], y: &mut [f32]) {
    assert!(row_bytes >= x.len() * 8 * Q4_0_BLOCK && w.len() >= rows * row_bytes && y.len() >= rows);
    unsafe {
        let m15 = vdupq_n_u8(15);
        for (r, yr) in y[..rows].iter_mut().enumerate() {
            let wr = w.as_ptr().add(r * row_bytes);
            let (mut acc0, mut acc1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            for (b, xb) in x.iter().enumerate() {
                let wb = wr.add(b * 8 * Q4_0_BLOCK);
                let mut s = [vdupq_n_s32(0); 8];
                for (k, sk) in s.iter_mut().enumerate() {
                    let q = vld1q_u8(wb.add(k * Q4_0_BLOCK + 2));
                    let a = xb.qs.as_ptr().add(32 * k);
                    let t = sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(q, m15)), vld1q_s8(a));
                    *sk = sdot(t, vreinterpretq_s8_u8(vshrq_n_u8::<4>(q)), vld1q_s8(a.add(16)));
                }
                let lo = vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
                let hi = vpaddq_s32(vpaddq_s32(s[4], s[5]), vpaddq_s32(s[6], s[7]));
                let bs = vld1q_s16(xb.sums.as_ptr());
                let lo = vsubq_s32(lo, vshll_n_s16::<3>(vget_low_s16(bs)));
                let hi = vsubq_s32(hi, vshll_high_n_s16::<3>(bs));
                let (d0, d1) = q4_0_scales(wb);
                let sc0 = vmulq_f32(d0, vld1q_f32(xb.d.as_ptr()));
                let sc1 = vmulq_f32(d1, vld1q_f32(xb.d.as_ptr().add(4)));
                acc0 = vfmaq_f32(acc0, sc0, vcvtq_f32_s32(lo));
                acc1 = vfmaq_f32(acc1, sc1, vcvtq_f32_s32(hi));
            }
            *yr = hsum4x2(acc0, acc1);
        }
    }
}

/// q4_0_dot_rows' pass: R rows (at w, w + row_bytes, ...) into y[0..R]. With R = 4, the four rows' last sums are added
/// up transposed, so the final adds across rows happen in lanes.
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn q4_0_dotr<const R: usize>(w: *const u8, row_bytes: usize, x: &[Groups32], y: *mut f32, pf: usize) {
    unsafe {
        let m15 = vdupq_n_u8(15);
        let mut acc0 = [vdupq_n_f32(0.0); R];
        let mut acc1 = [vdupq_n_f32(0.0); R];
        for (b, xb) in x.iter().enumerate() {
            let wb = |i: usize| w.add(i * row_bytes + b * 8 * Q4_0_BLOCK);
            if pf == 1 || pf == 2 {
                // each row's row for the next pass (R rows on), this block's 144 bytes
                for i in 0..R {
                    let p = wb(i).wrapping_add(R * row_bytes);
                    for l in 0..3 {
                        if pf == 1 {
                            asm!("prfm pldl1keep, [{p}]", p = in(reg) p.wrapping_add(64 * l),
                                options(nostack, readonly, preserves_flags));
                        } else {
                            asm!("prfm pldl2keep, [{p}]", p = in(reg) p.wrapping_add(64 * l),
                                options(nostack, readonly, preserves_flags));
                        }
                    }
                }
            }
            let bs = vld1q_s16(xb.sums.as_ptr());
            let bias = [vshll_n_s16::<3>(vget_low_s16(bs)), vshll_high_n_s16::<3>(bs)];
            let dx = [vld1q_f32(xb.d.as_ptr()), vld1q_f32(xb.d.as_ptr().add(4))];
            let mut dw = [[vdupq_n_f32(0.0); 2]; R];
            for (i, d) in dw.iter_mut().enumerate() {
                let (d0, d1) = q4_0_scales(wb(i));
                *d = [d0, d1];
            }
            // sub-blocks 0-3 go into acc0 and 4-7 into acc1, with each row's four sums pairwise-added into lanes. This
            // is a macro rather than a loop over the halves, because the compiler kept that loop, and the accumulators
            // in memory
            macro_rules! half {
                ($h:literal, $acc:ident) => {{
                    let mut p = [[vdupq_n_s32(0); 2]; R];
                    for j in 0..2 {
                        let mut s = [[vdupq_n_s32(0); 2]; R];
                        for e in 0..2 {
                            let k = 4 * $h + 2 * j + e;
                            let a = xb.qs.as_ptr().add(32 * k);
                            let (a0, a1) = (vld1q_s8(a), vld1q_s8(a.add(16)));
                            for i in 0..R {
                                let q = vld1q_u8(wb(i).add(k * Q4_0_BLOCK + 2));
                                let t = sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(q, m15)), a0);
                                s[i][e] = sdot(t, vreinterpretq_s8_u8(vshrq_n_u8::<4>(q)), a1);
                            }
                        }
                        for i in 0..R {
                            p[i][j] = vpaddq_s32(s[i][0], s[i][1]);
                        }
                    }
                    for i in 0..R {
                        let v = vsubq_s32(vpaddq_s32(p[i][0], p[i][1]), bias[$h]);
                        $acc[i] = vfmaq_f32($acc[i], vmulq_f32(dw[i][$h], dx[$h]), vcvtq_f32_s32(v));
                    }
                }};
            }
            half!(0, acc0);
            half!(1, acc1);
        }
        if R == 4 {
            // row i's lanes l: s_i[l] = acc0 + acc1. Transposed, lane i of c[l] is s_i[l], and each row's (s[0] +
            // s[2]) + (s[1] + s[3]) is q4_0_dot's last two adds, done lane by lane
            let s: [float32x4_t; 4] = core::array::from_fn(|i| vaddq_f32(acc0[i], acc1[i]));
            let c = tr4(s.map(|v| vreinterpretq_s8_f32(v))).map(|v| vreinterpretq_f32_s8(v));
            vst1q_f32(y, vaddq_f32(vaddq_f32(c[0], c[2]), vaddq_f32(c[1], c[3])));
        } else {
            for i in 0..R {
                *y.add(i) = hsum4x2(acc0[i], acc1[i]);
            }
        }
    }
}

/// Q8_0's block: an fp16 scale and 32 signed bytes.
pub const Q8_0_BLOCK: usize = 34;

/// Rows of Q8_0 against ONE vector (single-token decode: Qwen's attention k and v, the DeltaNet output and the shared
/// experts), 2 rows a pass sharing each sub-block's activations. For each row and block of 256, lane k is sub-block k:
/// the exact sum of weight x activation, then acc[k] = fma(d_w[k] x g.d[k], sum, acc[k]), and fold8 at the end. This is
/// q4_0_dot_rows' order without the nibbles or the 8 x sum bias, which is ggml's Q8_0 order, so the bits are the same
/// as the reference (parity test). The outputs go to y[r] for r < rows, and an odd last row runs alone.
///
/// # Safety
/// FEAT_DotProd. (w holds rows * row_bytes, row_bytes >= x.len() * 8 * 34 and y >= rows: all asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q8_0_dot_rows(w: &[u8], row_bytes: usize, rows: usize, x: &[Groups32], y: &mut [f32]) {
    assert!(row_bytes >= x.len() * 8 * Q8_0_BLOCK && w.len() >= rows * row_bytes && y.len() >= rows);
    let rr = rows / 2 * 2;
    unsafe {
        for r0 in (0..rr).step_by(2) {
            q8_0_dotr::<2>(w.as_ptr().add(r0 * row_bytes), row_bytes, x, y.as_mut_ptr().add(r0));
        }
        for i in rr..rows {
            q8_0_dotr::<1>(w.as_ptr().add(i * row_bytes), row_bytes, x, y.as_mut_ptr().add(i));
        }
    }
}

/// q8_0_dot_rows' pass: R rows (at w, w + row_bytes, ..) into y[0..R].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn q8_0_dotr<const R: usize>(w: *const u8, row_bytes: usize, x: &[Groups32], y: *mut f32) {
    unsafe {
        let mut acc0 = [vdupq_n_f32(0.0); R];
        let mut acc1 = [vdupq_n_f32(0.0); R];
        for (b, xb) in x.iter().enumerate() {
            let wb = |i: usize| w.add(i * row_bytes + b * 8 * Q8_0_BLOCK);
            let dx = [vld1q_f32(xb.d.as_ptr()), vld1q_f32(xb.d.as_ptr().add(4))];
            let mut dw = [[vdupq_n_f32(0.0); 2]; R];
            for (i, d) in dw.iter_mut().enumerate() {
                let h = |k: usize| core::ptr::read_unaligned(wb(i).add(k * Q8_0_BLOCK) as *const u16) as u64;
                let w4 = |k: usize| h(k) | h(k + 1) << 16 | h(k + 2) << 32 | h(k + 3) << 48;
                *d = [fcvtl_x(w4(0)), fcvtl_x(w4(4))];
            }
            macro_rules! half {
                ($h:literal, $acc:ident) => {{
                    let mut p = [[vdupq_n_s32(0); 2]; R];
                    for j in 0..2 {
                        let mut s = [[vdupq_n_s32(0); 2]; R];
                        for e in 0..2 {
                            let k = 4 * $h + 2 * j + e;
                            let a = xb.qs.as_ptr().add(32 * k);
                            let (a0, a1) = (vld1q_s8(a), vld1q_s8(a.add(16)));
                            for i in 0..R {
                                let q = wb(i).add(k * Q8_0_BLOCK + 2) as *const i8;
                                let t = sdot(vdupq_n_s32(0), vld1q_s8(q), a0);
                                s[i][e] = sdot(t, vld1q_s8(q.add(16)), a1);
                            }
                        }
                        for i in 0..R {
                            p[i][j] = vpaddq_s32(s[i][0], s[i][1]);
                        }
                    }
                    for i in 0..R {
                        let v = vpaddq_s32(p[i][0], p[i][1]);
                        $acc[i] = vfmaq_f32($acc[i], vmulq_f32(dw[i][$h], dx[$h]), vcvtq_f32_s32(v));
                    }
                }};
            }
            half!(0, acc0);
            half!(1, acc1);
        }
        for i in 0..R {
            *y.add(i) = hsum4x2(acc0[i], acc1[i]);
        }
    }
}

/// Q4_1's block: fp16 d, fp16 m and 16 bytes of nibbles (weights 0-15 in the low nibbles, 16-31 in the high), with w =
/// d x q + m.
pub const Q4_1_BLOCK: usize = 20;

/// Rows of Q4_1 against ONE vector (Qwen 3.6's expert down matrices), 2 rows a pass. For each block of 256 and block
/// position k: a[k] = fma(d[k] x g.d[k], the exact sum of q x a, a[k]) and m[k] = fma(m[k] x g.d[k], the activations'
/// sum, m[k]); the result is fold8(a) + fold8(m). The bits are the same as the reference (parity test). The outputs go
/// to y[r] for r < rows.
///
/// # Safety
/// FEAT_DotProd. (w holds rows * row_bytes, row_bytes >= x.len() * 8 * 20 and y >= rows: all asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q4_1_dot_rows(w: &[u8], row_bytes: usize, rows: usize, x: &[Groups32], y: &mut [f32]) {
    assert!(row_bytes >= x.len() * 8 * Q4_1_BLOCK && w.len() >= rows * row_bytes && y.len() >= rows);
    let rr = rows / 2 * 2;
    unsafe {
        for r0 in (0..rr).step_by(2) {
            q4_1_dotr::<1>(w.as_ptr().add(r0 * row_bytes), row_bytes, x, y.as_mut_ptr().add(r0));
            q4_1_dotr::<1>(w.as_ptr().add((r0 + 1) * row_bytes), row_bytes, x, y.as_mut_ptr().add(r0 + 1));
        }
        for i in rr..rows {
            q4_1_dotr::<1>(w.as_ptr().add(i * row_bytes), row_bytes, x, y.as_mut_ptr().add(i));
        }
    }
}

/// q4_1_dot_rows' pass: R rows (at w, w + row_bytes, ..) into y[0..R].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn q4_1_dotr<const R: usize>(w: *const u8, row_bytes: usize, x: &[Groups32], y: *mut f32) {
    unsafe {
        let m15 = vdupq_n_u8(15);
        let (mut a0, mut a1) = ([vdupq_n_f32(0.0); R], [vdupq_n_f32(0.0); R]);
        let (mut e0, mut e1) = ([vdupq_n_f32(0.0); R], [vdupq_n_f32(0.0); R]);
        for (b, xb) in x.iter().enumerate() {
            let wb = |i: usize| w.add(i * row_bytes + b * 8 * Q4_1_BLOCK);
            let dx = [vld1q_f32(xb.d.as_ptr()), vld1q_f32(xb.d.as_ptr().add(4))];
            // the activations' sums for each block position (exact), as s16 and then widened
            let s = vld1q_s16(xb.sums.as_ptr());
            let sf = [vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))), vcvtq_f32_s32(vmovl_high_s16(s))];
            for i in 0..R {
                let h =
                    |o: usize, k: usize| core::ptr::read_unaligned(wb(i).add(k * Q4_1_BLOCK + o) as *const u16) as u64;
                let w4 = |o: usize, k: usize| h(o, k) | h(o, k + 1) << 16 | h(o, k + 2) << 32 | h(o, k + 3) << 48;
                let (d0, d1) = (fcvtl_x(w4(0, 0)), fcvtl_x(w4(0, 4)));
                let (n0, n1) = (fcvtl_x(w4(2, 0)), fcvtl_x(w4(2, 4)));
                let mut sums = [vdupq_n_s32(0); 8];
                for (k, sk) in sums.iter_mut().enumerate() {
                    let q = vld1q_u8(wb(i).add(k * Q4_1_BLOCK + 4));
                    let a = xb.qs.as_ptr().add(32 * k);
                    let t = sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(q, m15)), vld1q_s8(a));
                    *sk = sdot(t, vreinterpretq_s8_u8(vshrq_n_u8::<4>(q)), vld1q_s8(a.add(16)));
                }
                let lo = vpaddq_s32(vpaddq_s32(sums[0], sums[1]), vpaddq_s32(sums[2], sums[3]));
                let hi = vpaddq_s32(vpaddq_s32(sums[4], sums[5]), vpaddq_s32(sums[6], sums[7]));
                a0[i] = vfmaq_f32(a0[i], vmulq_f32(d0, dx[0]), vcvtq_f32_s32(lo));
                a1[i] = vfmaq_f32(a1[i], vmulq_f32(d1, dx[1]), vcvtq_f32_s32(hi));
                e0[i] = vfmaq_f32(e0[i], vmulq_f32(n0, dx[0]), sf[0]);
                e1[i] = vfmaq_f32(e1[i], vmulq_f32(n1, dx[1]), sf[1]);
            }
        }
        for i in 0..R {
            *y.add(i) = hsum4x2(a0[i], a1[i]) + hsum4x2(e0[i], e1[i]);
        }
    }
}

/// Rows of Q4_1 against nv vectors (x is nv x nb blocks, and vector v's outputs go to y[v * y_stride + r]). Each row's
/// nibbles are unpacked once per half block and used for up to 4 vectors. Each vector goes through the steps of
/// q4_1_dot_rows, so the bits are the same (parity test).
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q4_1_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe { dm_matmul(false, w, row_bytes, rows, x, nv, y, y_stride) }
}

/// Q5_1's block: fp16 d, fp16 m, 32 fifth bits (qh, where bit l belongs to weight l) and 16 bytes of nibbles, with w =
/// d x q + m.
pub const Q5_1_BLOCK: usize = 24;

/// Rows of Q4_1 (five = false) or Q5_1 (five = true) against nv vectors, as q4_1_matmul does. For Q5_1 the fifth
/// bits are put over the nibbles first; the steps after that are the same, and so are the bits (parity test).
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn dm_matmul(
    five: bool,
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    let bs = if five { Q5_1_BLOCK } else { Q4_1_BLOCK };
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * 8 * bs && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while v0 < nv {
                let (xs, yr) = (x.as_ptr().add(v0 * nb), y.add(v0 * y_stride + r));
                macro_rules! go {
                    ($n:literal) => {
                        if five {
                            q4_1_dotv::<$n, true>(wr, xs, nb, yr, y_stride)
                        } else {
                            q4_1_dotv::<$n, false>(wr, xs, nb, yr, y_stride)
                        }
                    };
                }
                match nv - v0 {
                    1 => go!(1),
                    2 => go!(2),
                    3 => go!(3),
                    _ => go!(4),
                }
                v0 += 4;
            }
        }
    }
}

/// q4_1_matmul' pass: one row (w) against NV vectors (x, nb blocks each) into y[v * y_stride].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn q4_1_dotv<const NV: usize, const FIVE: bool>(
    w: *const u8,
    x: *const Groups32,
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe {
        let m15 = vdupq_n_u8(15);
        let bs = if FIVE { Q5_1_BLOCK } else { Q4_1_BLOCK };
        // Q5_1: byte l of a register takes byte l / 8 of qh's two (bytes 0 and 1 for weights 0-15, bytes 2 and 3 for
        // 16-31) and tests bit l % 8
        const SPREAD: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        const BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let (sp, bit, fifth) = (vld1q_u8(SPREAD.as_ptr()), vld1q_u8(BIT.as_ptr()), vdupq_n_u8(16));
        let mut a = [[vdupq_n_f32(0.0); 2]; NV];
        let mut e = [[vdupq_n_f32(0.0); 2]; NV];
        for b in 0..nb {
            let wb = w.add(b * 8 * bs);
            let h = |o: usize, k: usize| core::ptr::read_unaligned(wb.add(k * bs + o) as *const u16) as u64;
            let w4 = |o: usize, k: usize| h(o, k) | h(o, k + 1) << 16 | h(o, k + 2) << 32 | h(o, k + 3) << 48;
            let d = [fcvtl_x(w4(0, 0)), fcvtl_x(w4(0, 4))];
            let m = [fcvtl_x(w4(2, 0)), fcvtl_x(w4(2, 4))];
            for hf in 0..2 {
                let mut lo = [vdupq_n_s8(0); 4];
                let mut hi = [vdupq_n_s8(0); 4];
                for j in 0..4 {
                    let blk = wb.add((4 * hf + j) * bs);
                    let q = vld1q_u8(blk.add(if FIVE { 8 } else { 4 }));
                    let (mut l, mut u) = (vandq_u8(q, m15), vshrq_n_u8::<4>(q));
                    if FIVE {
                        let qh = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(blk.add(4) as *const u32)));
                        let fl = vandq_u8(vtstq_u8(vqtbl1q_u8(qh, sp), bit), fifth);
                        let fu = vandq_u8(vtstq_u8(vqtbl1q_u8(qh, vaddq_u8(sp, vdupq_n_u8(2))), bit), fifth);
                        l = vorrq_u8(l, fl);
                        u = vorrq_u8(u, fu);
                    }
                    lo[j] = vreinterpretq_s8_u8(l);
                    hi[j] = vreinterpretq_s8_u8(u);
                }
                for v in 0..NV {
                    let xb = &*x.add(v * nb + b);
                    let mut s = [vdupq_n_s32(0); 4];
                    for (j, sj) in s.iter_mut().enumerate() {
                        let p = xb.qs.as_ptr().add(32 * (4 * hf + j));
                        let t = sdot(vdupq_n_s32(0), lo[j], vld1q_s8(p));
                        *sj = sdot(t, hi[j], vld1q_s8(p.add(16)));
                    }
                    let sum = vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
                    let dx = vld1q_f32(xb.d.as_ptr().add(4 * hf));
                    // the activations' sums for each block position (exact integers)
                    let sf = vcvtq_f32_s32(vmovl_s16(vld1_s16(xb.sums.as_ptr().add(4 * hf))));
                    a[v][hf] = vfmaq_f32(a[v][hf], vmulq_f32(d[hf], dx), vcvtq_f32_s32(sum));
                    e[v][hf] = vfmaq_f32(e[v][hf], vmulq_f32(m[hf], dx), sf);
                }
            }
        }
        for v in 0..NV {
            *y.add(v * y_stride) = hsum4x2(a[v][0], a[v][1]) + hsum4x2(e[v][0], e[v][1]);
        }
    }
}

/// Rows of Q8_0 against nv vectors (as q4_1_matmul): each row's bytes are loaded once per half block for up to 4
/// vectors, and each vector goes through the steps of q8_0_dot_rows, so the bits are the same.
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q8_0_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * 8 * Q8_0_BLOCK && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while v0 < nv {
                let (xs, yr) = (x.as_ptr().add(v0 * nb), y.add(v0 * y_stride + r));
                match nv - v0 {
                    1 => q8_0_dotv::<1>(wr, xs, nb, yr, y_stride),
                    2 => q8_0_dotv::<2>(wr, xs, nb, yr, y_stride),
                    3 => q8_0_dotv::<3>(wr, xs, nb, yr, y_stride),
                    _ => q8_0_dotv::<4>(wr, xs, nb, yr, y_stride),
                }
                v0 += 4;
            }
        }
    }
}

/// q8_0_matmul' pass: one row (w) against NV vectors (x, nb blocks each) into y[v * y_stride].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn q8_0_dotv<const NV: usize>(w: *const u8, x: *const Groups32, nb: usize, y: *mut f32, y_stride: usize) {
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 2]; NV];
        for b in 0..nb {
            let wb = w.add(b * 8 * Q8_0_BLOCK);
            let h = |k: usize| core::ptr::read_unaligned(wb.add(k * Q8_0_BLOCK) as *const u16) as u64;
            let w4 = |k: usize| h(k) | h(k + 1) << 16 | h(k + 2) << 32 | h(k + 3) << 48;
            let dw = [fcvtl_x(w4(0)), fcvtl_x(w4(4))];
            for hf in 0..2 {
                let mut q = [[vdupq_n_s8(0); 2]; 4];
                for (j, qj) in q.iter_mut().enumerate() {
                    let p = wb.add((4 * hf + j) * Q8_0_BLOCK + 2) as *const i8;
                    *qj = [vld1q_s8(p), vld1q_s8(p.add(16))];
                }
                for v in 0..NV {
                    let xb = &*x.add(v * nb + b);
                    let mut s = [vdupq_n_s32(0); 4];
                    for (j, sj) in s.iter_mut().enumerate() {
                        let p = xb.qs.as_ptr().add(32 * (4 * hf + j));
                        let t = sdot(vdupq_n_s32(0), q[j][0], vld1q_s8(p));
                        *sj = sdot(t, q[j][1], vld1q_s8(p.add(16)));
                    }
                    let sum = vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
                    let dx = vld1q_f32(xb.d.as_ptr().add(4 * hf));
                    acc[v][hf] = vfmaq_f32(acc[v][hf], vmulq_f32(dw[hf], dx), vcvtq_f32_s32(sum));
                }
            }
        }
        for v in 0..NV {
            *y.add(v * y_stride) = hsum4x2(acc[v][0], acc[v][1]);
        }
    }
}

/// Q5_0's block (ggml): fp16 d, 32 fifth bits (qh, where bit l belongs to weight l) and 16 bytes of nibbles (weights
/// 0-15 low, 16-31 high), with w = d x (q - 16) and q = nibble | fifth bit << 4.
pub const Q5_0_BLOCK: usize = 22;

/// Rows of Q5_0 against nv vectors. Each weight is unpacked to q - 16, an integer in -16..15, which is exactly a Q8_0
/// weight under the same scale; from there on the steps are q8_0_dotv's, so the bits are those of Q8_0 rows holding the
/// same weights (parity test).
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q5_0_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * 8 * Q5_0_BLOCK && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while v0 < nv {
                let (xs, yr) = (x.as_ptr().add(v0 * nb), y.add(v0 * y_stride + r));
                match nv - v0 {
                    1 => q5_0_dotv::<1>(wr, xs, nb, yr, y_stride),
                    2 => q5_0_dotv::<2>(wr, xs, nb, yr, y_stride),
                    3 => q5_0_dotv::<3>(wr, xs, nb, yr, y_stride),
                    _ => q5_0_dotv::<4>(wr, xs, nb, yr, y_stride),
                }
                v0 += 4;
            }
        }
    }
}

/// q5_0_matmul' pass: one row (w) against NV vectors (x, nb blocks each) into y[v * y_stride].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn q5_0_dotv<const NV: usize>(w: *const u8, x: *const Groups32, nb: usize, y: *mut f32, y_stride: usize) {
    unsafe {
        let (m15, sixteen) = (vdupq_n_u8(15), vdupq_n_u8(16));
        // byte l of a register takes byte l / 8 of qh's low two (weights 0-15) or high two (16-31), and tests bit l % 8
        const SPREAD: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        const BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let (sp, bit) = (vld1q_u8(SPREAD.as_ptr()), vld1q_u8(BIT.as_ptr()));
        let sp_hi = vaddq_u8(sp, vdupq_n_u8(2));
        let mut acc = [[vdupq_n_f32(0.0); 2]; NV];
        for b in 0..nb {
            let wb = w.add(b * 8 * Q5_0_BLOCK);
            let h = |k: usize| core::ptr::read_unaligned(wb.add(k * Q5_0_BLOCK) as *const u16) as u64;
            let w4 = |k: usize| h(k) | h(k + 1) << 16 | h(k + 2) << 32 | h(k + 3) << 48;
            let dw = [fcvtl_x(w4(0)), fcvtl_x(w4(4))];
            for hf in 0..2 {
                let mut q = [[vdupq_n_s8(0); 2]; 4];
                for (j, qj) in q.iter_mut().enumerate() {
                    let blk = wb.add((4 * hf + j) * Q5_0_BLOCK);
                    let qh = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(blk.add(2) as *const u32)));
                    let ns = vld1q_u8(blk.add(6));
                    let lo = vorrq_u8(vandq_u8(ns, m15), vandq_u8(vtstq_u8(vqtbl1q_u8(qh, sp), bit), sixteen));
                    let hi = vorrq_u8(vshrq_n_u8::<4>(ns), vandq_u8(vtstq_u8(vqtbl1q_u8(qh, sp_hi), bit), sixteen));
                    // q - 16 lies in -16..15, and the wrapped byte is its two's complement
                    *qj = [vreinterpretq_s8_u8(vsubq_u8(lo, sixteen)), vreinterpretq_s8_u8(vsubq_u8(hi, sixteen))];
                }
                for v in 0..NV {
                    let xb = &*x.add(v * nb + b);
                    let mut s = [vdupq_n_s32(0); 4];
                    for (j, sj) in s.iter_mut().enumerate() {
                        let p = xb.qs.as_ptr().add(32 * (4 * hf + j));
                        let t = sdot(vdupq_n_s32(0), q[j][0], vld1q_s8(p));
                        *sj = sdot(t, q[j][1], vld1q_s8(p.add(16)));
                    }
                    let sum = vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
                    let dx = vld1q_f32(xb.d.as_ptr().add(4 * hf));
                    acc[v][hf] = vfmaq_f32(acc[v][hf], vmulq_f32(dw[hf], dx), vcvtq_f32_s32(sum));
                }
            }
        }
        for v in 0..NV {
            *y.add(v * y_stride) = hsum4x2(acc[v][0], acc[v][1]);
        }
    }
}

/// IQ4_NL's 16 levels (ggml's kvalues_iq4nl).
const IQ4NL_LEVELS: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// One row of IQ4_NL (18-byte blocks laid out as Q4_0's: an fp16 scale and 16 bytes of nibbles, weights 0-15 low and
/// 16-31 high) against nb Q8_K blocks. Each block position k has a lane, acc[k] = fma(d_w * g.d[k], dot_k, acc[k]), and
/// the eight are added as ((0+4) + (2+6)) + ((1+5) + (3+7)).
///
/// # Safety
/// FEAT_DotProd; w holds x.len() * 8 blocks.
#[target_feature(enable = "dotprod")]
pub unsafe fn iq4nl_dot(w: &[u8], x: &[Groups32]) -> f32 {
    assert!(w.len() >= x.len() * 8 * Q4_0_BLOCK);
    let mut y = [0f32; 1];
    unsafe { iq4nl_dotr::<1>(w.as_ptr(), 0, x, y.as_mut_ptr()) };
    y[0]
}

/// Rows of IQ4_NL against ONE vector (single-token decode: the experts of ggml-org's Flash Next IQ4_NL), R rows a pass
/// (2 or 4) sharing each sub-block's activations, with the scales of a block's 8 sub-blocks widened whole. Each row's
/// float steps are iq4nl_dot's, so the bits are the same. The outputs go to y[r] for r < rows, and an odd last row runs
/// alone.
///
/// # Safety
/// FEAT_DotProd. (w holds rows * row_bytes, row_bytes >= x.len() * 8 * 18 and y >= rows: all asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn iq4nl_dot_rows(w: &[u8], row_bytes: usize, rows: usize, x: &[Groups32], y: &mut [f32], r: usize) {
    assert!(row_bytes >= x.len() * 8 * Q4_0_BLOCK && w.len() >= rows * row_bytes && y.len() >= rows);
    assert!(r == 1 || r == 2 || r == 4);
    let rr = rows / r * r;
    unsafe {
        for r0 in (0..rr).step_by(r) {
            let (wp, yp) = (w.as_ptr().add(r0 * row_bytes), y.as_mut_ptr().add(r0));
            match r {
                1 => iq4nl_dotr::<1>(wp, row_bytes, x, yp),
                2 => iq4nl_dotr::<2>(wp, row_bytes, x, yp),
                _ => iq4nl_dotr::<4>(wp, row_bytes, x, yp),
            }
        }
        for i in rr..rows {
            iq4nl_dotr::<1>(w.as_ptr().add(i * row_bytes), row_bytes, x, y.as_mut_ptr().add(i));
        }
    }
}

/// iq4nl_dot_rows' pass: R rows (at w, w + row_bytes, ..) into y[0..R].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn iq4nl_dotr<const R: usize>(w: *const u8, row_bytes: usize, x: &[Groups32], y: *mut f32) {
    unsafe {
        let m15 = vdupq_n_u8(15);
        let lv = vld1q_s8(IQ4NL_LEVELS.as_ptr());
        let mut acc0 = [vdupq_n_f32(0.0); R];
        let mut acc1 = [vdupq_n_f32(0.0); R];
        for (b, xb) in x.iter().enumerate() {
            let wb = |i: usize| w.add(i * row_bytes + b * 8 * Q4_0_BLOCK);
            let dx = [vld1q_f32(xb.d.as_ptr()), vld1q_f32(xb.d.as_ptr().add(4))];
            let mut dw = [[vdupq_n_f32(0.0); 2]; R];
            for (i, d) in dw.iter_mut().enumerate() {
                let (d0, d1) = q4_0_scales(wb(i));
                *d = [d0, d1];
            }
            macro_rules! half {
                ($h:literal, $acc:ident) => {{
                    let mut p = [[vdupq_n_s32(0); 2]; R];
                    for j in 0..2 {
                        let mut s = [[vdupq_n_s32(0); 2]; R];
                        for e in 0..2 {
                            let k = 4 * $h + 2 * j + e;
                            let a = xb.qs.as_ptr().add(32 * k);
                            let (a0, a1) = (vld1q_s8(a), vld1q_s8(a.add(16)));
                            for i in 0..R {
                                let q = vld1q_u8(wb(i).add(k * Q4_0_BLOCK + 2));
                                let t = sdot(vdupq_n_s32(0), vqtbl1q_s8(lv, vandq_u8(q, m15)), a0);
                                s[i][e] = sdot(t, vqtbl1q_s8(lv, vshrq_n_u8::<4>(q)), a1);
                            }
                        }
                        for i in 0..R {
                            p[i][j] = vpaddq_s32(s[i][0], s[i][1]);
                        }
                    }
                    for i in 0..R {
                        let v = vpaddq_s32(p[i][0], p[i][1]);
                        $acc[i] = vfmaq_f32($acc[i], vmulq_f32(dw[i][$h], dx[$h]), vcvtq_f32_s32(v));
                    }
                }};
            }
            half!(0, acc0);
            half!(1, acc1);
        }
        if R == 4 {
            let s: [float32x4_t; 4] = core::array::from_fn(|i| vaddq_f32(acc0[i], acc1[i]));
            let c = tr4(s.map(|v| vreinterpretq_s8_f32(v))).map(|v| vreinterpretq_f32_s8(v));
            vst1q_f32(y, vaddq_f32(vaddq_f32(c[0], c[2]), vaddq_f32(c[1], c[3])));
        } else {
            for i in 0..R {
                *y.add(i) = hsum4x2(acc0[i], acc1[i]);
            }
        }
    }
}

/// Rows of Q4_0 against a few vectors (an MTP verify: 2-3 tokens) by SDOT, using q4_0_dot_rows' scheme with the vectors
/// side by side. Each pass is 2 rows x 2 vectors (an odd vector or row runs by itself); each row's sub-block is
/// unpacked once for both vectors, and each (row, vector) has its own pair of accumulators with q4_0_dot's float steps,
/// so every vector gets the same bits as q4_0_dot. The outputs go to y[v * y_stride + r]. q4_0_matmul_var takes 3 or 4
/// vectors a pass.
///
/// # Safety
/// FEAT_DotProd; y is valid for writes at v * y_stride + r, and nothing else touches those elements during the call. (w
/// holds rows * row_bytes and x holds nv * nb blocks: asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q4_0_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe { q4_0_matmul_var(w, row_bytes, rows, x, nv, y, y_stride, 2, 2) }
}

/// q4_0_matmul with r rows (1 or 2) and up to v vectors (2, 3 or 4) a pass. For kernel benchmarks.
///
/// # Safety
/// As for q4_0_matmul.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "dotprod")]
pub unsafe fn q4_0_matmul_var(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
    r: usize,
    v: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv > 0 && x.len() == nv * nb && row_bytes >= nb * 8 * Q4_0_BLOCK && w.len() >= rows * row_bytes);
    assert!((r == 1 || r == 2) && (2..=4).contains(&v));
    unsafe {
        let mut r0 = 0;
        while r0 < rows {
            let wp = w.as_ptr().add(r0 * row_bytes);
            let two = r == 2 && r0 + 1 < rows;
            // the vector groups run one after another over the same rows (which are in L1 by then)
            let mut v0 = 0;
            while v0 < nv {
                let g = if nv - v0 >= v {
                    v
                } else if nv - v0 >= 2 {
                    2
                } else {
                    1
                };
                let yp = y.add(v0 * y_stride + r0);
                match (two, g) {
                    (true, 4) => q4_0_dotrv::<2, 4>(wp, row_bytes, x, nb, v0, yp, y_stride),
                    (true, 3) => q4_0_dotrv::<2, 3>(wp, row_bytes, x, nb, v0, yp, y_stride),
                    (true, 2) => q4_0_dotrv::<2, 2>(wp, row_bytes, x, nb, v0, yp, y_stride),
                    (true, _) => q4_0_dotrv::<2, 1>(wp, row_bytes, x, nb, v0, yp, y_stride),
                    (false, 4) => q4_0_dotrv::<1, 4>(wp, row_bytes, x, nb, v0, yp, y_stride),
                    (false, 3) => q4_0_dotrv::<1, 3>(wp, row_bytes, x, nb, v0, yp, y_stride),
                    (false, 2) => q4_0_dotrv::<1, 2>(wp, row_bytes, x, nb, v0, yp, y_stride),
                    (false, _) => q4_0_dotrv::<1, 1>(wp, row_bytes, x, nb, v0, yp, y_stride),
                }
                v0 += g;
            }
            r0 += if two { 2 } else { 1 };
        }
    }
}

/// q4_0_matmul' pass: R rows (at w, w + row_bytes, ...) x vectors v0..v0 + V (vector v at x[v * nb..]) into y[v *
/// y_stride + i] (y already points at vector v0 and the pass's first row).
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn q4_0_dotrv<const R: usize, const V: usize>(
    w: *const u8,
    row_bytes: usize,
    x: &[Groups32],
    nb: usize,
    v0: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe {
        let m15 = vdupq_n_u8(15);
        let mut acc0 = [[vdupq_n_f32(0.0); V]; R];
        let mut acc1 = [[vdupq_n_f32(0.0); V]; R];
        for b in 0..nb {
            let wb = |i: usize| w.add(i * row_bytes + b * 8 * Q4_0_BLOCK);
            let xb = |v: usize| x.get_unchecked((v0 + v) * nb + b);
            let mut dw = [[vdupq_n_f32(0.0); 2]; R];
            for (i, d) in dw.iter_mut().enumerate() {
                let (d0, d1) = q4_0_scales(wb(i));
                *d = [d0, d1];
            }
            // as in q4_0_dot_rows: a macro per half, so the accumulators stay in registers
            macro_rules! half {
                ($h:literal, $acc:ident) => {{
                    let mut p = [[[vdupq_n_s32(0); 2]; V]; R];
                    for j in 0..2 {
                        let mut s = [[[vdupq_n_s32(0); 2]; V]; R];
                        for e in 0..2 {
                            let k = 4 * $h + 2 * j + e;
                            let mut q = [[vdupq_n_s8(0); 2]; R];
                            for (i, qi) in q.iter_mut().enumerate() {
                                let b = vld1q_u8(wb(i).add(k * Q4_0_BLOCK + 2));
                                *qi = [vreinterpretq_s8_u8(vandq_u8(b, m15)), vreinterpretq_s8_u8(vshrq_n_u8::<4>(b))];
                            }
                            for v in 0..V {
                                let a = xb(v).qs.as_ptr().add(32 * k);
                                let (a0, a1) = (vld1q_s8(a), vld1q_s8(a.add(16)));
                                for i in 0..R {
                                    s[i][v][e] = sdot(sdot(vdupq_n_s32(0), q[i][0], a0), q[i][1], a1);
                                }
                            }
                        }
                        for i in 0..R {
                            for v in 0..V {
                                p[i][v][j] = vpaddq_s32(s[i][v][0], s[i][v][1]);
                            }
                        }
                    }
                    for v in 0..V {
                        // 8 x the activation sums of sub-blocks 4h..4h+3, widened
                        let bias = vshll_n_s16::<3>(vld1_s16(xb(v).sums.as_ptr().add(4 * $h)));
                        let dx = vld1q_f32(xb(v).d.as_ptr().add(4 * $h));
                        for i in 0..R {
                            let t = vsubq_s32(vpaddq_s32(p[i][v][0], p[i][v][1]), bias);
                            $acc[i][v] = vfmaq_f32($acc[i][v], vmulq_f32(dw[i][$h], dx), vcvtq_f32_s32(t));
                        }
                    }
                }};
            }
            half!(0, acc0);
            half!(1, acc1);
        }
        for i in 0..R {
            for v in 0..V {
                *y.add(v * y_stride + i) = hsum4x2(acc0[i][v], acc1[i][v]);
            }
        }
    }
}

/// FCVTN + FCVTN2: 8 f32 values rounded to fp16 bits (round to nearest even, as vcvt_f16_f32 and F16C's mode 0 do).
#[inline]
unsafe fn fcvtn8(lo: float32x4_t, hi: float32x4_t) -> uint16x8_t {
    let o: uint16x8_t;
    unsafe {
        asm!("fcvtn {o:v}.4h, {a:v}.4s", "fcvtn2 {o:v}.8h, {b:v}.4s", o = out(vreg) o, a = in(vreg) lo, b = in(vreg) hi,
            options(pure, nomem, nostack));
    }
    o
}

/// FCVTL2: the high 4 of 8 fp16 values, widened.
#[inline]
unsafe fn fcvtl2(h: uint16x8_t) -> float32x4_t {
    let o: float32x4_t;
    unsafe {
        asm!("fcvtl2 {o:v}.4s, {h:v}.8h", o = out(vreg) o, h = in(vreg) h, options(pure, nomem, nostack));
    }
    o
}

/// ((0+4) + (2+6)) + ((1+5) + (3+7)) over lanes 0-3 in a and 4-7 in b.
#[inline]
unsafe fn hsum4x2(a: float32x4_t, b: float32x4_t) -> f32 {
    unsafe {
        let s = vaddq_f32(a, b);
        let t = vadd_f32(vget_low_f32(s), vget_high_f32(s));
        vget_lane_f32::<0>(t) + vget_lane_f32::<1>(t)
    }
}

/// The gated delta rule step on 4 fp16 state rows at a time. Each row's steps are its own and unchanged (times alpha,
/// then the dot with k, d = (v - sk) * beta, row + d k rounded to fp16, the dot with q, times q_scale); they are only
/// interleaved, so that four rows' dependent sums run side by side instead of one after another. The bits are the same.
/// `state` holds rows x dim fp16 bits, v and o hold rows values; dim must be a multiple of 16 (and <= 256), rows a
/// multiple of 4.
pub fn delta_rule_h(
    state: &mut [u16],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    beta: f32,
    q_scale: f32,
    o: &mut [f32],
    dim: usize,
) {
    let rows = v.len();
    assert!(
        dim.is_multiple_of(16) && dim <= 256 && rows.is_multiple_of(4) && o.len() == rows && state.len() >= rows * dim
    );
    assert!(q.len() >= dim && k.len() >= dim);
    // SAFETY: NEON and the FP16 conversions are baseline on aarch64, and every access is inside the asserted lengths
    unsafe {
        let alpha_v = vdupq_n_f32(alpha);
        let mut r = [[0.0f32; 256]; 4];
        for j0 in (0..rows).step_by(4) {
            let mut a = [[vdupq_n_f32(0.0); 4]; 4];
            for i in (0..dim).step_by(16) {
                let kv = [
                    vld1q_f32(k.as_ptr().add(i)),
                    vld1q_f32(k.as_ptr().add(i + 4)),
                    vld1q_f32(k.as_ptr().add(i + 8)),
                    vld1q_f32(k.as_ptr().add(i + 12)),
                ];
                for (rr, row) in r.iter_mut().enumerate() {
                    let mp = state.as_ptr().add((j0 + rr) * dim + i);
                    for h in 0..2 {
                        let f = vld1q_u16(mp.add(8 * h));
                        let x0 = vmulq_f32(fcvtl(vget_low_u16(f)), alpha_v);
                        let x1 = vmulq_f32(fcvtl2(f), alpha_v);
                        vst1q_f32(row.as_mut_ptr().add(i + 8 * h), x0);
                        vst1q_f32(row.as_mut_ptr().add(i + 8 * h + 4), x1);
                        a[rr][2 * h] = vfmaq_f32(a[rr][2 * h], x0, kv[2 * h]);
                        a[rr][2 * h + 1] = vfmaq_f32(a[rr][2 * h + 1], x1, kv[2 * h + 1]);
                    }
                }
            }
            let mut delta = [vdupq_n_f32(0.0); 4];
            for rr in 0..4 {
                let sk = hsum4x2(vaddq_f32(a[rr][0], a[rr][2]), vaddq_f32(a[rr][1], a[rr][3]));
                delta[rr] = vdupq_n_f32((v[j0 + rr] - sk) * beta);
            }
            let mut a = [[vdupq_n_f32(0.0); 4]; 4];
            for i in (0..dim).step_by(16) {
                let kv = [
                    vld1q_f32(k.as_ptr().add(i)),
                    vld1q_f32(k.as_ptr().add(i + 4)),
                    vld1q_f32(k.as_ptr().add(i + 8)),
                    vld1q_f32(k.as_ptr().add(i + 12)),
                ];
                let qv = [
                    vld1q_f32(q.as_ptr().add(i)),
                    vld1q_f32(q.as_ptr().add(i + 4)),
                    vld1q_f32(q.as_ptr().add(i + 8)),
                    vld1q_f32(q.as_ptr().add(i + 12)),
                ];
                for (rr, row) in r.iter().enumerate() {
                    let mp = state.as_mut_ptr().add((j0 + rr) * dim + i);
                    for h in 0..2 {
                        let x0 = vfmaq_f32(vld1q_f32(row.as_ptr().add(i + 8 * h)), delta[rr], kv[2 * h]);
                        let x1 = vfmaq_f32(vld1q_f32(row.as_ptr().add(i + 8 * h + 4)), delta[rr], kv[2 * h + 1]);
                        vst1q_u16(mp.add(8 * h), fcvtn8(x0, x1));
                        a[rr][2 * h] = vfmaq_f32(a[rr][2 * h], x0, qv[2 * h]);
                        a[rr][2 * h + 1] = vfmaq_f32(a[rr][2 * h + 1], x1, qv[2 * h + 1]);
                    }
                }
            }
            for rr in 0..4 {
                o[j0 + rr] = hsum4x2(vaddq_f32(a[rr][0], a[rr][2]), vaddq_f32(a[rr][1], a[rr][3])) * q_scale;
            }
        }
    }
}

/// The bytes of a Q6_K block: ql[128] (the low nibbles), qh[64] (the high pairs), 16 int8 group scales and fp16 d.
pub const Q6_K_BLOCK: usize = 210;

/// One pair of vectors' Q6_K block, packed for gemm_q6k: SMMLA's B operand for each of the 16 groups of 16 weights
/// (b[2g] holds activations 16g..16g+7 of both vectors and b[2g + 1] holds 16g+8..16g+15, from Q8_K's qs in weight
/// order), and the blocks' scales per lane as [d_v0, d_v1, d_v0, d_v1].
#[derive(Clone, Copy)]
pub struct PackedQ6 {
    b: [int8x16_t; 32],
    d: float32x4_t,
}

impl Default for PackedQ6 {
    fn default() -> PackedQ6 {
        // SAFETY: plain vectors, for which all-zero bits are valid
        unsafe { core::mem::zeroed() }
    }
}

/// The number of PackedQ6 entries for nv vectors of nb blocks.
pub fn pack_q6_len(nv: usize, nb: usize) -> usize {
    nv.div_ceil(2) * nb
}

/// Packs x (nv x nb blocks) for gemm_q6k: entry p * nb + b holds vectors 2p and 2p + 1 (an odd last vector is paired
/// with itself).
pub fn pack_q6(x: &[BlockQ8K], nv: usize, out: &mut [PackedQ6]) {
    let nb = x.len() / nv;
    assert!(nv > 0 && x.len() == nv * nb && out.len() >= pack_q6_len(nv, nb));
    unsafe {
        for p in 0..nv.div_ceil(2) {
            let (v0, v1) = (2 * p, (2 * p + 1).min(nv - 1));
            for b in 0..nb {
                let (x0, x1) = (&x[v0 * nb + b], &x[v1 * nb + b]);
                let e = &mut out[p * nb + b];
                for g in 0..16 {
                    let (lo, hi) = pair64(vld1q_s8(x0.qs.as_ptr().add(16 * g)), vld1q_s8(x1.qs.as_ptr().add(16 * g)));
                    e.b[2 * g] = lo;
                    e.b[2 * g + 1] = hi;
                }
                let dv = [x0.d, x1.d, x0.d, x1.d];
                e.d = vld1q_f32(dv.as_ptr());
            }
        }
    }
}

/// One Q6_K block of a row as 16 registers of signed weights (q - 32), where register g holds weights 16g..16g+15.
#[inline]
unsafe fn q6k_unpack(blk: *const u8, q: &mut [int8x16_t; 16]) {
    unsafe {
        let (m4, m2, c32) = (vdupq_n_u8(15), vdupq_n_u8(3), vdupq_n_s8(32));
        for n in 0..2 {
            let (ql, qh) = (blk.add(64 * n), blk.add(128 + 32 * n));
            for h in 0..2 {
                let l0 = vld1q_u8(ql.add(16 * h));
                let l1 = vld1q_u8(ql.add(32 + 16 * h));
                let hh = vld1q_u8(qh.add(16 * h));
                let base = 8 * n + h;
                let mk = |lo: uint8x16_t, hi2: uint8x16_t| {
                    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo, vshlq_n_u8::<4>(hi2))), c32)
                };
                q[base] = mk(vandq_u8(l0, m4), vandq_u8(hh, m2));
                q[base + 2] = mk(vandq_u8(l1, m4), vandq_u8(vshrq_n_u8::<2>(hh), m2));
                q[base + 4] = mk(vshrq_n_u8::<4>(l0), vandq_u8(vshrq_n_u8::<4>(hh), m2));
                q[base + 6] = mk(vshrq_n_u8::<4>(l1), vshrq_n_u8::<6>(hh));
            }
        }
    }
}

/// Rows of Q6_K against ONE vector (single-token decode: Qwen's vocabulary head and some FFN tensors), a row a pass.
/// For each block of 256: the exact integer sum of scale_g x sum((q - 32) a) over the 16 groups, then one fused step,
/// acc = fma(d_w * d_x, float(sum), acc). This is gemm_q6k's order for each (row, vector), giving the same bits as the
/// reference (parity test). The outputs go to y[r] for r < rows.
///
/// # Safety
/// FEAT_DotProd. (w holds rows * row_bytes, row_bytes >= x.len() * 210 and y >= rows: all asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q6k_dot_rows(w: &[u8], row_bytes: usize, rows: usize, x: &[BlockQ8K], y: &mut [f32]) {
    assert!(row_bytes >= x.len() * Q6_K_BLOCK && w.len() >= rows * row_bytes && y.len() >= rows);
    // a row a pass: a block's 16 unpacked registers and the 16 activation registers fill the register file
    unsafe {
        for i in 0..rows {
            q6k_dotr::<1>(w.as_ptr().add(i * row_bytes), row_bytes, x, y.as_mut_ptr().add(i));
        }
    }
}

/// Q4_K's block: fp16 d, fp16 dmin, 12 bytes of 6-bit scales and mins, and 128 bytes of nibbles.
pub const Q4_K_BLOCK: usize = 144;
/// Q5_K's block: as Q4_K's, with 32 bytes of fifth bits (qh) before the nibbles.
pub const Q5_K_BLOCK: usize = 176;

/// Q4_K (five = false) or Q5_K (five = true) weights times nv vectors (x is nv x nb blocks; output (v, r) goes to
/// y[v * y_stride + r]), one row per pass, with each block unpacked once and shared by up to 4 vectors. Per block, as
/// in ggml's Q4_K dot product: sumi = sum of sc[j] (q . a over sub-block j) and summs = sum of m[j] (bsum of sub-block
/// j), both exact in int32; the row sums d d8 sumi into one f32 chain and dmin d8 summs into a second (each a fused
/// step), and returns the first minus the second. The bits are the same as the reference (parity test).
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q45k_matmul(
    five: bool,
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    let bs = if five { Q5_K_BLOCK } else { Q4_K_BLOCK };
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * bs && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while v0 < nv {
                let (xs, yr) = (x.as_ptr().add(v0 * nb), y.add(v0 * y_stride + r));
                macro_rules! go {
                    ($n:literal) => {
                        if five {
                            q45k_vec_dot::<$n, true>(wr, xs, nb, yr, y_stride)
                        } else {
                            q45k_vec_dot::<$n, false>(wr, xs, nb, yr, y_stride)
                        }
                    };
                }
                match nv - v0 {
                    1 => go!(1),
                    2 => go!(2),
                    3 => go!(3),
                    _ => go!(4),
                }
                v0 += 4;
            }
        }
    }
}

/// q45k_matmul' pass: one row (w) against NV vectors (x, nb blocks each) into y[v * y_stride].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn q45k_vec_dot<const NV: usize, const FIVE: bool>(
    w: *const u8,
    x: *const BlockQ8K,
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe {
        let (m4, one) = (vdupq_n_u8(15), vdupq_n_u8(1));
        let bs = if FIVE { Q5_K_BLOCK } else { Q4_K_BLOCK };
        let (mut acc, mut accm) = ([0f32; NV], [0f32; NV]);
        for b in 0..nb {
            let blk = w.add(b * bs);
            let qs = blk.add(if FIVE { 48 } else { 16 });
            let (sc, mn) = crate::q45k_scales(&*(blk.add(4) as *const [u8; 12]));
            let sc16 = vmovl_u8(vld1_u8(sc.as_ptr()));
            let m16 = vmovl_u8(vld1_u8(mn.as_ptr()));
            let sc0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sc16)));
            let sc1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sc16)));
            let mm0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(m16)));
            let mm1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(m16)));
            let (h0, h1) =
                if FIVE { (vld1q_u8(blk.add(16)), vld1q_u8(blk.add(32))) } else { (vdupq_n_u8(0), vdupq_n_u8(0)) };
            // q[4j .. 4j + 3] holds sub-block 2j (the low nibbles) in two registers, then 2j + 1 (the high ones) in two
            let mut q = [vdupq_n_s8(0); 16];
            for j in 0..4 {
                let lo = vld1q_u8(qs.add(32 * j));
                let hi = vld1q_u8(qs.add(32 * j + 16));
                let (mut q0, mut q1) = (vandq_u8(lo, m4), vandq_u8(hi, m4));
                let (mut q2, mut q3) = (vshrq_n_u8::<4>(lo), vshrq_n_u8::<4>(hi));
                if FIVE {
                    // bit s of qh[l] is the fifth bit of weight 32 s + l
                    let r0 = vdupq_n_s8(-(2 * j as i8));
                    let r1 = vdupq_n_s8(-(2 * j as i8 + 1));
                    q0 = vorrq_u8(q0, vshlq_n_u8::<4>(vandq_u8(vshlq_u8(h0, r0), one)));
                    q1 = vorrq_u8(q1, vshlq_n_u8::<4>(vandq_u8(vshlq_u8(h1, r0), one)));
                    q2 = vorrq_u8(q2, vshlq_n_u8::<4>(vandq_u8(vshlq_u8(h0, r1), one)));
                    q3 = vorrq_u8(q3, vshlq_n_u8::<4>(vandq_u8(vshlq_u8(h1, r1), one)));
                }
                q[4 * j] = vreinterpretq_s8_u8(q0);
                q[4 * j + 1] = vreinterpretq_s8_u8(q1);
                q[4 * j + 2] = vreinterpretq_s8_u8(q2);
                q[4 * j + 3] = vreinterpretq_s8_u8(q3);
            }
            let fd = crate::f16_to_f32(core::ptr::read_unaligned(blk as *const u16));
            let fm = crate::f16_to_f32(core::ptr::read_unaligned(blk.add(2) as *const u16));
            for v in 0..NV {
                let xb = &*x.add(v * nb + b);
                let p: [int32x4_t; 8] = core::array::from_fn(|k| {
                    let a = xb.qs.as_ptr().add(32 * k);
                    sdot(sdot(vdupq_n_s32(0), q[2 * k], vld1q_s8(a)), q[2 * k + 1], vld1q_s8(a.add(16)))
                });
                let red = |a: int32x4_t, b: int32x4_t, c: int32x4_t, d: int32x4_t| {
                    vpaddq_s32(vpaddq_s32(a, b), vpaddq_s32(c, d))
                };
                let si = vmlaq_s32(vmulq_s32(red(p[0], p[1], p[2], p[3]), sc0), red(p[4], p[5], p[6], p[7]), sc1);
                // each sub-block's activation sum: its two bsums
                let bsv = vpaddq_s16(vld1q_s16(xb.bsums.as_ptr()), vld1q_s16(xb.bsums.as_ptr().add(8)));
                let sm = vmlaq_s32(vmulq_s32(vmovl_s16(vget_low_s16(bsv)), mm0), vmovl_high_s16(bsv), mm1);
                acc[v] = (fd * xb.d).mul_add(vaddvq_s32(si) as f32, acc[v]);
                accm[v] = (fm * xb.d).mul_add(vaddvq_s32(sm) as f32, accm[v]);
            }
        }
        for v in 0..NV {
            *y.add(v * y_stride) = acc[v] - accm[v];
        }
    }
}

/// Rows of Q3_K against nv vectors (x is nv x nb blocks, and vector v's outputs go to y[v * y_stride + r]), a row a
/// pass, each block unpacked once for up to 4 vectors. For each vector and block it forms the reference's exact integer
/// sumi = the sum over the 16 groups of scale_g x sum(value x a), computed here as the sum of scale_g x (u . a) less 4
/// x the sum of scale_g x the group's activation sum (with u = value + 4, in 0..7); then acc = fma(d x d8, sumi, acc).
/// The bits are the same (parity test).
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q3k_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * crate::Q3_K_BLOCK && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while v0 < nv {
                let (xs, yr) = (x.as_ptr().add(v0 * nb), y.add(v0 * y_stride + r));
                match nv - v0 {
                    1 => q3k_dotv::<1>(wr, xs, nb, yr, y_stride),
                    2 => q3k_dotv::<2>(wr, xs, nb, yr, y_stride),
                    3 => q3k_dotv::<3>(wr, xs, nb, yr, y_stride),
                    _ => q3k_dotv::<4>(wr, xs, nb, yr, y_stride),
                }
                v0 += 4;
            }
        }
    }
}

/// q3k_matmul' pass: one row (w) against NV vectors (x, nb blocks each) into y[v * y_stride].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn q3k_dotv<const NV: usize>(w: *const u8, x: *const BlockQ8K, nb: usize, y: *mut f32, y_stride: usize) {
    unsafe {
        let (three, one) = (vdupq_n_u8(3), vdupq_n_u8(1));
        let mut acc = [0f32; NV];
        for b in 0..nb {
            let blk = w.add(b * crate::Q3_K_BLOCK);
            let sc = crate::q3k_unpack_scales(&*(blk.add(96) as *const [u8; 12]));
            let sc8 = vld1q_s8(sc.as_ptr());
            let (sl, sh) = (vmovl_s8(vget_low_s8(sc8)), vmovl_high_s8(sc8));
            let sv = [vmovl_s16(vget_low_s16(sl)), vmovl_high_s16(sl), vmovl_s16(vget_low_s16(sh)), vmovl_high_s16(sh)];
            // group g = 8n + 2j + hh comes from qs[32n + 16hh ..] >> 2j and hmask[16hh ..] >> (j + 4n), with u =
            // value + 4
            let hm = [vld1q_u8(blk), vld1q_u8(blk.add(16))];
            let q: [int8x16_t; 16] = core::array::from_fn(|g| {
                let (n, j, hh) = (g / 8, g % 8 / 2, g % 2);
                let qs = vld1q_u8(blk.add(32 + 32 * n + 16 * hh));
                let lo = vandq_u8(vshlq_u8(qs, vdupq_n_s8(-(2 * j as i8))), three);
                let hi = vandq_u8(vshlq_u8(hm[hh], vdupq_n_s8(-((j + 4 * n) as i8))), one);
                vreinterpretq_s8_u8(vorrq_u8(lo, vshlq_n_u8::<2>(hi)))
            });
            let fd = crate::f16_to_f32(core::ptr::read_unaligned(blk.add(108) as *const u16));
            for v in 0..NV {
                let xb = &*x.add(v * nb + b);
                let red = |i: usize| {
                    let s: [int32x4_t; 4] = core::array::from_fn(|k| {
                        let g = 4 * i + k;
                        sdot(vdupq_n_s32(0), q[g], vld1q_s8(xb.qs.as_ptr().add(16 * g)))
                    });
                    vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]))
                };
                let mut t = vmulq_s32(red(0), sv[0]);
                t = vmlaq_s32(t, red(1), sv[1]);
                t = vmlaq_s32(t, red(2), sv[2]);
                t = vmlaq_s32(t, red(3), sv[3]);
                // the -4 in every value: 4 x the sum of scale_g x the group's activation sum
                let bs = [vld1q_s16(xb.bsums.as_ptr()), vld1q_s16(xb.bsums.as_ptr().add(8))];
                let mut c = vmull_s16(vget_low_s16(bs[0]), vget_low_s16(sl));
                c = vmlal_high_s16(c, bs[0], sl);
                c = vmlal_s16(c, vget_low_s16(bs[1]), vget_low_s16(sh));
                c = vmlal_high_s16(c, bs[1], sh);
                let sumi = vaddvq_s32(t) - 4 * vaddvq_s32(c);
                acc[v] = (fd * xb.d).mul_add(sumi as f32, acc[v]);
            }
        }
        for v in 0..NV {
            *y.add(v * y_stride) = acc[v];
        }
    }
}

/// Rows of IQ4_XS against nv vectors, a row a pass, with each block's levels looked up once for up to 4 vectors. For
/// each vector and block: sumi = the sum over the 8 groups of scale_g x (level . a) (exact), then acc = fma(d x d8,
/// sumi, acc). The bits are the same as the reference (parity test).
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn iq4xs_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * crate::IQ4_XS_BLOCK && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while v0 < nv {
                let (xs, yr) = (x.as_ptr().add(v0 * nb), y.add(v0 * y_stride + r));
                match nv - v0 {
                    1 => iq4xs_dotv::<1>(wr, xs, nb, yr, y_stride),
                    2 => iq4xs_dotv::<2>(wr, xs, nb, yr, y_stride),
                    3 => iq4xs_dotv::<3>(wr, xs, nb, yr, y_stride),
                    _ => iq4xs_dotv::<4>(wr, xs, nb, yr, y_stride),
                }
                v0 += 4;
            }
        }
    }
}

/// iq4xs_matmul' pass: one row (w) against NV vectors (x, nb blocks each) into y[v * y_stride].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn iq4xs_dotv<const NV: usize>(w: *const u8, x: *const BlockQ8K, nb: usize, y: *mut f32, y_stride: usize) {
    unsafe {
        let (m4, lv) = (vdupq_n_u8(15), vld1q_s8(crate::IQ4_LEVELS.as_ptr()));
        let mut acc = [0f32; NV];
        for b in 0..nb {
            let blk = w.add(b * crate::IQ4_XS_BLOCK);
            let sc = crate::iq4xs_scales(core::slice::from_raw_parts(blk, 8));
            let (sc0, sc1) = (vld1q_s32(sc.as_ptr()), vld1q_s32(sc.as_ptr().add(4)));
            let mut w0 = [vdupq_n_s8(0); 8];
            let mut w1 = [vdupq_n_s8(0); 8];
            for g in 0..8 {
                let q = vld1q_u8(blk.add(8 + 16 * g));
                w0[g] = vqtbl1q_s8(lv, vandq_u8(q, m4));
                w1[g] = vqtbl1q_s8(lv, vshrq_n_u8::<4>(q));
            }
            let fd = crate::f16_to_f32(core::ptr::read_unaligned(blk as *const u16));
            for v in 0..NV {
                let xb = &*x.add(v * nb + b);
                let p: [int32x4_t; 8] = core::array::from_fn(|g| {
                    let a = xb.qs.as_ptr().add(32 * g);
                    sdot(sdot(vdupq_n_s32(0), w0[g], vld1q_s8(a)), w1[g], vld1q_s8(a.add(16)))
                });
                let red = |a: int32x4_t, b: int32x4_t, c: int32x4_t, d: int32x4_t| {
                    vpaddq_s32(vpaddq_s32(a, b), vpaddq_s32(c, d))
                };
                let si = vmlaq_s32(vmulq_s32(red(p[0], p[1], p[2], p[3]), sc0), red(p[4], p[5], p[6], p[7]), sc1);
                acc[v] = (fd * xb.d).mul_add(vaddvq_s32(si) as f32, acc[v]);
            }
        }
        for v in 0..NV {
            *y.add(v * y_stride) = acc[v];
        }
    }
}

/// Rows of IQ3_S against nv vectors, a row a pass, with each block's levels and signs made once for up to 4 vectors.
/// For each vector and block it forms the reference's exact sumi = the sum over the 8 groups of scale_g x (signed level
/// . a), then acc = fma(d x d8, sumi, acc). The bits are the same (parity test).
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn iq3s_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * crate::IQ3_S_BLOCK && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while v0 < nv {
                let (xs, yr) = (x.as_ptr().add(v0 * nb), y.add(v0 * y_stride + r));
                match nv - v0 {
                    1 => iq3s_dotv::<1>(wr, xs, nb, yr, y_stride),
                    2 => iq3s_dotv::<2>(wr, xs, nb, yr, y_stride),
                    3 => iq3s_dotv::<3>(wr, xs, nb, yr, y_stride),
                    _ => iq3s_dotv::<4>(wr, xs, nb, yr, y_stride),
                }
                v0 += 4;
            }
        }
    }
}

/// iq3s_matmul' pass: one row (w) against NV vectors (x, nb blocks each) into y[v * y_stride].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn iq3s_dotv<const NV: usize>(w: *const u8, x: *const BlockQ8K, nb: usize, y: *mut f32, y_stride: usize) {
    unsafe {
        // a sign byte spread over its 8 weights: byte l takes byte l / 8 and tests bit l % 8
        const SPREAD: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        const BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let (sp, bit) = (vld1q_u8(SPREAD.as_ptr()), vld1q_u8(BIT.as_ptr()));
        let mut acc = [0f32; NV];
        for b in 0..nb {
            let blk = w.add(b * crate::IQ3_S_BLOCK);
            let bs = core::slice::from_raw_parts(blk, crate::IQ3_S_BLOCK);
            let sc: [i32; 8] = core::array::from_fn(|g| crate::iq3s_group_scale(bs, g));
            let (sc0, sc1) = (vld1q_s32(sc.as_ptr()), vld1q_s32(sc.as_ptr().add(4)));
            // the signed levels: (level ^ m) - m, with m all ones where the sign bit is set
            let mut q = [vdupq_n_s8(0); 16];
            for g in 0..8 {
                let (l, s) = crate::iq3s_group(bs, g);
                let sv = vreinterpretq_u8_u32(vdupq_n_u32(s));
                for hh in 0..2 {
                    let m = vtstq_u8(vqtbl1q_u8(sv, vaddq_u8(sp, vdupq_n_u8(2 * hh as u8))), bit);
                    let lv = vld1q_u8(l.as_ptr().add(16 * hh));
                    q[2 * g + hh] = vreinterpretq_s8_u8(vsubq_u8(veorq_u8(lv, m), m));
                }
            }
            let fd = crate::f16_to_f32(core::ptr::read_unaligned(blk as *const u16));
            for v in 0..NV {
                let xb = &*x.add(v * nb + b);
                let p: [int32x4_t; 8] = core::array::from_fn(|g| {
                    let a = xb.qs.as_ptr().add(32 * g);
                    sdot(sdot(vdupq_n_s32(0), q[2 * g], vld1q_s8(a)), q[2 * g + 1], vld1q_s8(a.add(16)))
                });
                let red = |a: int32x4_t, b: int32x4_t, c: int32x4_t, d: int32x4_t| {
                    vpaddq_s32(vpaddq_s32(a, b), vpaddq_s32(c, d))
                };
                let si = vmlaq_s32(vmulq_s32(red(p[0], p[1], p[2], p[3]), sc0), red(p[4], p[5], p[6], p[7]), sc1);
                acc[v] = (fd * xb.d).mul_add(vaddvq_s32(si) as f32, acc[v]);
            }
        }
        for v in 0..NV {
            *y.add(v * y_stride) = acc[v];
        }
    }
}

/// Rows of IQ3_XXS against nv vectors, a row a pass, with each block's levels and signs made once for up to 4 vectors.
/// For each vector and block it forms the reference's exact sumi = the sum over the 8 groups of scale_g x (signed level
/// . a), then acc = fma((d / 4) x d8, sumi, acc). The bits are the same (parity test).
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn iq3xxs_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * crate::IQ3_XXS_BLOCK && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while v0 < nv {
                let (xs, yr) = (x.as_ptr().add(v0 * nb), y.add(v0 * y_stride + r));
                match nv - v0 {
                    1 => iq3xxs_dotv::<1>(wr, xs, nb, yr, y_stride),
                    2 => iq3xxs_dotv::<2>(wr, xs, nb, yr, y_stride),
                    3 => iq3xxs_dotv::<3>(wr, xs, nb, yr, y_stride),
                    _ => iq3xxs_dotv::<4>(wr, xs, nb, yr, y_stride),
                }
                v0 += 4;
            }
        }
    }
}

/// iq3xxs_matmul' pass: one row (w) against NV vectors (x, nb blocks each) into y[v * y_stride].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn iq3xxs_dotv<const NV: usize>(w: *const u8, x: *const BlockQ8K, nb: usize, y: *mut f32, y_stride: usize) {
    unsafe {
        // a sign byte spread over its 8 weights: byte l takes byte l / 8 and tests bit l % 8
        const SPREAD: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        const BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let (sp, bit) = (vld1q_u8(SPREAD.as_ptr()), vld1q_u8(BIT.as_ptr()));
        let mut acc = [0f32; NV];
        for b in 0..nb {
            let blk = w.add(b * crate::IQ3_XXS_BLOCK);
            let bs = core::slice::from_raw_parts(blk, crate::IQ3_XXS_BLOCK);
            let sc: [i32; 8] = core::array::from_fn(|g| crate::iq3xxs_group_scale(bs, g));
            let (sc0, sc1) = (vld1q_s32(sc.as_ptr()), vld1q_s32(sc.as_ptr().add(4)));
            // the signed levels: (level ^ m) - m, with m all ones where the sign bit is set
            let mut q = [vdupq_n_s8(0); 16];
            for g in 0..8 {
                let (l, s) = crate::iq3xxs_group(bs, g);
                let sv = vreinterpretq_u8_u32(vdupq_n_u32(s));
                for hh in 0..2 {
                    let m = vtstq_u8(vqtbl1q_u8(sv, vaddq_u8(sp, vdupq_n_u8(2 * hh as u8))), bit);
                    let lv = vld1q_u8(l.as_ptr().add(16 * hh));
                    q[2 * g + hh] = vreinterpretq_s8_u8(vsubq_u8(veorq_u8(lv, m), m));
                }
            }
            let fd = crate::f16_to_f32(core::ptr::read_unaligned(blk as *const u16)) * 0.25;
            for v in 0..NV {
                let xb = &*x.add(v * nb + b);
                let p: [int32x4_t; 8] = core::array::from_fn(|g| {
                    let a = xb.qs.as_ptr().add(32 * g);
                    sdot(sdot(vdupq_n_s32(0), q[2 * g], vld1q_s8(a)), q[2 * g + 1], vld1q_s8(a.add(16)))
                });
                let red = |a: int32x4_t, b: int32x4_t, c: int32x4_t, d: int32x4_t| {
                    vpaddq_s32(vpaddq_s32(a, b), vpaddq_s32(c, d))
                };
                let si = vmlaq_s32(vmulq_s32(red(p[0], p[1], p[2], p[3]), sc0), red(p[4], p[5], p[6], p[7]), sc1);
                acc[v] = (fd * xb.d).mul_add(vaddvq_s32(si) as f32, acc[v]);
            }
        }
        for v in 0..NV {
            *y.add(v * y_stride) = acc[v];
        }
    }
}

/// Rows of IQ2_S against nv vectors, a row a pass, with each block's levels and signs made once for up to 4 vectors.
/// For each vector and block it forms the reference's exact sumi = the sum over the 16 runs of 16 of scale_c x (signed
/// level . a), then acc = fma((d / 8) x d8, sumi, acc). The bits are the same (parity test).
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn iq2s_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * crate::IQ2_S_BLOCK && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while v0 < nv {
                let (xs, yr) = (x.as_ptr().add(v0 * nb), y.add(v0 * y_stride + r));
                match nv - v0 {
                    1 => iq2s_dotv::<1>(wr, xs, nb, yr, y_stride),
                    2 => iq2s_dotv::<2>(wr, xs, nb, yr, y_stride),
                    3 => iq2s_dotv::<3>(wr, xs, nb, yr, y_stride),
                    _ => iq2s_dotv::<4>(wr, xs, nb, yr, y_stride),
                }
                v0 += 4;
            }
        }
    }
}

/// iq2s_matmul' pass: one row (w) against NV vectors (x, nb blocks each) into y[v * y_stride].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn iq2s_dotv<const NV: usize>(w: *const u8, x: *const BlockQ8K, nb: usize, y: *mut f32, y_stride: usize) {
    unsafe {
        // two sign bytes spread over 16 weights: lanes 0..8 take byte 0 and lanes 8..16 byte 1, and lane l tests bit l % 8
        const SPREAD: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        const BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let (sp, bit) = (vld1q_u8(SPREAD.as_ptr()), vld1q_u8(BIT.as_ptr()));
        let mut acc = [0f32; NV];
        for b in 0..nb {
            let blk = w.add(b * crate::IQ2_S_BLOCK);
            let bs = core::slice::from_raw_parts(blk, crate::IQ2_S_BLOCK);
            let sc: [i32; 16] = core::array::from_fn(|c| crate::iq2s_chunk_scale(bs, c));
            let scv: [int32x4_t; 4] = core::array::from_fn(|k| vld1q_s32(sc.as_ptr().add(4 * k)));
            // the signed levels of run c: (level ^ m) - m, with m all ones where the sign bit is set
            let mut q = [vdupq_n_s8(0); 16];
            for (c, qc) in q.iter_mut().enumerate() {
                let (l0, s0) = crate::iq2s_group(bs, 2 * c);
                let (l1, s1) = crate::iq2s_group(bs, 2 * c + 1);
                let mut lv = [0u8; 16];
                lv[..8].copy_from_slice(&l0);
                lv[8..].copy_from_slice(&l1);
                let mut sb = [0u8; 16];
                sb[0] = s0;
                sb[1] = s1;
                let m = vtstq_u8(vqtbl1q_u8(vld1q_u8(sb.as_ptr()), sp), bit);
                *qc = vreinterpretq_s8_u8(vsubq_u8(veorq_u8(vld1q_u8(lv.as_ptr()), m), m));
            }
            let fd = crate::f16_to_f32(core::ptr::read_unaligned(blk as *const u16)) * 0.125;
            for v in 0..NV {
                let xb = &*x.add(v * nb + b);
                let p: [int32x4_t; 16] =
                    core::array::from_fn(|c| sdot(vdupq_n_s32(0), q[c], vld1q_s8(xb.qs.as_ptr().add(16 * c))));
                let red = |a: int32x4_t, b: int32x4_t, c: int32x4_t, d: int32x4_t| {
                    vpaddq_s32(vpaddq_s32(a, b), vpaddq_s32(c, d))
                };
                let mut si = vmulq_s32(red(p[0], p[1], p[2], p[3]), scv[0]);
                for k in 1..4 {
                    si = vmlaq_s32(si, red(p[4 * k], p[4 * k + 1], p[4 * k + 2], p[4 * k + 3]), scv[k]);
                }
                acc[v] = (fd * xb.d).mul_add(vaddvq_s32(si) as f32, acc[v]);
            }
        }
        for v in 0..NV {
            *y.add(v * y_stride) = acc[v];
        }
    }
}

/// Rows of Q6_K against nv vectors (x is nv x nb blocks, and vector v's outputs go to y[v * y_stride + r]), a row a
/// pass, each block unpacked once for up to 8 vectors. Each vector follows q6k_dot_rows' steps (an exact integer sum
/// per block, then acc = fma(d_w * d_x, float(sum), acc)), so the bits are the same (parity test).
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q6k_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * Q6_K_BLOCK && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            for v0 in (0..nv).step_by(8) {
                let n = (nv - v0).min(8);
                let mut acc = [0f32; 8];
                for b in 0..nb {
                    let blk = wr.add(b * Q6_K_BLOCK);
                    let mut q = [vdupq_n_s8(0); 16];
                    q6k_unpack(blk, &mut q);
                    let sc8 = vld1q_s8(blk.add(192) as *const i8);
                    let (sl, sh) = (vmovl_s8(vget_low_s8(sc8)), vmovl_high_s8(sc8));
                    let sv = [
                        vmovl_s16(vget_low_s16(sl)),
                        vmovl_high_s16(sl),
                        vmovl_s16(vget_low_s16(sh)),
                        vmovl_high_s16(sh),
                    ];
                    let dw = crate::f16_to_f32(u16::from_le_bytes([*blk.add(208), *blk.add(209)]));
                    for (v, ac) in acc[..n].iter_mut().enumerate() {
                        let xb = &*x.as_ptr().add((v0 + v) * nb + b);
                        let red = |i: usize| {
                            let s: [int32x4_t; 4] = core::array::from_fn(|j| {
                                let g = 4 * i + j;
                                sdot(vdupq_n_s32(0), q[g], vld1q_s8(xb.qs.as_ptr().add(16 * g)))
                            });
                            vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]))
                        };
                        let mut t = vmulq_s32(red(0), sv[0]);
                        t = vmlaq_s32(t, red(1), sv[1]);
                        t = vmlaq_s32(t, red(2), sv[2]);
                        t = vmlaq_s32(t, red(3), sv[3]);
                        *ac = (dw * xb.d).mul_add(vaddvq_s32(t) as f32, *ac);
                    }
                }
                for (v, ac) in acc[..n].iter().enumerate() {
                    *y.add((v0 + v) * y_stride + r) = *ac;
                }
            }
        }
    }
}

/// Rows of IQ4_NL against nv vectors (as q8_0_matmul): each row's nibbles are looked up once per half block for up
/// to 4 vectors, and each vector follows the steps of iq4nl_dot_rows, so the bits are the same.
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn iq4nl_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * 8 * Q4_0_BLOCK && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while v0 < nv {
                let (xs, yr) = (x.as_ptr().add(v0 * nb), y.add(v0 * y_stride + r));
                match nv - v0 {
                    1 => iq4nl_dotv::<1>(wr, xs, nb, yr, y_stride),
                    2 => iq4nl_dotv::<2>(wr, xs, nb, yr, y_stride),
                    3 => iq4nl_dotv::<3>(wr, xs, nb, yr, y_stride),
                    _ => iq4nl_dotv::<4>(wr, xs, nb, yr, y_stride),
                }
                v0 += 4;
            }
        }
    }
}

/// iq4nl_matmul' pass: one row (w) against NV vectors (x, nb blocks each) into y[v * y_stride].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn iq4nl_dotv<const NV: usize>(w: *const u8, x: *const Groups32, nb: usize, y: *mut f32, y_stride: usize) {
    unsafe {
        let m15 = vdupq_n_u8(15);
        let lv = vld1q_s8(IQ4NL_LEVELS.as_ptr());
        let mut acc = [[vdupq_n_f32(0.0); 2]; NV];
        for b in 0..nb {
            let wb = w.add(b * 8 * Q4_0_BLOCK);
            let (d0, d1) = q4_0_scales(wb);
            let dw = [d0, d1];
            for hf in 0..2 {
                let mut q = [[vdupq_n_s8(0); 2]; 4];
                for (j, qj) in q.iter_mut().enumerate() {
                    let b8 = vld1q_u8(wb.add((4 * hf + j) * Q4_0_BLOCK + 2));
                    *qj = [vqtbl1q_s8(lv, vandq_u8(b8, m15)), vqtbl1q_s8(lv, vshrq_n_u8::<4>(b8))];
                }
                for v in 0..NV {
                    let xb = &*x.add(v * nb + b);
                    let mut s = [vdupq_n_s32(0); 4];
                    for (j, sj) in s.iter_mut().enumerate() {
                        let p = xb.qs.as_ptr().add(32 * (4 * hf + j));
                        let t = sdot(vdupq_n_s32(0), q[j][0], vld1q_s8(p));
                        *sj = sdot(t, q[j][1], vld1q_s8(p.add(16)));
                    }
                    let sum = vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
                    let dx = vld1q_f32(xb.d.as_ptr().add(4 * hf));
                    acc[v][hf] = vfmaq_f32(acc[v][hf], vmulq_f32(dw[hf], dx), vcvtq_f32_s32(sum));
                }
            }
        }
        for v in 0..NV {
            *y.add(v * y_stride) = hsum4x2(acc[v][0], acc[v][1]);
        }
    }
}

/// Rows of Q2_0 against nv vectors (as iq4nl_matmul): each row's 2-bit fields are unpacked once per 64 weights for up
/// to 4 vectors, and each vector's products are summed per 32 weights in integers, then scaled by the weights' fp16 d
/// times the activations' group scale. Every vector of a row follows the same steps, so the bits do not depend on how
/// many vectors are in the call.
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn q2_0_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * 4 * crate::Q2_0_BLOCK && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while v0 < nv {
                let (xs, yr) = (x.as_ptr().add(v0 * nb), y.add(v0 * y_stride + r));
                match nv - v0 {
                    1 => q2_0_dotv::<1>(wr, xs, nb, yr, y_stride),
                    2 => q2_0_dotv::<2>(wr, xs, nb, yr, y_stride),
                    3 => q2_0_dotv::<3>(wr, xs, nb, yr, y_stride),
                    _ => q2_0_dotv::<4>(wr, xs, nb, yr, y_stride),
                }
                v0 += 4;
            }
        }
    }
}

/// q2_0_matmul' pass: one row (w) against NV vectors (x, nb blocks of 256 each) into y[v * y_stride].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn q2_0_dotv<const NV: usize>(w: *const u8, x: *const Groups32, nb: usize, y: *mut f32, y_stride: usize) {
    unsafe {
        let (m3, one) = (vdupq_n_u8(3), vdupq_n_s8(1));
        let mut acc = [[vdupq_n_f32(0.0); 2]; NV];
        for b in 0..nb {
            let wb = w.add(b * 4 * crate::Q2_0_BLOCK);
            let ds: [f32; 4] = core::array::from_fn(|j| {
                crate::f16_to_f32(core::ptr::read_unaligned(wb.add(j * crate::Q2_0_BLOCK) as *const u16))
            });
            let dw =
                [vld1q_f32([ds[0], ds[0], ds[1], ds[1]].as_ptr()), vld1q_f32([ds[2], ds[2], ds[3], ds[3]].as_ptr())];
            // the levels (field - 1) of the 4 blocks, in weight order: q[j][h] holds weights 16 h .. 16 h + 15 of block j
            let mut q = [[vdupq_n_s8(0); 4]; 4];
            for (j, qj) in q.iter_mut().enumerate() {
                let by = vld1q_u8(wb.add(j * crate::Q2_0_BLOCK + 2));
                // field k of byte i is weight 4 i + k: t[k] lane i. Zipping (0, 2) with (1, 3), twice, restores the order
                let t = [
                    vandq_u8(by, m3),
                    vandq_u8(vshrq_n_u8::<2>(by), m3),
                    vandq_u8(vshrq_n_u8::<4>(by), m3),
                    vshrq_n_u8::<6>(by),
                ];
                let (a0, b0) = (vzip1q_u8(t[0], t[2]), vzip1q_u8(t[1], t[3]));
                let (a1, b1) = (vzip2q_u8(t[0], t[2]), vzip2q_u8(t[1], t[3]));
                let o = [vzip1q_u8(a0, b0), vzip2q_u8(a0, b0), vzip1q_u8(a1, b1), vzip2q_u8(a1, b1)];
                for h in 0..4 {
                    qj[h] = vsubq_s8(vreinterpretq_s8_u8(o[h]), one);
                }
            }
            for v in 0..NV {
                let xb = &*x.add(v * nb + b);
                // the 8 groups of 32 are the two halves of each of the 4 blocks
                let mut s = [vdupq_n_s32(0); 8];
                for (g, sg) in s.iter_mut().enumerate() {
                    let p = xb.qs.as_ptr().add(32 * g);
                    let t = sdot(vdupq_n_s32(0), q[g / 2][2 * (g % 2)], vld1q_s8(p));
                    *sg = sdot(t, q[g / 2][2 * (g % 2) + 1], vld1q_s8(p.add(16)));
                }
                let red = |a: int32x4_t, b: int32x4_t, c: int32x4_t, d: int32x4_t| {
                    vpaddq_s32(vpaddq_s32(a, b), vpaddq_s32(c, d))
                };
                for hf in 0..2 {
                    let sum = red(s[4 * hf], s[4 * hf + 1], s[4 * hf + 2], s[4 * hf + 3]);
                    let dx = vld1q_f32(xb.d.as_ptr().add(4 * hf));
                    acc[v][hf] = vfmaq_f32(acc[v][hf], vmulq_f32(dw[hf], dx), vcvtq_f32_s32(sum));
                }
            }
        }
        for v in 0..NV {
            *y.add(v * y_stride) = hsum4x2(acc[v][0], acc[v][1]);
        }
    }
}

/// Rows of MXFP4 against nv vectors, as iq4nl_matmul: each row's nibbles are looked up once per half block (in MXFP4's
/// levels, ggml's doubled E2M1 values) for up to 4 vectors, and each block's dot is scaled by its halved E8M0 scale
/// (crate::e8m0_half) times the activations' group scale. Every vector of a row follows the same steps, so the bits do
/// not depend on how many vectors are in the call.
///
/// # Safety
/// FEAT_DotProd; y[v * y_stride + r] must be writable for v < nv and r < rows. (The sizes are asserted.)
#[target_feature(enable = "dotprod")]
pub unsafe fn mxfp4_matmul(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * 8 * crate::MXFP4_BLOCK && w.len() >= rows * row_bytes);
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while v0 < nv {
                let (xs, yr) = (x.as_ptr().add(v0 * nb), y.add(v0 * y_stride + r));
                match nv - v0 {
                    1 => mxfp4_dotv::<1>(wr, xs, nb, yr, y_stride),
                    2 => mxfp4_dotv::<2>(wr, xs, nb, yr, y_stride),
                    3 => mxfp4_dotv::<3>(wr, xs, nb, yr, y_stride),
                    _ => mxfp4_dotv::<4>(wr, xs, nb, yr, y_stride),
                }
                v0 += 4;
            }
        }
    }
}

/// mxfp4_matmul's pass: one row (w) against NV vectors (x, nb groups of 256 each) into y[v * y_stride].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn mxfp4_dotv<const NV: usize>(w: *const u8, x: *const Groups32, nb: usize, y: *mut f32, y_stride: usize) {
    const B: usize = crate::MXFP4_BLOCK;
    unsafe {
        let m15 = vdupq_n_u8(15);
        let lv = vld1q_s8(crate::MXFP4_LEVELS.as_ptr());
        let mut acc = [[vdupq_n_f32(0.0); 2]; NV];
        for b in 0..nb {
            let wb = w.add(b * 8 * B);
            let sc: [f32; 8] = core::array::from_fn(|k| crate::e8m0_half(*wb.add(k * B)));
            let dw = [vld1q_f32(sc.as_ptr()), vld1q_f32(sc.as_ptr().add(4))];
            for hf in 0..2 {
                let mut q = [[vdupq_n_s8(0); 2]; 4];
                for (j, qj) in q.iter_mut().enumerate() {
                    let b8 = vld1q_u8(wb.add((4 * hf + j) * B + 1));
                    *qj = [vqtbl1q_s8(lv, vandq_u8(b8, m15)), vqtbl1q_s8(lv, vshrq_n_u8::<4>(b8))];
                }
                for v in 0..NV {
                    let xb = &*x.add(v * nb + b);
                    let mut s = [vdupq_n_s32(0); 4];
                    for (j, sj) in s.iter_mut().enumerate() {
                        let p = xb.qs.as_ptr().add(32 * (4 * hf + j));
                        let t = sdot(vdupq_n_s32(0), q[j][0], vld1q_s8(p));
                        *sj = sdot(t, q[j][1], vld1q_s8(p.add(16)));
                    }
                    let sum = vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
                    let dx = vld1q_f32(xb.d.as_ptr().add(4 * hf));
                    acc[v][hf] = vfmaq_f32(acc[v][hf], vmulq_f32(dw[hf], dx), vcvtq_f32_s32(sum));
                }
            }
        }
        for v in 0..NV {
            *y.add(v * y_stride) = hsum4x2(acc[v][0], acc[v][1]);
        }
    }
}

/// q6k_dot_rows' pass: R rows (at w, w + row_bytes, ..) into y[0..R].
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn q6k_dotr<const R: usize>(w: *const u8, row_bytes: usize, x: &[BlockQ8K], y: *mut f32) {
    unsafe {
        let mut acc = [0f32; R];
        for (b, xb) in x.iter().enumerate() {
            let a: [int8x16_t; 16] = core::array::from_fn(|g| vld1q_s8(xb.qs.as_ptr().add(16 * g)));
            for (i, ac) in acc.iter_mut().enumerate() {
                let blk = w.add(i * row_bytes + b * Q6_K_BLOCK);
                let mut q = [vdupq_n_s8(0); 16];
                q6k_unpack(blk, &mut q);
                // the 16 group scales as four int32 vectors. Each set of 4 groups' dots is reduced pairwise so that
                // lane j holds group 4i + j, then multiplied by its scale (exact integers)
                let sc8 = vld1q_s8(blk.add(192) as *const i8);
                let (sl, sh) = (vmovl_s8(vget_low_s8(sc8)), vmovl_high_s8(sc8));
                let sv =
                    [vmovl_s16(vget_low_s16(sl)), vmovl_high_s16(sl), vmovl_s16(vget_low_s16(sh)), vmovl_high_s16(sh)];
                let s: [int32x4_t; 16] = core::array::from_fn(|g| sdot(vdupq_n_s32(0), q[g], a[g]));
                let red =
                    |i: usize| vpaddq_s32(vpaddq_s32(s[4 * i], s[4 * i + 1]), vpaddq_s32(s[4 * i + 2], s[4 * i + 3]));
                let mut t = vmulq_s32(red(0), sv[0]);
                t = vmlaq_s32(t, red(1), sv[1]);
                t = vmlaq_s32(t, red(2), sv[2]);
                t = vmlaq_s32(t, red(3), sv[3]);
                let dw = crate::f16_to_f32(u16::from_le_bytes([*blk.add(208), *blk.add(209)]));
                *ac = (dw * xb.d).mul_add(vaddvq_s32(t) as f32, *ac);
            }
        }
        for (i, ac) in acc.iter().enumerate() {
            *y.add(i) = *ac;
        }
    }
}

/// Q6_K rows against many vectors by SMMLA, 4 rows (two 2x2 tiles) a pass. For each block, each (row, vector) integer
/// sum of scale_g x sum((q - 32) a) is exact in any order, followed by one fused step per block, acc = fma(d_w * d_x,
/// float(sum), acc), as q6k_matmul does (on NEON and AVX2 alike). The bits are the same as the reference. rows must be
/// a multiple of 4; the outputs go to y[v * y_stride + r].
///
/// # Safety
/// FEAT_I8MM; w holds rows * row_bytes (row_bytes >= nb * 210); pk is pack_q6 of nv vectors of nb blocks; y is valid
/// for writes at v * y_stride + r, and nothing else touches those elements during the call.
#[target_feature(enable = "i8mm")]
pub unsafe fn gemm_q6k(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    pk: &[PackedQ6],
    nv: usize,
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    assert!(nv > 0 && rows.is_multiple_of(4) && row_bytes >= nb * Q6_K_BLOCK && pk.len() >= pack_q6_len(nv, nb));
    assert!(w.len() >= rows * row_bytes);
    let np = nv.div_ceil(2);
    unsafe {
        // acc[p * 2 + t] is tile t (rows r0 + 2t and r0 + 2t + 1) of pair p, with the lanes [r_i v0, r_i v1, r_j v0,
        // r_j v1]
        let mut stack = core::mem::MaybeUninit::<[float32x4_t; 128]>::uninit();
        let mut heap = Vec::new();
        let acc: &mut [float32x4_t] = if np <= 64 {
            crate::zeroed_prefix(&mut stack, np * 2)
        } else {
            heap.resize(np * 2, vdupq_n_f32(0.0));
            &mut heap
        };
        for r0 in (0..rows).step_by(4) {
            acc.fill(vdupq_n_f32(0.0));
            for b in 0..nb {
                let blk = |i: usize| w.as_ptr().add((r0 + i) * row_bytes + b * Q6_K_BLOCK);
                let mut q = [[vdupq_n_s8(0); 16]; 4];
                for (i, qi) in q.iter_mut().enumerate() {
                    q6k_unpack(blk(i), qi);
                }
                // for each tile and group: the A halves, and the scale per lane as [s_i, s_i, s_j, s_j]
                let mut a = [[[vdupq_n_s8(0); 2]; 16]; 2];
                let mut sc = [[vdupq_n_s32(0); 16]; 2];
                for t in 0..2 {
                    let (si, sj) = (blk(2 * t).add(192) as *const i8, blk(2 * t + 1).add(192) as *const i8);
                    for g in 0..16 {
                        let (lo, hi) = pair64(q[2 * t][g], q[2 * t + 1][g]);
                        a[t][g] = [lo, hi];
                        let sv = [*si.add(g) as i32, *si.add(g) as i32, *sj.add(g) as i32, *sj.add(g) as i32];
                        sc[t][g] = vld1q_s32(sv.as_ptr());
                    }
                }
                let h = |i: usize| u16::from_le_bytes([*blk(i).add(208), *blk(i).add(209)]);
                let hv = [h(0), h(1), h(2), h(3)];
                let dw = fcvtl(vld1_u16(hv.as_ptr()));
                let dt = [vzip1q_f32(dw, dw), vzip2q_f32(dw, dw)];
                for p in 0..np {
                    let e = pk.get_unchecked(p * nb + b);
                    for t in 0..2 {
                        let mut isum = vdupq_n_s32(0);
                        for g in 0..16 {
                            let s = smmla(smmla(vdupq_n_s32(0), a[t][g][0], e.b[2 * g]), a[t][g][1], e.b[2 * g + 1]);
                            isum = vmlaq_s32(isum, s, sc[t][g]);
                        }
                        let ac = acc.as_mut_ptr().add(p * 2 + t);
                        *ac = vfmaq_f32(*ac, vmulq_f32(dt[t], e.d), vcvtq_f32_s32(isum));
                    }
                }
            }
            for p in 0..np {
                let (v0, v1) = (2 * p, 2 * p + 1);
                for t in 0..2 {
                    let mut o = [0.0f32; 4];
                    vst1q_f32(o.as_mut_ptr(), acc[p * 2 + t]);
                    let row = r0 + 2 * t;
                    *y.add(v0 * y_stride + row) = o[0];
                    *y.add(v0 * y_stride + row + 1) = o[2];
                    if v1 < nv {
                        *y.add(v1 * y_stride + row) = o[1];
                        *y.add(v1 * y_stride + row + 1) = o[3];
                    }
                }
            }
        }
    }
}

/// One pair of vectors' block packed for gemm_q45k: what pack_q6 packs (the activations as SMMLA's B, and d per lane),
/// plus each 32-weight sub-block's activation sum (its two bsums) per lane as [s_v0, s_v1, s_v0, s_v1].
#[derive(Clone, Copy)]
pub struct PackedK {
    q: PackedQ6,
    s: [int32x4_t; 8],
}

impl Default for PackedK {
    fn default() -> PackedK {
        // SAFETY: plain vectors, for which all-zero bits are valid
        unsafe { core::mem::zeroed() }
    }
}

/// The number of PackedK entries for nv vectors of nb blocks.
pub fn pack_k_len(nv: usize, nb: usize) -> usize {
    nv.div_ceil(2) * nb
}

/// Packs x (nv x nb blocks) for gemm_q45k: entry p * nb + b holds vectors 2p and 2p + 1 (an odd last vector is paired
/// with itself).
pub fn pack_k(x: &[BlockQ8K], nv: usize, out: &mut [PackedK]) {
    let nb = x.len() / nv;
    assert!(nv > 0 && x.len() == nv * nb && out.len() >= pack_k_len(nv, nb));
    unsafe {
        for p in 0..nv.div_ceil(2) {
            let (v0, v1) = (2 * p, (2 * p + 1).min(nv - 1));
            for b in 0..nb {
                let (x0, x1) = (&x[v0 * nb + b], &x[v1 * nb + b]);
                let e = &mut out[p * nb + b];
                for g in 0..16 {
                    let (lo, hi) = pair64(vld1q_s8(x0.qs.as_ptr().add(16 * g)), vld1q_s8(x1.qs.as_ptr().add(16 * g)));
                    e.q.b[2 * g] = lo;
                    e.q.b[2 * g + 1] = hi;
                }
                let dv = [x0.d, x1.d, x0.d, x1.d];
                e.q.d = vld1q_f32(dv.as_ptr());
                for j in 0..8 {
                    let s = |xb: &BlockQ8K| (xb.bsums[2 * j] as i32) + (xb.bsums[2 * j + 1] as i32);
                    let sv = [s(x0), s(x1), s(x0), s(x1)];
                    e.s[j] = vld1q_s32(sv.as_ptr());
                }
            }
        }
    }
}

/// One Q4_K (FIVE = false) or Q5_K block of a row as 16 registers of weights (0..15, or 0..31), where register g holds
/// weights 16g..16g+15. This is q45k_vec_dot's unpacking, with sub-block k in registers 2k and 2k + 1.
#[inline]
unsafe fn q45k_unpack<const FIVE: bool>(blk: *const u8, q: &mut [int8x16_t; 16]) {
    unsafe {
        let (m4, one) = (vdupq_n_u8(15), vdupq_n_u8(1));
        let qs = blk.add(if FIVE { 48 } else { 16 });
        let (h0, h1) =
            if FIVE { (vld1q_u8(blk.add(16)), vld1q_u8(blk.add(32))) } else { (vdupq_n_u8(0), vdupq_n_u8(0)) };
        for j in 0..4 {
            let lo = vld1q_u8(qs.add(32 * j));
            let hi = vld1q_u8(qs.add(32 * j + 16));
            let (mut q0, mut q1) = (vandq_u8(lo, m4), vandq_u8(hi, m4));
            let (mut q2, mut q3) = (vshrq_n_u8::<4>(lo), vshrq_n_u8::<4>(hi));
            if FIVE {
                let r0 = vdupq_n_s8(-(2 * j as i8));
                let r1 = vdupq_n_s8(-(2 * j as i8 + 1));
                q0 = vorrq_u8(q0, vshlq_n_u8::<4>(vandq_u8(vshlq_u8(h0, r0), one)));
                q1 = vorrq_u8(q1, vshlq_n_u8::<4>(vandq_u8(vshlq_u8(h1, r0), one)));
                q2 = vorrq_u8(q2, vshlq_n_u8::<4>(vandq_u8(vshlq_u8(h0, r1), one)));
                q3 = vorrq_u8(q3, vshlq_n_u8::<4>(vandq_u8(vshlq_u8(h1, r1), one)));
            }
            q[4 * j] = vreinterpretq_s8_u8(q0);
            q[4 * j + 1] = vreinterpretq_s8_u8(q1);
            q[4 * j + 2] = vreinterpretq_s8_u8(q2);
            q[4 * j + 3] = vreinterpretq_s8_u8(q3);
        }
    }
}

/// Q4_K (five = false) or Q5_K rows against many vectors by SMMLA, 4 rows (two 2x2 tiles) a pass. Every (row, vector)
/// pair gets the same integer sums as q45k_vec_dot (sumi from the scales, summs from the mins; integer, so their order
/// is free) and the same two fused f32 chains, d d_x sumi and dmin d_x summs, subtracted at the end. The bits are the
/// same as q45k_matmul. rows must be a multiple of 4; the outputs go to y[v * y_stride + r].
///
/// # Safety
/// FEAT_I8MM; w holds rows * row_bytes (row_bytes >= nb x the block size); pk is pack_k of nv vectors of nb blocks; y
/// is valid for writes at v * y_stride + r, and nothing else touches those elements during the call.
#[target_feature(enable = "i8mm")]
pub unsafe fn gemm_q45k(
    five: bool,
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    pk: &[PackedK],
    nv: usize,
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe {
        if five {
            gemm_q45k_t::<true>(w, row_bytes, rows, pk, nv, nb, y, y_stride)
        } else {
            gemm_q45k_t::<false>(w, row_bytes, rows, pk, nv, nb, y, y_stride)
        }
    }
}

#[target_feature(enable = "i8mm")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_q45k_t<const FIVE: bool>(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    pk: &[PackedK],
    nv: usize,
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let bs = if FIVE { Q5_K_BLOCK } else { Q4_K_BLOCK };
    assert!(nv > 0 && rows.is_multiple_of(4) && row_bytes >= nb * bs && pk.len() >= pack_k_len(nv, nb));
    assert!(w.len() >= rows * row_bytes);
    let np = nv.div_ceil(2);
    unsafe {
        // acc / accm[p * 2 + t] is tile t (rows r0 + 2t and r0 + 2t + 1) of pair p, with the lanes [r_i v0, r_i v1, r_j
        // v0, r_j v1]
        let mut heap = Vec::new();
        heap.resize(np * 4, vdupq_n_f32(0.0));
        let (acc, accm) = heap.split_at_mut(np * 2);
        for r0 in (0..rows).step_by(4) {
            acc.fill(vdupq_n_f32(0.0));
            accm.fill(vdupq_n_f32(0.0));
            for b in 0..nb {
                let blk = |i: usize| w.as_ptr().add((r0 + i) * row_bytes + b * bs);
                let mut q = [[vdupq_n_s8(0); 16]; 4];
                for (i, qi) in q.iter_mut().enumerate() {
                    q45k_unpack::<FIVE>(blk(i), qi);
                }
                // for each tile: the A halves per group, and the scale and min per sub-block and lane as [x_i, x_i,
                // x_j, x_j]
                let mut a = [[[vdupq_n_s8(0); 2]; 16]; 2];
                let mut sc = [[vdupq_n_s32(0); 8]; 2];
                let mut mn = [[vdupq_n_s32(0); 8]; 2];
                for t in 0..2 {
                    let (si, mi) = crate::q45k_scales(&*(blk(2 * t).add(4) as *const [u8; 12]));
                    let (sj, mj) = crate::q45k_scales(&*(blk(2 * t + 1).add(4) as *const [u8; 12]));
                    for g in 0..16 {
                        let (lo, hi) = pair64(q[2 * t][g], q[2 * t + 1][g]);
                        a[t][g] = [lo, hi];
                    }
                    for k in 0..8 {
                        let sv = [si[k] as i32, si[k] as i32, sj[k] as i32, sj[k] as i32];
                        let mv = [mi[k] as i32, mi[k] as i32, mj[k] as i32, mj[k] as i32];
                        sc[t][k] = vld1q_s32(sv.as_ptr());
                        mn[t][k] = vld1q_s32(mv.as_ptr());
                    }
                }
                let h = |i: usize, o: usize| u16::from_le_bytes([*blk(i).add(o), *blk(i).add(o + 1)]);
                let (hd, hm) = ([h(0, 0), h(1, 0), h(2, 0), h(3, 0)], [h(0, 2), h(1, 2), h(2, 2), h(3, 2)]);
                let (dw, mw) = (fcvtl(vld1_u16(hd.as_ptr())), fcvtl(vld1_u16(hm.as_ptr())));
                let dt = [vzip1q_f32(dw, dw), vzip2q_f32(dw, dw)];
                let mt = [vzip1q_f32(mw, mw), vzip2q_f32(mw, mw)];
                for p in 0..np {
                    let e = pk.get_unchecked(p * nb + b);
                    for t in 0..2 {
                        let (mut isum, mut imin) = (vdupq_n_s32(0), vdupq_n_s32(0));
                        for k in 0..8 {
                            let (g0, g1) = (2 * k, 2 * k + 1);
                            let s = smmla(
                                smmla(vdupq_n_s32(0), a[t][g0][0], e.q.b[2 * g0]),
                                a[t][g0][1],
                                e.q.b[2 * g0 + 1],
                            );
                            let s = smmla(smmla(s, a[t][g1][0], e.q.b[2 * g1]), a[t][g1][1], e.q.b[2 * g1 + 1]);
                            isum = vmlaq_s32(isum, s, sc[t][k]);
                            imin = vmlaq_s32(imin, e.s[k], mn[t][k]);
                        }
                        let i = p * 2 + t;
                        acc[i] = vfmaq_f32(acc[i], vmulq_f32(dt[t], e.q.d), vcvtq_f32_s32(isum));
                        accm[i] = vfmaq_f32(accm[i], vmulq_f32(mt[t], e.q.d), vcvtq_f32_s32(imin));
                    }
                }
            }
            for p in 0..np {
                let (v0, v1) = (2 * p, 2 * p + 1);
                for t in 0..2 {
                    let mut o = [0.0f32; 4];
                    vst1q_f32(o.as_mut_ptr(), vsubq_f32(acc[p * 2 + t], accm[p * 2 + t]));
                    let row = r0 + 2 * t;
                    *y.add(v0 * y_stride + row) = o[0];
                    *y.add(v0 * y_stride + row + 1) = o[2];
                    if v1 < nv {
                        *y.add(v1 * y_stride + row) = o[1];
                        *y.add(v1 * y_stride + row + 1) = o[3];
                    }
                }
            }
        }
    }
}

pub use crate::lay::{Q6_LAY_BYTES, q6_lay4, q45k_lay_bytes, q45k_lay4};

/// Q5_K: each operand's fifth bit (bit 4) is ORed in from the sub-block's fifth-bit register h (tile t's operand n is
/// in bit n + 4t of each byte).
#[inline(always)]
unsafe fn fifth(o: &mut [uint8x16_t; 4], h: uint8x16_t, t: usize, b16: uint8x16_t) {
    unsafe {
        let f = if t == 0 {
            [vshlq_n_u8::<4>(h), vshlq_n_u8::<3>(h), vshlq_n_u8::<2>(h), vshlq_n_u8::<1>(h)]
        } else {
            [h, vshrq_n_u8::<1>(h), vshrq_n_u8::<2>(h), vshrq_n_u8::<3>(h)]
        };
        for n in 0..4 {
            o[n] = vorrq_u8(o[n], vandq_u8(f[n], b16));
        }
    }
}

/// gemm_q45k over rows laid out by q45k_lay4 (rows / 4 groups of nb * q45k_lay_bytes(five) bytes), with nothing left to
/// unpack but a mask and a shift (and Q5_K's fifth bit), and the scales already bytes. For each block and (row, vector)
/// it forms the same exact integers isum and imin, takes the same two fused steps, and goes through the blocks in the
/// same order, so the bits are the same as gemm_q45k. It takes two quads (8 rows) a pass from TORC_KMM8_MIN vectors
/// (default 8) when the rows allow, and the vectors' pairs a panel at a time (TORC_KMM_PANEL, default 16). rows must be
/// a multiple of 4; the outputs go to y[v * y_stride + r].
///
/// # Safety
/// FEAT_I8MM; w holds rows / 4 laid-out quads; pk is pack_k of nv vectors of nb blocks; y is valid for writes at v *
/// y_stride + r, and nothing else touches those elements during the call.
#[target_feature(enable = "i8mm")]
pub unsafe fn gemm_q45k_lay(
    five: bool,
    w: &[u8],
    rows: usize,
    pk: &[PackedK],
    nv: usize,
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let quad = nb * q45k_lay_bytes(five);
    assert!(nv > 0 && rows.is_multiple_of(4) && pk.len() >= pack_k_len(nv, nb) && w.len() >= rows / 4 * quad);
    static MIN8: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_KMM8_MIN").unwrap_or(8));
    let r8 = if nv >= *MIN8 { rows / 8 * 8 } else { 0 };
    unsafe {
        let (rest, yr) = (&w[r8 / 4 * quad..], y.add(r8));
        match five {
            true => {
                if r8 > 0 {
                    gemm_q45k_lay_t::<true, 2>(w, r8, pk, nv, nb, y, y_stride);
                }
                if r8 < rows {
                    gemm_q45k_lay_t::<true, 1>(rest, rows - r8, pk, nv, nb, yr, y_stride);
                }
            }
            false => {
                if r8 > 0 {
                    gemm_q45k_lay_t::<false, 2>(w, r8, pk, nv, nb, y, y_stride);
                }
                if r8 < rows {
                    gemm_q45k_lay_t::<false, 1>(rest, rows - r8, pk, nv, nb, yr, y_stride);
                }
            }
        }
    }
}

/// Rows laid out by q45k_lay4 against 1-3 vectors by SDOT, 4 rows a pass with one row per lane. Each tile register is
/// dotted against the vector's 8 bytes twice over ([x0-7, x0-7], with lanes 0-1 for row i and 2-3 for row j), and one
/// pairwise add leaves the 4 rows' exact sub-block sums. For each (row, vector) and block it forms the same integers
/// isum and imin and takes the same two fused steps as q45k_matmul, so the bits are the same. The outputs go to
/// y[v * y_stride + r].
///
/// # Safety
/// FEAT_DotProd; w holds rows / 4 laid-out quads of x.len() / nv blocks; rows is a multiple of 4; 1 <= nv <= 3; y is
/// valid for writes at v * y_stride + r, and nothing else touches those elements during the call.
#[target_feature(enable = "dotprod")]
pub unsafe fn q45k_lay_matmul(
    five: bool,
    w: &[u8],
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv;
    assert!(
        rows.is_multiple_of(4)
            && (1..=3).contains(&nv)
            && x.len() == nv * nb
            && w.len() >= rows / 4 * nb * q45k_lay_bytes(five)
    );
    unsafe {
        match (five, nv) {
            (false, 1) => q45k_lay_pass::<false, 1>(w, rows, x, nb, y, y_stride),
            (false, 2) => q45k_lay_pass::<false, 2>(w, rows, x, nb, y, y_stride),
            (false, _) => q45k_lay_pass::<false, 3>(w, rows, x, nb, y, y_stride),
            (true, 1) => q45k_lay_pass::<true, 1>(w, rows, x, nb, y, y_stride),
            (true, 2) => q45k_lay_pass::<true, 2>(w, rows, x, nb, y, y_stride),
            (true, _) => q45k_lay_pass::<true, 3>(w, rows, x, nb, y, y_stride),
        }
    }
}

#[target_feature(enable = "dotprod")]
unsafe fn q45k_lay_pass<const FIVE: bool, const NV: usize>(
    w: &[u8],
    rows: usize,
    x: &[BlockQ8K],
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let lb = q45k_lay_bytes(FIVE);
    let quad = nb * lb;
    unsafe {
        let (m15, b16) = (vdupq_n_u8(15), vdupq_n_u8(16));
        for r0 in (0..rows).step_by(4) {
            let base = w.as_ptr().add(r0 / 4 * quad);
            let mut acc = [vdupq_n_f32(0.0); NV];
            let mut accm = [vdupq_n_f32(0.0); NV];
            for b in 0..nb {
                let g = base.add(b * lb);
                let mut isum = [vdupq_n_s32(0); NV];
                let mut imin = [vdupq_n_s32(0); NV];
                for k in 0..8 {
                    // the 4 rows' scale and min as lanes [r0 r1 r2 r3] (the layout holds each twice)
                    let lanes = |o: usize| {
                        let s = vmovl_u8(vreinterpret_u8_u32(vld1_dup_u32(g.add(o) as *const u32)));
                        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s)))
                    };
                    let (sc, mn) = (lanes(16 + 4 * k), lanes(48 + 4 * k));
                    let hq = if FIVE { vld1q_u8(g.add(592 + 16 * k)) } else { vdupq_n_u8(0) };
                    let mut a = [[vdupq_n_s8(0); 4]; 2];
                    for (t, at) in a.iter_mut().enumerate() {
                        let wp = g.add(80 + (2 * k + t) * 32);
                        let (r1, r2) = (vld1q_u8(wp), vld1q_u8(wp.add(16)));
                        let mut o = [vandq_u8(r1, m15), vshrq_n_u8::<4>(r1), vandq_u8(r2, m15), vshrq_n_u8::<4>(r2)];
                        if FIVE {
                            fifth(&mut o, hq, t, b16);
                        }
                        *at = [
                            vreinterpretq_s8_u8(o[0]),
                            vreinterpretq_s8_u8(o[1]),
                            vreinterpretq_s8_u8(o[2]),
                            vreinterpretq_s8_u8(o[3]),
                        ];
                    }
                    for v in 0..NV {
                        let xb = x.get_unchecked(v * nb + b);
                        let xp = xb.qs.as_ptr().add(32 * k);
                        let dup = |o: usize| vreinterpretq_s8_s64(vld1q_dup_s64(xp.add(o) as *const i64));
                        let (x0, x1, x2, x3) = (dup(0), dup(8), dup(16), dup(24));
                        let s =
                            sdot(sdot(sdot(sdot(vdupq_n_s32(0), a[0][0], x0), a[0][1], x1), a[0][2], x2), a[0][3], x3);
                        let t =
                            sdot(sdot(sdot(sdot(vdupq_n_s32(0), a[1][0], x0), a[1][1], x1), a[1][2], x2), a[1][3], x3);
                        isum[v] = vmlaq_s32(isum[v], vpaddq_s32(s, t), sc);
                        let bs = xb.bsums[2 * k] as i32 + xb.bsums[2 * k + 1] as i32;
                        imin[v] = vmlaq_s32(imin[v], mn, vdupq_n_s32(bs));
                    }
                }
                let (d, dm) = (fcvtl(vld1_u16(g as *const u16)), fcvtl(vld1_u16(g.add(8) as *const u16)));
                for v in 0..NV {
                    let xd = x.get_unchecked(v * nb + b).d;
                    acc[v] = vfmaq_f32(acc[v], vmulq_n_f32(d, xd), vcvtq_f32_s32(isum[v]));
                    accm[v] = vfmaq_f32(accm[v], vmulq_n_f32(dm, xd), vcvtq_f32_s32(imin[v]));
                }
            }
            for v in 0..NV {
                vst1q_f32(y.add(v * y_stride + r0), vsubq_f32(acc[v], accm[v]));
            }
        }
    }
}

/// gemm_q45k_lay's body: Q quads (4Q rows, 2Q tiles) a pass (rows must be a multiple of 4Q).
#[target_feature(enable = "i8mm")]
unsafe fn gemm_q45k_lay_t<const FIVE: bool, const Q: usize>(
    w: &[u8],
    rows: usize,
    pk: &[PackedK],
    nv: usize,
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let lb = q45k_lay_bytes(FIVE);
    let quad = nb * lb;
    assert!(rows % (4 * Q) == 0 && w.len() >= rows / 4 * quad);
    let np = nv.div_ceil(2);
    static PANEL: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_KMM_PANEL").unwrap_or(16));
    let panel = if *PANEL == 0 { np } else { (*PANEL).min(np) };
    unsafe {
        // acc / accm[p * 2Q + t] cover all the pairs, and the block's integer sums isum / imin[(p - p0) * 2Q + t] cover
        // the panel's pairs. They live on the stack for up to 1024 vectors, and on the heap beyond that
        let n = (2 * np + 2 * panel) * 2 * Q;
        let mut stack = core::mem::MaybeUninit::<[int32x4_t; 1024]>::uninit();
        let mut heap = Vec::new();
        let all: &mut [int32x4_t] = if n <= 1024 {
            crate::zeroed_prefix(&mut stack, n)
        } else {
            heap.resize(n, vdupq_n_s32(0));
            &mut heap
        };
        let (accs, sums) = all.split_at_mut(2 * np * 2 * Q);
        let accs = accs.as_mut_ptr() as *mut float32x4_t;
        let (acc, accm) = (accs, accs.add(np * 2 * Q));
        let (isum, imin) = sums.split_at_mut(panel * 2 * Q);
        let (m15, b16) = (vdupq_n_u8(15), vdupq_n_u8(16));
        for r0 in (0..rows).step_by(4 * Q) {
            for i in 0..2 * np * 2 * Q {
                *accs.add(i) = vdupq_n_f32(0.0);
            }
            let base = w.as_ptr().add(r0 / 4 * quad);
            for p0 in (0..np).step_by(panel) {
                let p1 = (p0 + panel).min(np);
                for b in 0..nb {
                    isum.fill(vdupq_n_s32(0));
                    imin.fill(vdupq_n_s32(0));
                    for k in 0..8 {
                        // for each quad: the tile lanes of the sub-block's scales and mins, and the 4 operands of each
                        // of its two tiles
                        let mut a = [[vdupq_n_s8(0); 4]; 4];
                        let mut sc = [vdupq_n_s32(0); 4];
                        let mut mn = [vdupq_n_s32(0); 4];
                        for q in 0..Q {
                            let g = base.add(q * quad + b * lb);
                            // bytes [r0 r1 r2 r3] go to the tile lanes [r0 r0 r1 r1] and [r2 r2 r3 r3]
                            let lanes = |o: usize| {
                                let v = vmovl_u8(vreinterpret_u8_u32(vld1_dup_u32(g.add(o) as *const u32)));
                                let z = vzip1q_u16(v, v);
                                (
                                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(z))),
                                    vreinterpretq_s32_u32(vmovl_high_u16(z)),
                                )
                            };
                            (sc[2 * q], sc[2 * q + 1]) = lanes(16 + 4 * k);
                            (mn[2 * q], mn[2 * q + 1]) = lanes(48 + 4 * k);
                            let hq = if FIVE { vld1q_u8(g.add(592 + 16 * k)) } else { vdupq_n_u8(0) };
                            for t in 0..2 {
                                let wp = g.add(80 + (2 * k + t) * 32);
                                let (r1, r2) = (vld1q_u8(wp), vld1q_u8(wp.add(16)));
                                let mut o =
                                    [vandq_u8(r1, m15), vshrq_n_u8::<4>(r1), vandq_u8(r2, m15), vshrq_n_u8::<4>(r2)];
                                if FIVE {
                                    fifth(&mut o, hq, t, b16);
                                }
                                a[2 * q + t] = [
                                    vreinterpretq_s8_u8(o[0]),
                                    vreinterpretq_s8_u8(o[1]),
                                    vreinterpretq_s8_u8(o[2]),
                                    vreinterpretq_s8_u8(o[3]),
                                ];
                            }
                        }
                        for p in p0..p1 {
                            let e = pk.get_unchecked(p * nb + b);
                            let (is, im) =
                                (isum.as_mut_ptr().add((p - p0) * 2 * Q), imin.as_mut_ptr().add((p - p0) * 2 * Q));
                            let bq = &e.q.b[4 * k..4 * k + 4];
                            for t in 0..2 * Q {
                                let mut s = smmla(vdupq_n_s32(0), a[t][0], bq[0]);
                                s = smmla(s, a[t][1], bq[1]);
                                s = smmla(s, a[t][2], bq[2]);
                                s = smmla(s, a[t][3], bq[3]);
                                *is.add(t) = vmlaq_s32(*is.add(t), s, sc[t]);
                                *im.add(t) = vmlaq_s32(*im.add(t), e.s[k], mn[t]);
                            }
                        }
                    }
                    // the block's two fused steps for each (pair, tile), going through the blocks in order as gemm_q45k
                    // does
                    let mut dt = [vdupq_n_f32(0.0); 4];
                    let mut mt = [vdupq_n_f32(0.0); 4];
                    for q in 0..Q {
                        let g = base.add(q * quad + b * lb);
                        let (d, m) = (fcvtl(vld1_u16(g as *const u16)), fcvtl(vld1_u16(g.add(8) as *const u16)));
                        dt[2 * q] = vzip1q_f32(d, d);
                        dt[2 * q + 1] = vzip2q_f32(d, d);
                        mt[2 * q] = vzip1q_f32(m, m);
                        mt[2 * q + 1] = vzip2q_f32(m, m);
                    }
                    for p in p0..p1 {
                        let e = pk.get_unchecked(p * nb + b);
                        for t in 0..2 * Q {
                            let (i, j) = (p * 2 * Q + t, (p - p0) * 2 * Q + t);
                            *acc.add(i) = vfmaq_f32(*acc.add(i), vmulq_f32(dt[t], e.q.d), vcvtq_f32_s32(isum[j]));
                            *accm.add(i) = vfmaq_f32(*accm.add(i), vmulq_f32(mt[t], e.q.d), vcvtq_f32_s32(imin[j]));
                        }
                    }
                }
            }
            for p in 0..np {
                let (v0, v1) = (2 * p, 2 * p + 1);
                for t in 0..2 * Q {
                    let mut o = [0.0f32; 4];
                    vst1q_f32(o.as_mut_ptr(), vsubq_f32(*acc.add(p * 2 * Q + t), *accm.add(p * 2 * Q + t)));
                    let row = r0 + 2 * t;
                    *y.add(v0 * y_stride + row) = o[0];
                    *y.add(v0 * y_stride + row + 1) = o[2];
                    if v1 < nv {
                        *y.add(v1 * y_stride + row) = o[1];
                        *y.add(v1 * y_stride + row + 1) = o[3];
                    }
                }
            }
        }
    }
}

/// Group g's two A operands for tile t, from a q6_lay4 block g0 (signed, q - 32).
#[inline(always)]
unsafe fn q6_lay_ops(g0: *const u8, g: usize, t: usize) -> [int8x16_t; 2] {
    unsafe {
        let (m15, m48, c32) = (vdupq_n_u8(15), vdupq_n_u8(48), vdupq_n_s8(32));
        let r = vld1q_u8(g0.add(72 + (2 * g + t) * 16));
        let h = vld1q_u8(g0.add(584 + 16 * g));
        let (h0, h1) = if t == 0 {
            (vandq_u8(vshlq_n_u8::<4>(h), m48), vandq_u8(vshlq_n_u8::<2>(h), m48))
        } else {
            (vandq_u8(h, m48), vandq_u8(vshrq_n_u8::<2>(h), m48))
        };
        let o0 = vorrq_u8(vandq_u8(r, m15), h0);
        let o1 = vorrq_u8(vshrq_n_u8::<4>(r), h1);
        [vsubq_s8(vreinterpretq_s8_u8(o0), c32), vsubq_s8(vreinterpretq_s8_u8(o1), c32)]
    }
}

/// gemm_q6k over rows laid out by q6_lay4 (rows / 4 groups of nb * Q6_LAY_BYTES bytes). The operands come out of a
/// mask, a shift and an OR, and the scales are already bytes in row order. For each block and (row, vector) it forms
/// the same exact integer sum, takes the same fused step, and goes through the blocks in the same order, so the bits
/// are the same as gemm_q6k. It takes two quads (8 rows) a pass from TORC_KMM8_MIN vectors (default 8) when the rows
/// allow, and the vectors' pairs a panel at a time (TORC_KMM_PANEL, default 16). rows must be a multiple of 4; the
/// outputs go to y[v * y_stride + r].
///
/// # Safety
/// FEAT_I8MM; w holds rows / 4 laid-out quads; pk is pack_q6 of nv vectors of nb blocks; y is valid for writes at v *
/// y_stride + r, and nothing else touches those elements during the call.
#[target_feature(enable = "i8mm")]
pub unsafe fn gemm_q6_lay(w: &[u8], rows: usize, pk: &[PackedQ6], nv: usize, nb: usize, y: *mut f32, y_stride: usize) {
    let quad = nb * Q6_LAY_BYTES;
    assert!(nv > 0 && rows.is_multiple_of(4) && pk.len() >= pack_q6_len(nv, nb) && w.len() >= rows / 4 * quad);
    static MIN8: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_KMM8_MIN").unwrap_or(8));
    let r8 = if nv >= *MIN8 { rows / 8 * 8 } else { 0 };
    unsafe {
        if r8 > 0 {
            gemm_q6_lay_t::<2>(w, r8, pk, nv, nb, y, y_stride);
        }
        if r8 < rows {
            gemm_q6_lay_t::<1>(&w[r8 / 4 * quad..], rows - r8, pk, nv, nb, y.add(r8), y_stride);
        }
    }
}

/// gemm_q6_lay's body: Q quads (4Q rows, 2Q tiles) a pass (rows must be a multiple of 4Q).
#[target_feature(enable = "i8mm")]
unsafe fn gemm_q6_lay_t<const Q: usize>(
    w: &[u8],
    rows: usize,
    pk: &[PackedQ6],
    nv: usize,
    nb: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let quad = nb * Q6_LAY_BYTES;
    assert!(rows % (4 * Q) == 0 && w.len() >= rows / 4 * quad);
    let np = nv.div_ceil(2);
    static PANEL: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_KMM_PANEL").unwrap_or(16));
    let panel = if *PANEL == 0 { np } else { (*PANEL).min(np) };
    unsafe {
        // acc[p * 2Q + t] covers all the pairs, and the block's integer sums isum[(p - p0) * 2Q + t] cover the panel's
        // pairs. They live on the stack for up to 1024 vectors, and on the heap beyond that
        let n = (np + panel) * 2 * Q;
        let mut stack = core::mem::MaybeUninit::<[int32x4_t; 1024]>::uninit();
        let mut heap = Vec::new();
        let all: &mut [int32x4_t] = if n <= 1024 {
            crate::zeroed_prefix(&mut stack, n)
        } else {
            heap.resize(n, vdupq_n_s32(0));
            &mut heap
        };
        let (accs, isum) = all.split_at_mut(np * 2 * Q);
        let acc = accs.as_mut_ptr() as *mut float32x4_t;
        for r0 in (0..rows).step_by(4 * Q) {
            for i in 0..np * 2 * Q {
                *acc.add(i) = vdupq_n_f32(0.0);
            }
            let base = w.as_ptr().add(r0 / 4 * quad);
            for p0 in (0..np).step_by(panel) {
                let p1 = (p0 + panel).min(np);
                for b in 0..nb {
                    isum.fill(vdupq_n_s32(0));
                    for g in 0..16 {
                        let mut a = [[vdupq_n_s8(0); 2]; 4];
                        let mut sc = [vdupq_n_s32(0); 4];
                        for q in 0..Q {
                            let g0 = base.add(q * quad + b * Q6_LAY_BYTES);
                            // int8 scales [r0 r1 r2 r3] go to the tile lanes [r0 r0 r1 r1] and [r2 r2 r3 r3]
                            let v = vmovl_s8(vreinterpret_s8_u32(vld1_dup_u32(g0.add(8 + 4 * g) as *const u32)));
                            let z = vzip1q_s16(v, v);
                            sc[2 * q] = vmovl_s16(vget_low_s16(z));
                            sc[2 * q + 1] = vmovl_high_s16(z);
                            a[2 * q] = q6_lay_ops(g0, g, 0);
                            a[2 * q + 1] = q6_lay_ops(g0, g, 1);
                        }
                        for p in p0..p1 {
                            let e = pk.get_unchecked(p * nb + b);
                            let is = isum.as_mut_ptr().add((p - p0) * 2 * Q);
                            for t in 0..2 * Q {
                                let s = smmla(smmla(vdupq_n_s32(0), a[t][0], e.b[2 * g]), a[t][1], e.b[2 * g + 1]);
                                *is.add(t) = vmlaq_s32(*is.add(t), s, sc[t]);
                            }
                        }
                    }
                    // the block's fused step for each (pair, tile), going through the blocks in order as gemm_q6k does
                    let mut dt = [vdupq_n_f32(0.0); 4];
                    for q in 0..Q {
                        let d = fcvtl(vld1_u16(base.add(q * quad + b * Q6_LAY_BYTES) as *const u16));
                        dt[2 * q] = vzip1q_f32(d, d);
                        dt[2 * q + 1] = vzip2q_f32(d, d);
                    }
                    for p in p0..p1 {
                        let e = pk.get_unchecked(p * nb + b);
                        for t in 0..2 * Q {
                            let (i, j) = (p * 2 * Q + t, (p - p0) * 2 * Q + t);
                            *acc.add(i) = vfmaq_f32(*acc.add(i), vmulq_f32(dt[t], e.d), vcvtq_f32_s32(isum[j]));
                        }
                    }
                }
            }
            for p in 0..np {
                let (v0, v1) = (2 * p, 2 * p + 1);
                for t in 0..2 * Q {
                    let mut o = [0.0f32; 4];
                    vst1q_f32(o.as_mut_ptr(), *acc.add(p * 2 * Q + t));
                    let row = r0 + 2 * t;
                    *y.add(v0 * y_stride + row) = o[0];
                    *y.add(v0 * y_stride + row + 1) = o[2];
                    if v1 < nv {
                        *y.add(v1 * y_stride + row) = o[1];
                        *y.add(v1 * y_stride + row + 1) = o[3];
                    }
                }
            }
        }
    }
}

/// Rows laid out by q6_lay4 against 1-3 vectors by SDOT, 4 rows a pass with one row per lane (q45k_lay_matmul's
/// scheme). For each (row, vector) and block it forms the same exact integer sum and takes the same fused step as
/// q6k_matmul, so the bits are the same. The outputs go to y[v * y_stride + r].
///
/// # Safety
/// FEAT_DotProd; w holds rows / 4 laid-out quads of x.len() / nv blocks; rows is a multiple of 4; 1 <= nv <= 3; y is
/// valid for writes at v * y_stride + r, and nothing else touches those elements during the call.
#[target_feature(enable = "dotprod")]
pub unsafe fn q6_lay_matmul(w: &[u8], rows: usize, x: &[BlockQ8K], nv: usize, y: *mut f32, y_stride: usize) {
    let nb = x.len() / nv;
    assert!(
        rows.is_multiple_of(4)
            && (1..=3).contains(&nv)
            && x.len() == nv * nb
            && w.len() >= rows / 4 * nb * Q6_LAY_BYTES
    );
    unsafe {
        match nv {
            1 => q6_lay_pass::<1>(w, rows, x, nb, y, y_stride),
            2 => q6_lay_pass::<2>(w, rows, x, nb, y, y_stride),
            _ => q6_lay_pass::<3>(w, rows, x, nb, y, y_stride),
        }
    }
}

#[target_feature(enable = "dotprod")]
unsafe fn q6_lay_pass<const NV: usize>(w: &[u8], rows: usize, x: &[BlockQ8K], nb: usize, y: *mut f32, y_stride: usize) {
    let quad = nb * Q6_LAY_BYTES;
    unsafe {
        for r0 in (0..rows).step_by(4) {
            let base = w.as_ptr().add(r0 / 4 * quad);
            let mut acc = [vdupq_n_f32(0.0); NV];
            for b in 0..nb {
                let g0 = base.add(b * Q6_LAY_BYTES);
                let mut isum = [vdupq_n_s32(0); NV];
                for g in 0..16 {
                    // the 4 rows' scales as lanes [r0 r1 r2 r3]
                    let v = vmovl_s8(vreinterpret_s8_u32(vld1_dup_u32(g0.add(8 + 4 * g) as *const u32)));
                    let sc = vmovl_s16(vget_low_s16(v));
                    let (a0, a1) = (q6_lay_ops(g0, g, 0), q6_lay_ops(g0, g, 1));
                    for v in 0..NV {
                        let xp = x.get_unchecked(v * nb + b).qs.as_ptr().add(16 * g);
                        let dup = |o: usize| vreinterpretq_s8_s64(vld1q_dup_s64(xp.add(o) as *const i64));
                        let (x0, x1) = (dup(0), dup(8));
                        let s = sdot(sdot(vdupq_n_s32(0), a0[0], x0), a0[1], x1);
                        let t = sdot(sdot(vdupq_n_s32(0), a1[0], x0), a1[1], x1);
                        isum[v] = vmlaq_s32(isum[v], vpaddq_s32(s, t), sc);
                    }
                }
                let d = fcvtl(vld1_u16(g0 as *const u16));
                for v in 0..NV {
                    acc[v] = vfmaq_f32(acc[v], vmulq_n_f32(d, x.get_unchecked(v * nb + b).d), vcvtq_f32_s32(isum[v]));
                }
            }
            for v in 0..NV {
                vst1q_f32(y.add(v * y_stride + r0), acc[v]);
            }
        }
    }
}

/// delta_rule_f32_scalar in NEON, matching it lane for lane, four rows a pass so their chains overlap. The state is
/// read from src (a revertible block's base) when one is given, else from `state`, and written to `state` - one read
/// and one write a step instead of a copy followed by two. dim must be a multiple of 16.
#[allow(clippy::too_many_arguments)]
pub fn delta_rule_f32(
    src: Option<&[f32]>,
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    beta: f32,
    q_scale: f32,
    o: &mut [f32],
    dim: usize,
) {
    let rows = v.len();
    assert!(dim.is_multiple_of(16) && o.len() >= rows && state.len() >= rows * dim && q.len() >= dim && k.len() >= dim);
    assert!(src.is_none_or(|s| s.len() >= rows * dim));
    const R: usize = 4;
    let rr = rows / R * R;
    // SAFETY: NEON is baseline on aarch64, and every access is inside the asserted lengths
    unsafe {
        let alpha_v = vdupq_n_f32(alpha);
        let (kp, qp) = (k.as_ptr(), q.as_ptr());
        let mp = state.as_mut_ptr();
        let sp: *const f32 = src.map_or(mp as *const f32, |s| s.as_ptr());
        let mut j0 = 0;
        while j0 < rows {
            let n = if j0 < rr { R } else { 1 };
            let mut a = [[vdupq_n_f32(0.0); 4]; R];
            for i in (0..dim).step_by(16) {
                let kv = [
                    vld1q_f32(kp.add(i)),
                    vld1q_f32(kp.add(i + 4)),
                    vld1q_f32(kp.add(i + 8)),
                    vld1q_f32(kp.add(i + 12)),
                ];
                for t in 0..n {
                    let (rs, rw) = (sp.add((j0 + t) * dim + i), mp.add((j0 + t) * dim + i));
                    for l in 0..4 {
                        let x = vmulq_f32(vld1q_f32(rs.add(4 * l)), alpha_v);
                        vst1q_f32(rw.add(4 * l), x);
                        a[t][l] = vfmaq_f32(a[t][l], x, kv[l]);
                    }
                }
            }
            let mut delta = [vdupq_n_f32(0.0); R];
            for t in 0..n {
                let sk = hsum4x2(vaddq_f32(a[t][0], a[t][2]), vaddq_f32(a[t][1], a[t][3]));
                delta[t] = vdupq_n_f32((*v.get_unchecked(j0 + t) - sk) * beta);
            }
            let mut a = [[vdupq_n_f32(0.0); 4]; R];
            for i in (0..dim).step_by(16) {
                let kv = [
                    vld1q_f32(kp.add(i)),
                    vld1q_f32(kp.add(i + 4)),
                    vld1q_f32(kp.add(i + 8)),
                    vld1q_f32(kp.add(i + 12)),
                ];
                let qv = [
                    vld1q_f32(qp.add(i)),
                    vld1q_f32(qp.add(i + 4)),
                    vld1q_f32(qp.add(i + 8)),
                    vld1q_f32(qp.add(i + 12)),
                ];
                for t in 0..n {
                    let rw = mp.add((j0 + t) * dim + i);
                    for l in 0..4 {
                        let x = vfmaq_f32(vld1q_f32(rw.add(4 * l)), delta[t], kv[l]);
                        vst1q_f32(rw.add(4 * l), x);
                        a[t][l] = vfmaq_f32(a[t][l], x, qv[l]);
                    }
                }
            }
            for t in 0..n {
                let sum = hsum4x2(vaddq_f32(a[t][0], a[t][2]), vaddq_f32(a[t][1], a[t][3]));
                *o.get_unchecked_mut(j0 + t) = sum * q_scale;
            }
            j0 += n;
        }
    }
}
