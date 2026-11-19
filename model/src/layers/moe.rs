//! moe.rs - the mixture-of-experts block.
//!
//! For one token: the router in fp16, a softmax and the top-k. Each chosen expert then runs gate and up over the Q8
//! input, the deterministic SiLU, its hidden state quantised without a division, then down. The outputs are added in
//! increasing expert order, with the shared expert last, behind its sigmoid gate. Prompt blocks, the NPU tier and the
//! two-machine worker all reduce to the same per-token sums, so every path gives the same bits.

use crate::pool::{self, SyncPtr};
use crate::weights::{TensorRef, Weights};
use format::{Result, err};
use kernels::{Q8, Q8m, Q8s, QK, f16_to_f32, ops as kops};

/// Blocks of at most this many tokens split their row tasks by guided claims of contiguous runs (pool::par_for_guided).
const RUNS_MAX: usize = 16;

/// Matrices smaller than this run on one thread.
const PAR_MIN_BYTES: usize = 4 << 20;

/// One matrix to multiply by: its type, rows and columns, and its bytes.
#[derive(Clone, Copy)]
pub struct Mat<'a> {
    pub ty: u32,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [u8],
}

impl<'a> Mat<'a> {
    pub fn of(t: &TensorRef<'a>) -> Mat<'a> {
        let (cols, rows) = t.shape();
        Mat { ty: t.t.ty, rows, cols, data: t.data }
    }

    /// y = W x, with x quantised. Every row is the same dot product whichever thread computes it.
    pub fn apply(&self, x: Q8s, y: &mut [f32]) {
        let row_bytes = self.data.len() / self.rows;
        assert_eq!(x.len(), self.cols / QK);
        let y = &mut y[..self.rows];
        // big matrices are split by rows over the pool; small ones (an expert's) run on the calling thread
        if self.rows * row_bytes < PAR_MIN_BYTES {
            dot_rows(self.ty, self.data, row_bytes, x, y);
            return;
        }
        let yp = SyncPtr::new(y);
        // SAFETY: chunk i owns its rows of y
        dense_deal()(self.chunks(), &|i| unsafe { self.apply_chunk(x, yp, i) });
    }

    /// The number of row chunks apply splits a big matrix into, so they can run as items of another sweep.
    pub fn chunks(&self) -> usize {
        self.rows.div_ceil(self.chunk_rows())
    }

    fn chunk_rows(&self) -> usize {
        self.rows.div_ceil(pool::get().threads * 4).max(16)
    }

    /// Chunk i of apply's rows (every row is the same dot product, whichever chunk or thread runs it).
    ///
    /// # Safety
    /// y holds self.rows elements, and no other task touches chunk i's rows of it while this runs.
    pub unsafe fn apply_chunk(&self, x: Q8s, y: SyncPtr<f32>, i: usize) {
        let (chunk, rb) = (self.chunk_rows(), self.data.len() / self.rows);
        let r0 = i * chunk;
        let n = chunk.min(self.rows - r0);
        let ys = unsafe { y.slice(r0, n) };
        dot_rows(self.ty, &self.data[r0 * rb..(r0 + n) * rb], rb, x, ys);
    }
}

/// One vector's dot products over the rows. On ARM64, Q4_0 goes through kernels::neon::q4_0_dot_rows1, a row at a time;
/// TORC_DOT=ours2 takes 2 rows a pass instead (rows over 3072 columns still go one at a time).
/// Other types: see below.
#[inline]
pub fn dot_rows(ty: u32, w: &[u8], rb: usize, x: Q8s, y: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // a row at a time by default; TORC_DOT=ours2 takes 2 rows a pass
        static OURS: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
            if !std::arch::is_aarch64_feature_detected!("dotprod") {
                return 0;
            }
            match crate::switches::get("TORC_DOT").as_deref() {
                Some("ours2") => 2,
                _ => 1,
            }
        });
        // a repacked copy of these rows (TORC_Q4X4=1) goes through q4x4_matmul, with the same bits, reading the 4 rows
        // of a pass as one stream
        if ty == kernels::GGML_Q4_0 && *OURS > 0 && q4x4(ty, 1) {
            if let Some(rp) = crate::repack::find(&w[..y.len() * rb], y.len()) {
                // SAFETY: FEAT_DotProd was checked above; y holds y.len() rows (a multiple of 4, as find guarantees)
                return unsafe { kernels::neon::q4x4_matmul(rp, y.len(), x.g, 1, y.as_mut_ptr(), y.len()) };
            }
        }
        if ty == kernels::GGML_Q4_0 && *OURS > 0 {
            // rows longer than 3072 columns go a row at a time: beyond that, the 2-row pass loses the hardware
            // prefetcher (Qwen 3.6's ssm_out is 4096 columns)
            if *OURS == 1 || rb > 3072 / 32 * kernels::Q4_0_BLOCK {
                // SAFETY: FEAT_DotProd was checked above; the kernel asserts the slice sizes
                return unsafe { kernels::neon::q4_0_dot_rows1(w, rb, y.len(), x.g, y) };
            }
            // SAFETY: FEAT_DotProd was checked above; the kernel asserts the slice sizes
            return unsafe { kernels::neon::q4_0_dot_rows_var(w, rb, y.len(), x.g, y, 2, 0) };
        }
        // IQ4_NL (the Flash Next experts): 2 rows a pass, bit-identical to the reference (parity test)
        static IQ: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::arch::is_aarch64_feature_detected!("dotprod"));
        if ty == 20 && *IQ {
            // SAFETY: FEAT_DotProd was checked above; the kernel asserts the slice sizes
            return unsafe { kernels::neon::iq4nl_dot_rows(w, rb, y.len(), x.g, y, 2) };
        }
        // Q8_0: bit-identical to the reference (parity test), as are Q6_K and Q4_1 below
        static Q8: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::arch::is_aarch64_feature_detected!("dotprod"));
        // Q5_0 is unpacked to Q8_0 weights (q - 16 under the same scale) and then follows Q8_0's steps, giving the bits
        // of a Q8_0 copy of the same weights (parity test)
        if ty == 6 && *Q8 {
            // SAFETY: FEAT_DotProd was checked above; the kernel asserts the sizes; y holds y.len() rows (one vector)
            return unsafe { kernels::neon::q5_0_matmul(w, rb, y.len(), x.g, 1, y.as_mut_ptr(), y.len()) };
        }
        if ty == 8 && *Q8 {
            // a row a pass (q8_0_matmul over one vector, the same bits): two rows a pass lose the hardware prefetcher
            // on long rows (Flash Next's MTP head and attention are Q8_0) SAFETY: FEAT_DotProd was checked above; the
            // kernel asserts the sizes; y holds y.len() rows (one vector)
            return unsafe { kernels::neon::q8_0_matmul(w, rb, y.len(), x.g, 1, y.as_mut_ptr(), y.len()) };
        }
        // Q6_K: bit-identical (parity test)
        static Q6: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::arch::is_aarch64_feature_detected!("dotprod"));
        // a laid-out copy of these rows (TORC_KLAY=1) goes through q6_lay_matmul, with the same bits
        if ty == 14 && unsafe { klay_matmul(ty, &w[..y.len() * rb], y.len(), x, 1, y.as_mut_ptr(), y.len()) } {
            return;
        }
        if ty == 14 && *Q6 {
            // SAFETY: FEAT_DotProd was checked above; the kernel asserts the slice sizes
            return unsafe { kernels::neon::q6k_dot_rows(w, rb, y.len(), x.k, y) };
        }
        // Q4_1 (Qwen 3.6's expert down): bit-identical (parity test)
        static Q41: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::arch::is_aarch64_feature_detected!("dotprod"));
        if ty == 3 && *Q41 {
            // SAFETY: FEAT_DotProd was checked above; the kernel asserts the slice sizes
            return unsafe { kernels::neon::q4_1_dot_rows(w, rb, y.len(), x.g, y) };
        }
        // Q5_1 (Flash Next Q4_K_XL): dm_matmul over one vector, the same bits (parity test)
        if ty == 7 && *Q41 {
            // SAFETY: FEAT_DotProd was checked above; the kernel asserts the sizes; y holds y.len() rows (one vector)
            return unsafe { kernels::neon::dm_matmul(true, w, rb, y.len(), x.g, 1, y.as_mut_ptr(), y.len()) };
        }
        // Q4_K / Q5_K: q45k_matmul over one vector, bit-identical (parity test)
        static KQ: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::arch::is_aarch64_feature_detected!("dotprod"));
        // a laid-out copy of these rows (TORC_KLAY=1) goes through q45k_lay_matmul, with the same bits
        if (ty == 12 || ty == 13)
            && unsafe { klay_matmul(ty, &w[..y.len() * rb], y.len(), x, 1, y.as_mut_ptr(), y.len()) }
        {
            return;
        }
        if (ty == 12 || ty == 13) && *KQ {
            // SAFETY: FEAT_DotProd was checked above; the kernel asserts the sizes; y holds y.len() rows (one vector)
            return unsafe { kernels::neon::q45k_matmul(ty == 13, w, rb, y.len(), x.k, 1, y.as_mut_ptr(), y.len()) };
        }
        if ty == 11 && *KQ {
            // Q3_K SAFETY: as above
            return unsafe { kernels::neon::q3k_matmul(w, rb, y.len(), x.k, 1, y.as_mut_ptr(), y.len()) };
        }
        if ty == 23 && *KQ {
            // IQ4_XS SAFETY: as above
            return unsafe { kernels::neon::iq4xs_matmul(w, rb, y.len(), x.k, 1, y.as_mut_ptr(), y.len()) };
        }
        if ty == 21 && *KQ {
            // IQ3_S SAFETY: as above
            return unsafe { kernels::neon::iq3s_matmul(w, rb, y.len(), x.k, 1, y.as_mut_ptr(), y.len()) };
        }
        if ty == 18 && *KQ {
            // IQ3_XXS SAFETY: as above
            return unsafe { kernels::neon::iq3xxs_matmul(w, rb, y.len(), x.k, 1, y.as_mut_ptr(), y.len()) };
        }
        if ty == 22 && *KQ {
            // IQ2_S SAFETY: as above
            return unsafe { kernels::neon::iq2s_matmul(w, rb, y.len(), x.k, 1, y.as_mut_ptr(), y.len()) };
        }
        if ty == 42 && *KQ {
            // Q2_0 SAFETY: as above
            return unsafe { kernels::neon::q2_0_matmul(w, rb, y.len(), x.g, 1, y.as_mut_ptr(), y.len()) };
        }
    }
    // x86-64 Q4_0: kernels::x86dot::q4_0_rows (the same bits)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
    {
        if ty == kernels::GGML_Q4_0 {
            // SAFETY: AVX2, FMA and F16C are the build's target (x86-64-v3); the kernel asserts the sizes
            return unsafe { kernels::x86dot::q4_0_rows(w, rb, y.len(), x.g, y) };
        }
        // Q8_0 (and Q5_0, whose weights go Q8_0's way), Q4_1, Q5_1, Q6_K, IQ4_NL, MXFP4, Q3_K, Q4_K, Q5_K, IQ3_S and
        // IQ4_XS: kernels::x86mm over one vector, bit-identical (parity test)
        if matches!(ty, 3 | 6 | 7 | 8 | 11 | 12 | 13 | 14 | 20 | 21 | 23 | 39) {
            // SAFETY: as above; y holds y.len() rows (one vector)
            return unsafe { kernels::x86mm::matmul(ty, w, rb, y.len(), x, 1, y.as_mut_ptr(), y.len()) };
        }
    }
    // SAFETY: y holds y.len() rows (one vector)
    unsafe { rest_multi(ty, w, rb, y.len(), x, 1, y.as_mut_ptr(), y.len()) }
}

/// How one vector's dense products (apply_group_and, Mat::apply) are dealt out to the pool: TORC_DENSE_DYN=2 uses
/// guided runs, 1 dynamic claims, and anything else (the default) par_for. Each task writes its own rows, so the bits
/// are the same whichever is used.
fn dense_deal() -> fn(usize, &(dyn Fn(usize) + Sync)) {
    static D: std::sync::LazyLock<u8> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_DENSE_DYN").unwrap_or(0));
    match *D {
        1 => pool::par_for_dynamic,
        2 => pool::par_for_guided,
        _ => pool::par_for,
    }
}

/// Several products over the same input in one sweep of the pool - one fork-join for a layer's products instead of one
/// per matrix. Each row is the same dot product as it would be alone.
pub fn apply_group(jobs: &mut [(Mat, &mut [f32])], x: Q8s) {
    apply_group_and(jobs, x, 0, &|_| {});
}

/// apply_group with `extra` more items in the same sweep, after the product tasks: item i calls f(i). Other work that
/// needs only the same input (a DeltaNet layer's beta and alpha dots, say) then costs no fork-join of its own. Each
/// item makes the same calls on whichever thread runs it, so the bits are the same.
pub fn apply_group_and(jobs: &mut [(Mat, &mut [f32])], x: Q8s, extra: usize, f: &(dyn Fn(usize) + Sync)) {
    // 64 rows a task (TORC_RC=r overrides it; every row is the same dot product however the rows are split)
    static RCV: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_RC").filter(|&r: &usize| r >= 1).unwrap_or(64));
    let rc = *RCV;
    let mut tasks: Vec<(usize, usize, usize, SyncPtr<f32>)> = Vec::new();
    for (j, (m, y)) in jobs.iter_mut().enumerate() {
        assert!(y.len() >= m.rows && x.len() == m.cols / QK);
        let p = SyncPtr::new(y);
        let mut r0 = 0;
        while r0 < m.rows {
            let n = rc.min(m.rows - r0);
            tasks.push((j, r0, n, p));
            r0 += n;
        }
    }
    let mats: Vec<Mat> = jobs.iter().map(|(m, _)| *m).collect();
    dense_deal()(tasks.len() + extra, &|t| {
        if t >= tasks.len() {
            return f(t - tasks.len());
        }
        let (j, r0, n, p) = tasks[t];
        let m = &mats[j];
        let rb = m.data.len() / m.rows;
        // SAFETY: the tasks cover disjoint row ranges of their own job's output, which outlives this call
        let ys = unsafe { p.slice(r0, n) };
        dot_rows(m.ty, &m.data[r0 * rb..(r0 + n) * rb], rb, x, ys);
    });
}

/// A router weight as it is held: fp16 on x86-64 (F16C) and ARM64 (FCVTL), widened in registers, which halves the bytes
/// read each token compared with f32; f32 elsewhere, widened once at load. fp16 to f32 is exact either way, so the
/// products are the same.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub type RouterW = u16;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub type RouterW = f32;

/// The activations packed once for SMMLA (kernels::neon::mmla_pack) and shared by every task over them; an empty slice
/// where nothing is packed (and on other targets, where this type is never filled).
#[cfg(target_arch = "aarch64")]
pub type Pk = kernels::neon::Packed;
#[cfg(not(target_arch = "aarch64"))]
pub type Pk = u8;
/// The activations packed for the Q6_K SMMLA kernel (kernels::neon::pack_q6); empty where there are none.
#[cfg(target_arch = "aarch64")]
pub type Pk6 = kernels::neon::PackedQ6;
#[cfg(not(target_arch = "aarch64"))]
pub type Pk6 = u8;

/// Whether Q6_K products over nv vectors go by SMMLA (ARM with FEAT_I8MM, from TORC_MMLA_MIN vectors).
#[allow(unused_variables)]
fn use_q6(ty: u32, nv: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        // TORC_Q6_MIN: the vocabulary head's own threshold (151,936 Q6_K rows, where SMMLA from 2 vectors nearly
        // doubled its time); when unset, TORC_MMLA_MIN applies, independently of the grouped Q4_0 range
        static Q6MIN: std::sync::LazyLock<Option<usize>> =
            std::sync::LazyLock::new(|| crate::switches::num("TORC_Q6_MIN"));
        return ty == 14
            && match *Q6MIN {
                Some(m) => nv >= m.max(2) && use_mmla(8, usize::MAX),
                None => use_mmla(8, nv),
            };
    }
    #[allow(unreachable_code)]
    false
}

/// use_q6 for a dense matrix: also true for a Q6_K matrix with a laid-out copy (TORC_KLAY_HEAD=1, the vocabulary head)
/// from 4 vectors, whose products then go through gemm_q6_lay (1-3 vectors go through q6_lay_matmul in dot_multi_rows),
/// with the same bits.
fn use_q6m(m: &Mat, nv: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    if m.ty == 14
        && nv >= 4
        && m.rows % 4 == 0
        && crate::repack::klay_on()
        && crate::repack::find(m.data, m.rows).is_some()
    {
        return true;
    }
    use_q6(m.ty, nv)
}

/// use_q6 for the experts: with their Q6_K laid out (TORC_KLAY=1, repack::klay_type_on), SMMLA from 4 vectors as for
/// the Q4_K / Q5_K experts (1-3 vectors go through q6_lay_matmul), so no product reads the original rows again.
fn use_q6e(ty: u32, nv: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    if ty == 14 && nv >= 4 && crate::repack::klay_on() && crate::repack::klay_type_on(14) {
        return true;
    }
    use_q6(ty, nv)
}

/// The activations packed for the Q4_K / Q5_K SMMLA kernel (kernels::neon::pack_k); empty where there are none.
#[cfg(target_arch = "aarch64")]
pub type Pkk = kernels::neon::PackedK;
#[cfg(not(target_arch = "aarch64"))]
pub type Pkk = u8;

/// Whether Q4_K / Q5_K products over nv vectors go by SMMLA (kernels::neon::gemm_q45k): ARM with FEAT_I8MM, from
/// TORC_KMM_MIN vectors (default 4); TORC_KMM=0 turns it off.
#[allow(unused_variables)]
fn use_kmm(ty: u32, nv: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        static MIN: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
            if !std::arch::is_aarch64_feature_detected!("i8mm") || crate::switches::is("TORC_KMM", "0") {
                return usize::MAX;
            }
            crate::switches::num("TORC_KMM_MIN").unwrap_or(4).max(2)
        });
        return matches!(ty, 12 | 13) && nv >= *MIN;
    }
    #[allow(unreachable_code)]
    false
}

#[allow(unused_variables)]
fn packk_len(nv: usize, nb: usize) -> usize {
    #[cfg(target_arch = "aarch64")]
    return kernels::neon::pack_k_len(nv, nb);
    #[allow(unreachable_code)]
    0
}

#[allow(unused_variables)]
fn packk(x: Q8s, nv: usize, out: &mut [Pkk]) {
    #[cfg(target_arch = "aarch64")]
    kernels::neon::pack_k(x.k, nv, out);
}

#[allow(unused_variables)]
fn pack6_len(nv: usize, nb: usize) -> usize {
    #[cfg(target_arch = "aarch64")]
    return kernels::neon::pack_q6_len(nv, nb);
    #[allow(unreachable_code)]
    0
}

#[allow(unused_variables)]
fn pack6(x: Q8s, nv: usize, out: &mut [Pk6]) {
    #[cfg(target_arch = "aarch64")]
    kernels::neon::pack_q6(x.k, nv, out);
}

/// Whether products of type ty over nv vectors go by SMMLA (and so whether their vectors are worth packing): ARM with
/// FEAT_I8MM, Q4_0 or Q8_0, from TORC_MMLA_MIN vectors (default 4, where packing inside the call starts to pay).
/// TORC_MMLA=0 turns it off.
#[allow(unused_variables)]
fn use_mmla(ty: u32, nv: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::arch::is_aarch64_feature_detected!("i8mm")
                && crate::switches::get("TORC_MMLA").map_or(true, |v| v != "0")
        });
        // from 16 vectors by default: short batches (MTP verify passes) stay on the dot kernels
        static MIN: std::sync::LazyLock<usize> =
            std::sync::LazyLock::new(|| crate::switches::num("TORC_MMLA_MIN").unwrap_or(16));
        // IQ4_NL (Flash Next): neon::IQ4NL, bit-identical (parity test) (and Q4_0 over a repacked copy from 4 vectors,
        // since gemm_q4x4 takes the packed vectors too)
        return (((ty == kernels::GGML_Q4_0 && !grouped(nv)) || ty == 8 || ty == 20) && nv >= (*MIN).max(2) && *ON)
            || nv >= 4 && q4x4(ty, nv);
    }
    #[allow(unreachable_code)]
    false
}

/// Whether a Q4_0 product over nv vectors goes by a repacked copy (repack.rs): q4x4_matmul for 1-3 vectors, gemm_q4x4
/// from 4, for nv in TORC_Q4X4_MIN..=TORC_Q4X4_MAX (all by default).
#[allow(unused_variables)]
fn q4x4(ty: u32, nv: usize) -> bool {
    // TORC_Q4X4_MIN (default 1): the number of vectors from which the repacked copy is used (1-3 go through
    // q4x4_matmul)
    static R: std::sync::LazyLock<(usize, usize)> = std::sync::LazyLock::new(|| {
        let var = |k: &str| crate::switches::num::<usize>(k);
        (var("TORC_Q4X4_MIN").unwrap_or(1).max(1), var("TORC_Q4X4_MAX").unwrap_or(usize::MAX))
    });
    ty == kernels::GGML_Q4_0 && nv >= R.0 && nv <= R.1 && crate::repack::on()
}

/// The number of packed entries for nv vectors of nb blocks (0 where SMMLA is not built).
fn pack_len(nv: usize, nb: usize) -> usize {
    #[cfg(target_arch = "aarch64")]
    return kernels::neon::pack_len(nv, nb);
    #[allow(unreachable_code)]
    {
        let _ = (nv, nb);
        0
    }
}

#[allow(unused_variables)]
fn pack(x: Q8s, nv: usize, out: &mut [Pk]) {
    #[cfg(target_arch = "aarch64")]
    kernels::neon::mmla_pack(x.g, nv, out);
}

/// Whether Q4_0 products over nv vectors go by kernels::neon_mm's rows (ARM): for nv in
/// TORC_GROUPED..=TORC_GROUPED_MAX, 4..=15 by default; TORC_GROUPED=0 turns it off. From 16 vectors the packed SMMLA
/// path takes over.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn grouped(nv: usize) -> bool {
    static R: std::sync::LazyLock<(usize, usize)> = std::sync::LazyLock::new(|| {
        let var = |k: &str| crate::switches::num::<usize>(k);
        match var("TORC_GROUPED") {
            Some(0) => (usize::MAX, 0),
            k => (k.unwrap_or(4).max(2), var("TORC_GROUPED_MAX").unwrap_or(15)),
        }
    });
    nv >= R.0 && nv <= R.1
}

/// A weight matrix times a small batch of vectors. Weight row r is the row_bytes starting at w[r * row_bytes], input v
/// is the blocks from x[v * cols / QK], and output (v, r) goes to y[v * y_stride + r]. Every output carries the bits a
/// one-vector product would give. The kernel used:
/// - Q4_0 on ARM over grouped(nv) vectors: kernels::neon_mm's rows;
/// - pk (x packed, use_mmla for ty and nv): ARM SMMLA, 2 rows x 2 vectors an instruction;
/// - pk6 (x packed for Q6_K, use_q6): ARM SMMLA for Q6_K;
/// - pkk (x packed for Q4_K / Q5_K, use_kmm): ARM SMMLA for those;
/// - Q4_0 on x86-64: AVX-VNNI with 8 rows a register, else AVX2 by groups of vectors;
/// - Q4_0 on ARM without I8MM: SDOT by element from 16 vectors;
/// - Q8_0 on x86-64 with AVX-VNNI: 8 rows a register;
/// - everything else row by row.
///
/// # Safety
/// w holds the rows, x holds nv vectors of their columns, and y has room for every output; pk is empty or mmla_pack of
/// x, pk6 is empty or pack_q6 of x, and pkk is empty or pack_k of x.
#[allow(unused_variables, clippy::too_many_arguments)]
unsafe fn dot_multi_rows(
    ty: u32,
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: Q8s,
    pk: &[Pk],
    pk6: &[Pk6],
    pkk: &[Pkk],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if !pkk.is_empty() && matches!(ty, 12 | 13) {
        let r4 = rows / 4 * 4;
        let nb = x.len() / nv;
        unsafe {
            if r4 > 0 {
                // a laid-out copy of these rows (TORC_KLAY=1) goes through gemm_q45k_lay, with the same bits
                match crate::repack::klay_on().then(|| crate::repack::find(&w[..r4 * row_bytes], r4)).flatten() {
                    Some(lay) => kernels::neon::gemm_q45k_lay(ty == 13, lay, r4, pkk, nv, nb, y, y_stride),
                    None => kernels::neon::gemm_q45k(
                        ty == 13,
                        &w[..r4 * row_bytes],
                        row_bytes,
                        r4,
                        pkk,
                        nv,
                        nb,
                        y,
                        y_stride,
                    ),
                }
            }
            if r4 < rows {
                rest_multi(ty, &w[r4 * row_bytes..], row_bytes, rows - r4, x, nv, y.add(r4), y_stride);
            }
        }
        return;
    }
    // a repacked copy of these rows (TORC_Q4X4=1) goes through gemm_q4x4 from 4 vectors and q4x4_matmul below that,
    // with the same bits
    #[cfg(target_arch = "aarch64")]
    if nv >= 4 && !pk.is_empty() && q4x4(ty, nv) {
        if let Some(rp) = crate::repack::find(&w[..rows * row_bytes], rows) {
            unsafe { kernels::neon::gemm_q4x4(rp, rows, pk, nv, x.len() / nv, y, y_stride) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if nv <= 3 && q4x4(ty, nv) && std::arch::is_aarch64_feature_detected!("dotprod") {
        if let Some(rp) = crate::repack::find(&w[..rows * row_bytes], rows) {
            unsafe { kernels::neon::q4x4_matmul(rp, rows, x.g, nv, y, y_stride) };
            return;
        }
    }
    // one vector (an expert with one token of a block): the single-vector row kernels, which are bit-identical to the
    // multi-vector ones over one vector (parity test)
    #[cfg(target_arch = "aarch64")]
    if nv == 1 && (ty == kernels::GGML_Q4_0 || ty == 20) {
        // SAFETY: y holds rows elements at y[r] (v = 0), and nothing else touches them during the call
        return dot_rows(ty, &w[..rows * row_bytes], row_bytes, x, unsafe { std::slice::from_raw_parts_mut(y, rows) });
    }
    #[cfg(target_arch = "aarch64")]
    if ty == kernels::GGML_Q4_0 && grouped(nv) {
        // the SMMLA kernels (kernels::neon_mm, the same bits). With the dense matrices on the repacked path, this
        // carries only the experts
        static OURS: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::arch::is_aarch64_feature_detected!("i8mm") && std::arch::is_aarch64_feature_detected!("dotprod")
        });
        if *OURS && x.len() / nv <= kernels::neon_mm::MAX_NB {
            unsafe { kernels::neon_mm::q4_0_rows_multi(w, row_bytes, rows, x.g, nv, y, y_stride) };
        } else {
            unsafe { rest_multi(ty, w, row_bytes, rows, x, nv, y, y_stride) };
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if !pk6.is_empty() && ty == 14 {
        let r4 = rows / 4 * 4;
        let nb = x.len() / nv;
        unsafe {
            if r4 > 0 {
                // a laid-out copy of these rows (TORC_KLAY=1) goes through gemm_q6_lay, with the same bits
                match crate::repack::klay_on().then(|| crate::repack::find(&w[..r4 * row_bytes], r4)).flatten() {
                    Some(lay) => kernels::neon::gemm_q6_lay(lay, r4, pk6, nv, nb, y, y_stride),
                    None => kernels::neon::gemm_q6k(&w[..r4 * row_bytes], row_bytes, r4, pk6, nv, nb, y, y_stride),
                }
            }
            if r4 < rows {
                rest_multi(ty, &w[r4 * row_bytes..], row_bytes, rows - r4, x, nv, y.add(r4), y_stride);
            }
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if !pk.is_empty() && (ty == kernels::GGML_Q4_0 || ty == 8 || ty == 20) {
        let r4 = rows / 4 * 4;
        let nb = x.len() / nv;
        unsafe {
            if r4 > 0 {
                let w4 = &w[..r4 * row_bytes];
                if ty == 8 {
                    kernels::neon::gemm_mmla_packed::<kernels::neon::Q80>(w4, row_bytes, r4, pk, nv, nb, y, y_stride);
                } else if ty == 20 {
                    kernels::neon::gemm_mmla_packed::<kernels::neon::IQ4NL>(w4, row_bytes, r4, pk, nv, nb, y, y_stride);
                } else {
                    kernels::neon::gemm_mmla_packed::<kernels::neon::Q40>(w4, row_bytes, r4, pk, nv, nb, y, y_stride);
                }
            }
            if r4 < rows {
                rest_multi(ty, &w[r4 * row_bytes..], row_bytes, rows - r4, x, nv, y.add(r4), y_stride);
            }
        }
        return;
    }
    // ARM, Q4_0 over 2..=TORC_DOTN vectors (default 3; 0 = one vector at a time): kernels::neon::q4_0_matmul, 2 rows
    // x 2 vectors a pass by SDOT, with the same bits; each pass reads its rows once for both vectors
    #[cfg(target_arch = "aarch64")]
    {
        static DOTN: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
            if !std::arch::is_aarch64_feature_detected!("dotprod") {
                return 0;
            }
            crate::switches::num("TORC_DOTN").unwrap_or(3)
        });
        if ty == kernels::GGML_Q4_0 && nv >= 2 && nv <= *DOTN {
            unsafe { kernels::neon::q4_0_matmul(w, row_bytes, rows, x.g, nv, y, y_stride) };
            return;
        }
    }
    // the packed GEMM microkernel (kernels::x86_gemm) from PACKED_MIN vectors, for Q4_0, Q8_0 and MXFP4 with VNNI
    // (AVX-VNNI, or AVX512-VNNI on 256-bit registers: kernels::x86mm::vnni), with the same bits (parity test). It is
    // well ahead of the 8-row kernel from 4 vectors up. TORC_PACKED_MIN=n tunes it, and a large n keeps the 8-row
    // kernel (which needs AVX-VNNI itself)
    #[cfg(target_arch = "x86_64")]
    static PACKED_MIN: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_PACKED_MIN").unwrap_or(4));
    #[cfg(target_arch = "x86_64")]
    if ty == kernels::GGML_Q4_0 {
        // many vectors and VNNI: 8 rows a lane each, so the unpacking is paid once for all the vectors
        let vnni = kernels::x86mm::vnni();
        // from 2 vectors on (the kernel alone is well ahead from 2; TORC_GEMM_MIN tunes it)
        static MIN: std::sync::LazyLock<usize> =
            std::sync::LazyLock::new(|| crate::switches::num("TORC_GEMM_MIN").unwrap_or(2));
        let packed = vnni > 0 && nv >= *MIN && nv >= *PACKED_MIN;
        let r8 = if packed || vnni == 1 && nv >= *MIN { rows / 8 * 8 } else { 0 };
        unsafe {
            if r8 > 0 && packed {
                kernels::x86_gemm::gemm_q4_0(&w[..r8 * row_bytes], row_bytes, r8, x.g, nv, y, y_stride);
            } else if r8 > 0 {
                kernels::x86::q4_0_gemm8(&w[..r8 * row_bytes], row_bytes, r8, x.g, nv, y, y_stride);
            }
            if r8 < rows {
                kernels::x86::q4_0_dot_multi(&w[r8 * row_bytes..], row_bytes, rows - r8, x.g, nv, y.add(r8), y_stride);
            }
        }
        return;
    }
    // ARM without I8MM: Q4_0 with 4 rows a register (SDOT by element), from 16 vectors
    #[cfg(target_arch = "aarch64")]
    if ty == kernels::GGML_Q4_0 && nv >= 16 && std::arch::is_aarch64_feature_detected!("dotprod") {
        let r4 = rows / 4 * 4;
        unsafe {
            if r4 > 0 {
                kernels::neon::q4_0_gemm4(&w[..r4 * row_bytes], row_bytes, r4, x.g, nv, y, y_stride);
            }
            if r4 < rows {
                rest_multi(ty, &w[r4 * row_bytes..], row_bytes, rows - r4, x, nv, y.add(r4), y_stride);
            }
        }
        return;
    }
    // Q6_K (attn_q, the vocabulary head) with AVX-VNNI from 4 vectors: 8 rows in the lanes
    #[cfg(target_arch = "x86_64")]
    static Q6MIN: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_Q6_MIN").unwrap_or(4));
    #[cfg(target_arch = "x86_64")]
    if ty == 14 && nv >= *Q6MIN && std::arch::is_x86_feature_detected!("avxvnni") {
        let r8 = rows / 8 * 8;
        unsafe {
            if r8 > 0 {
                kernels::x86::q6k_gemm8(&w[..r8 * row_bytes], row_bytes, r8, x.k, nv, y, y_stride);
            }
            if r8 < rows {
                x86_rest(ty, &w[r8 * row_bytes..], row_bytes, rows - r8, x, nv, y.add(r8), y_stride);
            }
        }
        return;
    }
    // Q4_K / Q5_K (the 35B UD-Q4_K_XL's experts) the same way, from TORC_KMM_X86_MIN vectors (default 4; 0 = never),
    // with the same bits as x86mm's q45k_pass (parity test)
    #[cfg(target_arch = "x86_64")]
    static KMIN: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| match crate::switches::num::<usize>("TORC_KMM_X86_MIN") {
            Some(0) => usize::MAX,
            Some(m) => m,
            None => 4,
        });
    #[cfg(target_arch = "x86_64")]
    if matches!(ty, 12 | 13) && nv >= *KMIN && std::arch::is_x86_feature_detected!("avxvnni") {
        let r8 = rows / 8 * 8;
        unsafe {
            if r8 > 0 {
                kernels::x86::q45k_gemm8(ty == 13, &w[..r8 * row_bytes], row_bytes, r8, x.k, nv, y, y_stride);
            }
            if r8 < rows {
                x86_rest(ty, &w[r8 * row_bytes..], row_bytes, rows - r8, x, nv, y.add(r8), y_stride);
            }
        }
        return;
    }
    // Q8_0 the same way (the resident Q8_0 products: ssm_out, k, v and the shared expert), and MXFP4 (gpt-oss's
    // experts) by the packed GEMM alone
    #[cfg(target_arch = "x86_64")]
    {
        let vnni = kernels::x86mm::vnni();
        let packed = vnni > 0 && nv >= 2 && nv >= *PACKED_MIN;
        if ty == 8 && (packed || vnni == 1 && nv >= 2) || ty == 39 && packed {
            let r8 = rows / 8 * 8;
            unsafe {
                if r8 > 0 && packed {
                    if ty == 8 {
                        kernels::x86_gemm::gemm_q8_0(&w[..r8 * row_bytes], row_bytes, r8, x.g, nv, y, y_stride);
                    } else {
                        kernels::x86_gemm::gemm_mxfp4(&w[..r8 * row_bytes], row_bytes, r8, x.g, nv, y, y_stride);
                    }
                } else if r8 > 0 {
                    kernels::x86::q8_0_gemm8(&w[..r8 * row_bytes], row_bytes, r8, x.g, nv, y, y_stride);
                }
                if r8 < rows {
                    x86_rest(ty, &w[r8 * row_bytes..], row_bytes, rows - r8, x, nv, y.add(r8), y_stride);
                }
            }
            return;
        }
    }
    // x86-64, Q8_0 / Q4_1 / Q6_K / IQ4_NL not taken above: kernels::x86mm, which unpacks a row's block once for up to
    // 4 vectors (the same bits: parity test)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
    if kernels::x86mm::covers(ty) {
        return unsafe { x86_rest(ty, w, row_bytes, rows, x, nv, y, y_stride) };
    }
    // ARM, the other types the row kernels here cover (Q4_1, Q8_0, Q6_K, IQ4_NL), below their SMMLA thresholds:
    // multi-vector rows, each row's block unpacked once for several vectors (kernels::neon::*_matmul), with the same
    // bits per vector (parity tests). TORC_MULTI=pervec sends each vector through the single-vector rows instead,
    // unpacking the blocks again for every vector
    #[cfg(target_arch = "aarch64")]
    {
        static MODE: std::sync::LazyLock<u8> = std::sync::LazyLock::new(|| {
            if !std::arch::is_aarch64_feature_detected!("dotprod") {
                return 0;
            }
            match crate::switches::get("TORC_MULTI").as_deref() {
                Some("pervec") => 2,
                _ => 1,
            }
        });
        let wr = &w[..rows * row_bytes];
        // a laid-out copy of these K-quant rows (TORC_KLAY=1) goes through q45k_lay_matmul / q6_lay_matmul for 1-3
        // vectors, with the same bits
        if matches!(ty, 12 | 13 | 14) && unsafe { klay_matmul(ty, wr, rows, x, nv, y, y_stride) } {
            return;
        }
        if *MODE == 1 && matches!(ty, 3 | 6 | 7 | 8 | 11 | 12 | 13 | 14 | 18 | 20 | 21 | 22 | 23 | 42) {
            // SAFETY: FEAT_DotProd was checked; the kernels assert the sizes; y is as this function's contract says
            unsafe {
                match ty {
                    3 => kernels::neon::q4_1_matmul(wr, row_bytes, rows, x.g, nv, y, y_stride),
                    6 => kernels::neon::q5_0_matmul(wr, row_bytes, rows, x.g, nv, y, y_stride),
                    7 => kernels::neon::dm_matmul(true, wr, row_bytes, rows, x.g, nv, y, y_stride),
                    8 => kernels::neon::q8_0_matmul(wr, row_bytes, rows, x.g, nv, y, y_stride),
                    14 => kernels::neon::q6k_matmul(wr, row_bytes, rows, x.k, nv, y, y_stride),
                    12 | 13 => kernels::neon::q45k_matmul(ty == 13, wr, row_bytes, rows, x.k, nv, y, y_stride),
                    11 => kernels::neon::q3k_matmul(wr, row_bytes, rows, x.k, nv, y, y_stride),
                    23 => kernels::neon::iq4xs_matmul(wr, row_bytes, rows, x.k, nv, y, y_stride),
                    21 => kernels::neon::iq3s_matmul(wr, row_bytes, rows, x.k, nv, y, y_stride),
                    18 => kernels::neon::iq3xxs_matmul(wr, row_bytes, rows, x.k, nv, y, y_stride),
                    22 => kernels::neon::iq2s_matmul(wr, row_bytes, rows, x.k, nv, y, y_stride),
                    42 => kernels::neon::q2_0_matmul(wr, row_bytes, rows, x.g, nv, y, y_stride),
                    _ => kernels::neon::iq4nl_matmul(wr, row_bytes, rows, x.g, nv, y, y_stride),
                }
            }
            return;
        }
        if *MODE == 2 && matches!(ty, 3 | 8 | 14 | 20) {
            let nb = x.len() / nv;
            for v in 0..nv {
                // SAFETY: vector v's outputs are y[v * y_stride + r] for r < rows, disjoint for each v, and nothing
                // else touches them
                let yv = unsafe { std::slice::from_raw_parts_mut(y.add(v * y_stride), rows) };
                dot_rows(ty, &w[..rows * row_bytes], row_bytes, x.sub(v * nb..(v + 1) * nb), yv);
            }
            return;
        }
    }
    unsafe { rest_multi(ty, w, row_bytes, rows, x, nv, y, y_stride) }
}

/// K-quant rows (Q4_K / Q5_K / Q6_K; w their bytes, rows a multiple of 4) against 1-3 vectors, using their laid-out
/// copy (TORC_KLAY=1, repack.rs): q45k_lay_matmul / q6_lay_matmul, with the same bits as q45k_matmul / q6k_matmul.
/// Returns false, having done nothing, when there is no copy or the vector count is not 1-3.
///
/// # Safety
/// As for dot_multi_rows.
#[cfg(target_arch = "aarch64")]
unsafe fn klay_matmul(ty: u32, w: &[u8], rows: usize, x: Q8s, nv: usize, y: *mut f32, y_stride: usize) -> bool {
    if !(matches!(ty, 12 | 13 | 14) && (1..=3).contains(&nv) && rows % 4 == 0 && crate::repack::klay_on()) {
        return false;
    }
    match crate::repack::find(w, rows) {
        // SAFETY: klay_on implies FEAT_I8MM, which implies DotProd on these cores; y is as dot_multi_rows says
        Some(lay) => unsafe {
            if ty == 14 {
                kernels::neon::q6_lay_matmul(lay, rows, x.k, nv, y, y_stride)
            } else {
                kernels::neon::q45k_lay_matmul(ty == 13, lay, rows, x.k, nv, y, y_stride)
            }
        },
        None => return false,
    }
    true
}

/// The rows the kernels above leave over, or every row of a type none of them takes: this crate's row kernels.
///
/// # Safety
/// As for dot_multi_rows.
#[allow(clippy::too_many_arguments)]
unsafe fn rest_multi(
    ty: u32,
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: Q8s,
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    if !unsafe { native_multi(ty, w, row_bytes, rows, x, nv, y, y_stride) } {
        panic!("ggml type {ty}: no kernel for it on this CPU");
    }
}

/// Rows against nv vectors using this crate's row kernels, for a type one of them covers (returns true). Each vector
/// gets the same bits as it would one vector at a time (parity tests).
///
/// # Safety
/// As for dot_multi_rows.
#[allow(clippy::too_many_arguments, unused_variables, dead_code)]
unsafe fn native_multi(
    ty: u32,
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: Q8s,
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) -> bool {
    let wr = &w[..rows * row_bytes];
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: FEAT_DotProd was checked; the kernels assert the sizes; y is as this function's contract says
        unsafe {
            match ty {
                2 => kernels::neon::q4_0_matmul(wr, row_bytes, rows, x.g, nv, y, y_stride),
                3 => kernels::neon::q4_1_matmul(wr, row_bytes, rows, x.g, nv, y, y_stride),
                6 => kernels::neon::q5_0_matmul(wr, row_bytes, rows, x.g, nv, y, y_stride),
                7 => kernels::neon::dm_matmul(true, wr, row_bytes, rows, x.g, nv, y, y_stride),
                8 => kernels::neon::q8_0_matmul(wr, row_bytes, rows, x.g, nv, y, y_stride),
                14 => kernels::neon::q6k_matmul(wr, row_bytes, rows, x.k, nv, y, y_stride),
                12 | 13 => kernels::neon::q45k_matmul(ty == 13, wr, row_bytes, rows, x.k, nv, y, y_stride),
                11 => kernels::neon::q3k_matmul(wr, row_bytes, rows, x.k, nv, y, y_stride),
                23 => kernels::neon::iq4xs_matmul(wr, row_bytes, rows, x.k, nv, y, y_stride),
                21 => kernels::neon::iq3s_matmul(wr, row_bytes, rows, x.k, nv, y, y_stride),
                18 => kernels::neon::iq3xxs_matmul(wr, row_bytes, rows, x.k, nv, y, y_stride),
                22 => kernels::neon::iq2s_matmul(wr, row_bytes, rows, x.k, nv, y, y_stride),
                42 => kernels::neon::q2_0_matmul(wr, row_bytes, rows, x.g, nv, y, y_stride),
                20 => kernels::neon::iq4nl_matmul(wr, row_bytes, rows, x.g, nv, y, y_stride),
                39 => kernels::neon::mxfp4_matmul(wr, row_bytes, rows, x.g, nv, y, y_stride),
                _ => return false,
            }
        }
        return true;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
    {
        if ty == kernels::GGML_Q4_0 {
            // SAFETY: AVX2, FMA and F16C are the build's target; y is as this function's contract says
            unsafe { kernels::x86::q4_0_dot_multi(wr, row_bytes, rows, x.g, nv, y, y_stride) };
            return true;
        }
        if kernels::x86mm::covers(ty) {
            // SAFETY: as above
            unsafe { kernels::x86mm::matmul(ty, wr, row_bytes, rows, x, nv, y, y_stride) };
            return true;
        }
    }
    // MXFP4 where there is no vector kernel for it: the portable one
    if ty == 39 {
        // SAFETY: y[v * y_stride + r] is writable for v < nv and r < rows (this function's contract), so the slice
        // up to the last of them is
        let ys = unsafe { std::slice::from_raw_parts_mut(y, (nv - 1) * y_stride + rows) };
        kernels::mxfp4_matmul_scalar(wr, row_bytes, rows, x.g, nv, ys, y_stride);
        return true;
    }
    false
}

/// x86-64: rows of a type kernels::x86mm covers go against nv vectors through it, and everything else through
/// rest_multi.
///
/// # Safety
/// As for dot_multi_rows.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn x86_rest(ty: u32, w: &[u8], row_bytes: usize, rows: usize, x: Q8s, nv: usize, y: *mut f32, y_stride: usize) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
    {
        if kernels::x86mm::covers(ty) {
            // SAFETY: AVX2, FMA and F16C are the build's target; y is as this function's contract says
            return unsafe { kernels::x86mm::matmul(ty, &w[..rows * row_bytes], row_bytes, rows, x, nv, y, y_stride) };
        }
    }
    unsafe { rest_multi(ty, w, row_bytes, rows, x, nv, y, y_stride) }
}

/// Runs several matrices against one shared batch of nv vectors (x holds nv x cols / QK blocks) as a single dispatch
/// to the pool; job j writes y[v * rows + r]. A weight row is streamed from memory once and used for all nv vectors,
/// and the results match the one-vector path bit for bit.
pub fn apply_group_multi(jobs: &mut [(Mat, &mut [f32])], x: Q8s, nv: usize) {
    apply_group_multi_and(jobs, x, nv, 0, &|_| {});
}

/// apply_group_multi with `extra` more items in the same sweep, after the product tasks (as apply_group_and does).
pub fn apply_group_multi_and(
    jobs: &mut [(Mat, &mut [f32])],
    x: Q8s,
    nv: usize,
    extra: usize,
    f: &(dyn Fn(usize) + Sync),
) {
    // 16 rows a task; TORC_RC_WIDE=R uses R rows from 64 vectors on (with TORC_VPANEL, a panel's packed
    // vectors then serve R rows from L1)
    static WIDE: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        crate::switches::num("TORC_RC_WIDE").filter(|&r: &usize| r >= 4 && r % 8 == 0).unwrap_or(16)
    });
    let rc = if nv >= 64 { *WIDE } else { 16 };
    let mut tasks: Vec<(usize, usize, usize, SyncPtr<f32>)> = Vec::new();
    for (j, (m, y)) in jobs.iter_mut().enumerate() {
        assert!(y.len() >= m.rows * nv && x.len() == nv * (m.cols / QK));
        let p = SyncPtr::new(y);
        let mut r0 = 0;
        while r0 < m.rows {
            let n = rc.min(m.rows - r0);
            tasks.push((j, r0, n, p));
            r0 += n;
        }
    }
    let mats: Vec<Mat> = jobs.iter().map(|(m, _)| *m).collect();
    // the vectors are packed once for every SMMLA task over them
    let mm = mats.iter().any(|m| use_mmla(m.ty, nv) || nv >= 4 && q4x4(m.ty, nv));
    let mut pk = crate::scratch::buf::<Pk>("multi.pk", if mm { pack_len(nv, x.len() / nv) } else { 0 });
    if mm {
        pack(x, nv, &mut pk);
    }
    let q6m: Vec<bool> = mats.iter().map(|m| use_q6m(m, nv)).collect();
    let m6 = q6m.iter().any(|&b| b);
    let mut pk6 = crate::scratch::buf::<Pk6>("multi.pk6", if m6 { pack6_len(nv, x.len() / nv) } else { 0 });
    if m6 {
        pack6(x, nv, &mut pk6);
    }
    let mk = mats.iter().any(|m| use_kmm(m.ty, nv));
    let mut pkk = crate::scratch::buf::<Pkk>("multi.pkk", if mk { packk_len(nv, x.len() / nv) } else { 0 });
    if mk {
        packk(x, nv, &mut pkk);
    }
    // small blocks (short MTP verify passes) give each thread contiguous runs of tasks; a 128-token prompt block and
    // one token's 64-row chunks stay round-robin. Larger blocks use dynamic scheduling on x86-64, where the cores
    // differ in speed (P- and E-cores); ARM keeps the static split. TORC_DYN=0 / 1 overrides the choice
    static DYN: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| match crate::switches::get("TORC_DYN") {
        Some(v) => v == "1",
        None => cfg!(target_arch = "x86_64"),
    });
    let split: fn(usize, &(dyn Fn(usize) + Sync)) = if nv <= RUNS_MAX {
        pool::par_for_guided
    } else if *DYN {
        pool::par_for_dynamic
    } else {
        pool::par_for
    };
    split(tasks.len() + extra, &|t| {
        if t >= tasks.len() {
            return f(t - tasks.len());
        }
        let (j, r0, n, p) = tasks[t];
        let m = &mats[j];
        let rb = m.data.len() / m.rows;
        let pk: &[Pk] = if use_mmla(m.ty, nv) || nv >= 4 && q4x4(m.ty, nv) { &pk } else { &[] };
        let pk6: &[Pk6] = if q6m[j] { &pk6 } else { &[] };
        let pkk: &[Pkk] = if use_kmm(m.ty, nv) { &pkk } else { &[] };
        // SAFETY: task t writes y[v * rows + r0 .. + n] for each v, disjoint from every other task's elements, and y
        // holds nv * rows (asserted above)
        unsafe {
            dot_multi_rows(m.ty, &m.data[r0 * rb..(r0 + n) * rb], rb, n, x, pk, pk6, pkk, nv, p.0.add(r0), m.rows)
        };
    });
}

pub struct MoeLayer {
    /// the layer's index (the MTP block's is n_layer), for the NPU's record of hot experts
    pub layer: u32,
    pub group: u32,
    /// the router, rounded to fp16 at load (see [`RouterW`])
    pub router: Vec<RouterW>,
    pub shexp_gate: Option<Vec<f32>>,
    /// gate, up and down of the shared expert (resident), when the model has one
    pub shared: Option<[String; 3]>,
    /// qwen4exp (hc.rs): the router and the shared gate are summed in f64 over the bf16 mix, and the hidden activations
    /// quantised as ggml does (set by the model after load)
    pub hc: bool,
    /// gemma4: GELU (tanh) instead of SiLU on the gate (set by the model after load)
    pub gelu: bool,
    /// gemma4: each expert's own output scale (ffn_down_exps.scale), folded into its routing weight after the weights
    /// are normalised
    pub expert_scale: Option<Vec<f32>>,
    /// ernie4_5-moe: a per-expert bias (exp_probs_b) added to the probabilities to choose the experts only; the
    /// weights stay the unbiased probabilities, renormalised (llama.cpp build_moe_ffn's selection_probs)
    pub sel_bias: Option<Vec<f32>>,
    /// deepseek2 (GLM-4.7-Flash): sigmoid gating instead of the softmax. Each expert's probability is the sigmoid of
    /// its logit; the k largest (with sel_bias) are taken, their probabilities renormalised when `norm_w`, then
    /// multiplied by `w_scale` (llama.cpp build_moe_ffn, LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID)
    pub sigmoid: bool,
    pub norm_w: bool,
    pub w_scale: f32,
    /// TORC_ROUTER_F32=1: the router also kept in f32, and the logits taken from it instead of the fp16 copy (a test of
    /// what the fp16 rounding does to the expert choice; empty otherwise)
    pub router32: Vec<f32>,
}

/// TORC_ROUTER_F32=1 (see MoeLayer::router32).
fn router_f32() -> bool {
    static R: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switches::on("TORC_ROUTER_F32"));
    *R
}

/// The logits of n tokens from the f32 router (TORC_ROUTER_F32), over the fp16 ones, one item per token.
fn logits_f32(ly: &MoeLayer, xn: &[f32], dm: usize, ne: usize, n: usize, logits: &mut [f32]) {
    let lp = SyncPtr::new(logits);
    pool::par_for(n, &|t| {
        // SAFETY: item t owns token t's logits
        let row = unsafe { lp.slice(t * ne, ne) };
        for (e, v) in row.iter_mut().enumerate() {
            *v = kops::dot_f32(&ly.router32[e * dm..(e + 1) * dm], &xn[t * dm..(t + 1) * dm]);
        }
    });
}

impl MoeLayer {
    pub fn load(w: &Weights, l: u32, group: u32) -> Result<MoeLayer> {
        if w.tensor(&format!("blk.{l}.ffn_gate_inp.weight")).is_none() && w.layer_experts(group).n_expert == 1 {
            // qwen35's dense FFN layers have no router: the FFN runs as a single expert behind an all-zero router, and
            // a softmax over one logit is exactly 1, so the expert's output passes through unscaled
            let dm = w.need("token_embd.weight")?.shape().0;
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let router = vec![0 as RouterW; dm];
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let router = vec![0.0 as RouterW; dm];
            return Ok(MoeLayer {
                layer: l,
                group,
                router,
                shexp_gate: None,
                shared: None,
                hc: false,
                gelu: false,
                expert_scale: None,
                sel_bias: None,
                sigmoid: false,
                norm_w: true,
                w_scale: 1.0,
                router32: Vec::new(),
            });
        }
        let r = w.layer(l, "ffn_gate_inp.weight")?;
        // F32, or BF16 (the MTP block's): widened to f32, then rounded to fp16
        if r.t.ty != 0 && r.t.ty != 30 {
            return err(format!("layer {l}: router type {} (F32 or BF16 expected)", r.t.ty));
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let router = r.f32s()?.into_iter().map(kops::fp32_to_fp16).collect();
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let router = r.f32s()?.into_iter().map(|v| f16_to_f32(kops::fp32_to_fp16(v))).collect();
        let shexp_gate = match w.tensor(&format!("blk.{l}.ffn_gate_inp_shexp.weight")) {
            Some(t) => Some(t.f32s()?),
            None => None,
        };
        let names = ["ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"];
        let shared = if w.tensor(&format!("blk.{l}.{}", names[0])).is_some() {
            Some(names.map(|n| format!("blk.{l}.{n}")))
        } else {
            None
        };
        let sel_bias = match w.tensor(&format!("blk.{l}.exp_probs_b.bias")) {
            Some(t) => Some(t.f32s()?),
            None => None,
        };
        let router32 = if router_f32() { r.f32s()? } else { Vec::new() };
        Ok(MoeLayer {
            layer: l,
            group,
            router,
            shexp_gate,
            shared,
            hc: false,
            gelu: false,
            expert_scale: None,
            sel_bias,
            sigmoid: false,
            norm_w: true,
            w_scale: 1.0,
            router32,
        })
    }
}

/// The fp16 dot product on x86-64: the weights are widened by F16C into two 8-lane accumulators, 16 at a time with
/// fused steps, summed as hsum8(a0 + a1), and then the tail is added unfused.
#[cfg(target_arch = "x86_64")]
pub fn dot_router(w: &[u16], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = x.len();
    assert!(w.len() >= n);
    // SAFETY: the build targets x86-64-v3 (AVX2, FMA and F16C, see .cargo/config.toml), and the loads stay inside w and
    // x
    unsafe {
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i + 16 <= n {
            let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i) as *const __m128i));
            let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i + 8) as *const __m128i));
            a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x.as_ptr().add(i)), a0);
            a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x.as_ptr().add(i + 8)), a1);
            i += 16;
        }
        // hsum8: ((0+4) + (2+6)) + ((1+5) + (3+7))
        let s = _mm256_add_ps(a0, a1);
        let mut t = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
        t = _mm_add_ps(t, _mm_movehl_ps(t, t));
        t = _mm_add_ss(t, _mm_movehdup_ps(t));
        let mut r = _mm_cvtss_f32(t);
        while i < n {
            r += f16_to_f32(w[i]) * x[i];
            i += 1;
        }
        r
    }
}

/// The fp16 dot product on ARM: four lanes x four accumulators, 16 at a time with fused steps, summed as hsum4x2(a0 +
/// a2, a1 + a3), with the tail fused onto that.
#[cfg(target_arch = "aarch64")]
pub fn dot_router(w: &[u16], x: &[f32]) -> f32 {
    dot_router_n::<1>(w, [x])[0]
}

/// dot_router for T tokens at once (x86-64): the row's fp16 is read and widened once for all T, and each token's two
/// accumulators, fold and tail are exactly dot_router's, so the bits are the same.
#[cfg(target_arch = "x86_64")]
pub fn dot_router_n<const T: usize>(w: &[u16], x: [&[f32]; T]) -> [f32; T] {
    use core::arch::x86_64::*;
    let n = x[0].len();
    assert!(w.len() >= n && x.iter().all(|v| v.len() == n));
    // SAFETY: the build targets x86-64-v3 (AVX2, FMA, F16C), and the loads stay inside w and the x rows
    unsafe {
        let mut a = [[_mm256_setzero_ps(); 2]; T];
        let mut i = 0;
        while i + 16 <= n {
            let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i) as *const __m128i));
            let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i + 8) as *const __m128i));
            for t in 0..T {
                let xp = x[t].as_ptr().add(i);
                a[t][0] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp), a[t][0]);
                a[t][1] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp.add(8)), a[t][1]);
            }
            i += 16;
        }
        let mut out = [0.0f32; T];
        for t in 0..T {
            let s = _mm256_add_ps(a[t][0], a[t][1]);
            let mut h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
            h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            h = _mm_add_ss(h, _mm_movehdup_ps(h));
            let mut r = _mm_cvtss_f32(h);
            for k in i..n {
                r += f16_to_f32(w[k]) * x[t][k];
            }
            out[t] = r;
        }
        out
    }
}

/// dot_router where there is neither F16C nor NEON: the weights were widened at load.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn dot_router(w: &[f32], x: &[f32]) -> f32 {
    let n = x.len();
    let mut a = [[0.0f32; 4]; 4];
    let mut i = 0;
    while i + 16 <= n {
        for (q, acc) in a.iter_mut().enumerate() {
            for l in 0..4 {
                let k = i + 4 * q + l;
                acc[l] = w[k].mul_add(x[k], acc[l]);
            }
        }
        i += 16;
    }
    let lo: [f32; 4] = std::array::from_fn(|l| a[0][l] + a[2][l]);
    let hi: [f32; 4] = std::array::from_fn(|l| a[1][l] + a[3][l]);
    let s: [f32; 4] = std::array::from_fn(|l| lo[l] + hi[l]);
    let mut r = (s[0] + s[2]) + (s[1] + s[3]);
    while i < n {
        r = w[i].mul_add(x[i], r);
        i += 1;
    }
    r
}

/// dot_router for T tokens at once (ARM NEON): each row is loaded a single time and serves all four tokens, and each
/// token's lanes and sums are exactly dot_router's, so the bits are the same. It runs 16 independent FMA chains instead
/// of 4.
#[cfg(target_arch = "aarch64")]
pub fn dot_router_n<const T: usize>(w: &[u16], x: [&[f32]; T]) -> [f32; T] {
    use core::arch::aarch64::*;
    // FCVTL / FCVTL2: 8 fp16 values (as raw bits) widened exactly to two f32 x 4 vectors (float16x8_t is unstable in
    // core::arch)
    #[inline(always)]
    unsafe fn widen8(h: uint16x8_t) -> [float32x4_t; 2] {
        let (lo, hi): (float32x4_t, float32x4_t);
        unsafe {
            core::arch::asm!("fcvtl {l:v}.4s, {h:v}.4h", "fcvtl2 {u:v}.4s, {h:v}.8h", l = out(vreg) lo, u = out(vreg) hi,
                h = in(vreg) h, options(pure, nomem, nostack));
        }
        [lo, hi]
    }
    let n = x[0].len();
    assert!(w.len() >= n && x.iter().all(|v| v.len() == n));
    // SAFETY: NEON is baseline on aarch64, and the loads stay inside w and the x rows (i + 16 <= n)
    unsafe {
        let mut a = [[vdupq_n_f32(0.0); 4]; T];
        let mut i = 0;
        while i + 16 <= n {
            let [w0, w1] = widen8(vld1q_u16(w.as_ptr().add(i)));
            let [w2, w3] = widen8(vld1q_u16(w.as_ptr().add(i + 8)));
            let wq = [w0, w1, w2, w3];
            for t in 0..T {
                let xp = x[t].as_ptr().add(i);
                for q in 0..4 {
                    a[t][q] = vfmaq_f32(a[t][q], wq[q], vld1q_f32(xp.add(4 * q)));
                }
            }
            i += 16;
        }
        let mut out = [0.0f32; T];
        for t in 0..T {
            let s = vaddq_f32(vaddq_f32(a[t][0], a[t][2]), vaddq_f32(a[t][1], a[t][3]));
            let mut sv = [0.0f32; 4];
            vst1q_f32(sv.as_mut_ptr(), s);
            let mut r = (sv[0] + sv[2]) + (sv[1] + sv[3]);
            for k in i..n {
                r = f16_to_f32(w[k]).mul_add(x[t][k], r);
            }
            out[t] = r;
        }
        out
    }
}

/// Scratch buffers, sized once.
pub struct Scratch {
    pub router: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub h: Vec<f32>,
    pub h_q8: Q8,
    pub down_out: Vec<f32>,
}

impl Scratch {
    pub fn new(n_expert: usize, d_model: usize, ff_max: usize) -> Scratch {
        let fq = ff_max.div_ceil(QK) * QK;
        Scratch {
            router: vec![0.0; n_expert],
            gate: vec![0.0; fq],
            up: vec![0.0; fq],
            h: vec![0.0; fq],
            h_q8: Q8::new(fq / QK),
            down_out: vec![0.0; d_model],
        }
    }
}

/// FAST MODE (not exact), TORC_TAIL_SKIP=r (1..k-1; unset or 0 keeps the exact path). A routed expert that every token
/// of the call ranks r or lower (0-based: rank >= r among its k, i.e. the smallest weights), and that is not in RAM, is
/// neither read nor computed: its contribution is zero and the others keep the router's weights, with no
/// renormalisation. On a machine streaming experts from the SSD, each skipped miss is a read saved. The output differs
/// from the exact path's, which is why it is off by default. Letting what is already in RAM shape the routed set is
/// the idea of Skliar et al., "Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference" (TMLR 2025;
/// they rerank the router by a cache prior, where this only drops tail experts that would miss).
/// (skipped, considered) pairs in TAIL_STATS.
pub static TAIL_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

static TAIL: std::sync::LazyLock<usize> =
    std::sync::LazyLock::new(|| crate::switches::num("TORC_TAIL_SKIP").unwrap_or(0));

/// Drops the skippable tail from the layer's routes (each token's (expert, weight) pairs); see TAIL.
fn tail_skip(w: &Weights, layer: u32, routes: &mut [Vec<(usize, f32)>]) {
    let r = *TAIL;
    if r == 0 {
        return;
    }
    let Some(c) = w.cache.as_ref() else { return };
    // per expert: whether it is used, and whether every token using it ranks it >= r
    let mut low: std::collections::BTreeMap<usize, bool> = std::collections::BTreeMap::new();
    for rt in routes.iter() {
        for &(e, wt) in rt {
            let rank = rt.iter().filter(|&&(_, v)| v > wt).count();
            let ok = low.entry(e).or_insert(true);
            *ok &= rank >= r;
        }
    }
    let cand: Vec<u32> = low.iter().filter(|&(_, &ok)| ok).map(|(&e, _)| e as u32).collect();
    let considered: u64 = routes.iter().map(|rt| rt.len() as u64).sum();
    TAIL_STATS[1].fetch_add(considered, std::sync::atomic::Ordering::Relaxed);
    if cand.is_empty() {
        return;
    }
    let present = c.held(layer, &cand);
    let skip: Vec<usize> = cand.iter().zip(&present).filter(|&(_, &p)| !p).map(|(&e, _)| e as usize).collect();
    if skip.is_empty() {
        return;
    }
    let mut n = 0u64;
    for rt in routes.iter_mut() {
        let before = rt.len();
        rt.retain(|(e, _)| !skip.contains(e));
        n += (before - rt.len()) as u64;
    }
    TAIL_STATS[0].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// One token's expert choice from its router logits r (overwritten with their exps): the softmax, then pick.
fn route(r: &mut [f32], k: usize, ly: &MoeLayer) -> Vec<(usize, f32)> {
    if ly.sigmoid {
        return pick_sigmoid(r, k, ly);
    }
    let bias = ly.sel_bias.as_deref();
    let top = r.iter().fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
    for v in r.iter_mut() {
        *v = (*v - top).exp();
    }
    pick(r, k, bias)
}

/// The k experts with the largest exps (an equal exp goes to the lower id), each as (expert, its probability exp /
/// total), the k probabilities then renormalised by their own sum, taken largest first; in increasing expert id.
/// With a selection bias (ernie4_5-moe), the k are those with the largest probability + bias, weighted as before.
fn pick(ex: &[f32], k: usize, bias: Option<&[f32]>) -> Vec<(usize, f32)> {
    let total = ex.iter().fold(0.0f32, |t, &v| t + v);
    let sel: Vec<f32>;
    let ex_rank: &[f32] = match bias {
        Some(b) => {
            sel = ex.iter().zip(b).map(|(&v, &bb)| v / total + bb).collect();
            &sel
        }
        None => ex,
    };
    let rank = |a: &usize, b: &usize| ex_rank[*b].total_cmp(&ex_rank[*a]).then(a.cmp(b));
    let mut ids: Vec<usize> = (0..ex.len()).collect();
    if k < ids.len() {
        ids.select_nth_unstable_by(k - 1, rank);
        ids.truncate(k);
    }
    ids.sort_unstable_by(rank);
    let probs: Vec<f32> = ids.iter().map(|&e| ex[e] / total).collect();
    let norm = probs.iter().fold(0.0f32, |t, &p| t + p);
    let mut out: Vec<(usize, f32)> = ids.into_iter().zip(probs).map(|(e, p)| (e, p / norm)).collect();
    out.sort_unstable_by_key(|x| x.0);
    out
}

/// Sigmoid gating (ly.sigmoid) from the router logits r (overwritten with their sigmoids): the k experts with the
/// largest sigmoid + selection bias (an equal value goes to the lower id), each weighted by its sigmoid, renormalised
/// by their sum when ly.norm_w (the sum clamped below at 6.1e-5 as llama.cpp does), times ly.w_scale; in increasing id.
fn pick_sigmoid(r: &mut [f32], k: usize, ly: &MoeLayer) -> Vec<(usize, f32)> {
    let logits: Vec<f32> = if crate::switches::on("TORC_TRACE_ROUTES") { r.to_vec() } else { Vec::new() };
    for v in r.iter_mut() {
        *v = crate::ops::sigmoid(*v);
    }
    let sel: Vec<f32> = match &ly.sel_bias {
        Some(b) => r.iter().zip(b).map(|(&p, &bb)| p + bb).collect(),
        None => r.to_vec(),
    };
    let rank = |a: &usize, b: &usize| sel[*b].total_cmp(&sel[*a]).then(a.cmp(b));
    if !logits.is_empty() {
        // TORC_TRACE_ROUTES=1: the best six by selection score, with their logits
        let mut o: Vec<usize> = (0..r.len()).collect();
        o.sort_by(rank);
        let t: Vec<String> = o.iter().take(6).map(|&e| format!("{e}:{:.5}(logit {:.4})", sel[e], logits[e])).collect();
        eprintln!("scores layer {}: {}", ly.layer, t.join(", "));
    }
    let mut ids: Vec<usize> = (0..r.len()).collect();
    if k < ids.len() {
        ids.select_nth_unstable_by(k - 1, rank);
        ids.truncate(k);
    }
    ids.sort_unstable_by(rank);
    let sum = if ly.norm_w { ids.iter().fold(0.0f32, |t, &e| t + r[e]).max(6.103_515_6e-5) } else { 1.0 };
    let mut out: Vec<(usize, f32)> = ids.into_iter().map(|e| (e, r[e] / sum * ly.w_scale)).collect();
    out.sort_unstable_by_key(|x| x.0);
    out
}

/// The time spent in each phase of moe_tokens, summed over calls (TORC_PROF=1; torc-run prints them).
pub static MOE_NS: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// The routed experts moe_tokens touched (TORC_PROF=1), by block size n (1..=16), as [calls, experts]. The bytes a
/// verify pass must read grow with the union of its tokens' experts, not with n.
pub static MOE_UNION: [[std::sync::atomic::AtomicU64; 2]; 17] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 17];

/// The routed pairs of the blocks that used the NPU (TORC_PROF=1), as [all, on the NPU (hot experts)].
pub static MOE_HOT: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

pub const MOE_PHASES: [&str; 11] = [
    "router",
    "jobs",
    "npu gather",
    "npu begin",
    "gather+pack x",
    "gate/up",
    "hidden",
    "pack h",
    "down",
    "npu wait",
    "combine",
];

/// TORC_NPU_ASYNC=1: moe_tokens gathers and starts the hot experts on a helper thread.
fn npu_async() -> bool {
    static A: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switches::on("TORC_NPU_ASYNC"));
    *A
}

/// (TORC_NPU_HOT_MIN: the fewest tokens a block's hot expert needs to go to the NPU - a smaller group stays on the CPU,
/// since each one costs a graph node set and gets padded; TORC_NPU_PAD: the fewest rows a group is padded to, 64 or 32)
fn npu_hot_limits() -> (usize, usize) {
    static L: std::sync::LazyLock<(usize, usize)> = std::sync::LazyLock::new(|| {
        let v = |k: &str, d: usize| crate::switches::num(k).unwrap_or(d);
        (v("TORC_NPU_HOT_MIN", 1), if v("TORC_NPU_PAD", 64) == 32 { 32 } else { 64 })
    });
    *L
}

/// When dropped, joins the helper thread that starts the NPU's work, if there is one (before the buffers it writes go
/// away).
struct Helper(Option<std::thread::JoinHandle<bool>>);

impl Drop for Helper {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            let _ = h.join();
            crate::npu::finish();
        }
    }
}

/// The MoE block for n tokens (a prompt block). Each token is routed as if it were alone; then every expert any token
/// chose is read once for all of its tokens (gate and up, then down, with the same bits as one token at a time), and
/// the shared expert once for all n. Each token's outputs are added in increasing expert order, the shared one last.
/// The result has the same bits as moe_token on each token in turn. xn, xq and x hold n rows each.
pub fn moe_tokens(w: &Weights, ly: &MoeLayer, xn: &[f32], xq: Q8s, x: &mut [f32], n: usize) -> Result<()> {
    moe_tokens_with(w, ly, xn, xq, x, n, None)
}

/// moe_tokens's experts computed somewhere else (a device tier's verify passes). Once the block's routed jobs are known
/// (each chosen expert in increasing id, with its tokens in order), the implementation is told them, starts the ones it
/// holds and says which (true); it always computes the shared expert. Its outputs are collected after the CPU's own
/// jobs into the same rows of down_out, so the combine is unchanged, and since each is the same values the CPU would
/// compute, the result has the same bits.
pub trait GpuMoeBlock {
    fn start(&mut self, jobs: &[(usize, &[usize])]) -> Vec<bool>;
    /// routed job j's output for its i-th token (for a job it started)
    fn take(&mut self, j: usize, i: usize, out: &mut [f32]);
    /// the shared expert's outputs for the n tokens (n rows)
    fn shared(&mut self, out: &mut [f32]);
}

/// What a model gives the NPU tier's graph alongside its experts (gemma4): the experts' f32 input when the router reads
/// a different one (xn is then the router's), and a feed-forward of its own to run in the same graph while the CPU does
/// the routed experts. `done` says whether that feed-forward ran there (its output written); if false, the caller
/// computes it.
pub struct NpuSide<'a> {
    pub xe: &'a [f32],
    pub ffn: Option<crate::npu::Ffn>,
    pub done: std::cell::Cell<bool>,
}

/// moe_tokens with a side for the NPU tier ([`NpuSide`]); without the tier, or for a small block, plain moe_tokens.
pub fn moe_tokens_side(
    w: &Weights,
    ly: &MoeLayer,
    xn: &[f32],
    xq: Q8s,
    x: &mut [f32],
    n: usize,
    side: &NpuSide,
) -> Result<()> {
    moe_tokens_inner(w, ly, xn, xq, x, n, None, Some(side))
}

/// moe_tokens with some experts (and the shared one) computed by `gpu`.
pub fn moe_tokens_with(
    w: &Weights,
    ly: &MoeLayer,
    xn: &[f32],
    xq: Q8s,
    x: &mut [f32],
    n: usize,
    gpu: Option<&mut dyn GpuMoeBlock>,
) -> Result<()> {
    moe_tokens_inner(w, ly, xn, xq, x, n, gpu, None)
}

#[allow(clippy::too_many_arguments)]
fn moe_tokens_inner(
    w: &Weights,
    ly: &MoeLayer,
    xn: &[f32],
    xq: Q8s,
    x: &mut [f32],
    n: usize,
    gpu: Option<&mut dyn GpuMoeBlock>,
    side: Option<&NpuSide>,
) -> Result<()> {
    let g = w.layer_experts(ly.group);
    let ne = g.n_expert as usize;
    let dm = xn.len() / n;
    let k = g.n_used as usize;
    let mut prof = crate::Laps::start(&MOE_NS);
    // 1. the router logits for every token (each row of the router is read once and used for all n), then each token's
    //    route
    let mut logits = crate::scratch::buf::<f32>("moe.logits", n * ne);
    // a long block (not qwen4exp): tiles of tokens, each tile against every row, so that a tile's activations stay in
    // the core's cache while the rows stream past (going by rows, every row would re-read all n tokens' activations).
    // Each logit is the same dot_router_n / dot_router call, so the bits are the same
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if !ly.hc && n >= 32 {
        let threads = pool::get().threads.max(1);
        let tile = (n.div_ceil(2 * threads).div_ceil(4) * 4).clamp(4, 64);
        let lp = SyncPtr::new(&mut logits);
        pool::par_for_dynamic(n.div_ceil(tile), &|c| {
            let (a, b) = (c * tile, ((c + 1) * tile).min(n));
            for e in 0..ne {
                let row = &ly.router[e * dm..(e + 1) * dm];
                let mut t0 = a;
                while t0 + 4 <= b {
                    let xs = std::array::from_fn(|i| &xn[(t0 + i) * dm..(t0 + i + 1) * dm]);
                    for (i, v) in dot_router_n::<4>(row, xs).into_iter().enumerate() {
                        // SAFETY: tile c owns its tokens' logits
                        unsafe { *lp.at((t0 + i) * ne + e) = v };
                    }
                    t0 += 4;
                }
                for t in t0..b {
                    // SAFETY: tile c owns its tokens' logits
                    unsafe { *lp.at(t * ne + e) = dot_router(row, &xn[t * dm..(t + 1) * dm]) };
                }
            }
        });
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    let by_rows = ly.hc || n < 32;
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let by_rows = true;
    if by_rows {
        let lp = SyncPtr::new(&mut logits);
        pool::par_for(ne.div_ceil(8), &|i| {
            for e in i * 8..(i * 8 + 8).min(ne) {
                let row = &ly.router[e * dm..(e + 1) * dm];
                #[allow(unused_mut)]
                let mut t0 = 0;
                if ly.hc {
                    // qwen4exp: a bf16 router summed in f64 as ggml sums it, four tokens for each read of the row
                    #[cfg(target_arch = "aarch64")]
                    while t0 + 4 <= n {
                        let xs = std::array::from_fn(|i| &xn[(t0 + i) * dm..(t0 + i + 1) * dm]);
                        for (i, v) in crate::hc::dot_dbl_f16_n::<4>(row, xs).into_iter().enumerate() {
                            // SAFETY: item i owns column e of every token's logits
                            unsafe { *lp.at((t0 + i) * ne + e) = v };
                        }
                        t0 += 4;
                    }
                    for t in t0..n {
                        // SAFETY: item i owns column e of every token's logits
                        unsafe { *lp.at(t * ne + e) = crate::hc::dot_dbl_f16(row, &xn[t * dm..(t + 1) * dm]) };
                    }
                    continue;
                }
                // four tokens for each read of the row (on x86-64, each row is then widened once per four tokens
                // instead of once per token)
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                while t0 + 4 <= n {
                    let xs = std::array::from_fn(|i| &xn[(t0 + i) * dm..(t0 + i + 1) * dm]);
                    for (i, v) in dot_router_n::<4>(row, xs).into_iter().enumerate() {
                        // SAFETY: item i owns column e of every token's logits
                        unsafe { *lp.at((t0 + i) * ne + e) = v };
                    }
                    t0 += 4;
                }
                for t in t0..n {
                    // SAFETY: item i owns column e of every token's logits
                    unsafe { *lp.at(t * ne + e) = dot_router(row, &xn[t * dm..(t + 1) * dm]) };
                }
            }
        });
    }
    if !ly.router32.is_empty() {
        logits_f32(ly, xn, dm, ne, n, &mut logits);
    }
    let mut routes: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    {
        let (lp, rp) = (SyncPtr::new(&mut logits), SyncPtr::new(&mut routes));
        // SAFETY: item t owns token t's logits and route
        let one = |t: usize| unsafe { rp.slice(t, 1)[0] = route(lp.slice(t * ne, ne), k, ly) };
        if n <= RUNS_MAX {
            (0..n).for_each(one);
        } else {
            pool::par_for(n, &one);
        }
    }
    if let Some(es) = &ly.expert_scale {
        // gemma4's ffn_down_exps.scale: applied through the routing weight (after the weights are normalised), which
        // scales that expert's contribution exactly as scaling its output would
        for r in routes.iter_mut() {
            for p in r.iter_mut() {
                p.1 *= es[p.0];
            }
        }
    }
    if gpu.is_none() {
        tail_skip(w, ly.layer, &mut routes);
    }
    crate::npu::record(ly.layer, &routes);
    prof.lap(0);
    // with the expert cache (ecache.rs), the routed experts go a chunk at a time in increasing id (a chunk's slots stay
    // valid until the next fetch), and the shared expert goes with the last chunk. Each token's outputs are still added
    // in increasing expert id, then the shared one, so the bits are the same as all at once
    if w.cache.is_some() && gpu.is_none() {
        return moe_tokens_cached(w, ly, xn, xq, x, n, &routes, &mut prof);
    }
    moe_tokens_jobs(w, ly, xn, xq, x, n, gpu, &routes, 0, ne, true, &mut prof, side)
}

/// moe_tokens through the expert cache (ecache.rs), with its reads overlapped with its products. The experts go a chunk
/// at a time (a chunk's slots stay valid until the next request). Within a chunk, the ones already in RAM (and the
/// shared expert, with the first chunk) are computed while the others are read, and then those. Every pair's output has
/// its own row, and each token then adds its experts' rows in increasing expert id with the shared one last, as
/// moe_tokens_jobs does, so the bits are the same however the waves fall.
#[allow(clippy::too_many_arguments)]
fn moe_tokens_cached(
    w: &Weights,
    ly: &MoeLayer,
    xn: &[f32],
    xq: Q8s,
    x: &mut [f32],
    n: usize,
    routes: &[Vec<(usize, f32)>],
    prof: &mut crate::Laps,
) -> Result<()> {
    let c = w.cache.as_ref().unwrap();
    let g = w.layer_experts(ly.group);
    let ne = g.n_expert as usize;
    let dm = xn.len() / n;
    let nb = dm / QK;
    // each used expert's tokens in order, and each pair's global row: the experts in increasing id, with the shared
    // expert after them
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); ne];
    for (t, r) in routes.iter().enumerate() {
        for &(e, _) in r {
            users[e].push(t);
        }
    }
    let used: Vec<usize> = (0..ne).filter(|&e| !users[e].is_empty()).collect();
    let mut gp0 = vec![0usize; ne];
    let mut np = 0;
    let mut pair_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &e in &used {
        gp0[e] = np;
        for (i, &t) in users[e].iter().enumerate() {
            pair_of[t].push(np + i);
        }
        np += users[e].len();
    }
    let shared_t = match &ly.shared {
        Some(names) => Some([w.need(&names[0])?, w.need(&names[1])?, w.need(&names[2])?]),
        None => None,
    };
    let sp0 = np;
    if shared_t.is_some() {
        np += n;
    }
    let mut dall = crate::scratch::buf::<f32>("moe.dall", np * dm);
    prof.lap(1);
    let run = |jobs: &[EJob], dall: &mut [f32]| run_ejobs(ly.hc, ly.gelu, jobs, xq, nb, dm, dall);
    // two machines (ep.rs): the odd-numbered experts go to the worker, sent first, while the even ones run here
    let mut local = used.clone();
    let mut remote: Vec<usize> = Vec::new();
    let mut link = None;
    if n >= crate::ep::min_tokens()
        && let Some(cl) = crate::ep::client()
    {
        remote = used.iter().copied().filter(|e| e % 2 == 1).collect();
        local = used.iter().copied().filter(|e| e % 2 == 0).collect();
        let mut s = cl.lock().unwrap();
        let jl: Vec<(usize, &[usize])> = remote.iter().map(|&e| (e, users[e].as_slice())).collect();
        crate::ep::send(&mut s, ly.layer, ly.group, ly.hc, ly.gelu, n, nb, dm, xq, &jl)
            .map_err(|e| format::Error(format!("two boxes: {e}")))?;
        link = Some(s);
    }
    let per = c.layer_slots(ly.layer).clamp(1, 128);
    let chunks: Vec<&[usize]> = local.chunks(per).collect();
    for (ci, chunk) in chunks.iter().enumerate() {
        let ids: Vec<u32> = chunk.iter().map(|&e| e as u32).collect();
        let (slots, inram) = w.experts_ask(ly.layer, ly.group, &ids)?;
        let mut a: Vec<EJob> = Vec::new();
        let mut b: Vec<EJob> = Vec::new();
        if ci == 0
            && let Some(t) = &shared_t
        {
            a.push(EJob { mats: [Mat::of(&t[0]), Mat::of(&t[1]), Mat::of(&t[2])], toks: (0..n).collect(), g0: sp0 });
        }
        for (i, &e) in chunk.iter().enumerate() {
            let (slot, lay) = slots[i];
            let j = EJob { mats: mats_of(slot, lay), toks: users[e].clone(), g0: gp0[e] };
            if inram[i] { a.push(j) } else { b.push(j) }
        }
        run(&a, &mut dall);
        w.experts_settle()?;
        run(&b, &mut dall);
    }
    if chunks.is_empty()
        && let Some(t) = &shared_t
    {
        run(
            &[EJob { mats: [Mat::of(&t[0]), Mat::of(&t[1]), Mat::of(&t[2])], toks: (0..n).collect(), g0: sp0 }],
            &mut dall,
        );
    }
    if let Some(mut s) = link {
        // the worker's rows go into their pairs' places
        let pairs: usize = remote.iter().map(|&e| users[e].len()).sum();
        let mut rows = vec![0.0f32; pairs * dm];
        crate::ep::recv(&mut s, pairs, dm, &mut rows)?;
        let mut off = 0;
        for &e in &remote {
            let k = users[e].len();
            dall[gp0[e] * dm..(gp0[e] + k) * dm].copy_from_slice(&rows[off * dm..(off + k) * dm]);
            off += k;
        }
    }
    prof.lap(8);
    // each token: its experts in increasing id, then the shared one with its sigmoid gate
    let xp = SyncPtr::new(x);
    pool::par_for(n, &|t| {
        // SAFETY: item t owns token t's row of x
        let xt = unsafe { xp.slice(t * dm, dm) };
        for (&p, &(_, wt)) in pair_of[t].iter().zip(routes[t].iter()) {
            kops::axpy_f32(xt, wt, &dall[p * dm..(p + 1) * dm]);
        }
        if shared_t.is_some() {
            let wt = match &ly.shexp_gate {
                Some(gv) if ly.hc => crate::ops::sigmoid(crate::hc::dot_dbl(gv, &xn[t * dm..(t + 1) * dm])),
                Some(gv) => crate::ops::sigmoid(kops::dot_f32(gv, &xn[t * dm..(t + 1) * dm])),
                None => 1.0,
            };
            kops::axpy_f32(xt, wt, &dall[(sp0 + t) * dm..(sp0 + t + 1) * dm]);
        }
    });
    prof.lap(10);
    Ok(())
}

/// A routed (or shared) expert's work in a block: its matrices, its tokens in order, and its first pair row in the
/// output (moe_tokens_cached's dall, or a two-machine worker's reply).
pub(crate) struct EJob<'a> {
    pub mats: [Mat<'a>; 3],
    pub toks: Vec<usize>,
    pub g0: usize,
}

/// One matrix of an expert's pack.
pub(crate) fn mat_of<'a>(pack: &'a [u8], p: &format::Piece) -> Mat<'a> {
    Mat { ty: p.ty, rows: p.rows as usize, cols: p.cols as usize, data: p.of(pack) }
}

/// An expert's gate, up and down matrices, from its pack.
pub(crate) fn mats_of<'a>(pack: &'a [u8], shape: &format::Pack) -> [Mat<'a>; 3] {
    [mat_of(pack, &shape.gate), mat_of(pack, &shape.up), mat_of(pack, &shape.down)]
}

/// A wave of jobs (for moe_tokens_cached and the two-machine worker): gate and up for each job's tokens, the hidden
/// activations (quantised as ggml does when `hc` is set), then down, with each pair's output going into its row (g0 +
/// i) of dall (dm wide).
pub(crate) fn run_ejobs(hc: bool, gelu: bool, jobs: &[EJob], xq: Q8s, nb: usize, dm: usize, dall: &mut [f32]) {
    let mut np = 0;
    let p0: Vec<usize> = jobs
        .iter()
        .map(|j| {
            let p = np;
            np += j.toks.len();
            p
        })
        .collect();
    if np == 0 {
        return;
    }
    let ffmax = jobs.iter().map(|j| j.mats[0].rows.div_ceil(QK) * QK).max().unwrap_or(QK);
    let mut rowsq = crate::scratch::q8("moe.rowsq", np * nb);
    for (ji, j) in jobs.iter().enumerate() {
        for (i, &t) in j.toks.iter().enumerate() {
            rowsq.sub_mut((p0[ji] + i) * nb..(p0[ji] + i + 1) * nb).copy_from(xq.sub(t * nb..(t + 1) * nb));
        }
    }
    const RC: usize = 16;
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (ji, j) in jobs.iter().enumerate() {
        for which in 0..2 {
            for r0 in (0..j.mats[which].rows).step_by(RC) {
                tasks.push((ji, which, r0));
            }
        }
    }
    let mut pkoff = vec![0usize; jobs.len() + 1];
    for (ji, j) in jobs.iter().enumerate() {
        let nv = j.toks.len();
        let k = if use_mmla(j.mats[0].ty, nv) || use_mmla(j.mats[1].ty, nv) { pack_len(nv, nb) } else { 0 };
        pkoff[ji + 1] = pkoff[ji] + k;
    }
    let mut pkx = crate::scratch::buf::<Pk>("moe.pkx", pkoff[jobs.len()]);
    if pkoff[jobs.len()] > 0 {
        let pp = SyncPtr::new(&mut pkx);
        pool::par_for(jobs.len(), &|ji| {
            let (j, len) = (&jobs[ji], pkoff[ji + 1] - pkoff[ji]);
            if len > 0 {
                // SAFETY: job ji owns pkx[pkoff[ji]..pkoff[ji + 1]]
                pack(rowsq.sub(p0[ji] * nb..(p0[ji] + j.toks.len()) * nb), j.toks.len(), unsafe {
                    pp.slice(pkoff[ji], len)
                });
            }
        });
    }
    let (kkoff, pkkx) = pack_jobs_k(
        jobs.iter().map(|j| (true, [j.mats[0].ty, j.mats[1].ty], j.toks.len())),
        |ji| rowsq.sub(p0[ji] * nb..(p0[ji] + jobs[ji].toks.len()) * nb),
        nb,
        "moe.pkkx",
    );
    let mut gu = crate::scratch::buf::<f32>("moe.gu", np * 2 * ffmax);
    {
        let gp = SyncPtr::new(&mut gu);
        pool::par_for_guided(tasks.len(), &|ti| {
            let (ji, which, r0) = tasks[ti];
            let j = &jobs[ji];
            let m = &j.mats[which];
            let rn = RC.min(m.rows - r0);
            let rb = m.data.len() / m.rows;
            let nv = j.toks.len();
            // SAFETY: this task writes rows r0..r0 + rn of matrix `which` of this job's pairs, and nothing else
            unsafe {
                let pk: &[Pk] = if use_mmla(m.ty, nv) { &pkx[pkoff[ji]..pkoff[ji + 1]] } else { &[] };
                let pkk: &[Pkk] = if use_kmm(m.ty, nv) { &pkkx[kkoff[ji]..kkoff[ji + 1]] } else { &[] };
                dot_multi_rows(
                    m.ty,
                    &m.data[r0 * rb..(r0 + rn) * rb],
                    rb,
                    rn,
                    rowsq.sub(p0[ji] * nb..(p0[ji] + nv) * nb),
                    pk,
                    &[],
                    pkk,
                    nv,
                    gp.at(p0[ji] * 2 * ffmax + which * ffmax + r0),
                    2 * ffmax,
                );
            }
        });
    }
    let pj: Vec<usize> = jobs.iter().enumerate().flat_map(|(ji, j)| std::iter::repeat_n(ji, j.toks.len())).collect();
    let fqb_of = |ji: usize| jobs[ji].mats[0].rows.div_ceil(QK);
    let mut hoff = vec![0usize; np + 1];
    for p in 0..np {
        hoff[p + 1] = hoff[p] + fqb_of(pj[p]);
    }
    let mut h_q8 = crate::scratch::q8("moe.h_q8", hoff[np]);
    {
        let hp = crate::pool::Q8Ptr::new(h_q8.view_mut());
        pool::par_for(np, &|p| {
            let ff = jobs[pj[p]].mats[0].rows;
            let fq = ff.div_ceil(QK) * QK;
            let (gt, up) =
                (&gu[p * 2 * ffmax..p * 2 * ffmax + ff], &gu[p * 2 * ffmax + ffmax..p * 2 * ffmax + ffmax + ff]);
            let mut h = vec![0.0f32; fq];
            for r in 0..ff {
                h[r] = crate::act::glu(gelu, gt[r], up[r]);
            }
            // SAFETY: pair p owns blocks hoff[p]..hoff[p + 1]
            let q = unsafe { hp.slice(hoff[p], fq / QK) };
            if hc { kops::q8k_quantize_gg(&h, q) } else { kops::q8k_quantize(&h, q) }
        });
    }
    let mut dtasks: Vec<(usize, usize)> = Vec::new();
    for (ji, j) in jobs.iter().enumerate() {
        for r0 in (0..j.mats[2].rows).step_by(RC) {
            dtasks.push((ji, r0));
        }
    }
    let mut hkoff = vec![0usize; jobs.len() + 1];
    for (ji, j) in jobs.iter().enumerate() {
        let k = if use_mmla(j.mats[2].ty, j.toks.len()) { pack_len(j.toks.len(), fqb_of(ji)) } else { 0 };
        hkoff[ji + 1] = hkoff[ji] + k;
    }
    let mut pkh = crate::scratch::buf::<Pk>("moe.pkh", hkoff[jobs.len()]);
    if hkoff[jobs.len()] > 0 {
        let pp = SyncPtr::new(&mut pkh);
        pool::par_for(jobs.len(), &|ji| {
            let (j, len) = (&jobs[ji], hkoff[ji + 1] - hkoff[ji]);
            if len > 0 {
                let hv = h_q8.sub(hoff[p0[ji]]..hoff[p0[ji] + j.toks.len()]);
                // SAFETY: job ji owns pkh[hkoff[ji]..hkoff[ji + 1]]
                pack(hv, j.toks.len(), unsafe { pp.slice(hkoff[ji], len) });
            }
        });
    }
    let (hkkoff, pkkh) = pack_jobs_k(
        jobs.iter().map(|j| (true, [j.mats[2].ty, 0], j.toks.len())),
        |ji| h_q8.sub(hoff[p0[ji]]..hoff[p0[ji] + jobs[ji].toks.len()]),
        0,
        "moe.pkkh",
    );
    let (h6off, pk6h) = pack_jobs_6(
        jobs.iter().map(|j| (true, [j.mats[2].ty, 0], j.toks.len())),
        |ji| h_q8.sub(hoff[p0[ji]]..hoff[p0[ji] + jobs[ji].toks.len()]),
        0,
        "moe.pk6h",
    );
    let dp = SyncPtr::new(dall);
    pool::par_for_guided(dtasks.len(), &|ti| {
        let (ji, r0) = dtasks[ti];
        let j = &jobs[ji];
        let m = &j.mats[2];
        assert_eq!(m.rows, dm);
        let rn = RC.min(m.rows - r0);
        let rb = m.data.len() / m.rows;
        let nv = j.toks.len();
        assert_eq!(m.cols / QK, fqb_of(ji));
        let hv = h_q8.sub(hoff[p0[ji]]..hoff[p0[ji] + nv]);
        // SAFETY: this task writes rows r0..r0 + rn of this job's pairs' global outputs, and nothing else
        unsafe {
            let pk: &[Pk] = if use_mmla(m.ty, nv) { &pkh[hkoff[ji]..hkoff[ji + 1]] } else { &[] };
            let pkk: &[Pkk] = if use_kmm(m.ty, nv) { &pkkh[hkkoff[ji]..hkkoff[ji + 1]] } else { &[] };
            let pk6: &[Pk6] = if use_q6e(m.ty, nv) { &pk6h[h6off[ji]..h6off[ji + 1]] } else { &[] };
            dot_multi_rows(
                m.ty,
                &m.data[r0 * rb..(r0 + rn) * rb],
                rb,
                rn,
                hv,
                pk,
                pk6,
                pkk,
                nv,
                dp.at(j.g0 * dm + r0),
                dm,
            );
        }
    });
}

/// Each job's vectors packed for its Q4_K / Q5_K SMMLA tasks (use_kmm), side by side, with job ji's at
/// out[off[ji]..off[ji + 1]] (empty where it has none). jobs gives, for each job, whether it runs here, its matrices'
/// types and its vector count; x_of(ji) gives its vectors (all nb blocks of each; 0 means x_of's whole length).
fn pack_jobs_k<'a, I>(
    jobs: I,
    x_of: impl Fn(usize) -> Q8s<'a> + Sync,
    nb: usize,
    name: &'static str,
) -> (Vec<usize>, crate::scratch::Buf<Pkk>)
where
    I: Iterator<Item = (bool, [u32; 2], usize)>,
{
    pack_jobs(jobs, x_of, nb, name, use_kmm, packk_len, packk)
}

/// pack_jobs_k for the Q6_K SMMLA tasks (use_q6).
fn pack_jobs_6<'a, I>(
    jobs: I,
    x_of: impl Fn(usize) -> Q8s<'a> + Sync,
    nb: usize,
    name: &'static str,
) -> (Vec<usize>, crate::scratch::Buf<Pk6>)
where
    I: Iterator<Item = (bool, [u32; 2], usize)>,
{
    pack_jobs(jobs, x_of, nb, name, use_q6e, pack6_len, pack6)
}

#[allow(clippy::too_many_arguments)]
fn pack_jobs<'a, T: Copy + Default + Send + 'static, I>(
    jobs: I,
    x_of: impl Fn(usize) -> Q8s<'a> + Sync,
    nb: usize,
    name: &'static str,
    used: fn(u32, usize) -> bool,
    len_of: fn(usize, usize) -> usize,
    pack_into: fn(Q8s, usize, &mut [T]),
) -> (Vec<usize>, crate::scratch::Buf<T>)
where
    I: Iterator<Item = (bool, [u32; 2], usize)>,
{
    let js: Vec<(bool, [u32; 2], usize)> = jobs.collect();
    let mut off = vec![0usize; js.len() + 1];
    for (ji, &(here, tys, nv)) in js.iter().enumerate() {
        let k = if here && nv > 0 && tys.iter().any(|&t| used(t, nv)) {
            len_of(nv, if nb > 0 { nb } else { x_of(ji).len() / nv })
        } else {
            0
        };
        off[ji + 1] = off[ji] + k;
    }
    let mut out = crate::scratch::buf::<T>(name, off[js.len()]);
    if off[js.len()] > 0 {
        let pp = SyncPtr::new(&mut out);
        pool::par_for(js.len(), &|ji| {
            let len = off[ji + 1] - off[ji];
            if len > 0 {
                // SAFETY: job ji owns out[off[ji]..off[ji + 1]]
                pack_into(x_of(ji), js[ji].2, unsafe { pp.slice(off[ji], len) });
            }
        });
    }
    (off, out)
}

/// Guided splits for big prompt blocks (pool::par_for_guided) instead of even ones. They were made the NPU tier's
/// default on Qwen3.6-35B-A3B to absorb the core its syncing thread takes, but on ERNIE-4.5-21B they cost a third of
/// the reading speed, so it is a tuned setting: set_guided (torc-tune), else TORC_NPU_GUIDED=0/1, else the value
/// torc-tune saved for this machine and model ("guided"), else even splits.
static GUIDED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

pub fn guided() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match GUIDED.load(Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = match crate::switches::get("TORC_NPU_GUIDED") {
                Some(v) => v == "1",
                None => crate::tune::get("guided") == Some(1),
            };
            GUIDED.store(on as u8, Relaxed);
            on
        }
    }
}

/// Fixes the split guided() reports (torc-tune measuring both).
pub fn set_guided(on: bool) {
    GUIDED.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}

/// moe_tokens from the routes onwards, for the routed experts lo..hi (and the shared expert when `with_shared` is set):
/// the jobs, their products, and each token's sum of their outputs into x, in increasing expert id.
#[allow(clippy::too_many_arguments)]
fn moe_tokens_jobs(
    w: &Weights,
    ly: &MoeLayer,
    xn: &[f32],
    xq: Q8s,
    x: &mut [f32],
    n: usize,
    mut gpu: Option<&mut dyn GpuMoeBlock>,
    routes: &[Vec<(usize, f32)>],
    lo: usize,
    hi: usize,
    with_shared: bool,
    prof: &mut crate::Laps,
    side: Option<&NpuSide>,
) -> Result<()> {
    let g = w.layer_experts(ly.group);
    let ne = g.n_expert as usize;
    let dm = xn.len() / n;
    let nb = dm / QK;
    let split: fn(usize, &(dyn Fn(usize) + Sync)) =
        if n <= RUNS_MAX || guided() { pool::par_for_guided } else { pool::par_for };
    // 2. the jobs: each chosen expert with its tokens (in token order), then the shared expert with all n
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); ne];
    for (t, r) in routes.iter().enumerate() {
        for &(e, _) in r {
            if (lo..hi).contains(&e) {
                users[e].push(t);
            }
        }
    }
    let shared_t = match (&ly.shared, with_shared) {
        (Some(names), true) => Some([w.need(&names[0])?, w.need(&names[1])?, w.need(&names[2])?]),
        _ => None,
    };
    let used_ids: Vec<u32> = users.iter().enumerate().filter(|(_, u)| !u.is_empty()).map(|(e, _)| e as u32).collect();
    let mut slots = w.experts_of(ly.layer, ly.group, &used_ids)?.into_iter();
    struct Job<'a> {
        mats: [Mat<'a>; 3],
        toks: Vec<usize>,
        /// the first pair index (a pair is one token of this job)
        p0: usize,
        /// the routed expert's id (usize::MAX for the shared expert)
        e: usize,
    }
    let mut jobs: Vec<Job> = Vec::new();
    let mut np = 0;
    for (e, toks) in users.into_iter().enumerate() {
        if toks.is_empty() {
            continue;
        }
        let (slot, lay) = slots.next().expect("a pack per used expert");
        let len = toks.len();
        jobs.push(Job { mats: mats_of(slot, lay), toks, p0: np, e });
        np += len;
    }
    if prof.on() && n <= 16 {
        MOE_UNION[n][0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        MOE_UNION[n][1].fetch_add(jobs.len() as u64, std::sync::atomic::Ordering::Relaxed);
    }
    let shared_job = shared_t.as_ref().map(|_| jobs.len());
    if let Some(t) = &shared_t {
        jobs.push(Job {
            mats: [Mat::of(&t[0]), Mat::of(&t[1]), Mat::of(&t[2])],
            toks: (0..n).collect(),
            p0: np,
            e: usize::MAX,
        });
        np += n;
    }
    let ffmax = jobs.iter().map(|j| j.mats[0].rows.div_ceil(QK) * QK).max().unwrap_or(QK);
    // the jobs the device takes (the routed jobs come first in increasing expert id, then the shared one)
    let mut gpu_job = vec![false; jobs.len()];
    if let Some(g) = gpu.as_mut() {
        let Some(sj) = shared_job else { return err("a shared expert from the GPU for a layer without one") };
        let routed: Vec<(usize, &[usize])> = jobs[..sj].iter().map(|j| (j.e, j.toks.as_slice())).collect();
        for (ji, h) in g.start(&routed).into_iter().enumerate() {
            gpu_job[ji] = h;
        }
        gpu_job[sj] = true;
    }
    // NPU tier (npu.rs): in a big block, the shared expert and the hot experts (at most 64, the graph's limit) run on
    // the NPU as one graph while the CPU does the other routed ones. Their outputs land in their own down_out rows at
    // the join below, so the CPU jobs' products and the combine order are unchanged
    let mut down_out = crate::scratch::buf::<f32>("moe.down_out", np * dm);
    let mut npu_job = vec![false; jobs.len()];
    // (job, NPU ids, first staging row, rows): the shared expert reads xn and writes sh_npu; a hot expert reads its
    // tokens' rows gathered into hot_x, padded to a multiple of 32 and at least 64, with the padding zeroed
    let mut ffns: Vec<(usize, [i32; 3], usize, usize)> = Vec::new();
    // (a GELU model's experts only go there once the NPU has GEGLU: npu::set_geglu)
    let gelu_ok = !ly.gelu || crate::npu::geglu();
    let hot = if gpu.is_some() || !gelu_ok { Vec::new() } else { crate::npu::hot(ly.layer, n) };
    let mut hot_rows = 0;
    let (hot_min, pad_min) = npu_hot_limits();
    prof.lap(1);
    if !hot.is_empty() {
        let mut h = hot.iter().peekable();
        for (ji, j) in jobs.iter().enumerate().filter(|(ji, _)| Some(*ji) != shared_job) {
            while h.next_if(|x| x.0 < j.e).is_some() {}
            if let Some(&&(_, ids)) = h.peek().filter(|x| x.0 == j.e && ffns.len() < 64 && j.toks.len() >= hot_min) {
                let pad = j.toks.len().div_ceil(32).max(pad_min / 32) * 32;
                ffns.push((ji, ids, hot_rows, pad));
                hot_rows += pad;
            }
        }
    }
    let mut hot_x = crate::scratch::buf::<f32>("moe.hot_x", hot_rows * dm);
    let mut hot_y = crate::scratch::buf::<f32>("moe.hot_y", hot_rows * dm);
    // TORC_NPU_SHARED=0: the shared expert stays on the CPU (an A/B of the NPU's effect on the CPU's routed experts)
    static NPU_SHARED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| crate::switches::get("TORC_NPU_SHARED").is_none_or(|v| v != "0"));
    let sh_ids = shared_job
        .filter(|_| gpu.is_none() && gelu_ok && *NPU_SHARED)
        .and_then(|sj| crate::npu::ffn_ids(&jobs[sj].mats, n));
    // the experts' f32 rows: xn, unless the side gives its own (the router read xn)
    let xe = side.map_or(xn, |s| s.xe);
    let side_ffn = side.and_then(|s| s.ffn).filter(|_| gpu.is_none());
    let mut sh_npu = crate::scratch::buf::<f32>("moe.sh_npu", if sh_ids.is_some() { n * dm } else { 0 });
    let mut groups: Vec<crate::npu::Ffn> = Vec::with_capacity(ffns.len() + 2);
    if let Some(f) = side_ffn {
        groups.push(f);
    }
    if let Some(ids) = sh_ids {
        groups.push(crate::npu::Ffn { ids, x: xe.as_ptr(), n, y: sh_npu.as_mut_ptr() });
    }
    for &(_, ids, r0, pad) in &ffns {
        groups.push(crate::npu::Ffn { ids, x: hot_x[r0 * dm..].as_ptr(), n: pad, y: hot_y[r0 * dm..].as_mut_ptr() });
    }
    // a hot group's rows: its tokens' xn rows, then the zeroed padding
    let rows: Vec<(Vec<usize>, usize, usize)> =
        ffns.iter().map(|&(ji, _, r0, pad)| (jobs[ji].toks.clone(), r0, pad)).collect();
    let gather = move |hx: SyncPtr<f32>, xs: SyncPtr<f32>, i: usize, rows: &[(Vec<usize>, usize, usize)]| {
        let (toks, r0, pad) = &rows[i];
        // SAFETY: group i owns staging rows r0..r0 + pad, and xs is xn (n x dm), read only
        let (hx, xs) = unsafe { (hx.slice(r0 * dm, pad * dm), std::slice::from_raw_parts(xs.0, n * dm)) };
        for (r, &t) in toks.iter().enumerate() {
            hx[r * dm..(r + 1) * dm].copy_from_slice(&xs[t * dm..(t + 1) * dm]);
        }
        hx[toks.len() * dm..].fill(0.0);
    };
    let (hx, xs) = (SyncPtr::new(&mut hot_x), SyncPtr(xe.as_ptr() as *mut f32));
    // the gather and begin run here or, with TORC_NPU_ASYNC=1, on a helper thread while the CPU starts its experts,
    // joined before the wait below (an asynchronous begin that fails aborts the block)
    let mut helper = Helper(None);
    let on_npu = if groups.is_empty() {
        false
    } else if npu_async() && !ffns.is_empty() {
        struct Send2(Vec<crate::npu::Ffn>);
        // SAFETY: the pointers stay valid until the helper is joined (Helper joins when dropped, before the buffers go
        // away)
        unsafe impl Send for Send2 {}
        let g = Send2(groups);
        helper.0 = Some(std::thread::spawn(move || {
            let g = g;
            for i in 0..rows.len() {
                gather(hx, xs, i, &rows);
            }
            // SAFETY: as below
            unsafe { crate::npu::begin_ffns(&g.0, dm) }
        }));
        true
    } else {
        pool::par_for(ffns.len(), &|i| gather(hx, xs, i, &rows));
        prof.lap(2);
        // SAFETY: xn and hot_x hold every group's rows (begin copies them in), and sh_npu and hot_y hold their outputs;
        // nothing touches them until finish, below
        unsafe { crate::npu::begin_ffns(&groups, dm) }
    };
    prof.lap(3);
    if on_npu {
        if let (Some(sj), Some(_)) = (shared_job, sh_ids) {
            npu_job[sj] = true;
        }
        for &(ji, ..) in &ffns {
            npu_job[ji] = true;
        }
        if prof.on() {
            let hp: usize = ffns.iter().map(|&(ji, ..)| jobs[ji].toks.len()).sum();
            MOE_HOT[0].fetch_add((np - shared_job.map_or(0, |_| n)) as u64, std::sync::atomic::Ordering::Relaxed);
            MOE_HOT[1].fetch_add(hp as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let cpu_job = |ji: usize| !npu_job[ji] && !gpu_job[ji];
    // each CPU job's input vectors, gathered side by side
    let mut rowsq = crate::scratch::q8("moe.rowsq", np * nb);
    for (_, j) in jobs.iter().enumerate().filter(|(ji, _)| cpu_job(*ji)) {
        for (i, &t) in j.toks.iter().enumerate() {
            rowsq.sub_mut((j.p0 + i) * nb..(j.p0 + i + 1) * nb).copy_from(xq.sub(t * nb..(t + 1) * nb));
        }
    }
    // 3. gate and up: tasks of RC rows of one matrix of one job, against that job's vectors. Pair p's gate row r goes
    //    to gu[p * 2 * ffmax + r], and its up row to gu[p * 2 * ffmax + ffmax + r]
    const RC: usize = 16;
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    let order: Vec<usize> =
        shared_job.into_iter().chain((0..jobs.len()).filter(|&ji| Some(ji) != shared_job)).collect();
    for (ji, j) in order.iter().map(|&ji| (ji, &jobs[ji])).filter(|(ji, _)| cpu_job(*ji)) {
        for which in 0..2 {
            for r0 in (0..j.mats[which].rows).step_by(RC) {
                tasks.push((ji, which, r0));
            }
        }
    }
    // each job's vectors are packed once for its SMMLA tasks (gate and up)
    let mut pkoff = vec![0usize; jobs.len() + 1];
    for (ji, j) in jobs.iter().enumerate() {
        let nv = j.toks.len();
        let n = if cpu_job(ji) && (use_mmla(j.mats[0].ty, nv) || use_mmla(j.mats[1].ty, nv)) {
            pack_len(nv, nb)
        } else {
            0
        };
        pkoff[ji + 1] = pkoff[ji] + n;
    }
    let mut pkx = crate::scratch::buf::<Pk>("moe.pkx", pkoff[jobs.len()]);
    if pkoff[jobs.len()] > 0 {
        let pp = SyncPtr::new(&mut pkx);
        pool::par_for(jobs.len(), &|ji| {
            let (j, len) = (&jobs[ji], pkoff[ji + 1] - pkoff[ji]);
            if len > 0 {
                // SAFETY: job ji owns pkx[pkoff[ji]..pkoff[ji + 1]]
                pack(rowsq.sub(j.p0 * nb..(j.p0 + j.toks.len()) * nb), j.toks.len(), unsafe {
                    pp.slice(pkoff[ji], len)
                });
            }
        });
    }
    let (kkoff, pkkx) = pack_jobs_k(
        jobs.iter().enumerate().map(|(ji, j)| (cpu_job(ji), [j.mats[0].ty, j.mats[1].ty], j.toks.len())),
        |ji| rowsq.sub(jobs[ji].p0 * nb..(jobs[ji].p0 + jobs[ji].toks.len()) * nb),
        nb,
        "moe.pkkx",
    );
    prof.lap(4);
    let mut gu = crate::scratch::buf::<f32>("moe.gu", np * 2 * ffmax);
    {
        let gp = SyncPtr::new(&mut gu);
        split(tasks.len(), &|ti| {
            let (ji, which, r0) = tasks[ti];
            let j = &jobs[ji];
            let m = &j.mats[which];
            let rn = RC.min(m.rows - r0);
            let rb = m.data.len() / m.rows;
            let nv = j.toks.len();
            // SAFETY: this task writes rows r0..r0 + rn of matrix `which` of this job's pairs, and nothing else
            unsafe {
                let pk: &[Pk] = if use_mmla(m.ty, nv) { &pkx[pkoff[ji]..pkoff[ji + 1]] } else { &[] };
                let pkk: &[Pkk] = if use_kmm(m.ty, nv) { &pkkx[kkoff[ji]..kkoff[ji + 1]] } else { &[] };
                dot_multi_rows(
                    m.ty,
                    &m.data[r0 * rb..(r0 + rn) * rb],
                    rb,
                    rn,
                    rowsq.sub(j.p0 * nb..(j.p0 + nv) * nb),
                    pk,
                    &[],
                    pkk,
                    nv,
                    gp.at(j.p0 * 2 * ffmax + which * ffmax + r0),
                    2 * ffmax,
                );
            }
        });
    }
    prof.lap(5);
    // 4. each pair's hidden state: SiLU(gate) * up, quantised without a division. A job's pairs sit side by side, each
    //    taking its expert's fq / QK blocks, so down reads them as they lie
    let pj: Vec<usize> = jobs.iter().enumerate().flat_map(|(ji, j)| std::iter::repeat_n(ji, j.toks.len())).collect();
    let fqb_of = |ji: usize| jobs[ji].mats[0].rows.div_ceil(QK);
    let mut hoff = vec![0usize; np + 1];
    for p in 0..np {
        hoff[p + 1] = hoff[p] + fqb_of(pj[p]);
    }
    let mut h_q8 = crate::scratch::q8("moe.h_q8", hoff[np]);
    {
        let hp = crate::pool::Q8Ptr::new(h_q8.view_mut());
        pool::par_for(np, &|p| {
            if !cpu_job(pj[p]) {
                return;
            }
            let ff = jobs[pj[p]].mats[0].rows;
            let fq = ff.div_ceil(QK) * QK;
            let (gt, up) =
                (&gu[p * 2 * ffmax..p * 2 * ffmax + ff], &gu[p * 2 * ffmax + ffmax..p * 2 * ffmax + ffmax + ff]);
            let mut h = vec![0.0f32; fq];
            for r in 0..ff {
                h[r] = crate::act::glu(ly.gelu, gt[r], up[r]);
            }
            // SAFETY: pair p owns blocks hoff[p]..hoff[p + 1]
            let q = unsafe { hp.slice(hoff[p], fq / QK) };
            if ly.hc { kops::q8k_quantize_gg(&h, q) } else { kops::q8k_quantize(&h, q) }
        });
    }
    // 5. down: tasks of RC rows of one job's down matrix against its pairs' hidden states, with pair p's output row r
    //    going to down_out[p * dm + r]
    let mut dtasks: Vec<(usize, usize)> = Vec::new();
    for (ji, j) in order.iter().map(|&ji| (ji, &jobs[ji])).filter(|(ji, _)| cpu_job(*ji)) {
        for r0 in (0..j.mats[2].rows).step_by(RC) {
            dtasks.push((ji, r0));
        }
    }
    prof.lap(6);
    // each job's hidden states are packed once for its SMMLA down tasks
    let mut hkoff = vec![0usize; jobs.len() + 1];
    for (ji, j) in jobs.iter().enumerate() {
        let n =
            if cpu_job(ji) && use_mmla(j.mats[2].ty, j.toks.len()) { pack_len(j.toks.len(), fqb_of(ji)) } else { 0 };
        hkoff[ji + 1] = hkoff[ji] + n;
    }
    let mut pkh = crate::scratch::buf::<Pk>("moe.pkh", hkoff[jobs.len()]);
    if hkoff[jobs.len()] > 0 {
        let pp = SyncPtr::new(&mut pkh);
        pool::par_for(jobs.len(), &|ji| {
            let (j, len) = (&jobs[ji], hkoff[ji + 1] - hkoff[ji]);
            if len > 0 {
                let hv = h_q8.sub(hoff[j.p0]..hoff[j.p0 + j.toks.len()]);
                // SAFETY: job ji owns pkh[hkoff[ji]..hkoff[ji + 1]]
                pack(hv, j.toks.len(), unsafe { pp.slice(hkoff[ji], len) });
            }
        });
    }
    let (hkkoff, pkkh) = pack_jobs_k(
        jobs.iter().enumerate().map(|(ji, j)| (cpu_job(ji), [j.mats[2].ty, 0], j.toks.len())),
        |ji| h_q8.sub(hoff[jobs[ji].p0]..hoff[jobs[ji].p0 + jobs[ji].toks.len()]),
        0,
        "moe.pkkh",
    );
    let (h6off, pk6h) = pack_jobs_6(
        jobs.iter().enumerate().map(|(ji, j)| (cpu_job(ji), [j.mats[2].ty, 0], j.toks.len())),
        |ji| h_q8.sub(hoff[jobs[ji].p0]..hoff[jobs[ji].p0 + jobs[ji].toks.len()]),
        0,
        "moe.pk6h",
    );
    prof.lap(7);
    {
        let dp = SyncPtr::new(&mut down_out);
        split(dtasks.len(), &|ti| {
            let (ji, r0) = dtasks[ti];
            let j = &jobs[ji];
            let m = &j.mats[2];
            assert_eq!(m.rows, dm);
            let rn = RC.min(m.rows - r0);
            let rb = m.data.len() / m.rows;
            let nv = j.toks.len();
            assert_eq!(m.cols / QK, fqb_of(ji));
            let hv = h_q8.sub(hoff[j.p0]..hoff[j.p0 + nv]);
            // SAFETY: this task writes rows r0..r0 + rn of this job's pairs' outputs, and nothing else
            unsafe {
                let pk: &[Pk] = if use_mmla(m.ty, nv) { &pkh[hkoff[ji]..hkoff[ji + 1]] } else { &[] };
                let pkk: &[Pkk] = if use_kmm(m.ty, nv) { &pkkh[hkkoff[ji]..hkkoff[ji + 1]] } else { &[] };
                let pk6: &[Pk6] = if use_q6e(m.ty, nv) { &pk6h[h6off[ji]..h6off[ji + 1]] } else { &[] };
                dot_multi_rows(
                    m.ty,
                    &m.data[r0 * rb..(r0 + rn) * rb],
                    rb,
                    rn,
                    hv,
                    pk,
                    pk6,
                    pkk,
                    nv,
                    dp.at(j.p0 * dm + r0),
                    dm,
                );
            }
        });
    }
    prof.lap(8);
    if let Some(g) = gpu.as_mut() {
        // the device's outputs go into their pairs' rows
        for (ji, j) in jobs.iter().enumerate().filter(|&(ji, _)| gpu_job[ji] && Some(ji) != shared_job) {
            for i in 0..j.toks.len() {
                g.take(ji, i, &mut down_out[(j.p0 + i) * dm..(j.p0 + i + 1) * dm]);
            }
        }
        let p0 = jobs[shared_job.unwrap()].p0;
        g.shared(&mut down_out[p0 * dm..(p0 + n) * dm]);
    }
    if on_npu {
        if let Some(h) = helper.0.take()
            && !h.join().unwrap_or(false)
        {
            return err("npu: the hot experts could not be started");
        }
        if !crate::npu::finish() {
            return err("npu: the shared and hot experts failed");
        }
        if let (Some(s), Some(_)) = (side, side_ffn) {
            s.done.set(true);
        }
        if let (Some(sj), Some(_)) = (shared_job, sh_ids) {
            let p0 = jobs[sj].p0;
            down_out[p0 * dm..(p0 + n) * dm].copy_from_slice(&sh_npu[..n * dm]);
        }
        // each hot job's output rows (its tokens in order, then the padding) are copied to its pairs' down_out rows
        let dp = SyncPtr::new(&mut down_out);
        pool::par_for(ffns.len(), &|i| {
            let (ji, _, r0, _) = ffns[i];
            let (p0, len) = (jobs[ji].p0, jobs[ji].toks.len());
            // SAFETY: job ji owns its pairs' down_out rows p0..p0 + len
            unsafe { dp.slice(p0 * dm, len * dm) }.copy_from_slice(&hot_y[r0 * dm..(r0 + len) * dm]);
        });
    }
    prof.lap(9);
    // 6. each token: its experts in increasing id (the job order), then the shared one with its sigmoid gate
    let mut pair_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (ji, j) in jobs.iter().enumerate() {
        if Some(ji) == shared_job {
            continue;
        }
        for (i, &t) in j.toks.iter().enumerate() {
            pair_of[t].push(j.p0 + i);
        }
    }
    let xp = SyncPtr::new(x);
    pool::par_for(n, &|t| {
        // SAFETY: item t owns token t's row of x
        let xt = unsafe { xp.slice(t * dm, dm) };
        for (&p, &(_, wt)) in pair_of[t].iter().zip(routes[t].iter().filter(|(e, _)| (lo..hi).contains(e))) {
            kops::axpy_f32(xt, wt, &down_out[p * dm..(p + 1) * dm]);
        }
        if let Some(sj) = shared_job {
            let wt = match &ly.shexp_gate {
                Some(gv) if ly.hc => crate::ops::sigmoid(crate::hc::dot_dbl(gv, &xn[t * dm..(t + 1) * dm])),
                Some(gv) => crate::ops::sigmoid(kops::dot_f32(gv, &xn[t * dm..(t + 1) * dm])),
                None => 1.0,
            };
            let p = jobs[sj].p0 + t;
            kops::axpy_f32(xt, wt, &down_out[p * dm..(p + 1) * dm]);
        }
    });
    prof.lap(10);
    Ok(())
}

/// x += the block's output for one token, where xn is the token normalised (the router's input) and xq is xn as Q8.
/// The phases of moe_token (TORC_PROF=1).
pub const MOE1_PHASES: [&str; 8] = [
    "router",
    "softmax+topk",
    "wave in RAM",
    "cache begin",
    "cache finish",
    "combine",
    "miss wave",
    "wave in RAM, reads in flight",
];
pub static MOE1_NS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// TORC_FUSE=0 runs moe_token with five pool sweeps. The default is three - the exps run on this thread, and each
/// expert's activations are done by whichever thread finishes its last gate/up chunk - and two where waiting is safe
/// (pool::waits_ok), with the down chunks inside the gate/up sweep, so their time then counts as gate/up.
fn fuse() -> bool {
    static F: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| !crate::switches::is("TORC_FUSE", "0"));
    *F
}

/// A raw output pointer shared by a group's claims (each claim writes rows no other claim does).
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
// SAFETY: the claims write disjoint rows through it
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// The k experts a layer's router ranks highest for the f32 row xn (normalised with that layer's FFN norm) - the guess
/// the expert cache reads ahead with. The logits come from dot_router spread over the threads, then a top-k.
pub fn predict_experts(ly: &MoeLayer, xn: &[f32], k: usize) -> Vec<u32> {
    let dm = xn.len();
    let ne = ly.router.len() / dm;
    if ne == 0 {
        return Vec::new();
    }
    let mut lg = vec![0.0f32; ne];
    let lp = SyncPtr::new(&mut lg);
    pool::par_for(ne.div_ceil(16), &|i| {
        let (e0, e1) = (i * 16, (i * 16 + 16).min(ne));
        // SAFETY: item i owns logits e0..e1
        let out = unsafe { lp.slice(e0, e1 - e0) };
        for (j, e) in (e0..e1).enumerate() {
            let row = &ly.router[e * dm..(e + 1) * dm];
            // qwen4exp: its router's own sum (this is only a guess at which reads to start early, never part of the
            // result)
            out[j] = if ly.hc { crate::hc::dot_dbl_f16(row, xn) } else { dot_router(row, xn) };
        }
    });
    let mut ids: Vec<u32> = (0..ne as u32).collect();
    let k = k.min(ne);
    if k == 0 {
        // TORC_READ_AHEAD=0,..: no experts are asked for
        return Vec::new();
    }
    ids.select_nth_unstable_by(k - 1, |a, b| lg[*b as usize].total_cmp(&lg[*a as usize]));
    ids.truncate(k);
    ids.sort_by(|a, b| lg[*b as usize].total_cmp(&lg[*a as usize]));
    ids
}

/// A prompt block's likeliest experts in another layer (for read-ahead, TORC_READ_AHEAD_BLOCK). Up to `sample` evenly
/// spaced tokens of the n each vote for their top k by that layer's router on these activations (using every token
/// would cost a router pass per token per layer), and the experts come back by votes, most first, ties by id. This is
/// only a guess at which reads to start early, never part of a result. The idea of reading ahead by the router's
/// votes is Strata's (github.com/Niko1221/Strata, MIT; THIRD-PARTY-NOTICES.md).
pub fn predict_block(ly: &MoeLayer, xn: &[f32], n: usize, k: usize, sample: usize) -> Vec<u32> {
    let dm = xn.len() / n.max(1);
    let ne = ly.router.len().checked_div(dm).unwrap_or(0);
    let k = k.min(ne);
    if ne == 0 || n == 0 || k == 0 {
        return Vec::new();
    }
    let toks: Vec<usize> = if n <= sample.max(1) {
        (0..n).collect()
    } else {
        let s = sample.max(1);
        (0..s).map(|i| i * n / s).collect()
    };
    let m = toks.len();
    let mut lg = vec![0.0f32; m * ne];
    let lp = SyncPtr::new(&mut lg);
    pool::par_for(ne.div_ceil(16), &|i| {
        for e in i * 16..(i * 16 + 16).min(ne) {
            let row = &ly.router[e * dm..(e + 1) * dm];
            for (j, &t) in toks.iter().enumerate() {
                let x = &xn[t * dm..(t + 1) * dm];
                let v = if ly.hc { crate::hc::dot_dbl_f16(row, x) } else { dot_router(row, x) };
                // SAFETY: item i owns column e of every sampled token's logits
                unsafe { *lp.at(j * ne + e) = v };
            }
        }
    });
    let mut votes = vec![0u32; ne];
    let mut ids: Vec<u32> = Vec::with_capacity(ne);
    for t in 0..m {
        let row = &lg[t * ne..(t + 1) * ne];
        ids.clear();
        ids.extend(0..ne as u32);
        ids.select_nth_unstable_by(k - 1, |a, b| row[*b as usize].total_cmp(&row[*a as usize]));
        for &e in &ids[..k] {
            votes[e as usize] += 1;
        }
    }
    let mut out: Vec<u32> = (0..ne as u32).filter(|&e| votes[e as usize] > 0).collect();
    out.sort_by(|a, b| votes[*b as usize].cmp(&votes[*a as usize]).then(a.cmp(b)));
    out
}

pub fn moe_token(w: &Weights, ly: &MoeLayer, xn: &[f32], xq: Q8s, x: &mut [f32], s: &mut Scratch) -> Result<()> {
    moe_token_with(w, ly, xn, xq, x, s, None)
}

/// Experts computed somewhere else (a device tier). Once the token's routed pairs are known (in increasing expert id),
/// the implementation is told them, starts the ones it holds and says which (true); their outputs, and the shared
/// expert's, are collected after the CPU's own experts. Each output is the same values the CPU would compute, and the
/// sum below keeps the CPU path's order (routed in increasing id, the shared one last), so the result has the same
/// bits.
pub trait GpuMoe {
    fn start(&mut self, pairs: &[(usize, f32)]) -> Vec<bool>;
    fn take(&mut self, j: usize, out: &mut [f32]);
    fn shared(&mut self, out: &mut [f32]);
}

/// moe_token with some experts (and the shared one) computed by `gpu`.
pub fn moe_token_with(
    w: &Weights,
    ly: &MoeLayer,
    xn: &[f32],
    xq: Q8s,
    x: &mut [f32],
    s: &mut Scratch,
    mut gpu: Option<&mut dyn GpuMoe>,
) -> Result<()> {
    if ly.gelu || ly.expert_scale.is_some() {
        // gemma4 runs its tokens as blocks of one (gemma.rs), because this fused path handles neither GELU nor the
        // expert scales
        return moe_tokens(w, ly, xn, xq, x, 1);
    }
    let g = w.layer_experts(ly.group);
    let ne = g.n_expert as usize;
    let dm = xn.len();
    let k = g.n_used as usize;
    let mut prof = crate::Laps::start(&MOE1_NS);
    // the shared expert's sigmoid gate (one f32 dot over xn) rides along as the last item of the router's sweep
    let mut sh_wt = [1.0f32];
    // the router's softmax exps, where waiting is safe: the sweep's second stage, one item per 32 experts, each finding
    // the maximum itself (the same scan, so the same maximum) and writing its exps to ex, leaving the logits readable
    let staged = fuse() && pool::waits_ok();
    let mut ex = crate::scratch::buf::<f32>("moe.ex", ne);
    {
        let rp = SyncPtr::new(&mut s.router[..ne]);
        let sp = SyncPtr::new(&mut sh_wt);
        let xp = SyncPtr::new(&mut ex);
        let nr = ne.div_ceil(16);
        let router = |i: usize| {
            if i == nr {
                // SAFETY: the last item owns sh_wt
                let gv = ly.shexp_gate.as_ref().unwrap();
                let d = if ly.hc { crate::hc::dot_dbl(gv, xn) } else { kops::dot_f32(gv, xn) };
                unsafe { sp.slice(0, 1)[0] = crate::ops::sigmoid(d) };
                return;
            }
            let e0 = i * 16;
            // SAFETY: item i owns router entries e0..e0 + 16
            let r = unsafe { rp.slice(e0, 16.min(ne - e0)) };
            for (j, v) in r.iter_mut().enumerate() {
                let e = e0 + j;
                let row = &ly.router[e * dm..(e + 1) * dm];
                // qwen4exp: a bf16 router summed in f64 as ggml sums it
                *v = if ly.hc { crate::hc::dot_dbl_f16(row, xn) } else { dot_router(row, xn) };
            }
        };
        let exps = |i: usize| {
            // SAFETY: the router stage is done and nothing writes the logits any more; item i owns ex[32 i..]
            let r = unsafe { std::slice::from_raw_parts(rp.at(0), ne) };
            let mut mx = f32::NEG_INFINITY;
            for &v in r {
                if v > mx {
                    mx = v;
                }
            }
            let e0 = i * 32;
            for (o, &v) in unsafe { xp.slice(e0, 32.min(ne - e0)) }.iter_mut().zip(&r[e0..]) {
                *o = (v - mx).exp();
            }
        };
        let n1 = nr + ly.shexp_gate.is_some() as usize;
        if staged {
            pool::par_stages(&[(n1, &router), (ne.div_ceil(32), &exps)]);
        } else {
            pool::par_for(n1, &router);
        }
    }
    prof.lap(0);
    // the softmax over the logits, then pick (or deepseek2's sigmoid gating, straight from the logits)
    let r = &mut s.router;
    let r32 = !ly.router32.is_empty();
    if r32 {
        logits_f32(ly, xn, dm, ne, 1, &mut r[..ne]);
    }
    let sigmoid_pairs = if ly.sigmoid { Some(pick_sigmoid(&mut r[..ne], k, ly)) } else { None };
    let mx = r[..ne].iter().fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
    // the exps (each is the same call on any thread, and 256 of them cost less than a pool sweep), then summed in order
    if staged && !r32 {
        r[..ne].copy_from_slice(&ex);
    } else if fuse() {
        for v in r[..ne].iter_mut() {
            *v = (*v - mx).exp();
        }
    } else {
        let rp = SyncPtr::new(&mut r[..ne]);
        pool::par_for(ne.div_ceil(32), &|i| {
            let e0 = i * 32;
            // SAFETY: item i owns entries e0..e0 + 32
            for v in unsafe { rp.slice(e0, 32.min(ne - e0)) } {
                *v = (*v - mx).exp();
            }
        });
    }
    let mut pairs = match sigmoid_pairs {
        Some(p) => p,
        None => pick(&r[..ne], k, ly.sel_bias.as_deref()),
    };
    // TORC_TRACE_ROUTES=1: each decoded token's chosen experts and weights, layer by layer (debugging a port)
    if crate::switches::on("TORC_TRACE_ROUTES") {
        let p: Vec<String> = pairs.iter().map(|(e, wt)| format!("{e}:{wt:.5}")).collect();
        eprintln!(
            "route layer {} (sigmoid {}, bias {}): {}",
            ly.layer,
            ly.sigmoid,
            ly.sel_bias.is_some(),
            p.join(", ")
        );
    }
    if gpu.is_none() {
        tail_skip(w, ly.layer, std::slice::from_mut(&mut pairs));
    }
    prof.lap(1);

    // the experts held elsewhere start now; the CPU computes the rest
    let hot = match gpu.as_mut() {
        Some(g) => g.start(&pairs),
        None => vec![false; pairs.len()],
    };
    // read the routed experts' slots, then compute every expert (and the shared one) in parallel, each with its own
    // buffers. Their outputs are added in the fixed order below, so the threads cannot change the bits
    let ids: Vec<u32> = pairs.iter().enumerate().filter(|(j, _)| !hot[*j]).map(|(_, p)| p.0 as u32).collect();
    // through the expert cache: the slots of the experts already in RAM come back at once (inready) and the others'
    // reads are started; the jobs in RAM and the shared one are computed while those reads land (begin / finish)
    let (slots, inram): (Vec<(&[u8], &format::Pack)>, Vec<bool>) = w.experts_ask(ly.layer, ly.group, &ids)?;
    prof.lap(3);
    #[allow(unused_mut)]
    let mut shared_t = match &ly.shared {
        Some(names) => Some([w.need(&names[0])?, w.need(&names[1])?, w.need(&names[2])?]),
        None => None,
    };
    if gpu.is_some() {
        assert!(shared_t.is_some(), "a shared expert from the GPU for a layer without one");
        shared_t = None;
    }
    let job_count = slots.len() + shared_t.is_some() as usize;
    let mats: Vec<[Mat; 3]> = (0..job_count)
        .map(|j| {
            if j < slots.len() {
                let (slot, lay) = slots[j];
                mats_of(slot, lay)
            } else {
                let t = shared_t.as_ref().unwrap();
                [Mat::of(&t[0]), Mat::of(&t[1]), Mat::of(&t[2])]
            }
        })
        .collect();
    // every expert's gate and up rows go in one flat sweep of row chunks, then the activations, then every down row.
    // The work spreads evenly over the threads, and each row is still the same dot product, so the bits are the same
    const RC: usize = 64;
    // the rows in a down chunk. A down matrix row is short (ff columns), so 64 rows is only about 18 KB of Q4_0. ARM64
    // uses 1024 and x86 64; TORC_MOE_DRC=r overrides it, with r dividing the model width
    static DRC: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        let d = if cfg!(target_arch = "aarch64") { 1024 } else { 64 };
        crate::switches::num("TORC_MOE_DRC").filter(|&r: &usize| r >= 16 && r % 16 == 0).unwrap_or(d)
    });
    let mut drc = if dm.is_multiple_of(*DRC) { *DRC } else { RC };
    // at least one down chunk per thread: a dense model's single job (the 27B, 5120 rows) in 1024-row chunks left 3 of
    // 8 threads idle for a third of its FFN (and the chunk count is a multiple of the thread count: 512 rows in 10
    // chunks ran as two waves, the second a quarter full)
    let th = pool::get().threads.max(1);
    if job_count * (dm / drc) < 4 * th {
        let eff = |r: usize| {
            let c = job_count * (dm / r);
            c as f64 / (c.div_ceil(th) * th) as f64
        };
        let mut best = drc;
        for r in (16..=drc).rev().step_by(16).filter(|r| dm.is_multiple_of(*r)) {
            if eff(r) > eff(best) + 1e-9 {
                best = r;
            }
            if eff(best) >= 0.99 {
                break;
            }
        }
        drc = best;
    }
    let ffmax = mats.iter().map(|m| m[0].rows.div_ceil(QK) * QK).max().unwrap_or(QK);
    // the wave's gate/up outputs, activations and down outputs, fresh and zeroed for each layer
    let mut gu_v = vec![0.0f32; job_count * 2 * ffmax];
    let mut hq_v = Q8::new(job_count * ffmax / QK);
    let mut df_v = vec![0.0f32; job_count * dm];
    let (gu, h_q8, dflat): (&mut [f32], Q8m, &mut [f32]) = (&mut gu_v, hq_v.view_mut(), &mut df_v);
    let gp = SyncPtr::new(gu);
    let hp = crate::pool::Q8Ptr::new(h_q8);
    // job j's activations: SiLU(gate) * up, quantised (the same calls whichever thread makes them)
    let hidden = |j: usize| {
        let ff = mats[j][0].rows;
        let fq = ff.div_ceil(QK) * QK;
        // SAFETY: called once per job, after every gate/up chunk of job j has been written (see below)
        let g = unsafe { gp.slice(j * 2 * ffmax, 2 * ffmax) };
        let mut h = vec![0.0f32; fq];
        for r in 0..ff {
            h[r] = crate::act::glu(ly.gelu, g[r], g[ffmax + r]);
        }
        // SAFETY: job j owns its ffmax / QK blocks
        let q = unsafe { hp.slice(j * ffmax / QK, fq / QK) };
        if ly.hc {
            kops::q8k_quantize_gg(&h, q)
        } else {
            kops::q8k_quantize(&h, q)
        }
        assert_eq!(mats[j][2].cols, fq);
    };
    // the rows in a gate/up chunk: 64 by default, or TORC_MOE_GRC=r with r dividing the expert width, so that no chunk
    // straddles gate and up
    static GRC: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        crate::switches::num("TORC_MOE_GRC").filter(|&r: &usize| r >= 16 && r % 16 == 0).unwrap_or(64)
    });
    // qwen4exp (768-wide padded experts): 256-row gate/up chunks unless TORC_MOE_GRC says otherwise
    let gwant = if ly.hc && !crate::switches::given("TORC_MOE_GRC") { 256 } else { *GRC };
    let grc = if ffmax % gwant == 0 { gwant } else { RC };
    let per_job = 2 * ffmax / grc;
    let done: Vec<std::sync::atomic::AtomicUsize> =
        (0..job_count).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect();
    let fused = fuse();
    let dp = SyncPtr::new(dflat);
    // a down chunk: rows r0..r0 + drc of the job's down matrix over its quantised activations
    let down = |job: usize, r0: usize| {
        let base = job * dm + r0;
        let m = &mats[job][2];
        let rb = m.data.len() / m.rows;
        let fq = m.cols;
        // SAFETY: chunk ci owns dflat[base..base + drc], and the job's activation blocks are written and no longer
        // change (either after the gate/up sweep, or once seen ready below)
        let (ys, hj) = unsafe { (dp.slice(base, drc), hp.part(job * ffmax / QK, fq / QK)) };
        dot_rows(m.ty, &m.data[r0 * rb..(r0 + drc) * rb], rb, hj, ys);
    };
    // where waiting is safe, the down chunks ride along in the gate/up sweep after its own chunks. Each one spins until
    // its own job's activations are made (every gate/up chunk of the job done, then hidden), not until the whole sweep
    // is done
    let flow = fused && pool::waits_ok();
    let ready: Vec<std::sync::atomic::AtomicUsize> =
        (0..job_count).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect();
    // the jobs of a wave: their gate/up chunks (and, when flowing, their down chunks) in one sweep of the pool
    let dpj = dm / drc;
    // TORC_MOE_DYN: how the wave's chunks are dealt out. 0 (the default) uses par_for, thread t taking
    // every T-th chunk; 1 is dynamic (each thread claims the next chunk); 2 uses guided runs. Any order is safe for the
    // flowing down chunks, because every gate/up chunk has a lower index than every down chunk, so it is claimed (and
    // running) first. Each chunk writes its own rows, so the bits are the same.
    static DYN: std::sync::LazyLock<u8> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_MOE_DYN").unwrap_or(0));
    let deal: fn(usize, &(dyn Fn(usize) + Sync)) = match *DYN {
        1 => pool::par_for_dynamic,
        2 => pool::par_for_guided,
        _ => pool::par_for,
    };
    let sweep = |jl: &[usize]| {
        let n_gu = jl.len() * per_job;
        deal(n_gu + if flow { jl.len() * dpj } else { 0 }, &|ci| {
            if ci >= n_gu {
                let (job, d) = (jl[(ci - n_gu) / dpj], (ci - n_gu) % dpj);
                pool::wait_for(&ready[job], 1);
                return down(job, d * drc);
            }
            let job = jl[ci / per_job];
            let rem = ci % per_job * grc;
            let base = job * 2 * ffmax + rem;
            let (which, r0) = (rem / ffmax, rem % ffmax);
            let m = &mats[job][which];
            if r0 < m.rows {
                // SAFETY: chunk ci owns gu[base..base + grc]
                let ys = unsafe { gp.slice(base, grc) };
                let n = ys.len().min(m.rows - r0);
                let rb = m.data.len() / m.rows;
                dot_rows(m.ty, &m.data[r0 * rb..(r0 + n) * rb], rb, xq, &mut ys[..n]);
            }
            // the last of a job's chunks to finish makes its activations (AcqRel makes every other chunk's rows
            // visible)
            if fused && done[job].fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1 == per_job {
                hidden(job);
                // Release: the activations become visible to the down chunks that see this
                ready[job].store(1, std::sync::atomic::Ordering::Release);
            }
        });
        if !fused {
            pool::par_for(jl.len(), &|i| hidden(jl[i]));
        }
        if !flow {
            pool::par_for(jl.len() * dpj, &|i| down(jl[i / dpj], i % dpj * drc));
        }
    };
    // TORC_MOE_JG=1 uses a guided group schedule: the wave's gate and up matrices as one group of their real rows (not
    // the padded width), claimed in guided chunks of 1/(2T) of what is left, 16..256 rows; then the activations; then
    // the down matrices as a second such group. Each row is the same dot product, so the bits are the same.
    static JG: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switches::on("TORC_MOE_JG"));
    let group = |tasks: &[(usize, usize, usize, SendPtr)]| {
        // (job, matrix, rows, out): the rows of matrix `which` of the job, with out its first output row
        let mut first = Vec::with_capacity(tasks.len() + 1);
        first.push(0usize);
        for t in tasks {
            first.push(first.last().unwrap() + t.2);
        }
        let total = *first.last().unwrap();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let nt = pool::get().threads.max(1);
        pool::par_for(nt, &|_| {
            let mut t = 0usize;
            loop {
                let seen = next.load(std::sync::atomic::Ordering::Relaxed);
                if seen >= total {
                    break;
                }
                let c = ((total - seen) / (2 * nt)).clamp(16, 256);
                let g0 = next.fetch_add(c, std::sync::atomic::Ordering::Relaxed);
                if g0 >= total {
                    break;
                }
                let g1 = (g0 + c).min(total);
                let mut g = g0;
                while g < g1 {
                    while first[t + 1] <= g {
                        t += 1;
                    }
                    let (job, which, _, _) = tasks[t];
                    let r0 = g - first[t];
                    let end = first[t + 1].min(g1);
                    let nr = end - g;
                    let m = &mats[job][which];
                    let rb = m.data.len() / m.rows;
                    let x: Q8s = if which == 2 {
                        // SAFETY: the job's activations are made (after the gate/up group)
                        unsafe { hp.part(job * ffmax / QK, m.cols / QK) }
                    } else {
                        xq
                    };
                    // SAFETY: rows r0..r0 + nr of this task's output belong to this claim alone
                    let ys = unsafe { std::slice::from_raw_parts_mut(tasks[t].3.0.add(r0), nr) };
                    dot_rows(m.ty, &m.data[r0 * rb..(r0 + nr) * rb], rb, x, ys);
                    g = end;
                }
            }
        });
    };
    let sweep_jg = |jl: &[usize]| {
        let mut t1 = Vec::with_capacity(2 * jl.len());
        for &j in jl {
            for which in 0..2 {
                // SAFETY: job j's gate (or up) rows start at gu[j * 2 * ffmax (+ ffmax)]
                t1.push((j, which, mats[j][which].rows, SendPtr(unsafe { gp.at(j * 2 * ffmax + which * ffmax) })));
            }
        }
        group(&t1);
        pool::par_for(jl.len(), &|i| hidden(jl[i]));
        // SAFETY: job j's down rows start at dflat[j * dm]
        let t2: Vec<_> = jl.iter().map(|&j| (j, 2, mats[j][2].rows, SendPtr(unsafe { dp.at(j * dm) }))).collect();
        group(&t2);
    };
    // wave 1 is the experts already in RAM plus the shared one; then the reads are finished; wave 2 is the rest. Each
    // job's output has its own row, summed below in the fixed order, so the waves cannot change the bits
    let (w1, w2): (Vec<usize>, Vec<usize>) = (0..job_count).partition(|&j| j >= slots.len() || inram[j]);
    if *JG {
        sweep_jg(&w1)
    } else {
        sweep(&w1)
    }
    // the wave's time is split by whether this layer has reads in flight (TORC_PROF), separating the I/O's cost to the
    // compute from the slots' own
    prof.lap(if w2.is_empty() { 2 } else { 7 });
    w.experts_settle()?;
    prof.lap(4);
    if !w2.is_empty() {
        if *JG { sweep_jg(&w2) } else { sweep(&w2) }
    }
    prof.lap(6);
    let outs: Vec<&[f32]> = dflat.chunks_exact(dm).collect();
    let mut so = crate::scratch::buf::<f32>("moe.shared", dm);
    let mut ci = 0;
    for (j, &(_, wt)) in pairs.iter().enumerate() {
        if hot[j] {
            gpu.as_mut().unwrap().take(j, &mut so);
            kops::axpy_f32(x, wt, &so);
        } else {
            kops::axpy_f32(x, wt, outs[ci]);
            ci += 1;
        }
    }
    if shared_t.is_some() {
        kops::axpy_f32(x, sh_wt[0], outs[ci]);
    }
    if let Some(g) = gpu.as_mut() {
        g.shared(&mut so);
        kops::axpy_f32(x, sh_wt[0], &so);
    }
    prof.lap(5);
    Ok(())
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    /// TORC_KLAY_HEAD's path: a Q6_K matrix with a laid-out copy (repack::register_many_klay) must give
    /// apply_group_multi the same bits as an unregistered copy of it (q6_lay_matmul for 1-3 vectors and gemm_q6_lay
    /// from 4, against q6k_dot_rows / q6k_matmul), from 1 to 9 vectors.
    #[test]
    fn laid_out_q6_same_bits() {
        if !crate::repack::klay_on() {
            return;
        }
        let _one = crate::pool::one_caller();
        let (rows, cols) = (96usize, 512usize);
        let nb = cols / QK;
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut w = vec![0u8; rows * nb * 210];
        for b in w.iter_mut() {
            *b = next() as u8;
        }
        // each block's fp16 scale (bytes 208-209) is set to a modest finite number
        for blk in w.chunks_exact_mut(210) {
            let h = (next() as u16 & 0x83ff) | ((((next() % 20) + 4) as u16) << 10);
            blk[208..210].copy_from_slice(&h.to_le_bytes());
        }
        let plain = w.clone();
        let w: &'static [u8] = Vec::leak(w);
        crate::repack::register_many_klay(&[(w, rows, cols, 14)]);
        for nv in 1..=9 {
            let xf: Vec<f32> = (0..nv * cols).map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0).collect();
            let mut xq = Q8::new(nv * nb);
            for v in 0..nv {
                kernels::q8k_quantize(&xf[v * cols..(v + 1) * cols], xq.sub_mut(v * nb..(v + 1) * nb));
            }
            let (mut a, mut b) = (vec![0f32; nv * rows], vec![0f32; nv * rows]);
            apply_group_multi(&mut [(Mat { ty: 14, rows, cols, data: w }, &mut a[..])], xq.view(), nv);
            apply_group_multi(&mut [(Mat { ty: 14, rows, cols, data: &plain }, &mut b[..])], xq.view(), nv);
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "nv {nv}: laid out differs");
        }
    }
}
