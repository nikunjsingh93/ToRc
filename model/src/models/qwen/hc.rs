//! hc.rs - Qwen3.8 Flash Next (arch qwen4exp): hyper-connections and the n-gram embeddings.
//!
//! The residual is hc (4) streams of d_model. Where other models read the normed residual, each block (the token mixer,
//! then the FFN) reads a mix of the streams: every stream is RMS-normed with its own weights (the squares summed in
//! f64), gated by the sigmoid of a low-rank product of all of them (down to hc_rank, SiLU, then up again), and the
//! gated streams are averaged. The block's output goes back into every stream, scaled by 2 sigmoid(inject . streams /
//! hc). There are no other norms; the head is one more mixer, without the scatter.
//!
//! The bf16 products are summed in f64 as ggml's CPU code does it: each product is exact in f32, the products are
//! summed in f64 in a fixed lane order (on ARM, NEON lanes into float64x2 accumulators), and the result is rounded once
//! to f32.
//!
//! Layer 1 first adds the n-gram embeddings. The token and the one or two before it are hashed into 16 rows of a 28.8
//! GB table (read in place from the mapping, one row each), giving a key per stream and a value. Each stream's normed
//! key meets the stream in a gate (the sigmoid of the signed square root of their scaled dot product); the gated value
//! is added to the stream, along with a causal convolution of its normed form over the positions 3, 6 and 9 back,
//! passed through SiLU.

use crate::hparams::Hparams;
use crate::pool::SyncPtr;
use crate::weights::Weights;
use format::meta::Meta;
use format::{Result, err};
use kernels::{Q8m, QK, ops as kops};

/// The positions the n-gram rings hold: one block (64) more than the convolution reaches back, so a draft that is taken
/// back never overwrites something a later token reads.
pub const NG_RING: usize = 128;
/// The n-gram layer's chunk size, in tokens.
const NG_CHUNK: usize = 64;

/// Rounds to bf16, to nearest even; a NaN stays as it is. The same rounding as ggml's ggml_compute_fp32_to_bf16 (MIT),
/// so the bits match a model rounded there.
pub fn to_bf16(f: f32) -> f32 {
    let u = f.to_bits();
    if u & 0x7fff_ffff > 0x7f80_0000 {
        return f;
    }
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

fn bf(w: u16) -> f32 {
    f32::from_bits((w as u32) << 16)
}

/// bf16 weights times inputs, with each product in f32 and the sum in f64. On ARM: 8 at a time into four float64x2
/// accumulators (element i + k into lane k % 2 of accumulator k / 2), then ((s0 + s1) + (s2 + s3)), lane 0 + lane 1,
/// then the tail in order. Elsewhere: in order.
pub fn dot_bf16_dbl(w: &[u16], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        dot_bf16_dbl_n::<1>(w, [x])[0]
    }
    #[cfg(not(target_arch = "aarch64"))]
    dot_bf16_dbl_scalar(w, x)
}

/// dot_bf16_dbl for T vectors against one row (ARM NEON: USHLL/USHLL2 to f32, FMUL, FCVTL/FCVTL2, FADD into four
/// float64x2 per vector). Each row is loaded a single time and serves all T vectors, and each vector's sum is exactly
/// dot_bf16_dbl's.
#[cfg(target_arch = "aarch64")]
pub fn dot_bf16_dbl_n<const T: usize>(w: &[u16], x: [&[f32]; T]) -> [f32; T] {
    use core::arch::aarch64::*;
    let n = x[0].len();
    assert!(w.len() >= n && x.iter().all(|v| v.len() == n));
    let mut out = [0.0f32; T];
    // SAFETY: NEON is baseline on aarch64, and the loads stay inside w and the x rows (i + 8 <= n)
    unsafe {
        let z = vdupq_n_f64(0.0);
        let mut s = [[z; 4]; T];
        let mut i = 0;
        while i + 8 <= n {
            let h = vld1q_u16(w.as_ptr().add(i));
            let w0 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(h)));
            let w1 = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(h));
            for t in 0..T {
                let p0 = vmulq_f32(w0, vld1q_f32(x[t].as_ptr().add(i)));
                let p1 = vmulq_f32(w1, vld1q_f32(x[t].as_ptr().add(i + 4)));
                s[t][0] = vaddq_f64(s[t][0], vcvt_f64_f32(vget_low_f32(p0)));
                s[t][1] = vaddq_f64(s[t][1], vcvt_high_f64_f32(p0));
                s[t][2] = vaddq_f64(s[t][2], vcvt_f64_f32(vget_low_f32(p1)));
                s[t][3] = vaddq_f64(s[t][3], vcvt_high_f64_f32(p1));
            }
            i += 8;
        }
        for t in 0..T {
            let mut r = vaddvq_f64(vaddq_f64(vaddq_f64(s[t][0], s[t][1]), vaddq_f64(s[t][2], s[t][3])));
            for k in i..n {
                r += (bf(w[k]) * x[t][k]) as f64;
            }
            out[t] = r as f32;
        }
    }
    out
}

/// The middle part of dot_bf16_dbl_n::<4> over columns i0..i1, from inputs already in f64 (wd is the row's
/// columns widened, xd the four vectors'), with the accumulators carried in `acc` between calls. Both
/// factors are bf16 values, so their product is exact in f32 and in f64, and an f32 product widened and added is the
/// same as one fused f64 multiply-add, on the same lanes in the same order.
#[cfg(target_arch = "aarch64")]
fn dot_bf16_dbl_seg4_f64(wd: &[f64], xd: [&[f64]; 4], acc: &mut [[f64; 8]; 4]) {
    use core::arch::aarch64::*;
    let len = wd.len();
    assert!(len % 8 == 0 && xd.iter().all(|v| v.len() >= len));
    // SAFETY: NEON is baseline on aarch64, and the loads stay inside wd and the xd rows (i + 8 <= len)
    unsafe {
        let mut s = [[vdupq_n_f64(0.0); 4]; 4];
        for t in 0..4 {
            for q in 0..4 {
                s[t][q] = vld1q_f64(acc[t].as_ptr().add(2 * q));
            }
        }
        let mut i = 0;
        while i < len {
            let w: [float64x2_t; 4] = std::array::from_fn(|q| vld1q_f64(wd.as_ptr().add(i + 2 * q)));
            for t in 0..4 {
                for q in 0..4 {
                    s[t][q] = vfmaq_f64(s[t][q], w[q], vld1q_f64(xd[t].as_ptr().add(i + 2 * q)));
                }
            }
            i += 8;
        }
        for t in 0..4 {
            for q in 0..4 {
                vst1q_f64(acc[t].as_mut_ptr().add(2 * q), s[t][q]);
            }
        }
    }
}

/// The end of a cut dot product: ((s0 + s1) + (s2 + s3)), lane 0 + lane 1, then the tail columns in order.
#[cfg(target_arch = "aarch64")]
fn dot_bf16_dbl_fin(acc: &[f64; 8], w: &[u16], x: &[f32], i: usize) -> f32 {
    // lanes: s_q = (acc[2q], acc[2q + 1])
    let l0 = (acc[0] + acc[2]) + (acc[4] + acc[6]);
    let l1 = (acc[1] + acc[3]) + (acc[5] + acc[7]);
    let mut r = l0 + l1;
    for k in i..x.len() {
        r += (bf(w[k]) * x[k]) as f64;
    }
    r as f32
}

#[allow(dead_code)]
fn dot_bf16_dbl_scalar(w: &[u16], x: &[f32]) -> f32 {
    let n = x.len();
    let mut i = 0;
    let mut s = 0.0f64;
    if cfg!(target_arch = "aarch64") {
        let mut a = [0.0f64; 8];
        while i + 8 <= n {
            for k in 0..8 {
                a[k] += (bf(w[i + k]) * x[i + k]) as f64;
            }
            i += 8;
        }
        let l0 = (a[0] + a[2]) + (a[4] + a[6]);
        let l1 = (a[1] + a[3]) + (a[5] + a[7]);
        s = l0 + l1;
    }
    while i < n {
        s += (bf(w[i]) * x[i]) as f64;
        i += 1;
    }
    s as f32
}

/// f32 x f32, with each product in f32 and the sum in f64. On ARM: 4 at a time into two float64x2 accumulators, then
/// (s0 + s1), lane 0 + lane 1, then the tail.
pub fn dot_dbl(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut i = 0;
    let mut s = 0.0f64;
    if cfg!(target_arch = "aarch64") {
        let mut acc = [0.0f64; 4];
        while i + 4 <= n {
            for k in 0..4 {
                acc[k] += (a[i + k] * b[i + k]) as f64;
            }
            i += 4;
        }
        s = (acc[0] + acc[2]) + (acc[1] + acc[3]);
    }
    while i < n {
        s += (a[i] * b[i]) as f64;
        i += 1;
    }
    s as f32
}

/// dot_dbl with fp16 weights (the router: bf16 values held exactly in fp16).
pub fn dot_dbl_f16(w: &[u16], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        dot_dbl_f16_n::<1>(w, [x])[0]
    }
    #[cfg(not(target_arch = "aarch64"))]
    dot_dbl_f16_scalar(w, x)
}

/// dot_dbl_f16 for T vectors against one row (ARM NEON: FCVTL of four fp16, FMUL, FCVTL / FCVTL2 to f64, FADD into two
/// float64x2 per vector). The row is read once, and each sum is exactly dot_dbl_f16's.
#[cfg(target_arch = "aarch64")]
pub fn dot_dbl_f16_n<const T: usize>(w: &[u16], x: [&[f32]; T]) -> [f32; T] {
    use core::arch::aarch64::*;
    let n = x[0].len();
    assert!(w.len() >= n && x.iter().all(|v| v.len() == n));
    let mut out = [0.0f32; T];
    // SAFETY: NEON is baseline on aarch64, and the loads stay inside w and the x rows (i + 4 <= n)
    unsafe {
        let z = vdupq_n_f64(0.0);
        let mut s = [[z; 2]; T];
        let mut i = 0;
        while i + 4 <= n {
            let h = vld1_u16(w.as_ptr().add(i));
            let wf: float32x4_t;
            // FCVTL: four fp16 values (raw bits) widened exactly to f32 (float16x4_t is unstable in core::arch)
            core::arch::asm!("fcvtl {o:v}.4s, {h:v}.4h", o = out(vreg) wf, h = in(vreg) h,
                options(pure, nomem, nostack));
            for t in 0..T {
                let p = vmulq_f32(wf, vld1q_f32(x[t].as_ptr().add(i)));
                s[t][0] = vaddq_f64(s[t][0], vcvt_f64_f32(vget_low_f32(p)));
                s[t][1] = vaddq_f64(s[t][1], vcvt_high_f64_f32(p));
            }
            i += 4;
        }
        for t in 0..T {
            let mut r = vaddvq_f64(vaddq_f64(s[t][0], s[t][1]));
            for k in i..n {
                r += (kernels::f16_to_f32(w[k]) * x[t][k]) as f64;
            }
            out[t] = r as f32;
        }
    }
    out
}

#[allow(dead_code)]
fn dot_dbl_f16_scalar(w: &[u16], x: &[f32]) -> f32 {
    let n = x.len();
    let mut i = 0;
    let mut s = 0.0f64;
    if cfg!(target_arch = "aarch64") {
        let mut acc = [0.0f64; 4];
        while i + 4 <= n {
            for k in 0..4 {
                acc[k] += (kernels::f16_to_f32(w[i + k]) * x[i + k]) as f64;
            }
            i += 4;
        }
        s = (acc[0] + acc[2]) + (acc[1] + acc[3]);
    }
    while i < n {
        s += (kernels::f16_to_f32(w[i]) * x[i]) as f64;
        i += 1;
    }
    s as f32
}

/// RMS norm as ggml computes it: the squares in f32, summed in f64, in order.
pub(crate) fn norm_gg(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v * v) as f64;
    }
    let scale = 1.0f32 / ((ss / n as f64) as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * w[i];
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A bf16 matrix: the file's own (by name, read in place), or one made here (the MTP block's Q8_0 mixers, converted).
pub struct Bf16 {
    name: String,
    own: Vec<u16>,
    pub rows: usize,
    pub cols: usize,
}

impl Bf16 {
    /// A d0 x d1 matrix in bf16: BF16 in the file, or Q8_0 (the MTP block's) converted to bf16 at load.
    pub fn load(w: &Weights, name: &str, d0: usize, d1: usize) -> Result<Bf16> {
        let t = w.need(name)?;
        if t.t.dims[0] as usize != d0 || t.t.n_dims < 2 || t.t.dims[1] as usize != d1 {
            return err(format!("tensor {name}: unexpected shape"));
        }
        let own = match t.t.ty {
            30 => {
                if !(t.data.as_ptr() as usize).is_multiple_of(2) {
                    return err(format!("tensor {name}: misaligned"));
                }
                Vec::new()
            }
            // Q8_0 (the MTP block's), and any other quantised type (ggml-org's IQ4_NL file keeps the mixers in IQ4_NL):
            // dequantised, then rounded to bf16
            // (block_size says None for Q5_0, which has its own weights path, but its dequantiser exists: the MTP file keeps
            // some mixers in it)
            ty if ty != 0 && (kops::block_size(ty).is_some() || ty == 6) => {
                let rb = t.row_bytes();
                let mut own = vec![0u16; d0 * d1];
                let mut row = vec![0.0f32; d0];
                for r in 0..d1 {
                    kops::dequantize(ty, &t.data[r * rb..(r + 1) * rb], &mut row);
                    for i in 0..d0 {
                        own[r * d0 + i] = (to_bf16(row[i]).to_bits() >> 16) as u16;
                    }
                }
                own
            }
            // F32 (unsloth's UD-Q4_K_XL keeps the inject matrices in F32): rounded to bf16 like the Q8_0 mixers, since
            // those are the numbers the mixer's bf16 products read either way
            0 => t.f32s()?.iter().map(|&v| (to_bf16(v).to_bits() >> 16) as u16).collect(),
            ty => return err(format!("tensor {name}: type {ty}, BF16, F32 or a quantised type expected")),
        };
        Ok(Bf16 { name: name.to_string(), own, rows: d1, cols: d0 })
    }

    pub fn get<'a>(&'a self, w: &'a Weights) -> &'a [u16] {
        if !self.own.is_empty() {
            return &self.own;
        }
        let d = w.need(&self.name).expect("bf16 tensor").data;
        // SAFETY: checked to be 2-aligned at load, and the mapping lives as long as w
        unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u16, d.len() / 2) }
    }
}

/// A mixer matrix: bf16, or a Q8_0 / IQ4_NL matrix kept as the file stores it and multiplied by kernels::mix (exact
/// weights and roughly 24-bit inputs, which means fewer bytes and arithmetic closer to exact than bf16 x bf16).
/// TORC_MIX=bf16 selects bf16 for every mixer.
pub enum MixW {
    Bf16(Bf16),
    Q { name: String, ty: u32, rows: usize, cols: usize, rb: usize },
}

/// Whether to use the quantised mixers (yes, unless TORC_MIX says bf16).
pub fn mix_native() -> bool {
    static N: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| !crate::switches::is("TORC_MIX", "bf16"));
    *N
}

/// TORC_MIX=exact: the quantised mixers in f64 on the unquantised inputs. Slow; it is the reference for the quantised
/// and bf16 products.
pub fn mix_exact() -> bool {
    static E: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switches::is("TORC_MIX", "exact"));
    *E
}

/// q_mm's reference: f32 inputs (n x cols) and f64 sums, over the pool.
pub fn q_mm_exact(w: &Weights, m: &MixW, x: &[f32], y: &mut [f32], n: usize) {
    let MixW::Q { name, ty, rows, cols, rb } = m else { panic!("q_mm_exact on a bf16 mixer") };
    let (ty, rows, cols, rb) = (*ty, *rows, *cols, *rb);
    let data = w.need(name).expect("mixer tensor").data;
    let yp = SyncPtr::new(y);
    crate::pool::par_for(rows.div_ceil(16), &|ci| {
        for r in ci * 16..((ci + 1) * 16).min(rows) {
            for j in 0..n {
                let v = kernels::mix::dot_exact(ty, &data[r * rb..(r + 1) * rb], &x[j * cols..(j + 1) * cols]);
                // SAFETY: chunk ci owns rows ci * 16.. of every vector's output
                unsafe { *yp.at(j * rows + r) = v as f32 };
            }
        }
    });
}

impl MixW {
    pub fn load(w: &Weights, name: &str, d0: usize, d1: usize) -> Result<MixW> {
        let t = w.need(name)?;
        let ty = t.t.ty;
        #[cfg(target_arch = "aarch64")]
        let dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        #[cfg(not(target_arch = "aarch64"))]
        let dotprod = true;
        if mix_native() && dotprod && kernels::mix::block_bytes(ty).is_some() && d0.is_multiple_of(32) {
            if t.t.dims[0] as usize != d0 || t.t.n_dims < 2 || t.t.dims[1] as usize != d1 {
                return err(format!("tensor {name}: unexpected shape"));
            }
            let rb = t.row_bytes();
            if rb != d0 / 32 * kernels::mix::block_bytes(ty).unwrap() {
                return err(format!("tensor {name}: unexpected row bytes"));
            }
            return Ok(MixW::Q { name: name.to_string(), ty, rows: d1, cols: d0, rb });
        }
        Ok(MixW::Bf16(Bf16::load(w, name, d0, d1)?))
    }

    pub fn is_q(&self) -> bool {
        matches!(self, MixW::Q { .. })
    }

    fn bf16(&self) -> &Bf16 {
        match self {
            MixW::Bf16(b) => b,
            MixW::Q { name, .. } => panic!("mixer {name}: quantized, not bf16"),
        }
    }
}

/// y[j][r] = row r . x[j] for n quantised input vectors (x is n x cols / 32 blocks) against a quantised mixer matrix,
/// with the rows claimed in chunks over the pool.
pub fn q_mm(w: &Weights, m: &MixW, x: &[kernels::mix::BlockX], y: &mut [f32], n: usize) {
    let MixW::Q { name, ty, rows, cols, rb } = m else { panic!("q_mm on a bf16 mixer") };
    q_mm_on(w.need(name).expect("mixer tensor").data, *ty, *rows, *cols, *rb, x, y, n)
}

/// q_mm on the matrix's bytes.
#[allow(clippy::too_many_arguments)]
fn q_mm_on(
    data: &[u8],
    ty: u32,
    rows: usize,
    cols: usize,
    rb: usize,
    x: &[kernels::mix::BlockX],
    y: &mut [f32],
    n: usize,
) {
    let nb = cols / 32;
    let th = crate::pool::get().threads.max(1);
    // a decode token's rows as one contiguous run per thread rather than about 4 chunks per thread. On by default on
    // ARM; TORC_MIX_SPLIT=0/1 overrides it
    static SPLIT: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| match crate::switches::get("TORC_MIX_SPLIT") {
            Some(v) => v != "0",
            None => cfg!(target_arch = "aarch64"),
        });
    let chunk = if *SPLIT && n == 1 {
        rows.div_ceil(th).div_ceil(4) * 4
    } else {
        rows.div_ceil(4 * th).clamp(4, if cols <= 1024 { 64 } else { 16 })
    };
    // a prompt's tokens go in groups of TB, so a task's inputs stay in cache across its rows (all of a prompt's inputs
    // to a long row run to megabytes, and would otherwise be re-read from memory for every row)
    const TB: usize = 16;
    let nt = n.div_ceil(TB);
    // and each task's rows are walked in pieces of KB blocks against its TB inputs (kernels::mix::neon::rows), so the
    // inputs stay close in cache instead of being re-read from L2 for every row. TORC_MIX_KB=0 turns it off
    #[cfg(target_arch = "aarch64")]
    static KB: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_MIX_KB").unwrap_or(16));
    // TORC_MIX_MM=1 (ARM with FEAT_I8MM): a prompt's products with SMMLA, two rows and two tokens at a time, with the
    // same bits (kernels::mix::neon::rows_mm). The tokens are paired up once here, over the pool, for every task. Not
    // for a matrix of a few rows (the inject's 4) or a short block (an MTP check of 2 tokens), where pairing the inputs
    // would cost more than the products
    #[cfg(target_arch = "aarch64")]
    static MM: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        crate::switches::on("TORC_MIX_MM") && std::arch::is_aarch64_feature_detected!("i8mm")
    });
    #[cfg(target_arch = "aarch64")]
    let xi = (*MM && n >= 32 && rows >= 64).then(|| {
        let pb = nb * kernels::mix::neon::PAIR_BLOCK;
        let mut xi = crate::scratch::buf::<i8>("hc.mixpair", n / 2 * pb);
        let xp = SyncPtr::new(&mut xi[..]);
        crate::pool::par_for(n / 2, &|p| {
            // SAFETY: item p owns pair p's bytes
            let o = unsafe { xp.slice(p * pb, pb) };
            kernels::mix::neon::interleave_pair(
                &x[2 * p * nb..(2 * p + 1) * nb],
                &x[(2 * p + 1) * nb..(2 * p + 2) * nb],
                o,
            );
        });
        xi
    });
    let yp = SyncPtr::new(y);
    let one = |i: usize| {
        let (ci, tj) = (i / nt, i % nt);
        let (r0, r1) = (ci * chunk, ((ci + 1) * chunk).min(rows));
        let (j0, j1) = (tj * TB, ((tj + 1) * TB).min(n));
        #[cfg(target_arch = "aarch64")]
        // SAFETY: FEAT_DotProd (MixW::load only makes a quantised matrix when it is present), and FEAT_I8MM whenever
        // xi is made; item i owns rows r0..r1 of vectors j0..j1's outputs; j0 is a multiple of TB, so even
        unsafe {
            let kb = if j1 - j0 > 1 { *KB } else { 0 };
            match &xi {
                Some(xi) => kernels::mix::neon::rows_mm(
                    ty,
                    data,
                    rb,
                    r0,
                    r1,
                    &x[j0 * nb..j1 * nb],
                    &xi[j0 / 2 * nb * kernels::mix::neon::PAIR_BLOCK..],
                    nb,
                    j1 - j0,
                    yp.at(j0 * rows),
                    rows,
                    kb,
                ),
                None => kernels::mix::neon::rows(
                    ty,
                    data,
                    rb,
                    r0,
                    r1,
                    &x[j0 * nb..j1 * nb],
                    nb,
                    j1 - j0,
                    yp.at(j0 * rows),
                    rows,
                    kb,
                ),
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        for r in r0..r1 {
            for j in j0..j1 {
                // SAFETY: as above
                unsafe {
                    *yp.at(j * rows + r) = kernels::mix::dot(ty, &data[r * rb..(r + 1) * rb], &x[j * nb..(j + 1) * nb])
                };
            }
        }
    };
    let ni = rows.div_ceil(chunk) * nt;
    if ni < 2 {
        (0..ni).for_each(one);
    } else {
        crate::pool::par_for(ni, &one);
    }
}

/// y[j][r] = dot_bf16_dbl(row r, x[j]) for n vectors x (n x cols) into y (n x rows), with the rows claimed in chunks
/// over the pool; each value is the same call on any thread.
pub fn bf16_mm(m: &[u16], rows: usize, cols: usize, x: &[f32], y: &mut [f32], n: usize) {
    // a block of tokens (a prompt) is tiled so each item's rows and tokens stay in cache (TORC_HC_TILE=0 turns it off)
    #[cfg(target_arch = "aarch64")]
    {
        static TILE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| !crate::switches::is("TORC_HC_TILE", "0"));
        if n >= 8 && *TILE {
            // long rows (the mixers' down and inject, the n-gram projections) go by column blocks; short rows (up, 320)
            // go row by row, each row widened once for all the tokens
            return if cols > 1024 {
                bf16_mm_tiled(m, rows, cols, x, y, n)
            } else {
                bf16_mm_rows_f64(m, rows, cols, x, y, n)
            };
        }
    }
    // about four chunks per thread, so a short matrix (the mixers' down, 320 rows) does not run as three uneven waves
    let th = crate::pool::get().threads.max(1);
    let chunk = rows.div_ceil(4 * th).clamp(4, if cols <= 1024 { 64 } else { 16 });
    let yp = SyncPtr::new(y);
    let one = |ci: usize| {
        let r0 = ci * chunk;
        for r in r0..(r0 + chunk).min(rows) {
            let wr = &m[r * cols..(r + 1) * cols];
            #[allow(unused_mut)]
            let mut j0 = 0;
            // four vectors for each read of the row (each sum is dot_bf16_dbl's own)
            #[cfg(target_arch = "aarch64")]
            while j0 + 4 <= n {
                let xs = std::array::from_fn(|t| &x[(j0 + t) * cols..(j0 + t + 1) * cols]);
                for (t, v) in dot_bf16_dbl_n::<4>(wr, xs).into_iter().enumerate() {
                    // SAFETY: chunk ci owns rows r0..r0 + chunk of every vector's output
                    unsafe { *yp.at((j0 + t) * rows + r) = v };
                }
                j0 += 4;
            }
            for j in j0..n {
                // SAFETY: chunk ci owns rows r0..r0 + chunk of every vector's output
                unsafe { *yp.at(j * rows + r) = dot_bf16_dbl(wr, &x[j * cols..(j + 1) * cols]) };
            }
        }
    };
    let nc = rows.div_ceil(chunk);
    if nc < 2 {
        (0..nc).for_each(one);
    } else {
        crate::pool::par_for(nc, &one);
    }
}

/// bf16_mm for a block of tokens over short rows (the inputs are bf16 values): the inputs are widened to f64 once and
/// each row once, then four tokens at a time go through dot_bf16_dbl_seg4_f64, with the same bits.
#[cfg(target_arch = "aarch64")]
fn bf16_mm_rows_f64(m: &[u16], rows: usize, cols: usize, x: &[f32], y: &mut [f32], n: usize) {
    let kmain = cols / 8 * 8;
    let mut xd = crate::scratch::buf::<f64>("hc.xd", n * cols);
    for (d, &v) in xd.iter_mut().zip(&x[..n * cols]) {
        *d = v as f64;
    }
    let xd = &xd[..];
    let yp = SyncPtr::new(y);
    const CH: usize = 64;
    crate::pool::par_for(rows.div_ceil(CH), &|ci| {
        let mut wd = vec![0.0f64; kmain];
        for r in ci * CH..((ci + 1) * CH).min(rows) {
            let wr = &m[r * cols..(r + 1) * cols];
            for (d, &h) in wd.iter_mut().zip(wr) {
                *d = bf(h) as f64;
            }
            let mut j = 0;
            while j + 4 <= n {
                let mut acc = [[0.0f64; 8]; 4];
                let xs: [&[f64]; 4] = std::array::from_fn(|t| &xd[(j + t) * cols..(j + t) * cols + kmain]);
                dot_bf16_dbl_seg4_f64(&wd, xs, &mut acc);
                for t in 0..4 {
                    let v = dot_bf16_dbl_fin(&acc[t], wr, &x[(j + t) * cols..(j + t + 1) * cols], kmain);
                    // SAFETY: chunk ci owns these rows of every token's output
                    unsafe { *yp.at((j + t) * rows + r) = v };
                }
                j += 4;
            }
            for j in j..n {
                // SAFETY: as above
                unsafe { *yp.at(j * rows + r) = dot_bf16_dbl(wr, &x[j * cols..(j + 1) * cols]) };
            }
        }
    });
}

/// bf16_mm for a block of tokens (the inputs are bf16 values, as every caller's are). Items of (8 rows, 32 tokens) each
/// walk the columns in blocks of 1024, keeping the four-token accumulators between blocks (dot_bf16_dbl_seg4_f64), so a
/// row's columns are read once per 32 tokens and a token's once per 8 rows, from cache. Every dot product adds the same
/// products in the same order as dot_bf16_dbl, so the bits are the same.
#[cfg(target_arch = "aarch64")]
fn bf16_mm_tiled(m: &[u16], rows: usize, cols: usize, x: &[f32], y: &mut [f32], n: usize) {
    const RC: usize = 8;
    const TB: usize = 32;
    const KB: usize = 1024;
    let kmain = cols / 8 * 8;
    let (nr, nt) = (rows.div_ceil(RC), n.div_ceil(TB));
    let yp = SyncPtr::new(y);
    // the inputs are widened to f64 once (every caller's inputs are bf16 values: see dot_bf16_dbl_seg4_f64)
    let mut xd = crate::scratch::buf::<f64>("hc.xd", n * cols);
    {
        let xp = SyncPtr::new(&mut xd);
        crate::pool::par_for(n, &|j| {
            // SAFETY: item j owns row j of xd
            let o = unsafe { xp.slice(j * cols, cols) };
            for (d, &v) in o.iter_mut().zip(&x[j * cols..(j + 1) * cols]) {
                *d = v as f64;
            }
        });
    }
    let xd = &xd[..];
    crate::pool::par_for(nr * nt, &|it| {
        let (r0, j0) = (it / nt * RC, it % nt * TB);
        let (r1, j1) = ((r0 + RC).min(rows), (j0 + TB).min(n));
        let ng = (j1 - j0) / 4;
        let mut acc = vec![[[0.0f64; 8]; 4]; (r1 - r0) * ng];
        let mut wd = vec![0.0f64; RC * KB];
        for kb in (0..kmain).step_by(KB) {
            let ke = (kb + KB).min(kmain);
            let kl = ke - kb;
            for r in r0..r1 {
                for (d, &h) in wd[(r - r0) * KB..(r - r0) * KB + kl].iter_mut().zip(&m[r * cols + kb..r * cols + ke]) {
                    *d = bf(h) as f64;
                }
            }
            for g in 0..ng {
                let jg = j0 + 4 * g;
                let xs: [&[f64]; 4] = std::array::from_fn(|t| &xd[(jg + t) * cols + kb..(jg + t) * cols + ke]);
                for r in r0..r1 {
                    dot_bf16_dbl_seg4_f64(&wd[(r - r0) * KB..(r - r0) * KB + kl], xs, &mut acc[(r - r0) * ng + g]);
                }
            }
        }
        for r in r0..r1 {
            let wr = &m[r * cols..(r + 1) * cols];
            for g in 0..ng {
                for t in 0..4 {
                    let j = j0 + 4 * g + t;
                    let v = dot_bf16_dbl_fin(&acc[(r - r0) * ng + g][t], wr, &x[j * cols..(j + 1) * cols], kmain);
                    // SAFETY: item it owns rows r0..r1 of tokens j0..j1's outputs
                    unsafe { *yp.at(j * rows + r) = v };
                }
            }
            for j in j0 + 4 * ng..j1 {
                // SAFETY: as above
                unsafe { *yp.at(j * rows + r) = dot_bf16_dbl(wr, &x[j * cols..(j + 1) * cols]) };
            }
        }
    });
}

/// One mixer: the streams' norms, the low-rank gate (down and up) and, except for the head's mixer, the scatter
/// weights.
pub struct Mixer {
    norm: Vec<f32>,
    down: MixW,
    up: MixW,
    inj: Option<MixW>,
}

impl Mixer {
    fn load(w: &Weights, h: &Hparams, norm: &str, down: &str, up: &str, inj: Option<&str>) -> Result<Mixer> {
        let (dm, hc, rank) = (h.d_model as usize, h.hc as usize, h.hc_rank as usize);
        let d = hc * dm;
        let norm = w.need(norm)?.f32s()?;
        if norm.len() != d {
            return err("hyper-connection norm: unexpected size");
        }
        let mut m = Mixer {
            norm,
            down: MixW::load(w, down, d, rank)?,
            up: MixW::load(w, up, rank, d)?,
            inj: inj.map(|n| MixW::load(w, n, d, hc)).transpose()?,
        };
        // the quantised path only runs when all of a mixer's matrices are quantised (hc_mix's qpath): a file that mixes
        // a native type with one the kernels lack (UD-Q4_K_XL) takes the bf16 path for all three
        // the inject matrix may be either: UD-Q4_K_XL keeps it in F32 (bf16 here) alongside Q8_0 down and up, and since
        // it is only 4 rows it is done beside the quantised pair (hc_mix)
        let all_q = m.down.is_q() && m.up.is_q();
        if !all_q {
            if m.down.is_q() {
                m.down = MixW::Bf16(Bf16::load(w, down, d, rank)?);
            }
            if m.up.is_q() {
                m.up = MixW::Bf16(Bf16::load(w, up, rank, d)?);
            }
        }
        Ok(m)
    }

    /// The MTP block's own final mixer (blk.<l>.nextn.hc_head_*, see mtp_hc.rs).
    pub fn nextn_head(w: &Weights, h: &Hparams, l: u32) -> Result<Mixer> {
        let n = |p: &str| format!("blk.{l}.nextn.hc_head_{p}.weight");
        Mixer::load(w, h, &n("norm"), &n("down"), &n("up"), None)
    }
}

/// A layer's two mixers: one before its token mixer (attn) and one before its FFN.
pub struct LayerHc {
    pub mix: [Mixer; 2],
}

impl LayerHc {
    pub fn load(w: &Weights, h: &Hparams, l: u32) -> Result<LayerHc> {
        let m = |p: &str| {
            Mixer::load(
                w,
                h,
                &format!("blk.{l}.hc_{p}_norm.weight"),
                &format!("blk.{l}.hc_{p}_down.weight"),
                &format!("blk.{l}.hc_{p}_up.weight"),
                Some(&format!("blk.{l}.hc_{p}_inject.weight")),
            )
        };
        Ok(LayerHc { mix: [m("attn")?, m("ffn")?] })
    }
}

/// The n-gram embeddings (one layer's): the hash constants, the table, its projections, norms and convolution, and the
/// rings of the sequence's tokens and normed gated values (a recurrent state, with position p at row p % NG_RING).
pub struct Ngram {
    pub layer: u32,
    n: usize,
    per: usize,
    kern: usize,
    eos: u32,
    dim: usize,
    mult: Vec<u64>,
    off: Vec<u32>,
    voc: Vec<u32>,
    key: Bf16,
    val: Bf16,
    nkey: Vec<f32>,
    nq: Vec<f32>,
    nconv: Vec<f32>,
    /// tap k of channel c is at k x D + c
    conv: Vec<f32>,
    hist: Vec<f32>,
    tok: Vec<i32>,
    /// a vocabulary-sliced file's ids mapped back to the original model's (TORC_NGRAM_IDMAP: a u32 per new id). The
    /// table was trained on hashes of the original ids, which a slice renumbers (nearly every id moves), so without the
    /// map every n-gram reads the wrong rows
    idmap: Vec<u32>,
}

impl Ngram {
    /// The rings, which are a recurrent state (a server's mark keeps them, see save_rec): the values, and the tokens as
    /// u16 pairs.
    pub fn ring(&self) -> (Vec<f32>, Vec<u16>) {
        (self.hist.clone(), self.tok.iter().flat_map(|&t| [t as u32 as u16, (t as u32 >> 16) as u16]).collect())
    }

    /// The rings' lengths as ring() gives them (values, and the u16 halves of the tokens); a saved state's must match.
    pub fn ring_lens(&self) -> (usize, usize) {
        (self.hist.len(), self.tok.len() * 2)
    }

    pub fn set_ring(&mut self, h: &[f32], t: &[u16]) {
        self.hist.copy_from_slice(h);
        for (d, c) in self.tok.iter_mut().zip(t.chunks_exact(2)) {
            *d = (c[0] as u32 | (c[1] as u32) << 16) as i32;
        }
    }

    /// Returns None when the file has no n-gram embeddings.
    pub fn load(w: &Weights, h: &Hparams) -> Result<Option<Ngram>> {
        let m = Meta::parse(&w.file.metadata)?;
        let ak = |k: &str| m.arch_key(k);
        let Some(layers) = m.ints(&ak("ple.layers")) else { return Ok(None) };
        if layers.is_empty() {
            return Ok(None);
        }
        let need = |k: &str| m.arch_u32(k).ok_or_else(|| format::Error(format!("n-gram embeddings: missing {k}")));
        let (dm, hc) = (h.d_model as usize, h.hc as usize);
        let d = hc * dm;
        let ngl = layers[0];
        let n = need("ple.ngram_size")? as usize;
        let per = need("ple.heads_per_ngram")? as usize;
        let kern = need("ple.conv_kernel")? as usize;
        let eos = need("ple.eos_token_id")?;
        let dim = need("embedding_length_per_layer_input")? as usize;
        if layers.len() != 1
            || ngl < 0
            || ngl as u32 >= h.n_layer
            || dm > 4096
            || !(2..=8).contains(&n)
            || per == 0
            || kern == 0
            || kern > 8
            || !dim.is_multiple_of(32)
            || (n - 1) * per > 64
            || (n - 1) * per * dim != dm
            || (kern - 1) * n + NG_CHUNK > NG_RING
        {
            return err("n-gram embeddings: inconsistent");
        }
        let heads = (n - 1) * per;
        let u64s = |k: &str, c: usize| -> Result<Vec<u64>> {
            let v = m.ints(&ak(k)).ok_or_else(|| format::Error(format!("n-gram embeddings: missing {k}")))?;
            if v.len() < c {
                return err("n-gram embeddings: hash constants");
            }
            Ok(v[..c].iter().map(|&x| x as u64).collect())
        };
        let mult = u64s("ple.layer_multipliers", n)?;
        let (off, voc) = (u64s("ple.head_offsets", heads)?, u64s("ple.head_vocab_sizes", heads)?);
        let mut rows = 0u64;
        for hh in 0..heads {
            if voc[hh] == 0 || off[hh] + voc[hh] > i32::MAX as u64 {
                return err(format!("n-gram embeddings: head {hh} range"));
            }
            rows = rows.max(off[hh] + voc[hh]);
        }
        let t = w.need("per_layer_token_embd.weight")?;
        if t.t.n_dims != 2 || t.t.dims[0] as usize != dim || t.t.dims[1] < rows || ![20, 8, 2].contains(&t.t.ty) {
            return err("per_layer_token_embd.weight missing or unexpected");
        }
        let l = ngl as u32;
        let f = |s: &str| -> Result<Vec<f32>> {
            let v = w.layer(l, s)?.f32s()?;
            if v.len() != d {
                return err(format!("blk.{l}.{s}: unexpected size"));
            }
            Ok(v)
        };
        let cw = w.layer(l, "ple_conv1d.weight")?;
        if !matches!(cw.t.ty, 0 | 1) || cw.t.dims[0] as usize != kern || cw.t.dims[1] as usize != d {
            return err("ple_conv1d.weight unexpected");
        }
        let cwv = cw.f32s()?;
        let mut conv = vec![0.0f32; kern * d];
        for ch in 0..d {
            for k in 0..kern {
                conv[k * d + ch] = cwv[ch * kern + k];
            }
        }
        let idmap: Vec<u32> = match crate::switches::get("TORC_NGRAM_IDMAP") {
            Some(p) if !p.is_empty() => {
                let b = std::fs::read(&p).map_err(|e| format::Error(format!("TORC_NGRAM_IDMAP {p}: {e}")))?;
                let m: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                eprintln!("n-gram layer: ids mapped back to the original vocabulary ({} ids, {p})", m.len());
                m
            }
            _ => Vec::new(),
        };
        // the end-of-sequence id the windows cut at is in the file's (sliced) numbering too
        let eos = if (eos as usize) < idmap.len() { idmap[eos as usize] } else { eos };
        Ok(Some(Ngram {
            layer: l,
            n,
            per,
            kern,
            eos,
            dim,
            mult,
            off: off.iter().map(|&v| v as u32).collect(),
            voc: voc.iter().map(|&v| v as u32).collect(),
            key: Bf16::load(w, &format!("blk.{l}.ple_key.weight"), dm, d)?,
            val: Bf16::load(w, &format!("blk.{l}.ple_value.weight"), dm, dm)?,
            nkey: f("ple_norm_key.weight")?,
            nq: f("ple_norm_query.weight")?,
            nconv: f("ple_norm_conv.weight")?,
            conv,
            hist: vec![0.0; NG_RING * d],
            tok: vec![0; NG_RING],
            idmap,
        }))
    }

    /// The table rows for the token at position p, one row per head. The window walks back from the token; a position
    /// before the first, or an end-of-sequence token, cuts it there, and that position and everything older read as
    /// that token (the token's own end-of-sequence does not cut its window).
    fn rows(&self, p: usize, rows: &mut [u32]) {
        let mut cc = [0i64; 8];
        cc[0] = self.tok[p % NG_RING] as i64;
        let mut cut = false;
        for s in 1..self.n {
            let t = if s <= p { self.tok[(p - s) % NG_RING] } else { -1 };
            cut = cut || t < 0 || t as u32 == self.eos;
            cc[s] = if cut { self.eos as i64 } else { t as i64 };
        }
        for g in 2..=self.n {
            let mut mixed = (cc[0] as u64).wrapping_mul(self.mult[0]);
            for k in 1..g {
                mixed ^= (cc[k] as u64).wrapping_mul(self.mult[k]);
            }
            for i in 0..self.per {
                let hh = (g - 2) * self.per + i;
                rows[hh] = (mixed % self.voc[hh] as u64 + self.off[hh] as u64) as u32;
            }
        }
    }

    /// Runs the layer over the residual streams of n consecutive tokens starting at position pos0 (res is n x hc
    /// d_model), a chunk of NG_CHUNK at a time, so the rings never lose a position a later token of the call reads.
    pub fn apply(&mut self, w: &Weights, h: &Hparams, tokens: &[u32], pos0: usize, res: &mut [f32]) -> Result<()> {
        let mut prof = crate::Laps::start(&HC_NS);
        let (dm, hc) = (h.d_model as usize, h.hc as usize);
        let d = hc * dm;
        let n = tokens.len();
        let tab = w.need("per_layer_token_embd.weight")?;
        let rb = tab.row_bytes();
        let heads = (self.n - 1) * self.per;
        let (wk, wv) = (self.key.get(w), self.val.get(w));
        let mut emb = vec![0.0f32; NG_CHUNK * dm];
        let mut key = vec![0.0f32; NG_CHUNK * d];
        let mut val = vec![0.0f32; NG_CHUNK * dm];
        for c0 in (0..n).step_by(NG_CHUNK) {
            let c1 = (c0 + NG_CHUNK).min(n);
            let cn = c1 - c0;
            for j in c0..c1 {
                let t = tokens[j];
                self.tok[(pos0 + j) % NG_RING] = self.idmap.get(t as usize).copied().unwrap_or(t) as i32;
            }
            {
                // each token's rows of the table (they page-fault on the mapping, so they are spread over the pool, one
                // row per item)
                let mut rows = vec![0u32; cn * heads];
                for j in 0..cn {
                    let mut r = [0u32; 64];
                    self.rows(pos0 + c0 + j, &mut r);
                    rows[j * heads..(j + 1) * heads].copy_from_slice(&r[..heads]);
                }
                let ep = SyncPtr::new(&mut emb);
                let dim = self.dim;
                crate::pool::par_for(cn * heads, &|i| {
                    let (j, hh) = (i / heads, i % heads);
                    let r = rows[i] as usize;
                    // SAFETY: item i owns head hh's slice of token j's row
                    let e = unsafe { ep.slice(j * dm + hh * dim, dim) };
                    kops::dequantize(tab.t.ty, &tab.data[r * rb..(r + 1) * rb], e);
                    for v in e.iter_mut() {
                        *v = to_bf16(*v);
                    }
                });
            }
            bf16_mm(wk, d, dm, &emb[..cn * dm], &mut key[..cn * d], cn);
            bf16_mm(wv, dm, dm, &emb[..cn * dm], &mut val[..cn * dm], cn);
            // the gates and the convolution's inputs, token by token (each token's on one thread)
            let sd = 1.0f32 / (dm as f32).sqrt();
            let eps = h.eps;
            {
                let (kp, hp) = (SyncPtr::new(&mut key), SyncPtr::new(&mut self.hist));
                let (res_r, val_r) = (&*res, &val);
                let (nkey, nq, nconv) = (&self.nkey, &self.nq, &self.nconv);
                let gate = |j: usize| {
                    let mut kn = vec![0.0f32; dm];
                    let mut qn = vec![0.0f32; dm];
                    // SAFETY: token j owns its key row and its ring row (the positions of a chunk are distinct mod
                    // RING)
                    let k = unsafe { kp.slice(j * d, d) };
                    let hist = unsafe { hp.slice(((pos0 + c0 + j) % NG_RING) * d, d) };
                    let vr = &val_r[j * dm..(j + 1) * dm];
                    for s in 0..hc {
                        let r = &res_r[(c0 + j) * d + s * dm..(c0 + j) * d + (s + 1) * dm];
                        norm_gg(&mut kn, &k[s * dm..(s + 1) * dm], &nkey[s * dm..(s + 1) * dm], eps);
                        norm_gg(&mut qn, r, &nq[s * dm..(s + 1) * dm], eps);
                        let mut dot = 0.0f64;
                        for i in 0..dm {
                            dot += (kn[i] * qn[i]) as f64;
                        }
                        let sc = dot as f32 * sd;
                        let mag = sc.abs().max(1e-6).sqrt();
                        let sg = if sc > 0.0 {
                            1.0
                        } else if sc < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                        let g = sigmoid(sg * mag);
                        let gv = &mut k[s * dm..(s + 1) * dm];
                        for i in 0..dm {
                            gv[i] = vr[i] * g;
                        }
                        norm_gg(&mut hist[s * dm..(s + 1) * dm], gv, &nconv[s * dm..(s + 1) * dm], eps);
                    }
                };
                if cn < 2 { (0..cn).for_each(gate) } else { crate::pool::par_for(cn, &gate) }
            }
            // the convolution, its SiLU, and the two additions to the streams
            {
                let rp = SyncPtr::new(res);
                let (hist, conv, key_r) = (&self.hist, &self.conv, &key);
                let (kern, dil) = (self.kern, self.n);
                let cv = |j: usize| {
                    let p = pos0 + c0 + j;
                    let tap: Vec<Option<&[f32]>> = (0..kern)
                        .map(|k| {
                            let back = (kern - 1 - k) * dil;
                            (back <= p).then(|| &hist[((p - back) % NG_RING) * d..][..d])
                        })
                        .collect();
                    // SAFETY: token j owns its streams
                    let r = unsafe { rp.slice((c0 + j) * d, d) };
                    let gv = &key_r[j * d..(j + 1) * d];
                    for ch in 0..d {
                        let mut acc = 0.0f32;
                        for (k, t) in tap.iter().enumerate() {
                            let xv = t.map_or(0.0, |t| t[ch]);
                            let tv = xv * conv[k * d + ch];
                            acc = if k > 0 { acc + tv } else { tv };
                        }
                        let co = silu(acc);
                        let add = gv[ch] + co;
                        r[ch] += add;
                    }
                };
                if cn < 2 { (0..cn).for_each(cv) } else { crate::pool::par_for(cn, &cv) }
            }
        }
        prof.lap(7);
        Ok(())
    }
}

/// The time spent in each step of the mixers and the n-gram layer (TORC_PROF=1; torc-run prints them).
pub static HC_NS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];
pub const HC_PHASES: [&str; 8] = ["norm", "down", "lo", "up", "inject", "mix", "combine+prep", "ngram"];

/// The streams' scratch space for a pass of up to n tokens.
#[derive(Default)]
pub struct HcBufs {
    xn: Vec<f32>,
    b: Vec<f32>,
    g: Vec<f32>,
    lo: Vec<f32>,
    pub inj: Vec<f32>,
}

/// The mixer for tokens t0..n: the streams in res (n x hc d_model) go into mix (rows t0..n of n x d_model), and, when
/// there is an inject matrix, the scatter weights' logits go into bufs.inj (n x hc).
pub fn hc_mix(
    w: &Weights,
    h: &Hparams,
    mx: &Mixer,
    res: &[f32],
    t0: usize,
    n: usize,
    mix: &mut [f32],
    bufs: &mut HcBufs,
) {
    let (dm, hc, rank) = (h.d_model as usize, h.hc as usize, h.hc_rank as usize);
    let d = hc * dm;
    let nt = n - t0;
    for v in [&mut bufs.xn, &mut bufs.b, &mut bufs.g] {
        v.resize(n * d, 0.0);
    }
    bufs.lo.resize(n * rank, 0.0);
    bufs.inj.resize(n * hc, 0.0);
    let eps = h.eps;
    let mut prof = crate::Laps::start(&HC_NS);
    // the quantised mixers (not TORC_MIX=exact): each stream's blocks for down are made in the norm's own item, giving
    // one pool sweep instead of two, and there is no bf16 copy (only the bf16 mixers read b). TORC_HC_FUSEQ=0 keeps the
    // two sweeps
    static FUSEQ: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| !crate::switches::is("TORC_HC_FUSEQ", "0"));
    let qpath = mx.down.is_q() && mx.up.is_q();
    let fuseq = *FUSEQ && qpath && !mix_exact();
    let mut xq_fused = fuseq.then(|| crate::scratch::buf::<kernels::mix::BlockX>("hc.mixq", nt * (d / 32)));
    // each stream normed with its weights (xn), and also at bf16 (b): one item per (token, stream)
    {
        let (xp, bp) = (SyncPtr::new(&mut bufs.xn), SyncPtr::new(&mut bufs.b));
        let qp = xq_fused.as_mut().map(|q| SyncPtr::new(&mut q[..]));
        let norm = |i: usize| {
            let (j, s) = (t0 + i / hc, i % hc);
            let o = j * d + s * dm;
            // SAFETY: item i owns stream s of token j in xn and in b
            let (xr, br) = unsafe { (xp.slice(o, dm), bp.slice(o, dm)) };
            norm_gg(xr, &res[o..o + dm], &mx.norm[s * dm..(s + 1) * dm], eps);
            if let Some(qp) = &qp {
                let sb = dm / 32;
                // SAFETY: item i owns stream s of token j - t0's blocks
                let q = unsafe { qp.slice((j - t0) * (d / 32) + s * sb, sb) };
                kernels::mix::quantize(xr, q);
                return;
            }
            for k in 0..dm {
                br[k] = to_bf16(xr[k]);
            }
        };
        crate::pool::par_for(nt * hc, &norm);
    }
    prof.lap(0);
    let inv = 1.0f32 / hc as f32;
    if qpath {
        // the quantised mixers (MixW): the normed streams go in without rounding to bf16, and so does lo
        use kernels::mix::{BlockX, quantize};
        if mix_exact() {
            q_mm_exact(w, &mx.down, &bufs.xn[t0 * d..n * d], &mut bufs.lo[t0 * rank..n * rank], nt);
            for v in bufs.lo[t0 * rank..n * rank].iter_mut() {
                *v = silu(*v * inv);
            }
            q_mm_exact(w, &mx.up, &bufs.lo[t0 * rank..n * rank], &mut bufs.g[t0 * d..n * d], nt);
            match &mx.inj {
                Some(inj) if inj.is_q() => {
                    q_mm_exact(w, inj, &bufs.xn[t0 * d..n * d], &mut bufs.inj[t0 * hc..n * hc], nt)
                }
                Some(inj) => {
                    bf16_mm(inj.bf16().get(w), hc, d, &bufs.xn[t0 * d..n * d], &mut bufs.inj[t0 * hc..n * hc], nt)
                }
                None => {}
            }
            prof.lap(4);
        } else {
            let (nbd, nbr) = (d / 32, rank / 32);
            let fused = xq_fused.is_some();
            let mut xq = xq_fused.take().unwrap_or_else(|| crate::scratch::buf::<BlockX>("hc.mixq", nt * nbd));
            if !fused {
                let qp = SyncPtr::new(&mut xq);
                let sb = dm / 32;
                // one item per (token, stream), a decode token's included
                crate::pool::par_for(nt * hc, &|i| {
                    let (j, s) = (i / hc, i % hc);
                    // SAFETY: item i owns stream s of token j's blocks
                    let o = unsafe { qp.slice(j * nbd + s * sb, sb) };
                    let r0 = (t0 + j) * d + s * dm;
                    quantize(&bufs.xn[r0..r0 + dm], o);
                });
            }
            q_mm(w, &mx.down, &xq, &mut bufs.lo[t0 * rank..n * rank], nt);
            prof.lap(1);
            let mut lq = crate::scratch::buf::<BlockX>("hc.mixlo", nt * nbr);
            for v in bufs.lo[t0 * rank..n * rank].iter_mut() {
                *v = silu(*v * inv);
            }
            quantize(&bufs.lo[t0 * rank..n * rank], &mut lq);
            prof.lap(2);
            q_mm(w, &mx.up, &lq, &mut bufs.g[t0 * d..n * d], nt);
            prof.lap(3);
            match &mx.inj {
                Some(inj) if inj.is_q() => q_mm(w, inj, &xq, &mut bufs.inj[t0 * hc..n * hc], nt),
                // an F32 inject (UD-Q4_K_XL): its 4 rows run at bf16 on the normed streams
                Some(inj) => {
                    bf16_mm(inj.bf16().get(w), hc, d, &bufs.xn[t0 * d..n * d], &mut bufs.inj[t0 * hc..n * hc], nt)
                }
                None => {}
            }
            prof.lap(4);
        }
    } else {
        let (down, up) = (mx.down.bf16().get(w), mx.up.bf16().get(w));
        bf16_mm(down, rank, d, &bufs.b[t0 * d..n * d], &mut bufs.lo[t0 * rank..n * rank], nt);
        prof.lap(1);
        for v in bufs.lo[t0 * rank..n * rank].iter_mut() {
            *v = to_bf16(silu(*v * inv));
        }
        prof.lap(2);
        bf16_mm(up, d, rank, &bufs.lo[t0 * rank..n * rank], &mut bufs.g[t0 * d..n * d], nt);
        prof.lap(3);
        if let Some(inj) = &mx.inj {
            bf16_mm(inj.bf16().get(w), hc, d, &bufs.b[t0 * d..n * d], &mut bufs.inj[t0 * hc..n * hc], nt);
        }
        prof.lap(4);
    }
    // the gated streams' mean, stream after stream, then the scale: one item per (token, 256 positions)
    let (xn, g) = (&bufs.xn, &bufs.g);
    let mp = SyncPtr::new(mix);
    let pieces = dm.div_ceil(256);
    crate::pool::par_for(nt * pieces, &|i| {
        let (j, c0) = (t0 + i / pieces, i % pieces * 256);
        let c1 = (c0 + 256).min(dm);
        // SAFETY: item i owns positions c0..c1 of token j's row
        let o = unsafe { mp.slice(j * dm + c0, c1 - c0) };
        let (xj, gj) = (&xn[j * d..(j + 1) * d], &g[j * d..(j + 1) * d]);
        for k in c0..c1 {
            let mut acc = xj[k] * sigmoid(gj[k]);
            for s in 1..hc {
                let p = xj[s * dm + k] * sigmoid(gj[s * dm + k]);
                acc += p;
            }
            o[k - c0] = acc * inv;
        }
    });
    prof.lap(5);
}

/// res[j][s] += out[j] * 2 sigmoid(inj[j][s] / hc), for tokens 0..n.
pub fn combine(h: &Hparams, res: &mut [f32], out: &[f32], inj: &[f32], n: usize) {
    let mut prof = crate::Laps::start(&HC_NS);
    let (dm, hc) = (h.d_model as usize, h.hc as usize);
    let d = hc * dm;
    let inv = 1.0f32 / hc as f32;
    let rp = SyncPtr::new(res);
    crate::pool::par_for(n * hc, &|i| {
        let (j, s) = (i / hc, i % hc);
        let wt = 2.0f32 * sigmoid(inj[j * hc + s] * inv);
        // SAFETY: item i owns stream s of token j
        let r = unsafe { rp.slice(j * d + s * dm, dm) };
        let b = &out[j * dm..(j + 1) * dm];
        for k in 0..dm {
            let p = b[k] * wt;
            r[k] += p;
        }
    });
    prof.lap(6);
}

/// A block's input from the mix: xq = Q8 as ggml quantises it, then xn rounded to bf16.
pub fn prep(dm: usize, xn: &mut [f32], xq: Q8m, n: usize) {
    let mut prof = crate::Laps::start(&HC_NS);
    let nb = dm / QK;
    let (xp, qp) = (SyncPtr::new(xn), crate::pool::Q8Ptr::new(xq));
    let one = |j: usize| {
        // SAFETY: item j owns row j of xn and of xq
        let (x, q) = unsafe { (xp.slice(j * dm, dm), qp.slice(j * nb, nb)) };
        kops::q8k_quantize_gg(x, q);
        for v in x.iter_mut() {
            *v = to_bf16(*v);
        }
    };
    if n < 2 {
        (0..n).for_each(one)
    } else {
        crate::pool::par_for(n, &one)
    }
    prof.lap(6);
}

/// The model-wide parts: the head's mixer, the n-gram layer and the streams.
pub struct HcModel {
    pub head: Mixer,
    pub ng: Option<Ngram>,
    pub res: Vec<f32>,
    pub bufs: HcBufs,
}

impl HcModel {
    pub fn load(w: &Weights, h: &Hparams) -> Result<HcModel> {
        Ok(HcModel {
            head: Mixer::load(w, h, "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight", None)?,
            ng: Ngram::load(w, h)?,
            res: Vec::new(),
            bufs: HcBufs::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding() {
        assert_eq!(to_bf16(1.0), 1.0);
        // 1 + 2^-8 is a tie between 1 and 1 + 2^-7, and rounds to even (1)
        assert_eq!(to_bf16(1.0 + 1.0 / 256.0), 1.0);
        assert_eq!(to_bf16(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
        assert!(to_bf16(f32::NAN).is_nan());
    }

    #[test]
    fn double_dots_agree_with_plain_sums() {
        let x: Vec<f32> = (0..37).map(|i| (i as f32 * 0.37).sin()).collect();
        let y: Vec<f32> = (0..37).map(|i| (i as f32 * 0.11).cos()).collect();
        let r: f64 = x.iter().zip(&y).map(|(a, b)| (a * b) as f64).sum();
        assert!((dot_dbl(&x, &y) as f64 - r).abs() < 1e-6);
    }
}
