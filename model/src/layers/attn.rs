//! attn.rs - the attention layer, gated or plain.
//!
//! For qwen35moe, whose q projection interleaves every query head with a same-sized output-gate head (attn_gate): the
//! q|gate, k and v products (qwen3moe's q projection has no gate heads, and its output is not gated); a per-head RMS
//! norm and NeoX RoPE (first n_rot dimensions) on q and k; K and V stored as int8 with one scale for each (position, KV
//! head); attention over the cache (fattn.rs); and the output gated, quantised, then multiplied by attn_output. Gemma
//! 4's layers and Flash Next's QSA indexer use the same engine with their own geometry.

use crate::hparams::Hparams;
use crate::moe::Mat;
use crate::ops::{rms_norm_inplace, rope_neox, sigmoid};
use crate::pool::SyncPtr;
use crate::weights::Weights;
use format::{Result, err};
use kernels::{Q8s, QK, ops as kops};

/// A layer's attention kernel (fattn.rs) for one geometry: the scores are scaled by `scale` instead of 1 /
/// sqrt(head_dim), and `window` gives a sliding window of that many positions (0 for all of them).
pub struct AttnEngine {
    f: crate::fattn::Fattn,
}

impl AttnEngine {
    /// The Qwen models' engine.
    pub fn new(h: &Hparams) -> Result<AttnEngine> {
        Self::create(h.n_head, h.n_head_kv, h.head_dim, h.attn_scale, 0)
    }

    /// An engine of any geometry (gemma.rs makes one for the full layers and one for the sliding-window ones). The
    /// cache itself says whether the keys and values are sixteen bits.
    pub fn create(n_head: u32, n_head_kv: u32, head_dim: u32, scale: Option<f32>, window: u32) -> Result<AttnEngine> {
        Ok(AttnEngine {
            f: crate::fattn::Fattn::new(n_head as usize, n_head_kv as usize, head_dim as usize, scale, window as usize),
        })
    }

    /// The same engine with a sink per query head (gpt-oss; Fattn::with_sinks).
    pub fn with_sinks(self, sinks: Vec<f32>) -> AttnEngine {
        AttnEngine { f: self.f.with_sinks(sinks) }
    }

    /// Attention over a block of n queries at positions pos0.. (q is n x qd, and out the same), on the keys and values
    /// in `kv`.
    pub fn run_block(&self, kv: &KvCache, q: &[f32], qd: usize, n: usize, pos0: u32, out: &mut [f32]) {
        assert!(q.len() >= n * qd && out.len() >= n * qd && pos0 as usize + n <= kv.cap);
        self.f.run_block(kv, q, qd, n, pos0, out, None)
    }

    /// run_block through the GPU tier when it is on and covers this shape (gpu.rs; it shares the block's tokens with
    /// the CPU), else on the CPU alone. For the models with their own forward pass (gpt-oss, Gemma); the Qwen family
    /// goes through attn_block's own call.
    pub fn run_block_tiered(&self, kv: &mut KvCache, q: &[f32], qd: usize, n: usize, pos0: u32, out: &mut [f32]) {
        assert!(q.len() >= n * qd && out.len() >= n * qd && pos0 as usize + n <= kv.cap);
        #[cfg(feature = "gpu")]
        if let Some(hi) = crate::gpu::run_block(kv, &self.f, q, qd, n, pos0 as usize, out) {
            kv.gpu_lo = hi;
            return;
        }
        self.f.run_block(kv, q, qd, n, pos0, out, None)
    }
}

/// One layer's cache: n_head_kv x n_ctx x head_dim bytes, with a scale per (KV head, position). For qwen4exp (kv_f16)
/// the numbers are held at sixteen bits in k16 / v16 instead, and k and v are empty.
pub struct KvCache {
    pub k: Vec<i8>,
    pub v: Vec<i8>,
    pub k16: Vec<i16>,
    pub v16: Vec<i16>,
    pub k_scale: Vec<f32>,
    pub v_scale: Vec<f32>,
    pub cap: usize,
    /// qwen4exp's indexer (qsa_prepare): the raw keys at f16 per position, the pooled keys, and the last block's picks
    pub ix_raw: Vec<u16>,
    pub ix_pk: Vec<f32>,
    pub ix_kp: usize,
    pub ix_npool: usize,
    pub sel: Vec<u8>,
    pub nv: Vec<u32>,
    pub qsa: bool,
    /// the indexer's scores for the last block, one row of ix_npool per token (scratch)
    pub ix_sc: Vec<f32>,
    /// the first position whose copy on the GPU (gpu.rs) may be stale: every write of keys and values lowers it
    pub gpu_lo: usize,
}

impl KvCache {
    /// Positions 0..npos as bytes (for Model::layer_states): K and V (int8, or 16 bits for qwen4exp) head by head (the
    /// two machines' caches may hold different contexts), their scales, then the whole indexer.
    pub fn put_bytes(&self, hd: usize, npos: usize, b: &mut Vec<u8>) {
        let (nh, cap) = (self.k_scale.len() / self.cap.max(1), self.cap);
        if self.k16.is_empty() {
            for x in [&self.k, &self.v] {
                for h in 0..nh {
                    b.extend(x[h * cap * hd..(h * cap + npos) * hd].iter().map(|&v| v as u8));
                }
            }
        } else {
            // qwen4exp keeps K and V at 16 bits
            for x in [&self.k16, &self.v16] {
                for h in 0..nh {
                    for v in &x[h * cap * hd..(h * cap + npos) * hd] {
                        b.extend(v.to_le_bytes());
                    }
                }
            }
        }
        for x in [&self.k_scale, &self.v_scale] {
            for h in 0..nh {
                for f in &x[h * cap..h * cap + npos] {
                    b.extend(f.to_le_bytes());
                }
            }
        }
        // qwen4exp's indexer (qsa_prepare): the raw keys of positions 0..npos and the pools they complete, as prefixes
        // of their layouts by position and by pool, so that a state loads into a cache of any size (sent whole, cap x
        // dim, a cache of a different size would read them with its own pool stride). Four u32s come first (the pool
        // size, the key width, and the counts that follow), all zero when there is no indexer. The picks (sel / nv) are
        // made again on every pass.
        let dim = if self.ix_raw.is_empty() { 0 } else { self.ix_raw.len() / cap.max(1) };
        let (nraw, npk) = if dim == 0 { (0, 0) } else { (npos * dim, npos / self.ix_kp * dim) };
        for v in [if dim == 0 { 0 } else { self.ix_kp }, dim, nraw, npk] {
            b.extend((v as u32).to_le_bytes());
        }
        for v in &self.ix_raw[..nraw] {
            b.extend(v.to_le_bytes());
        }
        for v in &self.ix_pk[..npk] {
            b.extend(v.to_le_bytes());
        }
    }

    /// Puts put_bytes' bytes back, reading them through `take(n)` (which returns the next n bytes, or fewer at the end
    /// of the data). Fails if they do not fit this cache (too short, or more positions than it holds), leaving it
    /// partly written.
    pub fn set_bytes(&mut self, hd: usize, npos: usize, take: &mut dyn FnMut(usize) -> Vec<u8>) -> Result<()> {
        let (nh, cap) = (self.k_scale.len() / self.cap.max(1), self.cap);
        if npos > cap {
            return err(format!("saved state: {npos} positions, this cache holds {cap}"));
        }
        self.gpu_lo = 0;
        let mut need = |n: usize| -> Result<Vec<u8>> {
            let r = take(n);
            if r.len() != n {
                return err("saved state: shorter than this cache's layout");
            }
            Ok(r)
        };
        if self.k16.is_empty() {
            for x in [&mut self.k, &mut self.v] {
                for h in 0..nh {
                    let r = need(npos * hd)?;
                    for (d, &v) in x[h * cap * hd..(h * cap + npos) * hd].iter_mut().zip(&r) {
                        *d = v as i8;
                    }
                }
            }
        } else {
            for x in [&mut self.k16, &mut self.v16] {
                for h in 0..nh {
                    let r = need(npos * hd * 2)?;
                    for (d, c) in x[h * cap * hd..(h * cap + npos) * hd].iter_mut().zip(r.chunks_exact(2)) {
                        *d = i16::from_le_bytes([c[0], c[1]]);
                    }
                }
            }
        }
        for x in [&mut self.k_scale, &mut self.v_scale] {
            for h in 0..nh {
                let r = need(npos * 4)?;
                for (d, c) in x[h * cap..h * cap + npos].iter_mut().zip(r.chunks_exact(4)) {
                    *d = f32::from_le_bytes(c.try_into().unwrap());
                }
            }
        }
        let u: Vec<usize> =
            need(16)?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize).collect();
        let (kp, dim, nraw, npk) = (u[0], u[1], u[2], u[3]);
        if dim == 0 {
            if nraw + npk != 0 {
                return err("saved state: indexer keys without a width");
            }
            self.ix_raw.clear();
            self.ix_pk.clear();
            (self.ix_kp, self.ix_npool) = (0, 0);
            return Ok(());
        }
        // laid out for this cache's size as qsa_prepare would lay it out, with the saved prefixes copied in
        let npool = cap / kp.max(1) + 1;
        if kp == 0 || nraw != npos * dim || npk != npos / kp * dim {
            return err("saved state: an indexer of another shape");
        }
        self.ix_raw = vec![0; cap * dim];
        self.ix_pk = vec![0.0; npool * dim];
        (self.ix_kp, self.ix_npool) = (kp, npool);
        for (d, c) in self.ix_raw.iter_mut().zip(need(nraw * 2)?.chunks_exact(2)) {
            *d = u16::from_le_bytes([c[0], c[1]]);
        }
        for (d, c) in self.ix_pk.iter_mut().zip(need(npk * 4)?.chunks_exact(4)) {
            *d = f32::from_le_bytes(c.try_into().unwrap());
        }
        Ok(())
    }

    pub fn new(h: &Hparams, n_ctx: usize) -> KvCache {
        Self::with(h.n_head_kv as usize, h.head_dim as usize, n_ctx, h.hc > 0 || h.kv16 || *KV16)
    }

    /// A cache of any geometry: n_head_kv heads of head_dim and n_ctx positions, at sixteen bits when `kv_f16` is set.
    pub fn with(n_head_kv: usize, head_dim: usize, n_ctx: usize, kv_f16: bool) -> KvCache {
        let n = n_head_kv * n_ctx;
        let len = n * head_dim;
        let (l8, l16) = if kv_f16 { (0, len) } else { (len, 0) };
        KvCache {
            k: vec![0; l8],
            v: vec![0; l8],
            k16: vec![0; l16],
            v16: vec![0; l16],
            k_scale: vec![0.0; n],
            v_scale: vec![0.0; n],
            cap: n_ctx,
            ix_raw: Vec::new(),
            ix_pk: Vec::new(),
            ix_kp: 0,
            ix_npool: 0,
            sel: Vec::new(),
            nv: Vec::new(),
            qsa: false,
            ix_sc: Vec::new(),
            gpu_lo: 0,
        }
    }
}

pub struct AttnLayer {
    pub l: u32,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    /// the q projection interleaves an output-gate head with every query head (qwen35moe, qwen3next, qwen4exp); false
    /// for a plain q projection (qwen3moe), whose attention output goes to attn_output ungated
    pub gated: bool,
    /// qwen4exp: the layer's indexer (load_indexer)
    pub ix: Option<Indexer>,
    /// deepseek2: multi-head latent attention (mla.rs) in place of the q, k and v products
    pub mla: Option<crate::mla::Mla>,
}

impl AttnLayer {
    pub fn load(w: &Weights, l: u32) -> Result<AttnLayer> {
        // gated when the q projection makes twice the attention output's width (its input: attn_output's rows)
        let q_rows = w.layer(l, "attn_q.weight")?.shape().1;
        let att_width = w.layer(l, "attn_output.weight")?.shape().0;
        Ok(AttnLayer {
            l,
            // ernie4_5-moe has no q and k norms (empty: none applied)
            q_norm: w.tensor(&format!("blk.{l}.attn_q_norm.weight")).map(|t| t.f32s()).transpose()?.unwrap_or_default(),
            k_norm: w.tensor(&format!("blk.{l}.attn_k_norm.weight")).map(|t| t.f32s()).transpose()?.unwrap_or_default(),
            gated: q_rows == 2 * att_width,
            ix: None,
            mla: None,
        })
    }

    /// A deepseek2 layer: its MLA weights (mla.rs), no q or k norms and no gate.
    pub fn load_mla(w: &Weights, h: &Hparams, l: u32) -> Result<AttnLayer> {
        Ok(AttnLayer {
            l,
            q_norm: Vec::new(),
            k_norm: Vec::new(),
            gated: false,
            ix: None,
            mla: Some(crate::mla::Mla::load(w, h, l)?),
        })
    }

    /// The q projection's rows per query head: the head, and its gate head when gated.
    fn q_stride(&self, hd: usize) -> usize {
        if self.gated { 2 * hd } else { hd }
    }
}

/// Symmetric int8 quantisation of one vector, returning its scale (max |v| / 127); each value is rounded to the
/// nearest step, ties to even.
pub(crate) fn quant8(src: &[f32], dst: &mut [i8]) -> f32 {
    let mut mx = 0.0f32;
    for &v in src {
        let a = v.abs();
        if a > mx {
            mx = a;
        }
    }
    let inv = if mx > 0.0 { 127.0 / mx } else { 0.0 };
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = (v * inv).round_ties_even() as i32 as i8;
    }
    mx / 127.0
}

/// One cache row as 16-bit integers (kv_f16), scaled so its biggest magnitude maps to 32767. Returns the scale.
pub(crate) fn quant16(src: &[f32], dst: &mut [i16]) -> f32 {
    let mut mx = 0.0f32;
    for &v in src {
        let a = v.abs();
        if a > mx {
            mx = a;
        }
    }
    let inv = if mx > 0.0 { 32767.0 / mx } else { 0.0 };
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = (v * inv).round_ties_even() as i32 as i16;
    }
    mx / 32767.0
}

#[allow(clippy::too_many_arguments)]
pub fn attn_token(
    w: &Weights,
    h: &Hparams,
    ly: &AttnLayer,
    eng: &AttnEngine,
    kv: &mut KvCache,
    pos: u32,
    cs: &[f32],
    sn: &[f32],
    xq: Q8s,
    x: &mut [f32],
) -> Result<()> {
    if let Some(m) = &ly.mla {
        return crate::mla::mla_block(w, h, m, eng, kv, pos, xq, x, 1);
    }
    let (hd, nh, nkv, dm) = (h.head_dim as usize, h.n_head as usize, h.n_head_kv as usize, h.d_model as usize);
    let (qd, kvd) = (nh * hd, nkv * hd);
    if pos as usize >= kv.cap {
        return err("position past the KV cache");
    }
    let buf = crate::scratch::buf::<f32>;
    let qw = nh * ly.q_stride(hd);
    let (mut big, mut kk, mut vv) = (buf("attn.big", qw), buf("attn.kk", kvd), buf("attn.vv", kvd));
    let (tq, tk, tv) =
        (w.layer(ly.l, "attn_q.weight")?, w.layer(ly.l, "attn_k.weight")?, w.layer(ly.l, "attn_v.weight")?);
    crate::moe::apply_group(
        &mut [(Mat::of(&tq), &mut big[..qw]), (Mat::of(&tk), &mut kk[..]), (Mat::of(&tv), &mut vv[..])],
        xq,
    );
    let (mut q, mut gate, mut att) = (buf("attn.q", qd), buf("attn.gate", qd), buf("attn.att", qd));
    attn_prep(h, ly, kv, pos, cs, sn, &big, &mut kk, &vv, &mut q, &mut gate);
    let t = std::time::Instant::now();
    eng.f.run_block(kv, &q, qd, 1, pos, &mut att, select_of(kv, 0).as_ref());
    ATTN_NS[CORE1].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    // one sweep: the output gate a Q8 block of 256 at a time, with each block quantised by the item that gated it
    // (blocks are quantised independently, so the bits are the same as the whole row at once), then attn_output's row
    // chunks once every block is done
    let wo = Mat::of(&w.layer(ly.l, "attn_output.weight")?);
    let mut attq = crate::scratch::q8("attn.attq", qd / QK);
    let mut out = buf("attn.out", dm);
    let (ap, qp, op, gate) =
        (SyncPtr::new(&mut att), crate::pool::Q8Ptr::new(attq.view_mut()), SyncPtr::new(&mut out), &gate);
    crate::pool::par_stages(&[
        (qd / QK, &|b| {
            // SAFETY: item b owns att[b * QK..(b + 1) * QK] and attq[b]
            let (a, y) = unsafe { (ap.slice(b * QK, QK), qp.slice(b, 1)) };
            if ly.gated {
                for (j, v) in a.iter_mut().enumerate() {
                    *v *= sigmoid(gate[b * QK + j]);
                }
            }
            // qwen4exp: quantised as ggml does it
            if h.hc > 0 { kops::q8k_quantize_gg(a, y) } else { kops::q8k_quantize(a, y) }
        }),
        // SAFETY: attq is complete and no longer written (the stage before is done), and chunk i owns its rows
        (wo.chunks(), &|i| unsafe { wo.apply_chunk(qp.part(0, qd / QK), op, i) }),
    ]);
    for i in 0..dm {
        x[i] += out[i];
    }
    Ok(())
}

/// The layer for n tokens at positions pos0.., in order (a prompt block). The products of all n run at once (each
/// weight row is read once, with the same bits); then every token's q, gate, K and V (with K and V into the cache);
/// then attention over the block in pieces of 64 queries (causal: a query attends to positions at or before its own);
/// then the output gate. The result has the same bits as attn_token on each token in turn.
#[allow(clippy::too_many_arguments)]
/// A prompt block's attention phases, in nanoseconds, for the profile: the q / k / v products; the core (scores,
/// weights, values and merge, whichever kernel runs); the output product (including its quantisation); that
/// quantisation alone; the per-token preparation (norms, rope, cache rows); and the output gate. Then two more:
/// Flash Next's indexer (qsa_prepare, prompt blocks and decode alike), the core of a one-token step (attn_token), and
/// the MTP block's indexer (qsa_prepare_mtp; the MTP block's other attention phases land in the block phases above).
pub const ATTN_PHASES: [&str; 9] =
    ["q/k/v", "core", "output", "output quant", "preparation", "gate", "indexer", "1-token core", "mtp indexer"];
pub static ATTN_NS: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];
/// TORC_KV16=1: every model's cache at sixteen bits (as qwen4exp's), not int8: a test of what the int8 rows' rounding
/// does to the whole model. The bits change, and the GPU tier stays off (it reads int8 rows only).
static KV16: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switches::on("TORC_KV16"));
/// TORC_ATTN_EXACT=1: prompt blocks' attention in f64 (Fattn::run_block_exact), to judge the faster paths against
static ATTN_EXACT: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switches::on("TORC_ATTN_EXACT"));
const PROJ_IN: usize = 0;
const CORE: usize = 1;
const PROJ_OUT: usize = 2;
const QUANT: usize = 3;
const PREP: usize = 4;
const GATE: usize = 5;
const INDEXER: usize = 6;
const CORE1: usize = 7;
const MTP_INDEXER: usize = 8;

pub fn attn_block(
    w: &Weights,
    h: &Hparams,
    ly: &AttnLayer,
    eng: &AttnEngine,
    kv: &mut KvCache,
    pos0: u32,
    xn: &[f32],
    xq: Q8s,
    x: &mut [f32],
    n: usize,
) -> Result<()> {
    if let Some(m) = &ly.mla {
        return crate::mla::mla_block(w, h, m, eng, kv, pos0, xq, x, n);
    }
    let (hd, nh, nkv, dm) = (h.head_dim as usize, h.n_head as usize, h.n_head_kv as usize, h.d_model as usize);
    let (qd, kvd) = (nh * hd, nkv * hd);
    if pos0 as usize + n > kv.cap {
        return err("position past the KV cache");
    }
    let qw = nh * ly.q_stride(hd);
    let mut big = crate::scratch::buf::<f32>("attn.big", n * qw);
    let mut kk = crate::scratch::buf::<f32>("attn.kk", n * kvd);
    let mut vv = crate::scratch::buf::<f32>("attn.vv", n * kvd);
    let (tq, tk, tv) =
        (w.layer(ly.l, "attn_q.weight")?, w.layer(ly.l, "attn_k.weight")?, w.layer(ly.l, "attn_v.weight")?);
    let mut jobs = [(Mat::of(&tq), &mut big[..n * qw]), (Mat::of(&tk), &mut kk[..]), (Mat::of(&tv), &mut vv[..])];
    let t0 = std::time::Instant::now();
    if !crate::npu::apply(&mut jobs, xn, n) {
        crate::moe::apply_group_multi(&mut jobs, xq, n);
    }
    ATTN_NS[PROJ_IN].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    let mut att = crate::scratch::buf::<f32>("attn.att", n * qd);
    attn_block_core(h, ly, eng, kv, pos0, &big, &mut kk, &vv, &mut att, n);
    let nbq = qd / QK;
    let mut out = crate::scratch::buf::<f32>("attn.out", n * dm);
    let two = w.layer(ly.l, "attn_output.weight")?;
    let mut jobs = [(Mat::of(&two), &mut out[..])];
    let t1 = std::time::Instant::now();
    if !crate::npu::apply(&mut jobs, &att, n) {
        let mut attq = crate::scratch::q8("attn.attq", n * nbq);
        let quant = if h.hc > 0 { kops::q8k_quantize_gg } else { kops::q8k_quantize };
        for t in 0..n {
            quant(&att[t * qd..(t + 1) * qd], attq.sub_mut(t * nbq..(t + 1) * nbq));
        }
        ATTN_NS[QUANT].fetch_add(t1.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        crate::moe::apply_group_multi(&mut jobs, attq.view(), n);
    }
    ATTN_NS[PROJ_OUT].fetch_add(t1.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    for (xr, o) in x.chunks_exact_mut(dm).zip(out.chunks_exact(dm)) {
        for i in 0..dm {
            xr[i] += o[i];
        }
    }
    Ok(())
}

/// Only attn_block's keys and values: the K and V products of the n rows, their norms, rope and cache rows at positions
/// pos0.., with the same bits attn_block writes, and no queries, attention or output. This is the MTP block's fill over
/// a prompt, where nothing reads the rows' outputs. TORC_MTP_FILL_FULL=1 runs attn_block instead.
#[allow(clippy::too_many_arguments)]
pub fn attn_fill_kv(
    w: &Weights,
    h: &Hparams,
    ly: &AttnLayer,
    kv: &mut KvCache,
    pos0: u32,
    xn: &[f32],
    xq: Q8s,
    n: usize,
) -> Result<()> {
    let (hd, nkv) = (h.head_dim as usize, h.n_head_kv as usize);
    let kvd = nkv * hd;
    if pos0 as usize + n > kv.cap {
        return err("position past the KV cache");
    }
    let mut kk = crate::scratch::buf::<f32>("attn.kk", n * kvd);
    let mut vv = crate::scratch::buf::<f32>("attn.vv", n * kvd);
    let (tk, tv) = (w.layer(ly.l, "attn_k.weight")?, w.layer(ly.l, "attn_v.weight")?);
    let mut jobs = [(Mat::of(&tk), &mut kk[..]), (Mat::of(&tv), &mut vv[..])];
    let t0 = std::time::Instant::now();
    if !crate::npu::apply(&mut jobs, xn, n) {
        crate::moe::apply_group_multi(&mut jobs, xq, n);
    }
    ATTN_NS[PROJ_IN].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    let tp = std::time::Instant::now();
    let half = (h.n_rot / 2) as usize;
    kv.gpu_lo = kv.gpu_lo.min(pos0 as usize);
    let (cap, wide) = (kv.cap, !kv.k16.is_empty());
    let (kp, vp) = (SyncPtr::new(&mut kv.k[..]), SyncPtr::new(&mut kv.v[..]));
    let (k16p, v16p) = (SyncPtr::new(&mut kv.k16[..]), SyncPtr::new(&mut kv.v16[..]));
    let (ksp, vsp) = (SyncPtr::new(&mut kv.k_scale[..]), SyncPtr::new(&mut kv.v_scale[..]));
    let kkp = SyncPtr::new(&mut kk[..]);
    crate::pool::par_for(n.div_ceil(8), &|c| {
        let (mut cs, mut sn) = (vec![0.0f32; half], vec![0.0f32; half]);
        for t in c * 8..(c * 8 + 8).min(n) {
            let pos = pos0 + t as u32;
            crate::vision::rope_slot(&mut cs, &mut sn, pos, h.n_rot, h.rope_base);
            // SAFETY: token t's row of kk and its cache rows (position pos, every KV head) belong to this task alone
            unsafe {
                let row = KvRow {
                    cap,
                    wide,
                    pos: pos as usize,
                    hd,
                    k: &kp,
                    v: &vp,
                    k16: &k16p,
                    v16: &v16p,
                    k_scale: &ksp,
                    v_scale: &vsp,
                };
                kv_prep_row(h, ly, &row, &cs, &sn, kkp.slice(t * kvd, kvd), &vv[t * kvd..(t + 1) * kvd]);
            }
        }
    });
    ATTN_NS[PREP].fetch_add(tp.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// attn_block from its products onwards: every token's q, gate, K and V (with K and V into the cache at its position),
/// attention over the block (fattn.rs), and the output gate, into att (n x n_head head_dim).
#[allow(clippy::too_many_arguments)]
fn attn_block_core(
    h: &Hparams,
    ly: &AttnLayer,
    eng: &AttnEngine,
    kv: &mut KvCache,
    pos0: u32,
    big: &[f32],
    kk: &mut [f32],
    vv: &[f32],
    att: &mut [f32],
    n: usize,
) {
    let (hd, nh, nkv) = (h.head_dim as usize, h.n_head as usize, h.n_head_kv as usize);
    let (qd, kvd) = (nh * hd, nkv * hd);
    let half = (h.n_rot / 2) as usize;
    let mut q = crate::scratch::buf::<f32>("attn.q", n * qd);
    let mut gate = crate::scratch::buf::<f32>("attn.gate", n * qd);
    // each token's norms, rope, gate split and cache row run on the pool, 8 tokens a task. A token's steps are its own,
    // and its cache row (position pos0 + t) is written by no other token, so the bits are those of the sequential loop
    let tp = std::time::Instant::now();
    kv.gpu_lo = kv.gpu_lo.min(pos0 as usize);
    let (cap, wide) = (kv.cap, !kv.k16.is_empty());
    let (kp, vp) = (SyncPtr::new(&mut kv.k[..]), SyncPtr::new(&mut kv.v[..]));
    let (k16p, v16p) = (SyncPtr::new(&mut kv.k16[..]), SyncPtr::new(&mut kv.v16[..]));
    let (ksp, vsp) = (SyncPtr::new(&mut kv.k_scale[..]), SyncPtr::new(&mut kv.v_scale[..]));
    let (qp, gp, kkp) = (SyncPtr::new(&mut q[..]), SyncPtr::new(&mut gate[..]), SyncPtr::new(&mut kk[..]));
    crate::pool::par_for(n.div_ceil(8), &|c| {
        let (mut cs, mut sn) = (vec![0.0f32; half], vec![0.0f32; half]);
        for t in c * 8..(c * 8 + 8).min(n) {
            let pos = pos0 + t as u32;
            crate::vision::rope_slot(&mut cs, &mut sn, pos, h.n_rot, h.rope_base);
            // SAFETY: token t's rows of q, gate and kk, and its cache rows (position pos, every KV head), are touched
            // by this task alone
            unsafe {
                let row = KvRow {
                    cap,
                    wide,
                    pos: pos as usize,
                    hd,
                    k: &kp,
                    v: &vp,
                    k16: &k16p,
                    v16: &v16p,
                    k_scale: &ksp,
                    v_scale: &vsp,
                };
                let qw = nh * ly.q_stride(hd);
                attn_prep_row(
                    h,
                    ly,
                    &row,
                    &cs,
                    &sn,
                    &big[t * qw..(t + 1) * qw],
                    kkp.slice(t * kvd, kvd),
                    &vv[t * kvd..(t + 1) * kvd],
                    qp.slice(t * qd, qd),
                    gp.slice(t * qd, qd),
                );
            }
        }
    });
    ATTN_NS[PREP].fetch_add(tp.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    let t = std::time::Instant::now();
    // the GPU tier (gpu feature, TORC_GPU_ATTN=1) runs the block when it covers it, sharing the tokens with the CPU.
    // Roadmap: the NPU and the GPU carrying the prompt between them, with the CPU nearly free.
    #[cfg(feature = "gpu")]
    let gpu = crate::gpu::run_block(kv, &eng.f, &q, qd, n, pos0 as usize, att);
    #[cfg(not(feature = "gpu"))]
    let gpu: Option<usize> = None;
    match gpu {
        Some(hi) => kv.gpu_lo = hi,
        None => match select_of(kv, 0) {
            // TORC_ATTN_EXACT=1: attention in f64 (Fattn::run_block_exact), the judge for the faster paths' rounding
            None if *ATTN_EXACT => eng.f.run_block_exact(kv, &q, qd, n, pos0, att),
            sel => eng.f.run_block(kv, &q, qd, n, pos0, att, sel.as_ref()),
        },
    }
    ATTN_NS[CORE].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    let t = std::time::Instant::now();
    if ly.gated {
        gate_mul(att, &gate);
    }
    ATTN_NS[GATE].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

/// One token's products, into its q (normed and rotated), its output gate, and its K and V (K normed and rotated, both
/// quantised into the cache at pos).
#[allow(clippy::too_many_arguments)]
/// A position's cache rows, written through the pool's shared pointers (attn_block_core).
struct KvRow<'a> {
    cap: usize,
    wide: bool,
    pos: usize,
    hd: usize,
    k: &'a SyncPtr<i8>,
    v: &'a SyncPtr<i8>,
    k16: &'a SyncPtr<i16>,
    v16: &'a SyncPtr<i16>,
    k_scale: &'a SyncPtr<f32>,
    v_scale: &'a SyncPtr<f32>,
}

/// One head of q or k: its RMS norm (when the model has one: ernie4_5-moe does not) and its rotary embedding, NeoX's
/// (i, i + n/2) pairs or, for ernie4_5-moe, adjacent pairs.
fn norm_rope(h: &Hparams, v: &mut [f32], norm: &[f32], n_rot: usize, cs: &[f32], sn: &[f32]) {
    if !norm.is_empty() {
        rms_norm_inplace(v, norm, h.eps);
    }
    if h.rope_pairs {
        crate::ops::rope_pairs(v, n_rot, cs, sn);
    } else {
        rope_neox(v, n_rot, cs, sn);
    }
}

/// attn_prep with the cache written through a KvRow: the same steps in the same order.
///
/// # Safety
/// The caller owns position row.pos of every KV head of the cache for the duration of the call.
#[allow(clippy::too_many_arguments)]
unsafe fn attn_prep_row(
    h: &Hparams,
    ly: &AttnLayer,
    kv: &KvRow,
    cs: &[f32],
    sn: &[f32],
    big: &[f32],
    kk: &mut [f32],
    vv: &[f32],
    q: &mut [f32],
    gate: &mut [f32],
) {
    let (hd, nh, nkv) = (h.head_dim as usize, h.n_head as usize, h.n_head_kv as usize);
    let n_rot = h.n_rot as usize;
    let qs = ly.q_stride(hd);
    for hh in 0..nh {
        let src = &big[hh * qs..(hh + 1) * qs];
        let v = &mut q[hh * hd..(hh + 1) * hd];
        v.copy_from_slice(&src[..hd]);
        if ly.gated {
            gate[hh * hd..(hh + 1) * hd].copy_from_slice(&src[hd..]);
        }
        norm_rope(h, v, &ly.q_norm, n_rot, cs, sn);
    }
    let _ = nkv;
    unsafe { kv_prep_row(h, ly, kv, cs, sn, kk, vv) }
}

/// The K and V half of attn_prep_row: the token's K (normed and rotated) and V, quantised into the cache at kv.pos.
///
/// # Safety
/// As for attn_prep_row.
unsafe fn kv_prep_row(h: &Hparams, ly: &AttnLayer, kv: &KvRow, cs: &[f32], sn: &[f32], kk: &mut [f32], vv: &[f32]) {
    let (hd, nkv) = (h.head_dim as usize, h.n_head_kv as usize);
    let n_rot = h.n_rot as usize;
    for hh in 0..nkv {
        let k = &mut kk[hh * hd..(hh + 1) * hd];
        norm_rope(h, k, &ly.k_norm, n_rot, cs, sn);
        let at = (hh * kv.cap + kv.pos) * kv.hd;
        let sc = hh * kv.cap + kv.pos;
        unsafe {
            if !kv.wide {
                *kv.k_scale.at(sc) = quant8(k, kv.k.slice(at, hd));
                *kv.v_scale.at(sc) = quant8(&vv[hh * hd..(hh + 1) * hd], kv.v.slice(at, hd));
            } else {
                *kv.k_scale.at(sc) = quant16(k, kv.k16.slice(at, hd));
                *kv.v_scale.at(sc) = quant16(&vv[hh * hd..(hh + 1) * hd], kv.v16.slice(at, hd));
            }
        }
    }
}

fn attn_prep(
    h: &Hparams,
    ly: &AttnLayer,
    kv: &mut KvCache,
    pos: u32,
    cs: &[f32],
    sn: &[f32],
    big: &[f32],
    kk: &mut [f32],
    vv: &[f32],
    q: &mut [f32],
    gate: &mut [f32],
) {
    let (hd, nh, nkv) = (h.head_dim as usize, h.n_head as usize, h.n_head_kv as usize);
    let n_rot = h.n_rot as usize;
    let qs = ly.q_stride(hd);
    for hh in 0..nh {
        let src = &big[hh * qs..(hh + 1) * qs];
        let v = &mut q[hh * hd..(hh + 1) * hd];
        v.copy_from_slice(&src[..hd]);
        if ly.gated {
            gate[hh * hd..(hh + 1) * hd].copy_from_slice(&src[hd..]);
        }
        norm_rope(h, v, &ly.q_norm, n_rot, cs, sn);
    }
    let row = pos as usize;
    kv.gpu_lo = kv.gpu_lo.min(row);
    for hh in 0..nkv {
        let k = &mut kk[hh * hd..(hh + 1) * hd];
        norm_rope(h, k, &ly.k_norm, n_rot, cs, sn);
        let at = (hh * kv.cap + row) * hd;
        let sc = hh * kv.cap + row;
        if kv.k16.is_empty() {
            kv.k_scale[sc] = quant8(k, &mut kv.k[at..at + hd]);
            kv.v_scale[sc] = quant8(&vv[hh * hd..(hh + 1) * hd], &mut kv.v[at..at + hd]);
        } else {
            kv.k_scale[sc] = quant16(k, &mut kv.k16[at..at + hd]);
            kv.v_scale[sc] = quant16(&vv[hh * hd..(hh + 1) * hd], &mut kv.v16[at..at + hd]);
        }
    }
}

/// att *= sigmoid(gate), element by element, in 256-wide pieces over the pool.
fn gate_mul(att: &mut [f32], gate: &[f32]) {
    let len = att.len();
    let ap = crate::pool::SyncPtr::new(att);
    crate::pool::par_for(len.div_ceil(256), &|c| {
        let i0 = c * 256;
        let n = 256.min(len - i0);
        // SAFETY: piece c owns att[i0..i0 + n]
        let a = unsafe { ap.slice(i0, n) };
        for (j, v) in a.iter_mut().enumerate() {
            *v *= sigmoid(gate[i0 + j]);
        }
    });
}

/// Flash Next's indexer (QSA), one per attention layer. Raw keys come from the layer's input (a bf16 product, kept at
/// f16 per position) and are pooled by kpool positions in order (summed, scaled, RMS-normed, rotated to the pool's
/// first position, and kept at f16). Past the budget (the top pools), each token's query (heads of dim, normed and
/// rotated) scores every complete pool (the sum over heads of relu(q . k) / sqrt(dim)), and the token attends to its
/// top pools plus the incomplete one, through the attention kernel's selection.
pub struct Indexer {
    q: crate::hc::Bf16,
    k: crate::hc::Bf16,
    qn: Vec<f32>,
    kn: Vec<f32>,
    heads: usize,
    dim: usize,
    kpool: usize,
    /// the pools a token attends past the budget (top_k / kpool)
    top: usize,
}

impl AttnLayer {
    /// Loads the layer's indexer when the file has one (qwen4exp), with its geometry from the metadata.
    pub fn load_indexer(&mut self, w: &Weights, h: &Hparams) -> Result<()> {
        let l = self.l;
        if w.tensor(&format!("blk.{l}.indexer.k_proj.weight")).is_none() {
            return Ok(());
        }
        let m = format::meta::Meta::parse(&w.file.metadata)?;
        let need = |k: &str| m.arch_u32(k).ok_or_else(|| format::Error(format!("the indexer's geometry: missing {k}")));
        let (heads, dim, topk) = (
            need("attention.indexer.head_count")? as usize,
            need("attention.indexer.key_length")? as usize,
            need("attention.indexer.top_k")? as usize,
        );
        let kpool = match m.ints(&m.arch_key("attention.compress_ratios")) {
            Some(v) => v.iter().copied().filter(|&r| r > 0).max().unwrap_or(0) as usize,
            None => m.arch_u32("attention.compress_ratios").unwrap_or(0) as usize,
        };
        if heads == 0 || dim == 0 || dim % 2 != 0 || kpool < 2 || topk % kpool != 0 || h.n_rot as usize > dim {
            return err("the indexer's geometry: inconsistent");
        }
        let dm = h.d_model as usize;
        let f = |s: &str| w.layer(l, s).and_then(|t| t.f32s());
        self.ix = Some(Indexer {
            q: crate::hc::Bf16::load(w, &format!("blk.{l}.indexer.q_proj.weight"), dm, heads * dim)?,
            k: crate::hc::Bf16::load(w, &format!("blk.{l}.indexer.k_proj.weight"), dm, dim)?,
            qn: f("indexer.q_norm.weight")?,
            kn: f("indexer.k_norm.weight")?,
            heads,
            dim,
            kpool,
            top: topk / kpool,
        });
        Ok(())
    }
}

/// The higher score first, then the earlier pool.
fn ix_before(s: &[f32], x: u32, y: u32) -> bool {
    s[x as usize] > s[y as usize] || (s[x as usize] == s[y as usize] && x < y)
}

/// Puts the k best of v[0..n) into v[0..k), unordered, by a quickselect with a fixed pivot rule, so ties land the same
/// way on every run.
fn ix_select(v: &mut [u32], n: usize, k: usize, s: &[f32]) {
    let (mut lo, mut hi, want) = (0usize, n, k - 1);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        v.swap(mid, hi - 1);
        let piv = v[hi - 1];
        let mut st = lo;
        for i in lo..hi - 1 {
            if ix_before(s, v[i], piv) {
                v.swap(i, st);
                st += 1;
            }
        }
        v.swap(st, hi - 1);
        if st == want {
            return;
        }
        if st < want {
            lo = st + 1;
        } else {
            hi = st;
        }
    }
}

/// The indexer's work for n tokens at pos0.. in a layer, before its attention: the raw and pooled keys and, past the
/// budget, each token's pick of pools into kv.sel / kv.nv (setting kv.qsa). xn is the layer's input at bf16.
pub fn qsa_prepare(w: &Weights, h: &Hparams, ly: &AttnLayer, kv: &mut KvCache, xn: &[f32], n: usize, pos0: u32) {
    let t = std::time::Instant::now();
    qsa_prepare_in(w, h, ly, kv, xn, n, pos0);
    ATTN_NS[INDEXER].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

/// qsa_prepare for the MTP block, timed apart from the model's own indexer.
pub fn qsa_prepare_mtp(w: &Weights, h: &Hparams, ly: &AttnLayer, kv: &mut KvCache, xn: &[f32], n: usize, pos0: u32) {
    let t = std::time::Instant::now();
    qsa_prepare_in(w, h, ly, kv, xn, n, pos0);
    ATTN_NS[MTP_INDEXER].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

#[allow(clippy::too_many_arguments)]
fn qsa_prepare_in(w: &Weights, h: &Hparams, ly: &AttnLayer, kv: &mut KvCache, xn: &[f32], n: usize, pos0: u32) {
    kv.qsa = false;
    let Some(ix) = &ly.ix else { return };
    let (dm, dim, kp, nh) = (h.d_model as usize, ix.dim, ix.kpool, ix.heads);
    let (cap, p0) = (kv.cap, pos0 as usize);
    if p0 + n > cap {
        return;
    }
    let npool = cap / kp + 1;
    if kv.ix_raw.is_empty() {
        kv.ix_raw = vec![0; cap * dim];
        kv.ix_pk = vec![0.0; npool * dim];
        kv.ix_kp = kp;
        kv.ix_npool = npool;
    }
    let eps = h.eps;
    // the raw keys, from the layer's input at bf16, stored at f16 in the cache
    let mut kr = vec![0.0f32; n * dim];
    crate::hc::bf16_mm(ix.k.get(w), dim, dm, xn, &mut kr, n);
    for j in 0..n {
        for i in 0..dim {
            kv.ix_raw[(p0 + j) * dim + i] = kops::fp32_to_fp16(kr[j * dim + i]);
        }
    }
    // the pools the block completes: the members summed in order, scaled, normed and rotated, stored at f16
    let n_rot = h.n_rot as usize;
    let half = n_rot / 2;
    let (mut cs, mut sn) = (vec![0.0f32; half], vec![0.0f32; half]);
    let mut acc = vec![0.0f32; dim];
    let mut nrm = vec![0.0f32; dim];
    for j in 0..n {
        let p = p0 + j;
        if p % kp != kp - 1 {
            continue;
        }
        let q = p / kp;
        let r = &kv.ix_raw[q * kp * dim..(q + 1) * kp * dim];
        for i in 0..dim {
            acc[i] = kernels::f16_to_f32(r[i]);
        }
        for k in 1..kp {
            for i in 0..dim {
                acc[i] += kernels::f16_to_f32(r[k * dim + i]);
            }
        }
        for v in acc.iter_mut() {
            *v *= 1.0f32 / kp as f32;
        }
        crate::hc::norm_gg(&mut nrm, &acc, &ix.kn, eps);
        crate::ops::rope_row(&mut cs, &mut sn, (q * kp) as u32, h.n_rot, h.rope_base);
        rope_neox(&mut nrm, n_rot, &cs, &sn);
        for i in 0..dim {
            kv.ix_pk[q * dim + i] = kernels::f16_to_f32(kops::fp32_to_fp16(nrm[i]));
        }
    }
    let nv_max = (p0 + n) / kp;
    if nv_max <= ix.top {
        return;
    }
    // past the budget: the queries, every complete pool's score, and each token's pick
    let mut qv = vec![0.0f32; n * nh * dim];
    crate::hc::bf16_mm(ix.q.get(w), nh * dim, dm, xn, &mut qv, n);
    for j in 0..n {
        crate::ops::rope_row(&mut cs, &mut sn, (p0 + j) as u32, h.n_rot, h.rope_base);
        for hh in 0..nh {
            let v = &mut qv[(j * nh + hh) * dim..(j * nh + hh + 1) * dim];
            nrm.copy_from_slice(v);
            crate::hc::norm_gg(v, &nrm, &ix.qn, eps);
            rope_neox(v, n_rot, &cs, &sn);
        }
    }
    kv.sel.resize(n * npool, 0);
    kv.nv.resize(n, 0);
    kv.ix_sc.resize(n * npool, 0.0);
    let wsc = 1.0f32 / (dim as f32).sqrt();
    let (top, pk) = (ix.top, &kv.ix_pk);
    // the scores, one item per (token, chunk of QSA_CHUNK pools), so a decode token's pools spread over every thread
    // (with one item per token, a single thread scored all of them, and the cost grew with the context on one core).
    // Each pool's score is the same sum in the same order, so the picks are the same.
    const QSA_CHUNK: usize = 256;
    let chunks = (p0 + n) / kp / QSA_CHUNK + 1;
    let scp = SyncPtr::new(&mut kv.ix_sc);
    crate::pool::par_for(n * chunks, &|it| {
        let (j, ch) = (it / chunks, it % chunks);
        let nv = (p0 + j + 1) / kp;
        if nv <= top || ch * QSA_CHUNK >= nv {
            return;
        }
        let (a, b) = (ch * QSA_CHUNK, (ch * QSA_CHUNK + QSA_CHUNK).min(nv));
        let q = &qv[j * nh * dim..(j + 1) * nh * dim];
        // SAFETY: item (j, ch) owns pools a..b of token j's score row
        let s = unsafe { scp.slice(j * npool + a, b - a) };
        for (p, sc) in (a..b).zip(s.iter_mut()) {
            let k = &pk[p * dim..(p + 1) * dim];
            let mut sum = 0.0f32;
            for hh in 0..nh {
                let d = kops::dot_f32(&q[hh * dim..(hh + 1) * dim], k);
                sum += wsc * if d > 0.0 { d } else { 0.0 };
            }
            *sc = sum;
        }
    });
    let (sp, np) = (SyncPtr::new(&mut kv.sel), SyncPtr::new(&mut kv.nv));
    let sc = &kv.ix_sc;
    crate::pool::par_for(n, &|j| {
        let nv = (p0 + j + 1) / kp;
        // SAFETY: item j owns token j's selection row and count
        let (sel, nvo) = unsafe { (sp.slice(j * npool, npool), np.slice(j, 1)) };
        if nv <= top {
            nvo[0] = u32::MAX;
            return;
        }
        let s = &sc[j * npool..j * npool + nv];
        let mut ord: Vec<u32> = (0..nv as u32).collect();
        ix_select(&mut ord, nv, top, s);
        sel[..nv].fill(0);
        for &o in &ord[..top] {
            sel[o as usize] = 1;
        }
        nvo[0] = nv as u32;
    });
    kv.qsa = true;
}

/// The last qsa_prepare's selection for its queries s.., in the form the attention kernel takes (None if there is
/// none).
fn select_of(kv: &KvCache, s: usize) -> Option<crate::fattn::Sel<'_>> {
    kv.qsa.then(|| crate::fattn::Sel {
        rows: &kv.sel[s * kv.ix_npool..],
        nv: &kv.nv[s..],
        stride: kv.ix_npool,
        kp: kv.ix_kp,
    })
}

#[cfg(test)]
mod kv_bytes_tests {
    use super::KvCache;

    /// A cache with every position filled and, when `dim` > 0, an indexer of pools of 4 laid out as qsa_prepare lays it
    /// out (raw keys cap x dim, pooled keys (cap / 4 + 1) x dim).
    fn cache(k16: bool, nh: usize, cap: usize, hd: usize, dim: usize) -> KvCache {
        let n = nh * cap * hd;
        let kp = 4;
        KvCache {
            k: if k16 { vec![] } else { (0..n).map(|i| (i % 251) as i8).collect() },
            v: if k16 { vec![] } else { (0..n).map(|i| (i % 241) as i8).collect() },
            k16: if k16 { (0..n).map(|i| (i % 30011) as i16).collect() } else { vec![] },
            v16: if k16 { (0..n).map(|i| (i % 30013) as i16 - 9000).collect() } else { vec![] },
            k_scale: (0..nh * cap).map(|i| i as f32 * 0.5).collect(),
            v_scale: (0..nh * cap).map(|i| i as f32 * 0.25).collect(),
            cap,
            ix_raw: (0..cap * dim).map(|i| (i * 7 + 1) as u16).collect(),
            ix_pk: (0..(cap / kp + 1) * dim).map(|i| i as f32 + 0.5).collect(),
            ix_kp: if dim > 0 { kp } else { 0 },
            ix_npool: if dim > 0 { cap / kp + 1 } else { 0 },
            sel: vec![],
            nv: vec![],
            qsa: false,
            ix_sc: Vec::new(),
            gpu_lo: 0,
        }
    }

    fn zeroed(mut z: KvCache) -> KvCache {
        for x in [&mut z.k, &mut z.v] {
            x.iter_mut().for_each(|v| *v = 0);
        }
        for x in [&mut z.k16, &mut z.v16] {
            x.iter_mut().for_each(|v| *v = 0);
        }
        z.k_scale.iter_mut().for_each(|v| *v = 0.0);
        z.v_scale.iter_mut().for_each(|v| *v = 0.0);
        z.ix_raw.clear();
        z.ix_pk.clear();
        z
    }

    fn reader(b: &[u8]) -> (impl FnMut(usize) -> Vec<u8> + '_, std::rc::Rc<std::cell::Cell<usize>>) {
        let at = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let a = at.clone();
        (
            move |n| {
                let e = (a.get() + n).min(b.len());
                let r = b[a.get()..e].to_vec();
                a.set(e);
                r
            },
            at,
        )
    }

    /// Saved from a cache of `cap` positions and loaded into one of `cap2` (a different context size).
    fn round_trip(k16: bool, cap: usize, cap2: usize) {
        let (nh, hd, npos, dim) = (2, 8, 11, 3);
        let a = cache(k16, nh, cap, hd, dim);
        let mut b = Vec::new();
        a.put_bytes(hd, npos, &mut b);
        let mut z = zeroed(cache(k16, nh, cap2, hd, dim));
        let (mut take, at) = reader(&b);
        z.set_bytes(hd, npos, &mut take).unwrap();
        assert_eq!(at.get(), b.len());
        // positions 0..npos of every head match, and the rest stay as they were (zero here)
        for h in 0..nh {
            let (r, r2) = (h * cap * hd..(h * cap + npos) * hd, h * cap2 * hd..(h * cap2 + npos) * hd);
            if k16 {
                assert_eq!(a.k16[r.clone()], z.k16[r2.clone()]);
                assert_eq!(a.v16[r], z.v16[r2]);
            } else {
                assert_eq!(a.k[r.clone()], z.k[r2.clone()]);
                assert_eq!(a.v[r], z.v[r2]);
            }
            assert_eq!(a.k_scale[h * cap..h * cap + npos], z.k_scale[h * cap2..h * cap2 + npos]);
            assert!(z.k_scale[h * cap2 + npos..(h + 1) * cap2].iter().all(|&v| v == 0.0));
        }
        // the indexer in this cache's own layout: the saved positions' keys and complete pools, and zeros past them
        assert_eq!(
            (z.ix_raw.len(), z.ix_pk.len(), z.ix_kp, z.ix_npool),
            (cap2 * dim, (cap2 / 4 + 1) * dim, 4, cap2 / 4 + 1)
        );
        assert_eq!(a.ix_raw[..npos * dim], z.ix_raw[..npos * dim]);
        assert!(z.ix_raw[npos * dim..].iter().all(|&v| v == 0));
        let pools = npos / 4 * dim;
        assert_eq!(a.ix_pk[..pools], z.ix_pk[..pools]);
        assert!(z.ix_pk[pools..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn put_set_round_trip() {
        for k16 in [false, true] {
            round_trip(k16, 16, 16);
        }
    }

    #[test]
    fn loads_into_another_context_size() {
        for k16 in [false, true] {
            round_trip(k16, 16, 24);
            round_trip(k16, 24, 16);
        }
    }

    #[test]
    fn no_indexer_round_trip() {
        let (nh, hd, npos) = (2, 8, 11);
        let a = cache(true, nh, 16, hd, 0);
        let mut b = Vec::new();
        a.put_bytes(hd, npos, &mut b);
        let mut z = zeroed(cache(true, nh, 16, hd, 3));
        let (mut take, at) = reader(&b);
        z.set_bytes(hd, npos, &mut take).unwrap();
        assert_eq!(at.get(), b.len());
        assert!(z.ix_raw.is_empty() && z.ix_pk.is_empty() && z.ix_kp == 0 && z.ix_npool == 0);
    }

    #[test]
    fn short_or_oversized_is_an_error() {
        let (nh, hd, npos) = (2, 8, 11);
        let a = cache(false, nh, 16, hd, 3);
        let mut b = Vec::new();
        a.put_bytes(hd, npos, &mut b);
        let mut z = zeroed(cache(false, nh, 16, hd, 3));
        let (mut take, _) = reader(&b[..b.len() - 1]);
        assert!(z.set_bytes(hd, npos, &mut take).is_err());
        // more positions than the cache holds
        let mut small = zeroed(cache(false, nh, 8, hd, 3));
        let (mut take, _) = reader(&b);
        assert!(small.set_bytes(hd, npos, &mut take).is_err());
    }
}
