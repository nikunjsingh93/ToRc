//! forward.rs - the model and its forward pass.
//!
//! The embedding first; then, for each layer, the attention norm (attn_norm) and Q8, the mixer, the FFN norm
//! (post_attention_norm for qwen35moe) and Q8, and the experts; and finally the output norm, Q8 and the vocabulary
//! product. Decode goes a token at a time and prompts go in blocks of tokens, and both give the same bits.

use crate::attn::{AttnEngine, AttnLayer, KvCache, attn_token};
use crate::hparams::Hparams;
use crate::moe::{Mat, MoeLayer, Scratch, moe_token};
use crate::ops::{rms_norm, rope_row};
use crate::rec::{RecLayer, RecState, rec_token};
use crate::weights::Weights;
use format::{BlockKind, Result, err};
use kernels::{Q8, Q8m, Q8s, QK, ops as kops};
use std::io::Write;

// a layer's mixer lives as long as the model, one per layer: its size doesn't matter, and boxing it would add a
// pointer chase on every token
#[allow(clippy::large_enum_variant)]
enum Mixer {
    Rec(RecLayer, RecState),
    Attn(AttnLayer, KvCache),
}

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    mixer: Mixer,
    moe: MoeLayer,
    /// qwen4exp: the layer's hyper-connection mixers (there are no norms then)
    hc: Option<crate::hc::LayerHc>,
}

/// One layer's state as Model::layer_states sends it: for attention, K and V (int8) and their scales for positions
/// 0..npos, head by head; for DeltaNet, the convolution window and the recurrent state.
fn put_layer_state(ly: &Layer, hd: usize, npos: usize, b: &mut Vec<u8>) {
    match &ly.mixer {
        Mixer::Attn(_, kv) => kv.put_bytes(hd, npos, b),
        Mixer::Rec(_, st) => {
            for f in &st.conv {
                b.extend(f.to_le_bytes());
            }
            for v in &st.ssm {
                b.extend(v.to_le_bytes());
            }
        }
    }
}

/// Seconds spent in each phase (TORC_PROF=1): the mixer (DeltaNet), attention, MoE, output and other.
#[derive(Default, Debug, Clone, Copy)]
pub struct Prof {
    pub rec: f64,
    pub attn: f64,
    pub moe: f64,
    pub out: f64,
    pub norm: f64,
}

/// Normalises and quantises each of n rows (x and xn are n x d_model, xq is n x d_model / QK), over the pool for 2 rows
/// or more. Each row makes the same calls on whichever thread runs it, so the bits are the same.
fn norm_rows(h: &Hparams, xn: &mut [f32], mut xq: Q8m, x: &[f32], wn: &[f32], n: usize) {
    let dm = h.d_model as usize;
    let nb = dm / QK;
    if n < 2 {
        for t in 0..n {
            rms_norm(&mut xn[t * dm..(t + 1) * dm], &x[t * dm..(t + 1) * dm], wn, h.eps);
            kops::q8k_quantize(&xn[t * dm..(t + 1) * dm], xq.sub(t * nb..(t + 1) * nb));
        }
        return;
    }
    let (np, qp) = (crate::pool::SyncPtr::new(xn), crate::pool::Q8Ptr::new(xq));
    crate::pool::par_for(n, &|t| {
        // SAFETY: item t owns row t of xn and of xq
        let (xr, qr) = unsafe { (np.slice(t * dm, dm), qp.slice(t * nb, nb)) };
        rms_norm(xr, &x[t * dm..(t + 1) * dm], wn, h.eps);
        kops::q8k_quantize(xr, qr);
    });
}

pub struct Model {
    pub prof: Prof,
    pub h: Hparams,
    pub w: Weights,
    layers: Vec<Layer>,
    out_norm: Vec<f32>,
    eng: AttnEngine,
    scratch: Scratch,
    n_ctx: u32,
    /// the MTP block, when the file has one
    mtp: Option<crate::mtp::MtpBlock>,
    /// qwen4exp's MTP block (TORC_MTP_FILE): its drafts read the streams before the final mixer (hidden is then n x hc
    /// d_model)
    mtp_hc: Option<crate::mtp_hc::MtpHc>,
    /// the out-normed hidden rows of the last prefill or verify that produced logits (n x d_model)
    pub hidden: Vec<f32>,
    /// log_hidden: every prefill block's hidden rows appended (the first one's position, then the rows), with or
    /// without logits - a server's prompt, for the MTP block's rows over it
    hidden_log: Option<(u32, Vec<f32>)>,
    /// qwen4exp: the head's mixer, the n-gram layer and the streams (hc.rs)
    hcm: Option<crate::hc::HcModel>,
    /// the other architectures' residual rows between the pieces of a pass (std_embed / std_layers / std_head)
    px: Vec<f32>,
    /// the pipeline worker's layers were pulled back here (pull_states), so passes run on this machine alone
    pipe_off: bool,
    /// the next prefill_blocks_at is the prompt's last stretch before a pull: its last block brings the worker's layer
    /// states back as they are done (pipe::WANT_STATES), and the pull then has nothing left to do
    pub stream_pull: bool,
    /// this sequence's split (0 until it is chosen). It is fixed from the sequence's first pass until the next
    /// clear_states, since each machine holds the state of its own layers (the saved split that a prompt rewrites
    /// steers the next sequence)
    psplit: std::cell::Cell<usize>,
    /// when half_send ran, for the round trip half_recv adds to the profile
    half_t0: Option<std::time::Instant>,
    /// half_send / half_recv: only the top K logits come back (0 for the whole row) - 1 for greedy, or the sampler's
    /// top_k for top-k sampling (pipe.rs TOPK_SHIFT); TORC_PIPE_TOPK sets it for torc-run
    pub half_topk: usize,
    /// TORC_HEAD_SPLIT=f (0 < f < 1): the main machine computes that share of the head's rows itself (pipe::HEAD_SPLIT)
    pub head_split: u32,
    /// the rows 0.. the main machine computes for the request in flight (0 for none)
    half_rows: usize,
    /// a pipeline worker's marks (OP_SAVE / OP_LOAD), by slot (pipe::MARK_SLOTS)
    wmark: [Option<Vec<(Vec<f32>, Vec<u16>)>>; crate::pipe::MARK_SLOTS],
}

/// A sequence's state at one point, to resume from there (save_mark / load_mark): the split, this machine's DeltaNet
/// states, and whether it was taken through a pipeline worker (whose half is then in its slot of the same number).
pub type Mark = (usize, Vec<(Vec<f32>, Vec<u16>)>, bool);

/// Read-ahead on single tokens: the next layer's gate applied to this layer's state guesses which experts it will
/// want, since each layer only adds to the residual stream (Eliseev and Mazur, "Fast Inference of Mixture-of-Experts
/// Language Models with Offloading", 2023). TORC_READ_AHEAD=rank,reads: the `rank` best guesses are asked for, and at
/// most `reads` reads started. Unset (or either number 0) turns it off. Returns (rank, reads).
fn read_ahead() -> (usize, usize) {
    static P: std::sync::LazyLock<(usize, usize)> = std::sync::LazyLock::new(|| {
        crate::switches::get("TORC_READ_AHEAD")
            .and_then(|v| {
                let (r, b) = v.split_once(',')?;
                Some((r.trim().parse().ok()?, b.trim().parse().ok()?))
            })
            .unwrap_or((0, 0))
    });
    *P
}

/// qwen4exp read-ahead on prompt blocks: after a block's experts in one layer, the next layer's likeliest experts (each
/// of up to `tokens` sampled tokens votes for its `rank` best by that layer's router) are read into the cache while
/// that layer mixes, at most `max` of them. TORC_READ_AHEAD_BLOCK=rank,max[,tokens] (tokens defaults to 32); unset
/// turns it off. The SSD is otherwise idle while a layer computes.
fn block_read_ahead() -> Option<(usize, usize, usize)> {
    static P: std::sync::LazyLock<Option<(usize, usize, usize)>> = std::sync::LazyLock::new(|| {
        let v = crate::switches::get("TORC_READ_AHEAD_BLOCK")?;
        let f: Vec<usize> = v.split(',').map(|x| x.trim().parse().ok()).collect::<Option<Vec<_>>>()?;
        match f.as_slice() {
            [r, m] if *r > 0 && *m > 0 => Some((*r, *m, 32)),
            [r, m, t] if *r > 0 && *m > 0 && *t > 0 => Some((*r, *m, *t)),
            _ => None,
        }
    });
    *P
}

impl Model {
    pub fn load(path: &str, n_ctx: u32) -> Result<Model> {
        // TORC_PROF=1: where the load's time goes
        let tl = std::time::Instant::now();
        let say = |what: &str| {
            if crate::switches::get("TORC_PROF").is_some_and(|v| v != "0") {
                eprintln!("load: {what} at {:.1} s", tl.elapsed().as_secs_f64());
            }
        };
        // before the weights load: the expert cache and the NPU load only this machine's layers on a pipeline
        // (pipe::own_layers), which needs to know the model pipelines and whether the file is a shard
        crate::pipe::set_piped();
        crate::pipe::set_shard(path);
        let w = Weights::load(path)?;
        say("weights");
        let h = Hparams::from_file(&w.file)?;
        crate::tune::set_arch(&h.arch);
        crate::tune::set_resident(w.cache.is_none());
        // the tiled attention path is this architecture's default on ARM (fattn::set_tile_default)
        crate::fattn::set_tile_default(h.arch == "qwen35moe");
        crate::pipe::set_shape(h.n_layer, h.d_model, h.hc);
        crate::pipe::set_ident(
            &w.file.header.model_id,
            w.file.header.file_bytes,
            h.n_layer,
            h.d_model,
            h.hc,
            w.file.tensors.len(),
            n_ctx,
        );
        let mut layers = Vec::with_capacity(h.n_layer as usize);
        for l in 0..h.n_layer {
            let group = w
                .file
                .blocks
                .iter()
                .find(|b| b.layer == l && b.kind == BlockKind::Moe && b.experts.is_some())
                // a dense model (qwen35): its FFN as the layer's one expert, recorded on the mixer's block
                .or_else(|| w.file.blocks.iter().find(|b| b.layer == l && b.experts.is_some()))
                .and_then(|b| b.experts)
                .ok_or_else(|| format::Error(format!("layer {l}: no routed experts")))?;
            let mixer = if h.is_attention(l) {
                // deepseek2: multi-head latent attention (mla.rs)
                let mut a = if h.mla.is_some() { AttnLayer::load_mla(&w, &h, l)? } else { AttnLayer::load(&w, l)? };
                if h.hc > 0 {
                    a.load_indexer(&w, &h)?;
                }
                Mixer::Attn(a, KvCache::new(&h, n_ctx as usize))
            } else {
                Mixer::Rec(RecLayer::load(&w, &h, l)?, RecState::new(&h))
            };
            let hc = h.hc > 0;
            let mut moe = MoeLayer::load(&w, l, group)?;
            moe.hc = hc;
            // deepseek2's gating (sigmoid, renormalised, scaled), for routed layers only: a dense layer's one expert
            // has no router and keeps its weight of 1
            if w.layer_experts(group).n_expert > 1 {
                (moe.sigmoid, moe.norm_w, moe.w_scale) = (h.gate_sigmoid, h.norm_w, h.w_scale);
            }
            layers.push(Layer {
                attn_norm: if hc { Vec::new() } else { w.layer(l, "attn_norm.weight")?.f32s()? },
                // qwen35moe names the FFN norm post_attention_norm, qwen3moe ffn_norm
                ffn_norm: if hc {
                    Vec::new()
                } else {
                    w.layer(l, "post_attention_norm.weight").or_else(|_| w.layer(l, "ffn_norm.weight"))?.f32s()?
                },
                mixer,
                moe,
                hc: if hc { Some(crate::hc::LayerHc::load(&w, &h, l)?) } else { None },
            });
        }
        // TORC_NPU=1 (ARM64 Windows): the dense products of big prompt blocks, and the hot experts, go on the NPU
        // (npu.rs)
        crate::npu::load(&w, h.n_layer, &layers.iter().map(|ly| ly.moe.group).collect::<Vec<_>>());
        // lfm2moe names its final norm token_embd_norm
        let out_norm = if h.hc > 0 {
            Vec::new()
        } else {
            w.tensor("output_norm.weight")
                .or_else(|| w.tensor("token_embd_norm.weight"))
                .ok_or_else(|| format::Error("missing tensor output_norm.weight".into()))?
                .f32s()?
        };
        let hcm = if h.hc > 0 { Some(crate::hc::HcModel::load(&w, &h)?) } else { None };
        say("layers");
        let eng = AttnEngine::new(&h)?;
        let ff_max = w.file.experts.iter().map(|x| x.width as usize).max().unwrap_or(0).max(
            layers
                .iter()
                .filter_map(|ly| ly.moe.shared.as_ref())
                .map(|s| w.need(&s[0]).map(|t| t.shape().1))
                .next()
                .transpose()?
                .unwrap_or(0),
        );
        let scratch = Scratch::new(h.n_expert as usize, h.d_model as usize, ff_max);
        say("engine + scratch");
        let (mtp, mtp_hc) = if h.hc > 0 {
            (None, crate::mtp_hc::MtpHc::load(&w, &h, n_ctx as usize)?)
        } else {
            // deepseek2's next-token block is not run (yet): its layer is MLA, which mtp.rs does not build
            (if h.mla.is_some() { None } else { crate::mtp::MtpBlock::load(&w, &h, n_ctx as usize)? }, None)
        };
        say("mtp");
        Ok(Model {
            prof: Prof::default(),
            h,
            w,
            layers,
            out_norm,
            eng,
            scratch,
            n_ctx,
            mtp,
            mtp_hc,
            hidden: Vec::new(),
            hidden_log: None,
            px: Vec::new(),
            pipe_off: false,
            stream_pull: false,
            psplit: std::cell::Cell::new(0),
            half_t0: None,
            half_topk: crate::switches::num("TORC_PIPE_TOPK").unwrap_or(0),
            head_split: crate::switches::num::<f64>("TORC_HEAD_SPLIT")
                .filter(|f| *f > 0.0 && *f < 1.0)
                .map_or(0, |f| ((f * 64.0).round() as u32).clamp(1, 63)),
            half_rows: 0,
            wmark: Default::default(),
            hcm,
        })
    }

    /// A revertible block (MTP verification): a prefill of `tokens` at pos0 (with all their logits when `logits` is not
    /// empty), with the recurrent layers logging it so that `commit` can keep just a prefix. The KV rows past what is
    /// kept are simply rewritten by the next pass, since attention never reads a row before it is rewritten.
    pub fn verify(&mut self, tokens: &[u32], pos0: u32, logits: &mut [f32]) -> Result<()> {
        // a reply step: the decode width (pool::Width), fewer threads where that writes faster
        let _w = crate::pool::Width::decode();
        if tokens.len() > crate::rec::LOG_MAX {
            return err("verify block longer than the rollback log");
        }
        // with a pipeline worker (pipe.rs), this machine's layers end at the split and the worker's layers begin their
        // block there
        let piped = self.pipe().is_some();
        let s = if piped { self.split() } else { self.layers.len() };
        for ly in &mut self.layers[..s] {
            if let Mixer::Rec(_, st) = &mut ly.mixer {
                st.begin();
            }
        }
        if piped {
            crate::pipe::control(s, crate::pipe::OP_BEGIN)?;
        }
        let r = self.prefill(tokens, pos0, logits);
        if r.is_err() {
            // a failed block keeps nothing: the state stays as it was before it
            self.commit(0);
        }
        r
    }

    /// Keeps the first `kept` tokens of the last verify block, leaving the recurrent state as if only they had been
    /// read (bit for bit: the rolled-back layers replay them from the log).
    pub fn commit(&mut self, kept: usize) {
        let h = self.h.clone();
        if self.pipe().is_some()
            && let Err(e) =
                crate::pipe::control(self.split(), crate::pipe::OP_COMMIT | (kept as u32) << crate::pipe::KEPT_SHIFT)
        {
            eprintln!("{e}");
        }
        for ly in &mut self.layers {
            if let Mixer::Rec(r, st) = &mut ly.mixer {
                crate::rec::commit(&h, r, st, kept);
            }
        }
    }

    /// Two-machine pipelined speculation (torc-run's pipespec), qwen4exp: this machine's half of one token at `pos`
    /// (the embedding and layers 0..split), returning its streams for half_send. Between spec_begin and spec_end the
    /// DeltaNet layers write a revertible tip (a draft's half, run while the worker has the token before it).
    pub fn half(&mut self, token: u32, pos: u32) -> Result<Vec<f32>> {
        if self.hcm.is_none() || self.pipe().is_none() {
            return err("half: qwen4exp with a pipeline worker only");
        }
        if pos >= self.n_ctx {
            return err("position past the context");
        }
        self.hc_embed(&[token])?;
        let s = self.split();
        self.hc_layers(&[token], pos, 0, s)?;
        Ok(self.hcm.as_ref().unwrap().res[..self.hc_width()].to_vec())
    }

    /// Sends the worker its half of a token, from the token's streams (half): the request is written, and the reply is
    /// left to half_recv.
    pub fn half_send(&mut self, token: u32, pos: u32, streams: &[f32]) -> Result<()> {
        let cl = self.pipe().ok_or_else(|| format::Error("half_send: no pipeline worker".into()))?;
        let mut want = 1
            | if self.mtp_hc.is_some() { crate::pipe::WANT_HIDDEN } else { 0 }
            | (self.half_topk.min(0xffff) as u32) << crate::pipe::TOPK_SHIFT;
        self.half_rows = 0;
        if self.head_split > 0 && self.half_topk > 0 {
            want |= self.head_split << crate::pipe::HEAD_SPLIT_SHIFT;
            self.half_rows = crate::pipe::split_rows(self.n_vocab()?, self.head_split);
        }
        let rq = crate::pipe::request(self.split(), pos, &[token], want, streams);
        cl.lock()
            .unwrap_or_else(|p| p.into_inner())
            .write_all(&rq)
            .map_err(|e| format::Error(format!("two boxes (pipeline): {e}")))?;
        self.half_t0 = Some(std::time::Instant::now());
        Ok(())
    }

    /// The reply to half_send: the token's logits and, with an MTP block here, its streams before the head's mixer,
    /// into self.hidden.
    pub fn half_recv(&mut self, logits: &mut [f32]) -> Result<()> {
        let cl = self.pipe().ok_or_else(|| format::Error("half_recv: no pipeline worker".into()))?;
        let mut st = cl.lock().unwrap_or_else(|p| p.into_inner());
        // the head split: the worker's early frame (the head's input) first, then rows 0..half_rows are computed here
        // while the worker does the rest; they are filled into logits after its top K below
        let mut mine: Vec<f32> = Vec::new();
        if self.half_rows > 0 {
            use std::io::Read;
            let mut lb = [0u8; 4];
            st.read_exact(&mut lb).map_err(|e| format::Error(format!("two boxes (pipeline): {e}")))?;
            let nbytes = u32::from_le_bytes(lb) as usize;
            let nb = self.h.d_model as usize / QK;
            let mut xq = Q8::new(nb);
            let mut xm = xq.view_mut();
            let (kb, gb) = xm.bytes_mut();
            if nbytes != kb.len() + gb.len() {
                return err(format!("two boxes (pipeline): head split frame of {nbytes} bytes"));
            }
            st.read_exact(kb)
                .and_then(|_| st.read_exact(gb))
                .map_err(|e| format::Error(format!("two boxes (pipeline): {e}")))?;
            let tm = std::time::Instant::now();
            let out = self.w.head()?;
            let m = Mat::of(&out);
            let rb = m.data.len() / m.rows;
            let r = self.half_rows;
            mine = vec![0.0f32; r];
            Mat { ty: m.ty, rows: r, cols: m.cols, data: &m.data[..r * rb] }.apply(xq.view(), &mut mine);
            self.prof.out += tm.elapsed().as_secs_f64();
        }
        let k = self.half_topk.min(logits.len());
        let mut kv = vec![0.0f32; 2 * k];
        let out: &mut [f32] = if k > 0 { &mut kv } else { logits };
        let r = if self.mtp_hc.is_some() {
            self.hidden.resize(self.hc_width(), 0.0);
            crate::pipe::recv_with_hidden(&mut st, out, &mut self.hidden)
        } else {
            crate::pipe::recv(&mut st, out)
        };
        if k > 0 && r.is_ok() {
            // the top K in a row of -inf: argmax and top-k sampling (with k <= K) pick exactly as they would from the
            // full row
            logits.fill(f32::NEG_INFINITY);
            for i in 0..k {
                logits[kv[k + i].to_bits() as usize] = kv[i];
            }
            // this machine's rows (all lower ids than the worker's, so its top K and these hold the whole row's top K)
            logits[..mine.len()].copy_from_slice(&mine);
        }
        if let Some(t0) = self.half_t0.take() {
            crate::pipe::trip_add(t0.elapsed().as_micros() as u64);
        }
        r
    }

    /// A revertible step on this machine's layers only (layers 0..split; the worker never sees a draft).
    pub fn spec_begin(&mut self) {
        let s = self.split();
        for ly in &mut self.layers[..s] {
            if let Mixer::Rec(_, st) = &mut ly.mixer {
                st.begin();
            }
        }
    }

    /// Keeps the step since spec_begin (the draft was the right token) or drops it (leaving the state as before it).
    pub fn spec_end(&mut self, keep: bool) {
        let h = self.h.clone();
        let s = self.split();
        for ly in &mut self.layers[..s] {
            if let Mixer::Rec(r, st) = &mut ly.mixer {
                crate::rec::commit(&h, r, st, keep as usize);
            }
        }
    }

    /// Whether pipelined speculation (half / half_send / spec_begin ..) can run: it needs qwen4exp's MTP block here and
    /// a pipeline worker.
    pub fn can_pipespec(&self) -> bool {
        self.mtp_hc.is_some() && self.hcm.is_some() && self.pipe().is_some()
    }

    /// From now on every prefill (on one machine, through the block path) keeps its hidden rows in a log, with or
    /// without logits, until take_hidden_log. This gives the MTP block its rows over a prompt read without logits
    /// (torc-server). The positions must follow on from each other; a block that does not starts the log again.
    pub fn log_hidden(&mut self) {
        self.hidden_log = Some((0, Vec::new()));
    }

    /// Returns the logged rows (their first position, then n x d_model) and stops the log.
    pub fn take_hidden_log(&mut self) -> Option<(u32, Vec<f32>)> {
        self.hidden_log.take()
    }

    /// Takes the hidden-row log back to position `pos` (a prompt resumed from a checkpoint, where the rows after it are
    /// read again), so it stays one run of rows from its start.
    pub fn truncate_hidden_log(&mut self, pos: u32) {
        let d = self.hidden_width();
        if let Some((s, log)) = self.hidden_log.as_mut() {
            if pos >= *s {
                log.truncate((pos - *s) as usize * d);
            } else {
                log.clear();
                *s = pos;
            }
        }
    }

    /// Whether the MTP block is the other architectures' kind (mtp.rs) rather than qwen4exp's.
    pub fn mtp_std(&self) -> bool {
        self.mtp.is_some()
    }

    /// Whether qwen4exp's MTP block (TORC_MTP_FILE) is loaded.
    pub fn mtp_hc_loaded(&self) -> bool {
        self.mtp_hc.is_some()
    }

    /// Runs only the MTP block's keys and values for the rows tokens[j] / hid[j] at pos0 + j (mtp::MtpBlock::fill).
    pub fn mtp_fill(&mut self, tokens: &[u32], hid: &[f32], pos0: u32) -> Result<()> {
        let Model { mtp, w, h, eng, .. } = self;
        match mtp.as_mut() {
            Some(b) => b.fill(w, h, eng, tokens, hid, pos0),
            None => err("no MTP block in this file"),
        }
    }

    /// The number of floats in a row of `hidden`: the streams (hc x d_model) with qwen4exp's MTP block, else d_model.
    pub fn hidden_width(&self) -> usize {
        if self.mtp_hc.is_some() { self.hc_width() } else { self.h.d_model as usize }
    }

    /// MTP rows (see mtp::MtpBlock::forward): tokens[j] with main hidden rows hid[j] at positions pos0 + j. Returns the
    /// MTP hidden rows, and fills `logits` (empty, or draft_width() long) with the last row's draft logits.
    pub fn mtp(&mut self, tokens: &[u32], hid: &[f32], pos0: u32, logits: &mut [f32]) -> Result<Vec<f32>> {
        let _w = crate::pool::Width::decode();
        // with a pipeline worker, the main hidden rows (for qwen4exp, the streams before the head's mixer) come back
        // with the logits (pipe.rs WANT_HIDDEN), so the MTP block runs here either way
        let Model { mtp, mtp_hc, w, h, eng, .. } = self;
        if let Some(b) = mtp_hc.as_mut() {
            return b.forward(w, h, eng, tokens, hid, pos0, logits);
        }
        match mtp.as_mut() {
            Some(b) => b.forward(w, h, eng, tokens, hid, pos0, logits),
            None => err("no MTP block in this file"),
        }
    }

    /// Notes the ids seen in the prompt or the reply for the draft head (mtp::MtpBlock::note).
    pub fn mtp_note(&mut self, ids: &[u32]) -> Result<()> {
        if let Some(b) = self.mtp_hc.as_mut() {
            return b.note(&self.w, ids);
        }
        match self.mtp.as_mut() {
            Some(b) => b.note(&self.w, ids),
            None => err("no MTP block in this file"),
        }
    }

    /// The current width of a draft logit row (DRAFT_V plus the noted ids; 0 without an MTP block).
    pub fn draft_width(&self) -> usize {
        if let Some(b) = &self.mtp_hc {
            return b.draft_width();
        }
        self.mtp.as_ref().map_or(0, |b| b.draft_width())
    }

    /// The token id of draft logit index i.
    pub fn draft_id(&self, i: usize) -> u32 {
        if let Some(b) = &self.mtp_hc {
            return b.draft_id(i);
        }
        self.mtp.as_ref().map_or(i as u32, |b| b.draft_id(i))
    }

    pub fn n_vocab(&self) -> Result<usize> {
        Ok(self.w.head()?.shape().1)
    }

    /// The logits of `token` at position `pos` (positions must go in order from 0).
    pub fn decode(&mut self, token: u32, pos: u32, logits: &mut [f32]) -> Result<()> {
        let _w = crate::pool::Width::decode();
        if pos >= self.n_ctx {
            return err("position past the context");
        }
        if self.hcm.is_some() {
            return self.forward_hc(&[token], pos, logits, false);
        }
        // a pipeline worker holds the later layers, so the token goes as a block of one (the same bits as the token
        // path)
        if self.pipe().is_some() {
            return self.prefill(&[token], pos, logits);
        }
        let h = self.h.clone();
        let dm = h.d_model as usize;
        let emb = self.w.need("token_embd.weight")?;
        let rb = emb.row_bytes();
        if token as usize >= emb.shape().1 {
            return err("token out of range");
        }
        let mut x = vec![0.0f32; dm];
        kops::dequantize(emb.t.ty, &emb.data[token as usize * rb..(token as usize + 1) * rb], &mut x);
        let half = (h.n_rot / 2) as usize;
        let (mut cs, mut sn) = (vec![0.0f32; half], vec![0.0f32; half]);
        crate::vision::rope_slot(&mut cs, &mut sn, pos, h.n_rot, h.rope_base);
        let mut xn = vec![0.0f32; dm];
        let mut xq = Q8::new(dm / QK);
        let trace = crate::switches::get("TORC_TRACE").is_some_and(|v| v != "0");
        use std::time::Instant;
        let mut t = Instant::now();
        let lap = |acc: &mut f64, t: &mut Instant| {
            let n = Instant::now();
            *acc += (n - *t).as_secs_f64();
            *t = n;
        };
        let mut p = self.prof;
        // TORC_READ_AHEAD: after a layer's token mixer, the next layer's likely experts are asked for (see read_ahead)
        let (ahead_rank, ahead_reads) = read_ahead();
        let mut xp = vec![0.0f32; dm];
        let nl = self.layers.len();
        for l in 0..nl {
            let (head, tail) = self.layers.split_at_mut(l + 1);
            let ly = &mut head[l];
            let next = tail.first();
            rms_norm(&mut xn, &x, &ly.attn_norm, h.eps);
            kops::q8k_quantize(&xn, xq.view_mut());
            lap(&mut p.norm, &mut t);
            match &mut ly.mixer {
                Mixer::Rec(r, st) => {
                    rec_token(&self.w, &h, r, st, &xn, xq.view(), &mut x)?;
                    lap(&mut p.rec, &mut t);
                }
                Mixer::Attn(a, kv) => {
                    attn_token(&self.w, &h, a, &self.eng, kv, pos, &cs, &sn, xq.view(), &mut x)?;
                    lap(&mut p.attn, &mut t);
                }
            }
            // TORC_TRACE=2: also the state after the mixer, as layer 1000 + l
            if trace && crate::switches::is("TORC_TRACE", "2") {
                crate::ops::trace_state(1000 + l, &x);
            }
            if ahead_reads > 0
                && let (Some(nx), Some(cache)) = (next, self.w.cache.as_ref())
            {
                rms_norm(&mut xp, &x, &nx.ffn_norm, h.eps);
                let ids = crate::moe::predict_experts(&nx.moe, &xp, ahead_rank);
                if !ids.is_empty() {
                    cache.read_ahead(nx.moe.layer, &ids, ahead_reads);
                }
            }
            rms_norm(&mut xn, &x, &ly.ffn_norm, h.eps);
            kops::q8k_quantize(&xn, xq.view_mut());
            lap(&mut p.norm, &mut t);
            moe_token(&self.w, &ly.moe, &xn, xq.view(), &mut x, &mut self.scratch)?;
            lap(&mut p.moe, &mut t);
            if trace {
                crate::ops::trace_state(l, &x);
            }
        }
        rms_norm(&mut xn, &x, &self.out_norm, h.eps);
        kops::q8k_quantize(&xn, xq.view_mut());
        let out = self.w.head()?;
        let m = Mat::of(&out);
        // the worker's layers were pulled back (it sits idle while this machine decodes): with TORC_HEAD_SPLIT, the
        // head's last rows run on the worker and the rest here at the same time (pipe::head_split); each row is the
        // same dot product
        match head_split_at(m.rows).filter(|_| self.pipe_off && crate::pipe::client().is_some()) {
            Some(r0) => {
                let rb = m.data.len() / m.rows;
                let (lo, hi) = logits[..m.rows].split_at_mut(r0);
                let mine = Mat { ty: m.ty, rows: r0, cols: m.cols, data: &m.data[..r0 * rb] };
                crate::pipe::head_split(xq.view(), r0, m.rows, hi, &mut || mine.apply(xq.view(), lo))?;
            }
            None => m.apply(xq.view(), logits),
        }
        lap(&mut p.out, &mut t);
        self.prof = p;
        Ok(())
    }

    /// The output head's rows r0..r1 over one quantised vector (a pipeline worker's answer to a head request).
    pub fn head_rows(&self, xq: Q8s, r0: usize, r1: usize) -> Result<Vec<f32>> {
        let out = self.w.head()?;
        let m = Mat::of(&out);
        if r0 >= r1 || r1 > m.rows || xq.len() != m.cols / QK {
            return err(format!("bad head request: rows {r0}..{r1} of {}, {} blocks", m.rows, xq.len()));
        }
        let rb = m.data.len() / m.rows;
        let mut y = vec![0.0f32; r1 - r0];
        Mat { ty: m.ty, rows: r1 - r0, cols: m.cols, data: &m.data[r0 * rb..r1 * rb] }.apply(xq, &mut y);
        Ok(y)
    }

    /// Prefill: the n tokens that start at position pos0 go through each layer together, so a weight row is fetched
    /// once for the whole block; the recurrent state and the KV cache still step one token at a time. logits is n x
    /// n_vocab (every position's, exactly as decode would give them one by one), or empty for none (a prompt only needs
    /// the last block's last row, and decode gives that).
    pub fn prefill(&mut self, tokens: &[u32], pos0: u32, logits: &mut [f32]) -> Result<()> {
        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }
        if pos0 as usize + n > self.n_ctx as usize {
            return err("block past the context");
        }
        if self.hcm.is_some() {
            return self.forward_hc(tokens, pos0, logits, true);
        }
        let nl = self.layers.len();
        self.std_embed(tokens)?;
        // two machines as a layer pipeline (pipe.rs): this machine's layers, then the worker's and its head
        if let Some(cl) = self.pipe() {
            let s = self.split();
            self.std_layers(pos0, n, 0, s)?;
            // with logits, the hidden rows come back too (the MTP block drafts from them on this machine)
            // a prompt read under log_hidden (without logits): the worker's out-normed rows come back for the log, so
            // the MTP block can draft here once the worker's layers are pulled back (TORC_PIPE_PULL)
            let hlog = logits.is_empty() && self.hidden_log.is_some();
            let want = if logits.is_empty() { 0 } else { 2 }
                | if hlog || !logits.is_empty() { crate::pipe::WANT_HIDDEN } else { 0 };
            let rq = crate::pipe::request(s, pos0, tokens, want, &self.px[..n * self.stream_width()]);
            let mut st = cl.lock().unwrap_or_else(|p| p.into_inner());
            st.write_all(&rq).map_err(|e| format::Error(format!("two boxes (pipeline): {e}")))?;
            if logits.is_empty() && !hlog {
                return crate::pipe::recv(&mut st, logits);
            }
            let dm = self.h.d_model as usize;
            self.hidden.resize(n * dm, 0.0);
            crate::pipe::recv_with_hidden(&mut st, logits, &mut self.hidden[..n * dm])?;
            if hlog {
                let (start, log) = self.hidden_log.as_mut().unwrap();
                if log.is_empty() || *start as usize + log.len() / dm != pos0 as usize {
                    *start = pos0;
                    log.clear();
                }
                log.extend_from_slice(&self.hidden[..n * dm]);
            }
            return Ok(());
        }
        self.std_layers(pos0, n, 0, nl)?;
        if logits.is_empty() && self.hidden_log.is_some() {
            self.std_hidden(n);
        } else {
            self.std_head(n, logits)?;
        }
        if let Some((start, log)) = self.hidden_log.as_mut() {
            let dm = self.h.d_model as usize;
            if log.is_empty() || *start as usize + log.len() / dm != pos0 as usize {
                *start = pos0;
                log.clear();
            }
            log.extend_from_slice(&self.hidden[..n * dm]);
        }
        Ok(())
    }

    /// The out-normed rows of n residual rows, into `hidden`, without the vocabulary product (log_hidden).
    fn std_hidden(&mut self, n: usize) {
        let h = self.h.clone();
        let dm = h.d_model as usize;
        let mut xq = Q8::new(n * (dm / QK));
        self.hidden.resize(n * dm, 0.0);
        let mut xn = std::mem::take(&mut self.hidden);
        norm_rows(&h, &mut xn, xq.view_mut(), &self.px, &self.out_norm, n);
        self.hidden = xn;
    }

    /// The residual rows of n tokens start as their embeddings (px).
    fn std_embed(&mut self, tokens: &[u32]) -> Result<()> {
        let dm = self.h.d_model as usize;
        let n = tokens.len();
        let emb = self.w.need("token_embd.weight")?;
        let rb = emb.row_bytes();
        self.px.resize(n * dm, 0.0);
        for (t, &tok) in tokens.iter().enumerate() {
            // an image row (vision.rs): its encoded row stands in for the token's embedding
            if crate::vision::is_row(tok) {
                if !crate::vision::copy_row(tok, &mut self.px[t * dm..(t + 1) * dm]) {
                    return err("an image in the prompt is no longer known to the server");
                }
                continue;
            }
            if tok as usize >= emb.shape().1 {
                return err("token out of range");
            }
            kops::dequantize(
                emb.t.ty,
                &emb.data[tok as usize * rb..(tok as usize + 1) * rb],
                &mut self.px[t * dm..(t + 1) * dm],
            );
        }
        Ok(())
    }

    /// Layers l0..l1 for n rows at positions pos0.., on the residual rows (px), through the block paths (the same bits
    /// as the token paths).
    fn std_layers(&mut self, pos0: u32, n: usize, l0: usize, l1: usize) -> Result<()> {
        self.std_layers_then(pos0, n, l0, l1, &mut |_| Ok(()))
    }

    /// std_layers, with `after` called on each layer once it is done.
    fn std_layers_then(
        &mut self,
        pos0: u32,
        n: usize,
        l0: usize,
        l1: usize,
        after: &mut dyn FnMut(&Layer) -> Result<()>,
    ) -> Result<()> {
        let h = self.h.clone();
        let dm = h.d_model as usize;
        let nb = dm / QK;
        let mut x = std::mem::take(&mut self.px);
        let mut xn = vec![0.0f32; n * dm];
        let mut xq = Q8::new(n * nb);
        let norm_all = |xn: &mut [f32], xq: Q8m, x: &[f32], wn: &[f32]| norm_rows(&h, xn, xq, x, wn, n);
        use std::time::Instant;
        let mut tm = Instant::now();
        let lap = |acc: &mut f64, tm: &mut Instant| {
            let now = Instant::now();
            *acc += (now - *tm).as_secs_f64();
            *tm = now;
        };
        let mut p = self.prof;
        let mut r = Ok(());
        let trace = crate::switches::get("TORC_TRACE").map_or(0, |v| {
            if v == "2" {
                2
            } else if v != "0" {
                1
            } else {
                0
            }
        });
        for (l, ly) in self.layers.iter_mut().enumerate().take(l1).skip(l0) {
            norm_all(&mut xn, xq.view_mut(), &x, &ly.attn_norm);
            lap(&mut p.norm, &mut tm);
            let rec = matches!(ly.mixer, Mixer::Rec(..));
            r = match &mut ly.mixer {
                Mixer::Rec(rl, st) => crate::rec::rec_block(&self.w, &h, rl, st, &xn, xq.view(), &mut x, n),
                Mixer::Attn(a, kv) => {
                    crate::attn::attn_block(&self.w, &h, a, &self.eng, kv, pos0, &xn, xq.view(), &mut x, n)
                }
            };
            if r.is_err() {
                break;
            }
            lap(if rec { &mut p.rec } else { &mut p.attn }, &mut tm);
            // TORC_TRACE=2: the last row after the mixer, as layer 1000 + l
            if trace == 2 {
                crate::ops::trace_state(1000 + l, &x[(n - 1) * dm..n * dm]);
            }
            norm_all(&mut xn, xq.view_mut(), &x, &ly.ffn_norm);
            lap(&mut p.norm, &mut tm);
            r = crate::moe::moe_tokens(&self.w, &ly.moe, &xn, xq.view(), &mut x, n);
            if r.is_err() {
                break;
            }
            lap(&mut p.moe, &mut tm);
            if trace > 0 {
                crate::ops::trace_state(l, &x[(n - 1) * dm..n * dm]);
            }
            r = after(ly);
            if r.is_err() {
                break;
            }
        }
        self.px = x;
        self.prof = p;
        r
    }

    /// The head of n residual rows: every row's logits (or none, if logits is empty), keeping the out-normed rows in
    /// hidden.
    fn std_head(&mut self, n: usize, logits: &mut [f32]) -> Result<()> {
        if logits.is_empty() {
            return Ok(());
        }
        let h = self.h.clone();
        let dm = h.d_model as usize;
        let nb = dm / QK;
        let tm = std::time::Instant::now();
        let mut xn = vec![0.0f32; n * dm];
        let mut xq = Q8::new(n * nb);
        norm_rows(&h, &mut xn, xq.view_mut(), &self.px, &self.out_norm, n);
        self.hidden.clear();
        self.hidden.extend_from_slice(&xn);
        let out = self.w.head()?;
        crate::moe::apply_group_multi(&mut [(Mat::of(&out), logits)], xq.view(), n);
        self.prof.out += tm.elapsed().as_secs_f64();
        Ok(())
    }

    /// A prompt in blocks of `block` from position 0, with each block's logits (every row's when `rows` is set, else
    /// none) handed to `f` in order. With a pipeline worker (pipe.rs) the blocks overlap - this machine's layers run
    /// block k while the worker does block k-1; otherwise it goes block by block through prefill.
    pub fn prefill_blocks(
        &mut self,
        tokens: &[u32],
        block: usize,
        rows: bool,
        f: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<()> {
        self.prefill_blocks_at(tokens, 0, block, rows, f)
    }

    /// prefill_blocks from position pos0 (for a server reading only what is new past the kept state).
    pub fn prefill_blocks_at(
        &mut self,
        tokens: &[u32],
        pos0: u32,
        block: usize,
        rows: bool,
        f: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<()> {
        let nv = self.n_vocab()?;
        let chunks: Vec<&[u32]> = tokens.chunks(block.max(1)).collect();
        // the worker's time from here on belongs to this prompt (not to the last reply's decode passes), for balancing
        // the split
        crate::pipe::worker_time_reset();
        let cl = match self.pipe() {
            Some(cl) => cl,
            _ => {
                let mut lg = Vec::new();
                for (b, ch) in chunks.iter().enumerate() {
                    lg.resize(if rows { ch.len() * nv } else { 0 }, 0.0);
                    self.prefill(ch, pos0 + (b * block) as u32, &mut lg)?;
                    f(b, &lg);
                }
                return Ok(());
            }
        };
        if pos0 as usize + tokens.len() > self.n_ctx as usize {
            return err("block past the context");
        }
        let io = |e: std::io::Error| format::Error(format!("two boxes (pipeline): {e}"));
        let mut rs = cl.lock().unwrap_or_else(|p| p.into_inner());
        let mut ws = rs.try_clone().map_err(io)?;
        let s = self.split();
        let d = self.stream_width();
        // log_hidden: the worker's rows come back too (a server's prompt, for the MTP block's rows over it) - with
        // qwen4exp's MTP block, every row's streams before the head's mixer, else the out-normed rows (the MTP block
        // drafts here once the worker's layers are pulled back, TORC_PIPE_PULL)
        let hlog = self.hidden_log.is_some() && (self.hcm.is_none() || self.mtp_hc.is_some());
        if let Some((start, log)) = self.hidden_log.as_mut()
            && (log.is_empty() || *start as usize + log.len() / d != pos0 as usize)
        {
            *start = pos0;
            log.clear();
        }
        let want = if rows { 2 } else { 0 } | if hlog { crate::pipe::WANT_HIDDEN } else { 0 };
        // the prompt's last stretch before a pull (stream_pull): its last block asks for the worker's layer states,
        // which come back while the worker computes, in place of pull_states' copy afterwards
        let stream = std::mem::take(&mut self.stream_pull)
            && self.hcm.is_none()
            && crate::pipe::shard().is_none()
            && pos0 as usize + tokens.len() <= self.n_ctx as usize;
        let last_b = chunks.len().saturating_sub(1);
        let mut hrow = Vec::new();
        std::thread::scope(|sc| -> Result<()> {
            let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
            let sender = sc.spawn(move || -> std::io::Result<()> {
                for b in rx {
                    ws.write_all(&b)?;
                }
                ws.flush()
            });
            let mut lg = Vec::new();
            let mut ours = 0.0f64;
            for (b, ch) in chunks.iter().enumerate() {
                let pos = pos0 + (b * block) as u32;
                let t = std::time::Instant::now();
                let st: &[f32] = if self.hcm.is_some() {
                    self.hc_embed(ch)?;
                    self.hc_layers(ch, pos, 0, s)?;
                    &self.hcm.as_ref().unwrap().res
                } else {
                    self.std_embed(ch)?;
                    self.std_layers(pos, ch.len(), 0, s)?;
                    &self.px
                };
                ours += t.elapsed().as_secs_f64();
                let w = if stream && b == last_b { want | crate::pipe::WANT_STATES } else { want };
                let rq = crate::pipe::request(s, pos, ch, w, &st[..ch.len() * d]);
                if tx.send(rq).is_err() {
                    break;
                }
                if b > 0 {
                    lg.resize(if rows { chunks[b - 1].len() * nv } else { 0 }, 0.0);
                    if hlog {
                        hrow.resize(chunks[b - 1].len() * d, 0.0);
                        crate::pipe::recv_with_hidden(&mut rs, &mut lg, &mut hrow)?;
                        self.hidden_log.as_mut().unwrap().1.extend_from_slice(&hrow);
                    } else {
                        crate::pipe::recv(&mut rs, &mut lg)?;
                    }
                    f(b - 1, &lg);
                }
            }
            drop(tx);
            if let Some(last) = chunks.last() {
                // the worker's layer states, one early frame a layer, come before the last block's reply
                let mut sb: Vec<u8> = Vec::new();
                if stream {
                    use std::io::Read;
                    for _ in s..self.layers.len() {
                        let mut lb = [0u8; 4];
                        rs.read_exact(&mut lb).map_err(io)?;
                        let k = u32::from_le_bytes(lb) as usize;
                        if k > 1 << 30 {
                            return err(format!(
                                "two boxes (pipeline): a layer state of {k} bytes (out of step with the worker?)"
                            ));
                        }
                        let at = sb.len();
                        sb.resize(at + k, 0);
                        rs.read_exact(&mut sb[at..]).map_err(io)?;
                    }
                }
                lg.resize(if rows { last.len() * nv } else { 0 }, 0.0);
                if hlog {
                    hrow.resize(last.len() * d, 0.0);
                    crate::pipe::recv_with_hidden(&mut rs, &mut lg, &mut hrow)?;
                    self.hidden_log.as_mut().unwrap().1.extend_from_slice(&hrow);
                } else {
                    crate::pipe::recv(&mut rs, &mut lg)?;
                }
                if stream {
                    let (nl, npos) = (self.layers.len(), pos0 as usize + tokens.len());
                    if self.set_layer_states(s, nl, npos, &sb)? != sb.len() {
                        return err("two boxes (pipeline): the worker's layer states do not fit this machine's layers");
                    }
                    self.pipe_off = true;
                    eprintln!(
                        "two boxes (pipeline): the worker's layers {s}.. came back with the last block ({:.1} MB); \
                        this box alone from here",
                        sb.len() as f64 / 1e6
                    );
                }
                f(chunks.len() - 1, &lg);
            }
            // a prompt of several blocks: keep the split that balances the two machines for the next run
            if chunks.len() >= 4 {
                crate::pipe::save_split(self.layers.len(), s, ours);
            }
            sender.join().unwrap().map_err(io)
        })
    }

    /// qwen4exp's forward pass for n tokens at positions pos0... The streams start as the embedding. Layer by layer:
    /// the n-gram embeddings (on their layer); then, for the token mixer and for the FFN, a mixer of the streams into
    /// the block's input (Q8 as ggml makes it, then rounded to bf16), the block run on a zeroed x, and x scattered
    /// back into the streams. Finally the head's mixer and the vocabulary product. A single token (n = 1) goes through
    /// the blocks' token paths and a block through their block paths, with the same bits. logits gets the last row's
    /// (all = false) or every row's (all), or is empty for none.
    fn forward_hc(&mut self, tokens: &[u32], pos0: u32, logits: &mut [f32], all: bool) -> Result<()> {
        let n = tokens.len();
        if pos0 as usize + n > self.n_ctx as usize {
            return err("position past the context");
        }
        let nl = self.layers.len();
        self.hc_embed(tokens)?;
        // two machines as a layer pipeline (pipe.rs): this machine's layers, then the worker's and its head
        if let Some(cl) = self.pipe() {
            let s = self.split();
            self.hc_layers(tokens, pos0, 0, s)?;
            let want = if logits.is_empty() {
                0
            } else if all {
                2
            } else {
                1
            };
            // with an MTP block here, the streams before the head's mixer come back with the logits
            let hid = want != 0 && self.mtp_hc.is_some();
            let rq = crate::pipe::request(
                s,
                pos0,
                tokens,
                want | if hid { crate::pipe::WANT_HIDDEN } else { 0 },
                &self.hcm.as_ref().unwrap().res[..n * self.hc_width()],
            );
            let mut st = cl.lock().unwrap_or_else(|p| p.into_inner());
            let t0 = std::time::Instant::now();
            st.write_all(&rq).map_err(|e| format::Error(format!("two boxes (pipeline): {e}")))?;
            let k = if want == 2 {
                logits.len()
            } else if want == 1 {
                logits.len().min(self.n_vocab()?)
            } else {
                0
            };
            let r = if hid {
                self.hidden.resize(n * self.hc_width(), 0.0);
                crate::pipe::recv_with_hidden(&mut st, &mut logits[..k], &mut self.hidden)
            } else {
                crate::pipe::recv(&mut st, &mut logits[..k])
            };
            crate::pipe::trip_add(t0.elapsed().as_micros() as u64);
            return r;
        }
        self.hc_layers(tokens, pos0, 0, nl)?;
        // the MTP head needs the residual streams of every token, taken before the final mixer
        if self.mtp_hc.is_some() {
            let d = self.hc_width();
            self.hidden.clear();
            self.hidden.extend_from_slice(&self.hcm.as_ref().unwrap().res[..n * d]);
        }
        self.hc_head(n, logits, all)
    }

    fn hc_width(&self) -> usize {
        (self.h.hc * self.h.d_model) as usize
    }

    /// The pipeline worker, unless its layers were pulled back here.
    fn pipe(&self) -> Option<&'static std::sync::Mutex<std::net::TcpStream>> {
        if self.pipe_off { None } else { crate::pipe::client() }
    }

    /// Layers l0.. of this model's states as bytes packed in f32 words (a pipeline worker's reply to OP_PULL). Each
    /// layer in order: for attention, K and V (int8) and their scales for positions 0..npos, head by head (the two
    /// machines' caches may hold different contexts); for DeltaNet, the convolution window and the recurrent state.
    fn layer_states(&self, l0: usize, l1: usize, npos: usize) -> Vec<f32> {
        let mut b = self.layer_state_bytes(l0, l1, npos);
        b.resize(b.len().div_ceil(4) * 4, 0);
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    /// layer_states as plain bytes (unpadded).
    fn layer_state_bytes(&self, l0: usize, l1: usize, npos: usize) -> Vec<u8> {
        let hd = self.h.head_dim as usize;
        let mut b: Vec<u8> = Vec::new();
        for ly in &self.layers[l0..l1] {
            put_layer_state(ly, hd, npos, &mut b);
        }
        b
    }

    /// After a prompt of npos tokens through a pipeline worker: copies its layers' states here, and the passes after
    /// that run on this machine alone (a model that fits on one machine decodes faster without a network hop per
    /// token). The state is the same as if this machine had read the prompt itself, so the bits are the same from here
    /// on. Not yet for qwen4exp.
    pub fn pull_states(&mut self, npos: usize) -> Result<()> {
        let Some(cl) = self.pipe() else { return Ok(()) };
        if crate::pipe::shard().is_some() {
            return err("pulling the pipeline worker's states: this box holds a shard, not every layer");
        }
        if self.hcm.is_some() {
            return err("pulling the pipeline worker's states: not for qwen4exp yet");
        }
        if npos > self.n_ctx as usize {
            return err("pulling the pipeline worker's states: past the context");
        }
        let s = self.split();
        let nl = self.layers.len();
        let b = self.pull_range(cl, s, nl, npos)?;
        self.set_layer_states(s, nl, npos, &b)?;
        self.pipe_off = true;
        eprintln!(
            "two boxes (pipeline): the worker's layers {s}.. pulled back ({:.1} MB); this box alone from here",
            b.len() as f64 / 1e6
        );
        Ok(())
    }

    /// The worker's states for layers l0..l1 at positions 0..npos (OP_PULL, with the end layer in bits 16..), as bytes.
    fn pull_range(
        &self,
        cl: &std::sync::Mutex<std::net::TcpStream>,
        l0: usize,
        l1: usize,
        npos: usize,
    ) -> Result<Vec<u8>> {
        let end = if l1 == self.layers.len() { 0 } else { (l1 as u32) << 16 };
        let mut st = cl.lock().unwrap_or_else(|p| p.into_inner());
        // the position count rides in pos0
        st.write_all(&crate::pipe::request(l0, npos as u32, &[], crate::pipe::OP_PULL | end, &[]))
            .map_err(|e| format::Error(format!("two boxes (pipeline): {e}")))?;
        let got = crate::pipe::recv_any(&mut st)?;
        Ok(got.iter().flat_map(|f| f.to_le_bytes()).collect())
    }

    /// Moves this sequence's pipeline split up to `to` after npos positions: the worker's states for layers split..to
    /// are copied here (the KV for positions 0..npos, and DeltaNet's conv and recurrence), and from then on this
    /// machine runs those layers and the worker only to.. (each request names its split). The state is the same as if
    /// this machine had run those layers all along, so the bits are the same. This lets a prompt run at the split that
    /// suits prompts and the reply at the one that suits replies (the two can differ).
    pub fn shift_split(&mut self, to: usize, npos: usize) -> Result<()> {
        let Some(cl) = self.pipe() else { return err("shift_split: no pipeline worker") };
        if crate::pipe::shard().is_some() {
            return err("shift_split: a shard's split is fixed");
        }
        let s = self.split();
        if to <= s || to >= self.layers.len() || npos > self.n_ctx as usize {
            return err(format!("shift_split: {s} to {to} ({} layers)", self.layers.len()));
        }
        let t = std::time::Instant::now();
        let b = self.pull_range(cl, s, to, npos)?;
        self.set_layer_states(s, to, npos, &b)?;
        self.psplit.set(to);
        eprintln!(
            "two boxes (pipeline): split {s} -> {to}: layers {s}..{to} from the worker ({:.1} MB) in {:.3} s",
            b.len() as f64 / 1e6,
            t.elapsed().as_secs_f64()
        );
        Ok(())
    }

    /// Writes layer_states' bytes into layers l0..l1 (positions 0..npos) and returns the count of bytes read. Fails if
    /// they do not fit these layers (too short, or a cache too small); the layers are then partly written, and the
    /// caller starts over.
    fn set_layer_states(&mut self, l0: usize, l1: usize, npos: usize, b: &[u8]) -> Result<usize> {
        let mut at = 0usize;
        let mut take = |n: usize| {
            let e = (at + n).min(b.len());
            let r = b[at..e].to_vec();
            at = e;
            r
        };
        let hd = self.h.head_dim as usize;
        for ly in &mut self.layers[l0..l1] {
            match &mut ly.mixer {
                Mixer::Attn(_, kv) => kv.set_bytes(hd, npos, &mut take)?,
                Mixer::Rec(_, st) => {
                    let (nc, ns) = (st.conv.len(), st.ssm.len());
                    let (c, s) = (take(nc * 4), take(ns * 2));
                    if c.len() != nc * 4 || s.len() != ns * 2 {
                        return err("saved state: shorter than these layers");
                    }
                    st.conv = c.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
                    st.ssm = s.chunks_exact(2).map(|x| u16::from_le_bytes(x.try_into().unwrap())).collect();
                }
            }
        }
        Ok(at)
    }

    /// The number of floats a token's state has between layers: the hc streams, or the residual row.
    pub(crate) fn stream_width(&self) -> usize {
        if self.hcm.is_some() { self.hc_width() } else { self.h.d_model as usize }
    }

    /// The streams of n tokens start as their embeddings (hc copies of each).
    fn hc_embed(&mut self, tokens: &[u32]) -> Result<()> {
        let (dm, hc) = (self.h.d_model as usize, self.h.hc as usize);
        let d = hc * dm;
        let n = tokens.len();
        let Model { w, hcm, .. } = self;
        let hm = hcm.as_mut().unwrap();
        let emb = w.need("token_embd.weight")?;
        let rb = emb.row_bytes();
        let mut x = vec![0.0f32; dm];
        hm.res.resize(n * d, 0.0);
        for (t, &tok) in tokens.iter().enumerate() {
            if tok as usize >= emb.shape().1 {
                return err("token out of range");
            }
            kops::dequantize(emb.t.ty, &emb.data[tok as usize * rb..(tok as usize + 1) * rb], &mut x);
            for s in 0..hc {
                hm.res[t * d + s * dm..t * d + (s + 1) * dm].copy_from_slice(&x);
            }
        }
        Ok(())
    }

    /// A new sequence (a pipeline worker's next connection): every DeltaNet layer goes back to a zero state, with no
    /// open revertible block (attention only reads the positions a run writes).
    pub fn clear_states(&mut self) {
        // a new sequence goes back through the pipeline worker if one was pulled from (its layers are reset below)
        self.pipe_off = false;
        for ly in &mut self.layers {
            if let Mixer::Rec(_, st) = &mut ly.mixer {
                st.clear();
            }
        }
        // a pipeline worker's layers too (a server's new conversation on the same connection)
        if self.pipe().is_some() {
            let s = self.split();
            if let Err(e) = crate::pipe::control(s, crate::pipe::OP_RESET) {
                eprintln!("{e}");
            }
        }
        // the next sequence takes the split saved by then
        self.psplit.set(0);
    }

    /// The pipeline split of the current sequence (see psplit).
    fn split(&self) -> usize {
        if self.psplit.get() == 0 {
            self.psplit.set(crate::pipe::split(self.layers.len()));
        }
        self.psplit.get()
    }

    /// Whether a pass goes through a pipeline worker (TORC_PIPE, and not pulled back).
    pub fn piped(&self) -> bool {
        self.pipe().is_some()
    }

    /// The DeltaNet layers' state (conv and recurrence), to go back to with `load_rec`. Together with the KV cache,
    /// whose positions past a point are only read once rewritten, it is the whole state of a sequence at that point (a
    /// server's next turn resumes where the last prompt ended). This machine's layers only, so not while piped.
    pub fn save_rec(&self) -> Vec<(Vec<f32>, Vec<u16>)> {
        let mut v: Vec<_> = self
            .layers
            .iter()
            .filter_map(|ly| match &ly.mixer {
                Mixer::Rec(_, st) => Some((st.conv.clone(), st.ssm.clone())),
                _ => None,
            })
            .collect();
        // qwen4exp: the n-gram rings too (the last 128 positions; a reply past the mark overwrites them)
        if let Some(ng) = self.hcm.as_ref().and_then(|h| h.ng.as_ref()) {
            v.push(ng.ring());
        }
        v
    }

    /// The state of the sequence here, on both machines when piped. The worker keeps its own half in `slot`, one mark
    /// per slot: a server keeps its last prompt in slot 0 and its system prompt in slot 1.
    pub fn save_mark(&self, slot: usize) -> Mark {
        let piped = self.pipe().is_some();
        if piped && let Err(e) = crate::pipe::control(self.split(), crate::pipe::OP_SAVE | crate::pipe::slot_bits(slot))
        {
            eprintln!("{e}");
        }
        (self.split(), self.save_rec(), piped)
    }

    /// Goes back to a mark of this sequence (with no clear_states since it was taken), restoring from the worker's
    /// `slot` too when it was taken through the worker. A mark taken before a pull (TORC_PIPE_PULL, which copies the
    /// worker's layers here at the prompt's end) goes back through the worker, because this machine's copies of its
    /// layers are past the mark.
    pub fn load_mark(&mut self, mk: &Mark, slot: usize) {
        if mk.2 {
            self.pipe_off = false;
        }
        self.psplit.set(mk.0);
        self.load_rec(&mk.1);
        if self.pipe().is_some()
            && let Err(e) = crate::pipe::control(mk.0, crate::pipe::OP_LOAD | crate::pipe::slot_bits(slot))
        {
            eprintln!("{e}");
        }
    }

    /// This machine's layers l0..l1 at positions 0..npos, with everything a sequence carries there, as bytes: the
    /// layers' states (layer_state_bytes), the n-gram ring (with a u32 flag first) and, with `mtp`, the MTP block's
    /// cache (also flagged; its noted ids are made again from every prompt). Used for a server's saved system prompt
    /// (markfile).
    pub fn state_bytes(&mut self, l0: usize, l1: usize, npos: usize, mtp: bool) -> Vec<u8> {
        let mut b = self.layer_state_bytes(l0, l1, npos);
        match self.hcm.as_ref().and_then(|h| h.ng.as_ref()) {
            Some(ng) => {
                let (h, t) = ng.ring();
                b.extend(1u32.to_le_bytes());
                b.extend((h.len() as u32).to_le_bytes());
                for v in &h {
                    b.extend(v.to_le_bytes());
                }
                b.extend((t.len() as u32).to_le_bytes());
                for v in &t {
                    b.extend(v.to_le_bytes());
                }
            }
            None => b.extend(0u32.to_le_bytes()),
        }
        let hd = self.h.head_dim as usize;
        let kv = match (mtp, self.mtp_hc.as_mut(), self.mtp.as_mut()) {
            (false, ..) => None,
            (true, Some(m), _) => Some(m.kv()),
            (true, None, m) => m.map(|m| m.kv()),
        };
        match kv {
            Some(kv) => {
                b.extend(1u32.to_le_bytes());
                kv.put_bytes(hd, npos, &mut b);
            }
            None => b.extend(0u32.to_le_bytes()),
        }
        b
    }

    /// Puts state_bytes back into this machine. Fails if they do not fit this model (a ring or an MTP block it does not
    /// have).
    pub fn set_state_bytes(&mut self, l0: usize, l1: usize, npos: usize, b: &[u8]) -> Result<()> {
        let mut at = self.set_layer_states(l0, l1, npos, b)?;
        let mut take = |n: usize| -> Vec<u8> {
            let e = (at + n).min(b.len());
            let r = b[at..e].to_vec();
            at = e;
            r
        };
        let u32_of = |v: Vec<u8>| v.try_into().map(u32::from_le_bytes).unwrap_or(0) as usize;
        if u32_of(take(4)) == 1 {
            let nh = u32_of(take(4));
            let h: Vec<f32> = take(nh * 4).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let nt = u32_of(take(4));
            let t: Vec<u16> = take(nt * 2).chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            match self.hcm.as_mut().and_then(|h| h.ng.as_mut()) {
                // checked against this model's rings, not only the file's own counts (set_ring copies, and a wrong
                // length would panic)
                Some(ng) if h.len() == nh && t.len() == nt && ng.ring_lens() == (nh, nt) => ng.set_ring(&h, &t),
                _ => return err("saved state: an n-gram ring this model does not take"),
            }
        }
        if u32_of(take(4)) == 1 {
            let hd = self.h.head_dim as usize;
            let kv = match (self.mtp_hc.as_mut(), self.mtp.as_mut()) {
                (Some(m), _) => m.kv(),
                (None, Some(m)) => m.kv(),
                (None, None) => return err("saved state: an MTP cache and no MTP block here"),
            };
            kv.set_bytes(hd, npos, &mut take)?;
        }
        if at != b.len() {
            return err(format!("saved state: {} bytes left over", b.len() - at));
        }
        Ok(())
    }

    /// The layers a saved system prompt holds on this machine now: 0..split through a pipeline worker, else all of
    /// them.
    fn mark_layers(&self) -> usize {
        if self.pipe().is_some() { self.split() } else { self.layers.len() }
    }

    /// A saved state's key: this model, its layout (through a pipeline worker, or every layer on this machine) and the
    /// tokens. Not the split: a two-machine model may move it after a prompt (pipe::save_split); the file records it,
    /// and a loaded sequence runs at it.
    fn mark_key(&self, tokens: &[u32], piped: bool) -> u64 {
        let mut h = blake3::Hasher::new();
        h.update(&crate::pipe::ident().to_le_bytes());
        h.update(&[piped as u8]);
        for t in tokens {
            h.update(&t.to_le_bytes());
        }
        u64::from_le_bytes(h.finalize().as_bytes()[..8].try_into().unwrap())
    }

    /// Saves the sequence's state after `tokens` (which the layers hold exactly now) to disk on both machines
    /// (markfile), the worker's half first, so a main machine's file only exists alongside the worker's. A sequence
    /// pulled back (TORC_PIPE_PULL) is saved whole from this machine.
    pub fn mark_save(&mut self, tokens: &[u32]) -> Result<()> {
        if crate::markfile::keep() == 0 {
            return Ok(());
        }
        let piped = self.pipe().is_some();
        let (key, npos, l1) = (self.mark_key(tokens, piped), tokens.len(), self.mark_layers());
        if l1 < self.layers.len() && !crate::pipe::mark_op(l1, crate::pipe::KEY_SAVE, key, npos)? {
            return err("the worker did not save its half");
        }
        let b = self.state_bytes(0, l1, npos, true);
        let ident = crate::pipe::ident();
        crate::markfile::save(key, "m", ident, 0, l1 as u32, npos as u32, &b)
            .and_then(|_| crate::markfile::save_tokens(key, ident, tokens))
            .map_err(|e| format::Error(format!("saving the state: {e}")))
    }

    /// Loads the state after `tokens` from disk, if it is there (for a fresh sequence: clear_states first). Through the
    /// pipeline worker it loads both machines' halves; otherwise, with `whole` (TORC_PIPE_PULL, where this machine
    /// holds every layer), it loads one saved whole here, and the sequence then runs on this machine alone, as after a
    /// pull. Ok(false) means it was not saved and nothing changed.
    pub fn mark_load(&mut self, tokens: &[u32], whole: bool) -> Result<bool> {
        if crate::markfile::keep() == 0 {
            return Ok(false);
        }
        let (npos, nl, ident) = (tokens.len(), self.layers.len(), crate::pipe::ident());
        if npos > self.n_ctx as usize {
            return Ok(false);
        }
        if self.pipe().is_some() {
            let key = self.mark_key(tokens, true);
            if let Some((b, l1)) = crate::markfile::load(key, "m", ident, 0, None, npos as u32) {
                let l1 = l1 as usize;
                if l1 > 0 && l1 < nl && crate::pipe::mark_op(l1, crate::pipe::KEY_LOAD, key, npos)? {
                    // this sequence runs at the split the state was saved with
                    self.psplit.set(l1);
                    self.set_state_bytes(0, l1, npos, &b)?;
                    return Ok(true);
                }
            }
            if !whole {
                return Ok(false);
            }
        }
        let key = self.mark_key(tokens, false);
        let Some((b, _)) = crate::markfile::load(key, "m", ident, 0, Some(nl as u32), npos as u32) else {
            return Ok(false);
        };
        self.set_state_bytes(0, nl, npos, &b)?;
        // every layer is here, so the passes after this run on this machine (as after pull_states)
        self.pipe_off = true;
        Ok(true)
    }

    /// A pipeline worker's half of a saved state (pipe KEYOP): layers l0.. at npos positions, saved or loaded under
    /// `key`. Ok(false) means nothing was loaded (no such file, or it does not match).
    pub fn worker_mark(&mut self, op: u32, key: u64, l0: usize, npos: usize) -> Result<bool> {
        let nl = self.layers.len();
        if l0 == 0 || l0 >= nl || npos > self.n_ctx as usize {
            return err(format!("bad saved-state request: layers {l0}.., {npos} positions"));
        }
        let who = format!("w{l0}");
        let ident = crate::pipe::ident();
        match op {
            crate::pipe::KEY_SAVE => {
                let b = self.state_bytes(l0, nl, npos, false);
                crate::markfile::save(key, &who, ident, l0 as u32, nl as u32, npos as u32, &b)
                    .map_err(|e| format::Error(format!("saving the state: {e}")))?;
                Ok(true)
            }
            crate::pipe::KEY_LOAD => {
                let Some((b, _)) = crate::markfile::load(key, &who, ident, l0 as u32, Some(nl as u32), npos as u32)
                else {
                    return Ok(false);
                };
                self.set_state_bytes(l0, nl, npos, &b)?;
                Ok(true)
            }
            _ => err(format!("bad saved-state request: op {op}")),
        }
    }

    /// Drops a pipeline worker's marks (on a new connection: a new main machine, or one that started again from zero).
    pub fn clear_marks(&mut self) {
        self.wmark = Default::default();
    }

    /// save_mark, failing when the worker did not take its half (a checkpoint must be both machines' or neither's).
    pub fn try_save_mark(&self, slot: usize) -> Result<Mark> {
        let piped = self.pipe().is_some();
        if piped {
            crate::pipe::control(self.split(), crate::pipe::OP_SAVE | crate::pipe::slot_bits(slot))?;
        }
        Ok((self.split(), self.save_rec(), piped))
    }

    /// load_mark, failing when the worker did not go back to its half (the worker answers an empty slot with an error,
    /// so the two machines are never left at different points).
    pub fn try_load_mark(&mut self, mk: &Mark, slot: usize) -> Result<()> {
        if mk.2 {
            self.pipe_off = false;
        }
        self.psplit.set(mk.0);
        self.load_rec(&mk.1);
        if self.pipe().is_some() {
            crate::pipe::control(mk.0, crate::pipe::OP_LOAD | crate::pipe::slot_bits(slot))?;
        }
        Ok(())
    }

    pub fn load_rec(&mut self, s: &[(Vec<f32>, Vec<u16>)]) {
        let mut it = s.iter();
        for ly in &mut self.layers {
            if let Mixer::Rec(_, st) = &mut ly.mixer {
                let (c, r) = it.next().expect("a state per DeltaNet layer");
                st.clear();
                st.conv.copy_from_slice(c);
                st.ssm.copy_from_slice(r);
            }
        }
        if let (Some((h, t)), Some(ng)) = (it.next(), self.hcm.as_mut().and_then(|h| h.ng.as_mut())) {
            ng.set_ring(h, t);
        }
    }

    /// The context this model was loaded with (a pipeline worker's limit on a request's rows).
    pub fn n_ctx(&self) -> usize {
        self.n_ctx as usize
    }

    /// The worker's half of a pass (pipe.rs): the streams come in, layers l0.. and the head run, and the logits
    /// asked for go back (want 0: none, 1: the last row's, 2: every row's).
    pub fn hc_back(
        &mut self,
        tokens: &[u32],
        pos0: u32,
        l0: usize,
        res: Vec<f32>,
        want: u32,
        early: &mut dyn FnMut(&[u8]) -> std::io::Result<()>,
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        // a mismatched or corrupt main machine is refused, never allowed to panic on a slice (the worker would die with
        // it)
        if l0 == 0 || l0 >= self.layers.len() || pos0 > self.n_ctx {
            return err(format!(
                "bad pipeline request: split {l0}, position {pos0} ({} layers, ctx {})",
                self.layers.len(),
                self.n_ctx
            ));
        }
        // the control requests (no tokens): a verify block begins, or keeps its first `kept` tokens
        if n == 0 && want & crate::pipe::OP_PULL != 0 {
            // bits 16..: the end layer (0 for the last), see Model::shift_split
            let l1 = match (want >> 16) as usize {
                0 => self.layers.len(),
                e => e,
            };
            if l1 <= l0 || l1 > self.layers.len() {
                return err(format!("bad pipeline request: pull layers {l0}..{l1}"));
            }
            return Ok(self.layer_states(l0, l1, pos0 as usize));
        }
        if n == 0 && want & crate::pipe::OP_SAVE != 0 {
            self.wmark[crate::pipe::slot_of(want)] = Some(self.save_rec());
            return Ok(Vec::new());
        }
        if n == 0 && want & crate::pipe::OP_LOAD != 0 {
            // an empty slot is an error (the connection drops and the main machine starts again from zero), never a
            // quiet carry-on with whatever state the layers hold
            let k = crate::pipe::slot_of(want);
            let Some(mk) = self.wmark[k].take() else { return err(format!("no mark in slot {k}")) };
            self.load_rec(&mk);
            self.wmark[k] = Some(mk);
            return Ok(Vec::new());
        }
        if n == 0 && want & crate::pipe::OP_RESET != 0 {
            for ly in &mut self.layers {
                if let Mixer::Rec(_, st) = &mut ly.mixer {
                    st.clear();
                }
            }
            return Ok(Vec::new());
        }
        if n == 0 {
            let h = self.h.clone();
            for ly in &mut self.layers[l0..] {
                if let Mixer::Rec(r, st) = &mut ly.mixer {
                    if want & crate::pipe::OP_BEGIN != 0 {
                        st.begin();
                    } else if want & crate::pipe::OP_COMMIT != 0 {
                        // the count comes off the network: one past the block is refused, not left to commit's assert
                        let kept = (want >> crate::pipe::KEPT_SHIFT) as usize;
                        if st.logged().is_some_and(|n| kept > n) {
                            return err("two boxes: a commit past the end of its block");
                        }
                        crate::rec::commit(&h, r, st, kept);
                    }
                }
            }
            return Ok(Vec::new());
        }
        // bits 16.. of a request with tokens are TOPK (pipe.rs), and only a one-token logits request (mode 1) may carry
        // it, because KEPT shares those bits on the n = 0 ops above
        if want >> crate::pipe::TOPK_SHIFT != 0 && want & 3 != 1 {
            return err(format!("bad pipeline request: a top-K count on mode {} ({n} tokens)", want & 3));
        }
        if pos0 as usize + n > self.n_ctx as usize || res.len() != n * self.stream_width() {
            return err("bad pipeline request");
        }
        let nl = self.layers.len();
        let nv = self.n_vocab()?;
        if self.hcm.is_none() {
            // the other architectures: the residual rows, this machine's layers, then every row's logits (only the last
            // for want 1)
            self.px = res;
            if want & crate::pipe::WANT_STATES != 0 {
                // each layer's state goes out as it is done (pipe::WANT_STATES)
                let (hd, npos) = (self.h.head_dim as usize, pos0 as usize + n);
                self.std_layers_then(pos0, n, l0, nl, &mut |ly| {
                    let mut b = Vec::new();
                    put_layer_state(ly, hd, npos, &mut b);
                    early(&b).map_err(|e| format::Error(format!("two boxes (pipeline): {e}")))
                })?;
            } else {
                self.std_layers(pos0, n, l0, nl)?;
            }
            let mode = want & 3;
            let mut lg = vec![0.0f32; if mode == 0 { 0 } else { n * nv }];
            // a prompt read with WANT_HIDDEN and no logits (a server's prompt before a pull, with its MTP block
            // drafting here afterwards): the out-normed rows alone
            if mode == 0 && want & crate::pipe::WANT_HIDDEN != 0 {
                self.std_hidden(n);
            } else {
                self.std_head(n, &mut lg)?;
            }
            if mode == 1 {
                lg.drain(..(n - 1) * nv);
            }
            if want & crate::pipe::WANT_HIDDEN != 0 {
                lg.extend_from_slice(&self.hidden[..n * self.h.d_model as usize]);
            }
            return Ok(lg);
        }
        self.hcm.as_mut().unwrap().res = res;
        self.hc_layers(tokens, pos0, l0, nl)?;
        let mode = want & 3;
        // WANT_HIDDEN: every row's streams before the head's mixer (the main machine's MTP block drafts from them)
        let hid = (want & crate::pipe::WANT_HIDDEN != 0)
            .then(|| self.hcm.as_ref().unwrap().res[..n * self.hc_width()].to_vec());
        let mut lg = vec![
            0.0f32;
            match mode {
                0 => 0,
                1 => nv,
                _ => n * nv,
            }
        ];
        let k = (want >> crate::pipe::TOPK_SHIFT) as usize;
        let v = (want >> crate::pipe::HEAD_SPLIT_SHIFT) & crate::pipe::HEAD_SPLIT_MASK;
        if v > 0 && !(mode == 1 && k > 0) {
            return err(format!("bad pipeline request: a head split on mode {mode}, top {k}"));
        }
        if v > 0 {
            // the head split (pipe::HEAD_SPLIT): the head's input goes to the main machine at once, then rows r.. are
            // computed here
            let xq = self.hc_head_input(n)?;
            let (kb, gb) = xq.view().bytes();
            early(&[kb, gb].concat()).map_err(|e| format::Error(format!("two boxes (pipeline): {e}")))?;
            let tm = std::time::Instant::now();
            let out = self.w.head()?;
            let m = Mat::of(&out);
            let rb = m.data.len() / m.rows;
            let r = crate::pipe::split_rows(m.rows, v);
            let mut part = vec![0.0f32; m.rows - r];
            Mat { ty: m.ty, rows: m.rows - r, cols: m.cols, data: &m.data[r * rb..] }.apply(xq.view(), &mut part);
            self.prof.out += tm.elapsed().as_secs_f64();
            let mut lg = crate::pipe::topk(&part, k);
            let kk = lg.len() / 2;
            for id in &mut lg[kk..] {
                *id = f32::from_bits(id.to_bits() + r as u32);
            }
            if let Some(h) = hid {
                lg.extend_from_slice(&h);
            }
            return Ok(lg);
        }
        if mode == 1 && k > 0 {
            self.hc_head(n, &mut lg, false)?;
            lg = crate::pipe::topk(&lg, k);
        } else {
            self.hc_head(n, &mut lg, mode == 2)?;
        }
        if let Some(h) = hid {
            lg.extend_from_slice(&h);
        }
        Ok(lg)
    }

    /// Layers l0..l1 for n tokens at positions pos0.., on the streams (hcm.res).
    fn hc_layers(&mut self, tokens: &[u32], pos0: u32, l0: usize, l1: usize) -> Result<()> {
        let h = self.h.clone();
        let (dm, hc) = (h.d_model as usize, h.hc as usize);
        let (d, nb) = (hc * dm, dm / QK);
        let n = tokens.len();
        let Model { w, layers, eng, scratch, hcm, prof, .. } = self;
        let hm = hcm.as_mut().unwrap();
        let mut x = vec![0.0f32; n * dm];
        let mut xn = vec![0.0f32; n * dm];
        let mut xq = Q8::new(n * nb);
        let half = (h.n_rot / 2) as usize;
        let (mut cs, mut sn) = (vec![0.0f32; half], vec![0.0f32; half]);
        if n == 1 {
            rope_row(&mut cs, &mut sn, pos0, h.n_rot, h.rope_base);
        }
        use std::time::Instant;
        let mut tm = Instant::now();
        let lap = |acc: &mut f64, tm: &mut Instant| {
            let now = Instant::now();
            *acc += (now - *tm).as_secs_f64();
            *tm = now;
        };
        let mut p = *prof;
        let trace = crate::switches::get("TORC_TRACE").is_some_and(|v| v != "0");
        let (ahead_rank, ahead_reads) = read_ahead();
        for l in l0..l1 {
            let (head, tail) = layers.split_at_mut(l + 1);
            let ly = &mut head[l];
            // the next layer, if this machine runs it (a pipeline's main machine does not read the worker's experts)
            let next = tail.first().filter(|_| l + 1 < l1);
            if let Some(ng) = hm.ng.as_mut().filter(|g| g.layer as usize == l) {
                ng.apply(w, &h, tokens, pos0 as usize, &mut hm.res)?;
            }
            let lh = ly.hc.as_ref().unwrap();
            for part in 0..2 {
                crate::hc::hc_mix(w, &h, &lh.mix[part], &hm.res, 0, n, &mut xn, &mut hm.bufs);
                crate::hc::prep(dm, &mut xn, xq.view_mut(), n);
                x.fill(0.0);
                lap(&mut p.norm, &mut tm);
                if part == 1 {
                    if n == 1 {
                        // read ahead: the next layer's likeliest experts, guessed from this layer's MoE input (its own
                        // is only one mix away), with the reads running under this layer's experts and the next layer's
                        // mixer
                        if let (Some(nx), Some(cache), true) = (next, w.cache.as_ref(), ahead_reads > 0) {
                            let ids = crate::moe::predict_experts(&nx.moe, &xn, ahead_rank);
                            if !ids.is_empty() {
                                cache.read_ahead(nx.moe.layer, &ids, ahead_reads);
                            }
                        }
                        moe_token(w, &ly.moe, &xn, xq.view(), &mut x, scratch)?;
                    } else {
                        crate::moe::moe_tokens(w, &ly.moe, &xn, xq.view(), &mut x, n)?;
                        // read ahead (TORC_READ_AHEAD_BLOCK): the next layer's likeliest experts for this block, read
                        // while that layer mixes. They are only asked for now, so no read can push out an expert this
                        // layer still uses
                        if let (Some(nx), Some(cache), Some((r, m, s))) = (next, w.cache.as_ref(), block_read_ahead()) {
                            let ids = crate::moe::predict_block(&nx.moe, &xn, n, r, s);
                            if !ids.is_empty() {
                                cache.read_ahead(nx.moe.layer, &ids, m);
                            }
                        }
                    }
                    lap(&mut p.moe, &mut tm);
                } else {
                    match &mut ly.mixer {
                        Mixer::Rec(r, st) => {
                            if n == 1 {
                                rec_token(w, &h, r, st, &xn, xq.view(), &mut x)?;
                            } else {
                                crate::rec::rec_block(w, &h, r, st, &xn, xq.view(), &mut x, n)?;
                            }
                            lap(&mut p.rec, &mut tm);
                        }
                        Mixer::Attn(a, kv) => {
                            // the indexer (QSA) runs on the layer's input, then the attention reads its picks
                            crate::attn::qsa_prepare(w, &h, a, kv, &xn, n, pos0);
                            if n == 1 {
                                attn_token(w, &h, a, eng, kv, pos0, &cs, &sn, xq.view(), &mut x)?;
                            } else {
                                crate::attn::attn_block(w, &h, a, eng, kv, pos0, &xn, xq.view(), &mut x, n)?;
                            }
                            lap(&mut p.attn, &mut tm);
                        }
                    }
                }
                crate::hc::combine(&h, &mut hm.res, &x, &hm.bufs.inj, n);
                lap(&mut p.norm, &mut tm);
            }
            if trace {
                // the last token's streams
                crate::ops::trace_state(l, &hm.res[(n - 1) * d..n * d]);
            }
        }
        *prof = p;
        Ok(())
    }

    /// The head's input for the last of n rows: its mixer, then Q8, as hc_head makes it.
    fn hc_head_input(&mut self, n: usize) -> Result<Q8> {
        let h = self.h.clone();
        let dm = h.d_model as usize;
        let Model { w, hcm, prof, .. } = self;
        let hm = hcm.as_mut().unwrap();
        let mut xn = vec![0.0f32; n * dm];
        let mut xq = Q8::new(dm / QK);
        let tm = std::time::Instant::now();
        crate::hc::hc_mix(w, &h, &hm.head, &hm.res, n - 1, n, &mut xn, &mut hm.bufs);
        kops::q8k_quantize_gg(&xn[(n - 1) * dm..n * dm], xq.view_mut());
        prof.out += tm.elapsed().as_secs_f64();
        Ok(xq)
    }

    /// The head of n tokens' streams: the last row's logits (all = false) or every row's, or none (if logits is empty).
    fn hc_head(&mut self, n: usize, logits: &mut [f32], all: bool) -> Result<()> {
        if logits.is_empty() {
            return Ok(());
        }
        let h = self.h.clone();
        let dm = h.d_model as usize;
        let nb = dm / QK;
        let Model { w, hcm, prof, .. } = self;
        let hm = hcm.as_mut().unwrap();
        let mut xn = vec![0.0f32; n * dm];
        let mut xq = Q8::new(n * nb);
        let tm = std::time::Instant::now();
        {
            let first = if all { 0 } else { n - 1 };
            crate::hc::hc_mix(w, &h, &hm.head, &hm.res, first, n, &mut xn, &mut hm.bufs);
            for j in first..n {
                kops::q8k_quantize_gg(&xn[j * dm..(j + 1) * dm], xq.sub_mut(j * nb..(j + 1) * nb));
            }
            let out = w.head()?;
            if n - first == 1 {
                Mat::of(&out).apply(xq.sub(first * nb..n * nb), logits);
            } else {
                crate::moe::apply_group_multi(&mut [(Mat::of(&out), logits)], xq.sub(first * nb..), n - first);
            }
            prof.out += tm.elapsed().as_secs_f64();
        }
        Ok(())
    }
}

/// TORC_HEAD_SPLIT=f (0 < f < 1; off by default): with the worker's layers pulled back, the fraction f
/// of the head's rows is computed on the worker while this machine decodes (Model::decode). Returns the split row, a
/// multiple of 64.
fn head_split_at(rows: usize) -> Option<usize> {
    static F: std::sync::LazyLock<Option<f64>> =
        std::sync::LazyLock::new(|| crate::switches::num::<f64>("TORC_HEAD_SPLIT").filter(|&f| f > 0.0 && f < 1.0));
    let f = (*F)?;
    let r0 = (((rows as f64) * (1.0 - f)) as usize) / 64 * 64;
    (r0 > 0 && r0 < rows).then_some(r0)
}
