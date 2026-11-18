//! serve.rs - the Qwen family and its relatives for torc-server (chat/serve.rs): Qwen3.5 / 3.6, Qwen3 MoE,
//! Qwen3-Next, Flash Next, ERNIE 4.5, GLM-4.7-Flash, Moonlight and LFM2, all on model::Model. Each model's chat format
//! (its prompt and its tool calls) is in its own folder (qwen/chat.rs, models/ernie, glm, moonlight and lfm2); this
//! file picks one from the file's metadata (Chat) and runs the conversation.
//!
//! Replies are greedy unless "temperature" > 0, in which case top_k defaults to 40 and top_p to 1, and "seed" makes a
//! run repeatable. The state of the last conversation is kept. A request whose prompt extends it reads only the new
//! tokens, and so does the next turn of a chat: its prompt starts where the last one's messages ended, and that state
//! is kept as the mark. Any other request starts again from zero, because DeltaNet state cannot be rewound - except
//! that it can start past its system prompt when that is the one the last prompt read from zero began with (a second
//! mark, kept where the system prompt ends), or past a chat saved on disk (markfile). TORC_PIPE reads the prompt over
//! two machines; with TORC_PIPE_PULL=1 the worker's state then comes back and the reply (and the chat's next turns) run
//! on this machine. Thinking streams as reasoning_content. TORC_METRICS=file writes progress lines for a front end. MTP
//! drafts (for a model with a qwen35moe or qwen3next MTP block, on one machine): up to TORC_MTP drafts a pass (default
//! 8, 0 = off), chosen by the draft planner (model::plan), and each pass is verified by one exact block. Each position
//! of the block is sampled exactly as decode would sample it (the same logits, the same random stream, one draw a
//! token), and a draft is kept only while it equals that draw. The reply is therefore the same tokens as without
//! drafts, sampled or greedy.

use super::sample::Sampling;
use crate::api;
use crate::chat::json_calls;
use crate::plan::{Plans, conf};
use crate::serve::{CallPiece, Reply, Serve, defused, defused_tools};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::io::Write;

struct Engine {
    m: crate::Model,
    tok: crate::tok::Tokenizer,
    nv: usize,
    stops: Vec<u32>,
    /// the tokens the model's state has read, in order
    seen: Vec<u32>,
    name: String,
    reason_start: Option<u32>,
    reason_end: Option<u32>,
    /// the model reasons in <think> tags (its chat template opens the reply with one); false for an instruct-only
    /// model such as Qwen3-30B-A3B-Instruct-2507, whose turns open with no reasoning block
    reasons: bool,
    /// whether it reasons when a request doesn't say (Qwen3.6 does, Qwen3.5 only when asked)
    think_default: bool,
    tool_start: Option<u32>,
    tool_end: Option<u32>,
    /// the context the model was loaded with: a prompt plus its reply must fit in it
    ctx: usize,
    /// TORC_METRICS=file: JSON lines (the prompt, progress after each block, the first token, the reply) that a front
    /// end reads to show "reading N / T tokens" during a long prompt
    metrics: Option<std::fs::File>,
    replies: u64,
    /// the tokens of the last prompt's base and the state where it ended (on both machines when piped), for the chat's
    /// next turn
    mark: Option<(Vec<u32>, crate::Mark)>,
    /// the tokens of the last system prompt read from zero and the state where it ended (the worker keeps its half in
    /// slot 1). A new chat with the same system prompt starts there while the cache still holds it: `seen` starts with
    /// it, and its positions are only rewritten by a prompt read from zero, which takes a new mark
    sysmark: Option<(Vec<u32>, crate::Mark)>,
    /// how many of the last prompt's tokens were not read again (usage: prompt_tokens_details.cached_tokens)
    kept: usize,
    /// the tokens of the states saved on disk (markfile::index): system prompts, and chats that were left for another
    saved: Vec<Vec<u32>>,
    /// how many requests the current sequence has had (1 for a one-shot request; 2 or more for a chat past its first
    /// turn)
    turns: usize,
    /// whether the request in progress has changed any state yet; an error before that point keeps the warm state
    touched: bool,
    /// TORC_PIPE_PULL=1: after a prompt read on two machines, the worker's state comes back and the reply runs here
    pull: bool,
    /// the prompt block size: TORC_BLOCK, else torc-tune's result for this machine and layout, else the built-in rule
    /// (tune::block_rule)
    block: usize,
    /// the most MTP drafts in a pass (0 for none); the planner learns what passes cost across replies
    mtp_d: usize,
    pl: Box<dyn Plans>,
    /// the last reply's own timing: prompt tokens read and the seconds reading them, seconds to the first token going
    /// out, reply tokens and their seconds (the usage chunk's `timings`, as llama-server reports them, plus ttft_ms)
    timed: (usize, f64, f64, usize, f64),
}

/// A reply's MTP state: the next pass's first draft and its confidence, and the block's last hidden row; the tokens of
/// the open verify pass (t and the drafts kept, in the state but not yet committed) and how many of them the reply has
/// taken; and the queue of tokens to hand out before the next pass (the kept drafts, then the bonus token).
struct Drafting {
    d0: u32,
    c0: f64,
    mh_last: Vec<f32>,
    pass: Vec<u32>,
    kept: usize,
    queue: VecDeque<u32>,
    passes: usize,
    drafted: usize,
    accepted: usize,
}

/// A reply's pipelined speculation (two machines, with qwen4exp's MTP block on the main machine; see torc-run's
/// pipespec): the draft for the position after the token in flight, which has been sent to the worker but whose reply
/// has not been read yet, and the running tally.
struct PipeSpec {
    dft: u32,
    sent: bool,
    hit: usize,
    tried: usize,
}

impl Engine {
    fn metric(&mut self, v: Value) {
        put(&mut self.metrics, v);
    }
}

fn put(mf: &mut Option<std::fs::File>, v: Value) {
    if let Some(f) = mf.as_mut() {
        let _ = writeln!(f, "{v}");
        let _ = f.flush();
    }
}

/// The attention phase table's nanoseconds so far (crate::attn::ATTN_NS).
fn attn_ns() -> Vec<u64> {
    crate::attn::ATTN_NS.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed)).collect()
}

/// With TORC_PROF=1, prints the milliseconds per row of each phase between two readings of the profile.
fn prof_say(what: &str, a: crate::forward::Prof, b: crate::forward::Prof, attn_a: &[u64], rows: usize) {
    if rows == 0 || !crate::switches::get("TORC_PROF").is_some_and(|v| v != "0") {
        return;
    }
    let ms = |x: f64, y: f64| 1e3 * (y - x) / rows as f64;
    let at: Vec<String> = crate::attn::ATTN_PHASES
        .iter()
        .zip(attn_a.iter().zip(attn_ns()))
        .map(|(n, (x, y))| format!("{n} {:.3}", (y - x) as f64 / 1e6 / rows as f64))
        .collect();
    eprintln!(
        "torc-server: {what}: ms a row: deltanet {:.3} attention {:.3} moe {:.3} output {:.3} norms {:.3}; attention: {}",
        ms(a.rec, b.rec),
        ms(a.attn, b.attn),
        ms(a.moe, b.moe),
        ms(a.out, b.out),
        ms(a.norm, b.norm),
        at.join(", ")
    );
}

fn argmax(l: &[f32]) -> u32 {
    crate::ops::argmax(l) as u32
}

impl Engine {
    /// The MTP block's keys and values over what this reply's prompt read (positions start..last, as logged by
    /// prefill): the rows before the last, each with the token after it (token q + 1 with hidden row q). Returns the
    /// last row, which mtp_start runs in full with the reply's first token, or None if the log does not cover those
    /// positions.
    fn mtp_fill(&mut self, ids: &[u32], start: usize) -> format::Result<Option<Vec<f32>>> {
        let last = ids.len() - 1;
        let dm = self.m.hidden_width();
        let Some((s0, log)) = self.m.take_hidden_log() else { return Ok(None) };
        let rows = log.len() / dm;
        if rows == 0 || s0 as usize != start || start + rows != last + 1 {
            return Ok(None);
        }
        self.m.mtp_note(ids)?;
        if rows > 1 {
            self.m.mtp_fill(&ids[start + 1..=last], &log[..(rows - 1) * dm], start as u32)?;
        }
        Ok(Some(log[(rows - 1) * dm..].to_vec()))
    }

    /// Makes the first draft from the prompt's last row (tail, from mtp_fill) with token t, the first token of the
    /// reply. Returns None without a tail.
    fn mtp_start(&mut self, ids: &[u32], tail: Option<Vec<f32>>, t: u32) -> format::Result<Option<Drafting>> {
        let Some(tail) = tail else { return Ok(None) };
        let last = ids.len() - 1;
        self.m.mtp_note(&[t])?;
        let mut dl = vec![0.0f32; self.m.draft_width()];
        let mh = self.m.mtp(&[t], &tail, last as u32, &mut dl)?;
        Ok(Some(Drafting {
            d0: self.m.draft_id(argmax(&dl) as usize),
            c0: conf(&dl),
            mh_last: mh,
            pass: Vec::new(),
            kept: 0,
            queue: VecDeque::new(),
            passes: 0,
            drafted: 0,
            accepted: 0,
        }))
    }

    /// The token after t, which the reply has just taken: the next token of the open pass, or else a new pass of t and
    /// its drafts (from the draft planner) verified in one block, with its tokens queued. The block's rows are picked
    /// in order exactly as decode picks them (sp: greedy, or one draw of the random stream a token) until one differs
    /// from its draft, and that pick is the pass's last token.
    fn mtp_next(&mut self, t: u32, d: &mut Drafting, sp: &mut Sampling) -> format::Result<u32> {
        if let Some(nx) = d.queue.pop_front() {
            d.kept += 1;
            return Ok(nx);
        }
        self.mtp_close(d);
        let p = self.seen.len();
        let dm = self.m.hidden_width();
        // the chain: draft g comes from draft g - 1 and the block's last hidden row, at position p - 1 + g
        let (mut drafts, mut surv) = (vec![d.d0], vec![d.c0]);
        let mut d1 = Vec::new();
        let room = self.ctx.saturating_sub(p + 1);
        while drafts.len() < self.mtp_d.min(room) && self.pl.plan(&surv) == drafts.len() {
            let g = drafts.len();
            let tc = std::time::Instant::now();
            d1.resize(self.m.draft_width(), 0.0);
            let h1 = self.m.mtp(&[drafts[g - 1]], &d.mh_last, (p - 1 + g) as u32, &mut d1)?;
            surv.push(surv[g - 1] * conf(&d1));
            drafts.push(self.m.draft_id(argmax(&d1) as usize));
            d.mh_last = h1;
            self.pl.time_chain(tc.elapsed().as_secs_f64());
        }
        let nd = self.pl.pick(&surv).min(room);
        drafts.truncate(nd);
        surv.truncate(nd);
        let mut input = vec![t];
        input.extend_from_slice(&drafts);
        let nv = self.nv;
        let mut vl = vec![0.0f32; input.len() * nv];
        let tvv = std::time::Instant::now();
        self.m.verify(&input, p as u32, &mut vl)?;
        let verify_secs = tvv.elapsed().as_secs_f64();
        let mut acc = 0;
        let bonus = loop {
            let pick = sp.pick(&vl[acc * nv..(acc + 1) * nv]);
            if acc == nd || pick != drafts[acc] {
                break pick;
            }
            acc += 1;
        };
        self.pl.record(&surv, acc, 1 + nd, verify_secs);
        // the MTP rows of the kept positions, run on the main model's own hidden rows
        let vh = self.m.hidden[..(acc + 1) * dm].to_vec();
        let mut rows = drafts[..acc].to_vec();
        rows.push(bonus);
        self.m.mtp_note(&rows)?;
        let mut dl = vec![0.0f32; self.m.draft_width()];
        let mut mh = self.m.mtp(&rows, &vh, p as u32, &mut dl)?;
        d.d0 = self.m.draft_id(argmax(&dl) as usize);
        d.c0 = conf(&dl);
        d.mh_last = mh.split_off(acc * dm);
        d.pass = input[..acc + 1].to_vec();
        d.kept = 1;
        d.queue = rows.into_iter().collect();
        d.passes += 1;
        d.drafted += nd;
        d.accepted += acc;
        Ok(d.queue.pop_front().unwrap())
    }

    /// Starts pipelined speculation after the prompt: runs the MTP block's rows over what this reply's prompt read
    /// (logged along with the worker's streams), makes the first draft from the last row and t, then runs t's half and
    /// sends it to the worker.
    fn ps_start(&mut self, ids: &[u32], start: usize, t: u32) -> format::Result<PipeSpec> {
        let last = ids.len() - 1;
        let dm = self.m.hidden_width();
        self.m.mtp_note(ids)?;
        self.m.mtp_note(&[t])?;
        if let Some((s0, log)) = self.m.take_hidden_log() {
            let rows = log.len() / dm;
            if rows > 0 && s0 as usize == start && start + rows == last {
                // rows before the last: token q + 1 with q's streams, and no draft logits
                for c0 in (0..rows).step_by(256) {
                    let c1 = (c0 + 256).min(rows);
                    self.m.mtp(
                        &ids[start + c0 + 1..start + c1 + 1],
                        &log[c0 * dm..c1 * dm],
                        (start + c0) as u32,
                        &mut [],
                    )?;
                }
            }
        }
        let hl = self.m.hidden[..dm].to_vec();
        let mut dl = vec![0.0f32; self.m.draft_width()];
        self.m.mtp(&[t], &hl, last as u32, &mut dl)?;
        let p = self.seen.len();
        let st = self.m.half(t, p as u32)?;
        self.m.half_send(t, p as u32, &st)?;
        Ok(PipeSpec { dft: self.m.draft_id(argmax(&dl) as usize), sent: true, hit: 0, tried: 0 })
    }

    /// The token after t, while t is in flight on the worker. The draft's half runs in the meantime (revertibly); then
    /// t's logits arrive and are picked from. A right draft goes to the worker straight away; a wrong one is dropped
    /// and the pick's half is run instead. Then the next draft is made.
    fn ps_next(&mut self, t: u32, ps: &mut PipeSpec, sp: &mut Sampling) -> format::Result<u32> {
        let p = self.seen.len();
        let spec = p + 1 < self.ctx;
        let sd = if spec {
            self.m.spec_begin();
            Some(self.m.half(ps.dft, (p + 1) as u32)?)
        } else {
            None
        };
        let mut one = vec![0.0f32; self.nv];
        self.m.half_recv(&mut one)?;
        ps.sent = false;
        self.seen.push(t);
        let nt = sp.pick(&one);
        let Some(sd) = sd else { return Ok(nt) };
        let hp = self.m.hidden.clone();
        ps.tried += 1;
        let st = if nt == ps.dft {
            ps.hit += 1;
            self.m.spec_end(true);
            sd
        } else {
            self.m.spec_end(false);
            self.m.half(nt, (p + 1) as u32)?
        };
        self.m.half_send(nt, (p + 1) as u32, &st)?;
        ps.sent = true;
        self.m.mtp_note(&[nt])?;
        let mut dl = vec![0.0f32; self.m.draft_width()];
        self.m.mtp(&[nt], &hp, p as u32, &mut dl)?;
        ps.dft = self.m.draft_id(argmax(&dl) as usize);
        Ok(nt)
    }

    /// Closes a reply that ended with a token in flight: its pass is read back and the token is left in the state,
    /// since both machines ran it.
    fn ps_close(&mut self, t: u32, ps: &mut PipeSpec) -> format::Result<()> {
        if ps.sent {
            let mut one = vec![0.0f32; self.nv];
            self.m.half_recv(&mut one)?;
            self.seen.push(t);
            ps.sent = false;
        }
        Ok(())
    }

    /// After a link error in the middle of a reply: reconnects with the worker keeping its states and marks
    /// (reconnect_resume), takes both machines back to the mark at the end of this prompt's base (slot 0) and reads
    /// again what followed it: the rest of the prompt and the reply so far as a block, then t (the token whose step
    /// failed, already sent to the client) as a decode, whose logits land in `one`. The reply then goes on with plain
    /// decode. With the NPU the read-again rows can differ in the last bits from the first time (its products depend
    /// on the block), so a greedy reply may take another path after a drop. `committed`: t is already in `seen` (the
    /// step failed after the worker had returned t's pass, e.g. sending the next token in pipelined speculation), so
    /// it is read again as the last token of the block but not added a second time.
    fn reply_resume(&mut self, t: u32, committed: bool, one: &mut [f32]) -> format::Result<()> {
        crate::pipe::reconnect_resume()?;
        let Some((mt, mk)) = self.mark.as_ref() else { return format::err("no mark to resume the reply from") };
        let p0 = mt.len();
        if self.seen.len() < p0 || self.seen[..p0] != mt[..] {
            return format::err("the mark is not this reply's prompt");
        }
        self.m.try_load_mark(mk, 0)?;
        self.m.truncate_hidden_log(p0 as u32);
        let mut seq = self.seen[p0..].to_vec();
        if !committed {
            seq.push(t);
        }
        let k = seq.len();
        if k > 1 {
            self.m.prefill_blocks_at(&seq[..k - 1], p0 as u32, self.block, false, &mut |_, _| {})?;
        }
        self.m.decode(seq[k - 1], (p0 + k - 1) as u32, one)?;
        if !committed {
            self.seen.push(t);
        }
        eprintln!("torc-server: connected again; {k} tokens read again from the mark at {p0}");
        Ok(())
    }

    /// Commits the open pass up to what the reply took of it.
    fn mtp_close(&mut self, d: &mut Drafting) {
        if !d.pass.is_empty() {
            self.m.commit(d.kept);
            self.seen.extend_from_slice(&d.pass[..d.kept]);
            d.pass.clear();
            d.queue.clear();
        }
    }

    /// The prompt, in the chat format of the model loaded (Chat): each format is in its model's folder.
    fn prompt(msgs: &[Value], tools: &[Value]) -> String {
        match chat() {
            Chat::Qwen => super::chat::prompt(msgs, tools),
            Chat::Glm => crate::models::glm::chat::prompt(msgs, tools),
            Chat::Ernie => crate::models::ernie::chat::prompt(msgs, tools),
            Chat::Kimi => crate::models::moonlight::chat::prompt(msgs, tools),
            Chat::Lfm2 => crate::models::lfm2::chat::prompt(msgs, tools),
        }
    }

    /// Reads the prompt and generates up to max tokens, handing each piece of valid UTF-8 text to `emit` along with
    /// whether it is reasoning. The prompt is `base` (the messages, with the assistant's turn opened) followed by
    /// `opener` (the think block). Only what is new gets read: past the kept sequence when the prompt extends it, else
    /// past the mark (the state where the last prompt's base ended - the next turn's prompt starts with it, followed by
    /// the reply without its reasoning), else from zero. The prompt is read without logits, so there is no vocabulary
    /// product per row and, on two machines, no rows sent over the link; its last token is run as a decode, as torc-run
    /// does. With `think` the reply starts as reasoning: the <think> tag is dropped, and </think> ends it (that tag is
    /// dropped too, along with the blank line after it). Returns the prompt tokens, reply tokens, reasoning tokens, the
    /// finish reason and the bodies of any tool calls made.
    fn run(
        &mut self,
        base: &str,
        opener: &str,
        max: usize,
        think: bool,
        sp: &mut Sampling,
        emit: &mut dyn FnMut(bool, &str) -> bool,
        tools: &[Value],
        on_call: &mut dyn FnMut(CallPiece) -> bool,
    ) -> format::Result<(usize, usize, usize, &'static str, Vec<String>)> {
        self.touched = false;
        let r = self.run_seq(base, opener, max, think, sp, emit, tools, on_call);
        if let Err(e) = &r
            && self.touched
        {
            // an error can leave the two machines out of step (a request's replies unread, the worker gone, a pass half
            // done), so the worker gets a new connection and the next request starts its sequence from zero
            eprintln!("torc-server: {e}; starting the next request from zero");
            if let Err(e2) = crate::pipe::reconnect() {
                eprintln!("torc-server: {e2}");
            }
            self.seen.clear();
            self.mark = None;
            self.sysmark = None;
        }
        r
    }

    /// Called when a request for another chat comes in, or when none has come for a while (TORC_IDLE_SAVE): saves the
    /// current chat (its last prompt's mark) to disk, if it is long enough to be worth it and not saved already, so
    /// that coming back to it reads only what is new. The state is left at the mark (`seen` holds its tokens), which is
    /// where the chat's next turn begins.
    fn save_leaving(&mut self) {
        const MIN: usize = 512;
        let Some((mt, mk)) = self.mark.take() else { return };
        if mt.len() < MIN || self.saved.contains(&mt) || crate::markfile::keep() == 0 {
            self.mark = Some((mt, mk));
            return;
        }
        let t = std::time::Instant::now();
        // takes the layers back to where the mark was taken, since the reply after it is past what gets saved. A worker
        // that could not go back saves nothing (its half would be from a different point in the chat), and the next
        // request starts from zero
        if let Err(e) = self.m.try_load_mark(&mk, 0) {
            eprintln!("torc-server: saving the chat: {e}; the next request starts from zero");
            self.seen.clear();
            return;
        }
        match self.m.mark_save(&mt) {
            Ok(()) => {
                eprintln!("torc-server: chat ({} tokens) saved in {:.2} s", mt.len(), t.elapsed().as_secs_f64());
                self.note_saved(&mt);
            }
            Err(e) => eprintln!("torc-server: saving the chat: {e}"),
        }
        self.seen = mt.clone();
        self.mark = Some((mt, mk));
    }

    /// TORC_IDLE_SAVE: saves the current chat to disk by the same rule a switch uses (two turns or more), and also any
    /// single prompt of TORC_IDLE_SAVE_MIN tokens or more (default 8192). A long read is the most expensive thing to
    /// lose to a reload, while an agent's short one-shot calls would only push real chats out of the kept files.
    fn save_idle(&mut self) {
        let min = crate::switches::num::<usize>("TORC_IDLE_SAVE_MIN").unwrap_or(8192);
        if self.turns >= 2 || self.mark.as_ref().is_some_and(|(mt, _)| mt.len() >= min) {
            self.save_leaving();
        }
    }

    fn note_saved(&mut self, t: &[u32]) {
        if !self.saved.iter().any(|x| x[..] == t[..]) {
            self.saved.push(t.to_vec());
        }
        // the disk keeps at most markfile::keep() files; the index is trimmed to match now and then
        if self.saved.len() > 2 * crate::markfile::keep().max(1) {
            self.saved = crate::markfile::index(crate::pipe::ident());
        }
    }

    /// Loads the longest saved state that `bt` begins with, after clearing the states for a fresh sequence, and returns
    /// its tokens.
    fn load_saved(&mut self, bt: &[u32]) -> Option<Vec<u32>> {
        let mut c: Vec<Vec<u32>> =
            self.saved.iter().filter(|t| t.len() < bt.len() && bt[..t.len()] == t[..]).cloned().collect();
        c.sort_by_key(|t| std::cmp::Reverse(t.len()));
        for t in c {
            match self.m.mark_load(&t, self.pull) {
                Ok(true) => return Some(t),
                // its files are gone (only the newest few are kept), so it comes out of the index
                Ok(false) => self.saved.retain(|x| *x != t),
                Err(e) => {
                    eprintln!("torc-server: loading a saved state: {e}");
                    // part of it may have loaded: take the layers back to zero
                    self.m.clear_states();
                }
            }
        }
        None
    }

    /// The tokens of the prompt's system turn (its first message, when that is the system's, up to and including
    /// <|im_end|> and the newline), if they are a prefix of the prompt's tokens and long enough to be worth a mark.
    fn system_tokens(&self, base: &str, bt: &[u32]) -> Option<Vec<u32>> {
        const MIN: usize = 64;
        let rest = base.strip_prefix("<|im_start|>system\n")?;
        let end = "<|im_start|>system\n".len() + rest.find("<|im_end|>\n")? + "<|im_end|>\n".len();
        let st = self.tok.encode(&base[..end], true).ok()?;
        (st.len() >= MIN && st.len() < bt.len() && bt[..st.len()] == st[..]).then_some(st)
    }

    fn run_seq(
        &mut self,
        base: &str,
        opener: &str,
        max: usize,
        think: bool,
        sp: &mut Sampling,
        emit: &mut dyn FnMut(bool, &str) -> bool,
        tools: &[Value],
        on_call: &mut dyn FnMut(CallPiece) -> bool,
    ) -> format::Result<(usize, usize, usize, &'static str, Vec<String>)> {
        // a SentencePiece model's BOS first (llama.cpp adds it the same way)
        let bt: Vec<u32> = self.tok.bos().into_iter().chain(self.tok.encode(base, true)?).collect();
        let mut ids = bt.clone();
        ids.extend(self.tok.encode(opener, true)?);
        let nd = self.tok.take_dropped();
        if nd > 0 {
            // a vocabulary-sliced model: the prompt is read without these characters, and the log says so rather than
            // hiding it
            eprintln!(
                "torc-server: {nd} character(s) of the prompt have no token in this model's vocabulary; left out"
            );
            self.metric(json!({"event": "dropped_characters", "count": nd}));
        }
        if ids.len() >= self.ctx {
            return Err(crate::chat::serve::too_long(ids.len(), self.ctx));
        }
        let max = max.min(self.ctx - ids.len());
        let st = self.system_tokens(base, &bt);
        // its characters were counted above
        let _ = self.tok.take_dropped();
        // an empty `seen` (a new server, or after an error) is read from zero below: the states are cleared first (an
        // error leaves this machine's DeltaNet layers mid-prompt) and the system prompt's mark is taken
        // from here on the request changes the model's state, so an error after this point resets it (see run)
        self.touched = true;
        let seen_ok = !self.seen.is_empty() && ids.len() > self.seen.len() && ids[..self.seen.len()] == self.seen[..];
        let mark_ok = self.mark.as_ref().is_some_and(|(mt, _)| bt.len() >= mt.len() && bt[..mt.len()] == mt[..]);
        // the longest saved state this prompt begins with. It is preferred over the system prompt's mark in RAM when it
        // saves more than a second of reading (a chat coming back)
        let best_disk =
            self.saved.iter().filter(|t| t.len() < bt.len() && bt[..t.len()] == t[..]).map(|t| t.len()).max();
        let sys_ok = self.sysmark.as_ref().is_some_and(|(smt, _)| {
            bt.len() > smt.len()
                && bt[..smt.len()] == smt[..]
                && self.seen.len() >= smt.len()
                && self.seen[..smt.len()] == smt[..]
                && best_disk.is_none_or(|d| d <= smt.len() + 512)
        });
        if !seen_ok && !mark_ok {
            // a conversation is being left; one-shot requests are not saved on the way out
            if self.turns >= 2 {
                self.save_leaving();
            }
            self.turns = 0;
        }
        self.turns += 1;
        // where the system prompt's mark is taken: a prompt read from zero that has a system turn
        let mut sys_at = None;
        let start = if seen_ok {
            self.seen.len()
        } else if mark_ok {
            let (mt, ms) = self.mark.as_ref().unwrap();
            self.m.try_load_mark(ms, 0)?;
            self.seen = mt.clone();
            mt.len()
        } else if sys_ok {
            // another chat with the same system prompt: the last prompt's mark is past what gets rewritten now
            let (smt, ms) = self.sysmark.as_ref().unwrap();
            self.m.try_load_mark(ms, 1)?;
            self.seen = smt.clone();
            self.mark = None;
            smt.len()
        } else {
            self.m.clear_states();
            self.seen.clear();
            self.mark = None;
            self.sysmark = None;
            // the longest state saved on disk (markfile, on both machines) that the prompt begins with; otherwise the
            // prompt is read, and its system prompt's state is saved below
            match self.load_saved(&bt) {
                Some(c) => {
                    if st.as_ref() == Some(&c) {
                        self.sysmark = Some((c.clone(), self.m.try_save_mark(1)?));
                    }
                    eprintln!("torc-server: {} tokens loaded from disk", c.len());
                    self.metric(json!({"event": "state_loaded", "tokens": c.len()}));
                    self.seen = c;
                    self.seen.len()
                }
                None => {
                    sys_at = st.as_ref().map(|s| s.len());
                    0
                }
            }
        };
        let last = ids.len() - 1;
        // drafts run on one machine, or on two once the worker's layers are pulled back after the prompt (its prompt
        // rows come back with their hidden rows for the MTP block, see forward.rs)
        let drafting = self.mtp_d > 0 && (!self.m.piped() || self.pull);
        // two machines with qwen4exp's MTP block use pipelined speculation (TORC_PIPESPEC=0 turns it off)
        let pipespec = self.m.can_pipespec() && !crate::switches::is("TORC_PIPESPEC", "0");
        if drafting || pipespec {
            self.m.log_hidden();
        }
        self.replies += 1;
        let rn = self.replies;
        self.kept = start;
        let total = ids.len() - start;
        self.metric(json!({"event": "prompt", "reply": rn, "input_tokens": total, "kept": start}));
        let t0 = std::time::Instant::now();
        let (prof_a, attn_a) = (self.m.prof, attn_ns());
        let mut tb = t0;
        // the base, then the opener (once the state is pulled back and the mark taken); the last token is run as the
        // decode below
        let cut = bt.len().clamp(start, last);
        // also stop at the system prompt's end when its mark is taken there (it is inside the base: 0 < s < cut)
        let sys = sys_at.filter(|&s| s > start && s < cut).unwrap_or(start);
        let segs: &[(usize, usize)] =
            if sys > start { &[(start, sys), (sys, cut), (cut, last)] } else { &[(start, cut), (cut, last)] };
        // a long stretch read through a pipeline worker goes in chunks of TORC_CHECKPOINT tokens (default 4096), with
        // the sequence's state saved before each one (on both machines, mark slot 2). If the link breaks mid-prompt,
        // the server reconnects with the worker keeping its state (reconnect_resume), goes back to the start of the
        // chunk and reads it again, up to 3 times a chunk, instead of failing the prompt. 0 reads it in one piece
        let ckn = crate::switches::num::<usize>("TORC_CHECKPOINT").unwrap_or(4096);
        // a checkpoint is also taken at the start of each piped stretch of TORC_CHECKPOINT_MIN tokens or more (default
        // 512), and the last one is kept across stretches. A drop while the system prompt or a turn's last stretch is
        // being read then goes back to it instead of failing the request. A shorter stretch is cheaper to read again
        // from the last checkpoint than to checkpoint
        let ck_min = crate::switches::num::<usize>("TORC_CHECKPOINT_MIN").unwrap_or(512);
        // the last checkpoint both machines took: (position, state)
        let mut good: Option<(usize, crate::Mark)> = None;
        for &(a, b) in segs {
            let ck = if ckn > 0 && self.m.piped() && b - a > ckn { ckn } else { b - a };
            let (mut p, mut tries) = (a, 0);
            while p < b || a == b {
                // going back to before this stretch (to an earlier stretch's checkpoint): catch up in full-size chunks
                let q = (p + if p < a { ck.max(ckn) } else { ck }).min(b);
                let mut fail = None;
                if ck < b - a || (ckn > 0 && self.m.piped() && b - a >= ck_min && p == a) {
                    match self.m.try_save_mark(2) {
                        Ok(mk) => good = Some((p, mk)),
                        Err(e) => fail = Some(e),
                    }
                }
                // read in blocks, overlapped across the two machines when piped (prefill_blocks_at). The base's last
                // stretch before a pull brings the worker's states back with its last block (Model::stream_pull)
                self.m.stream_pull =
                    self.pull && fail.is_none() && q == b && b == cut && cut == bt.len() && self.m.piped();
                let mf = &mut self.metrics;
                // TORC_BLOCK_LONG=b[,n] (off by default): a prompt of n tokens or more (default 2048) is read in blocks
                // of b. On the CPU a token's bits do not depend on its block (the NPU's products do, npu.rs)
                static LONG: std::sync::LazyLock<Option<(usize, usize)>> = std::sync::LazyLock::new(|| {
                    let v = crate::switches::get("TORC_BLOCK_LONG")?;
                    let mut it = v.split(',').map(|x| x.trim().parse::<usize>().ok());
                    let b = it.next()??;
                    let n = it.next().flatten().unwrap_or(2048);
                    (b > 0).then_some((b, n))
                });
                let bs = match *LONG {
                    Some((b, n)) if total >= n => b,
                    _ => self.block,
                };
                let r = if let Some(e) = fail {
                    Err(e)
                } else {
                    self.m.prefill_blocks_at(&ids[p..q], p as u32, bs, false, &mut |bi, _| {
                        let p1 = (p + (bi + 1) * bs).min(q);
                        let (done, el) = (p1 - start, t0.elapsed().as_secs_f64());
                        let (now, avg) = ((p1 - (p + bi * bs)) as f64 / tb.elapsed().as_secs_f64(), done as f64 / el);
                        tb = std::time::Instant::now();
                        put(
                            mf,
                            json!({"event": "progress", "reply": rn, "input_done": done, "input_tokens": total,
                        "tokens_per_second": now, "tokens_per_second_avg": avg, "elapsed_seconds": el,
                        "eta_seconds_estimate": (total - done) as f64 / avg}),
                        );
                    })
                };
                self.m.stream_pull = false;
                match r {
                    // a chunk was read, so the tries count from zero again (a long prompt may meet several drops)
                    Ok(()) => {
                        p = q;
                        tries = 0;
                    }
                    Err(e) if good.is_some() && tries < 3 => {
                        tries += 1;
                        let (gp, mk) = good.as_ref().unwrap();
                        p = *gp;
                        eprintln!("torc-server: {e}; reconnecting, the prompt goes on from {p} (try {tries} of 3)");
                        self.metric(json!({"event": "resumed", "reply": rn, "at": p, "try": tries}));
                        crate::pipe::reconnect_resume()?;
                        self.m.try_load_mark(mk, 2)?;
                        self.m.truncate_hidden_log(p as u32);
                    }
                    Err(e) => return Err(e),
                }
                if a == b {
                    break;
                }
            }
            if b == sys && sys > start {
                self.sysmark = Some((ids[..sys].to_vec(), self.m.try_save_mark(1)?));
                // already on disk (read again after its state was dropped from RAM), so it is not written again
                if !self.saved.iter().any(|x| x[..] == ids[..sys]) {
                    let t = std::time::Instant::now();
                    match self.m.mark_save(&ids[..sys]) {
                        Ok(()) => {
                            eprintln!(
                                "torc-server: system prompt ({sys} tokens) saved in {:.2} s",
                                t.elapsed().as_secs_f64()
                            );
                            self.note_saved(&ids[..sys]);
                        }
                        Err(e) => eprintln!("torc-server: saving the system prompt's state: {e}"),
                    }
                }
            }
            if b == cut && cut == bt.len() {
                if self.pull
                    && self.m.piped()
                    && let Err(e) = self.m.pull_states(cut)
                {
                    eprintln!("torc-server: {e}");
                }
                self.mark = Some((bt.clone(), self.m.try_save_mark(0)?));
            }
        }
        let mut one = vec![0.0f32; self.nv];
        if drafting {
            // a block of one: the same bits as decode, with its hidden row logged
            self.m.prefill(&ids[last..], last as u32, &mut one)?;
        } else {
            self.m.decode(ids[last], last as u32, &mut one)?;
        }
        self.seen.extend_from_slice(&ids[self.seen.len()..]);
        // the MTP block's keys and values over the prompt are part of reading it (pipelined speculation takes the
        // prompt's rows itself)
        let mut mtp_tail = if drafting && !(pipespec && max > 0) { self.mtp_fill(&ids, start)? } else { None };
        let prompt_s = t0.elapsed().as_secs_f64();
        self.metric(json!({"event": "first_token", "reply": rn, "prompt_seconds": prompt_s}));
        // TORC_PROF=1: this machine's share of the prompt, then of the reply, per row or token (the main machine's side
        // of what the worker prints)
        prof_say(&format!("reply {rn} prompt, {total} rows"), prof_a, self.m.prof, &attn_a, total);
        let (prof_b, attn_b) = (self.m.prof, attn_ns());
        let mut t = sp.pick(&one);
        // the MTP block's first draft (the prompt's last row with the first token) waits until the first token has
        // gone out, so the client does not wait for it
        let (mut dr, mut dr_due, t1) = (None, drafting, t);
        // the worker sends back only what the sampler can pick: the top 1 (greedy) or the top_k (top-k sampling)
        // with a penalty, the whole row: a penalised token can drop out of the top_k and let the next one in
        self.m.half_topk = if sp.penalised() {
            0
        } else if sp.temp <= 0.0 {
            1
        } else if (1..=4096).contains(&sp.top_k) {
            sp.top_k
        } else {
            0
        };
        let mut ps = if pipespec && max > 0 { Some(self.ps_start(&ids, start, t)?) } else { None };
        let (mut n, mut pending, mut reason, mut resumes) = (0usize, Vec::<u8>::new(), "length", 0);
        let (mut inside, mut nthink, mut trim) = (think, 0usize, false);
        // the calls the reply makes: the body of each <tool_call> .. </tool_call>, kept out of the text
        let (mut calls, mut call, mut in_tool) = (Vec::<String>::new(), Vec::<u8>::new(), false);
        // the Qwen format's calls also go out piece by piece while they are written (reply_calls), numbered as the
        // calls that read as calls are; the other formats' come whole at the end
        let streams = chat() == Chat::Qwen;
        let (mut cs, mut ncalls) = (None::<super::chat::CallStream>, 0usize);
        // a model with no tool-call tokens writes its calls as text: they are found in what would be sent
        let mut text_calls = matches!(chat(), Chat::Ernie | Chat::Kimi).then(json_calls::TextCalls::default);
        // the reply's own timing starts as the first token goes out. The time to it includes the pipelined
        // speculation's start above, which the client waits for too; the prompt's reading time does not
        let tg = std::time::Instant::now();
        let ttft_s = (tg - t0).as_secs_f64();
        while n < max {
            if self.stops.contains(&t) {
                reason = "stop";
                break;
            }
            if inside {
                nthink += 1;
            }
            if Some(t) == self.tool_start && !inside {
                in_tool = true;
                call.clear();
                if streams {
                    cs = Some(super::chat::CallStream::new(ncalls));
                }
            } else if Some(t) == self.tool_end && in_tool {
                in_tool = false;
                calls.push(String::from_utf8_lossy(&call).into_owned());
                // a body that never named its function is not a call, and takes no number
                if cs.take().is_some_and(|s| s.started()) {
                    ncalls += 1;
                }
            } else if in_tool {
                call.extend(self.tok.decode(t));
                if let Some(s) = cs.as_mut() {
                    let ok = match std::str::from_utf8(&call) {
                        Ok(b) => b.len(),
                        Err(e) => e.valid_up_to(),
                    };
                    let body = std::str::from_utf8(&call[..ok]).unwrap_or("");
                    if !s.feed(body, tools, on_call) {
                        eprintln!("torc-server: reply {rn}: the client closed the connection; stopped");
                        reason = "stop";
                        break;
                    }
                }
            } else if Some(t) == self.reason_start {
                // the tag itself is not text; a model that opens its own reasoning (LFM2) reasons from here
                inside = true;
            } else if Some(t) == self.reason_end {
                inside = false;
                trim = true;
                pending.clear();
            } else {
                pending.extend(self.tok.decode(t));
            }
            let ok = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if ok > 0 {
                let s = std::str::from_utf8(&pending[..ok]).unwrap();
                let s = if trim { s.trim_start_matches(['\n', '\r']) } else { s };
                if !s.is_empty() {
                    trim = false;
                    let shown = match text_calls.as_mut() {
                        Some(tc) if !inside => std::borrow::Cow::Owned(tc.feed(s, &mut calls)),
                        _ => std::borrow::Cow::Borrowed(s),
                    };
                    if !shown.is_empty() && !emit(inside, &shown) {
                        // the client has gone (a Stop button closes the connection): end here as at a stop token,
                        // with t not yet run, so the pipe, the drafts and `seen` close as they do after a stop
                        eprintln!("torc-server: reply {rn}: the client closed the connection; stopped");
                        reason = "stop";
                        break;
                    }
                }
                pending.drain(..ok);
            }
            if dr_due {
                dr = self.mtp_start(&ids, mtp_tail.take(), t1)?;
                dr_due = false;
            }
            let seen0 = self.seen.len();
            let r = match (dr.as_mut(), ps.as_mut()) {
                (Some(d), _) => self.mtp_next(t, d, sp),
                (None, Some(q)) => self.ps_next(t, q, sp),
                (None, None) => match self.m.decode(t, self.seen.len() as u32, &mut one) {
                    Ok(()) => {
                        self.seen.push(t);
                        Ok(sp.pick(&one))
                    }
                    Err(e) => Err(e),
                },
            };
            t = match r {
                Ok(nt) => nt,
                // a link error mid-reply (a drop, or a read that timed out, TORC_PIPE_TIMEOUT): up to 3 times a reply,
                // reconnect and read the reply so far again from the prompt's mark, then go on with plain decode
                Err(e) if dr.is_none() && self.m.piped() && resumes < 3 => {
                    resumes += 1;
                    eprintln!(
                        "torc-server: reply {rn}: {e}; reconnecting, the reply goes on from token {n} (try {resumes} \
                        of 3)"
                    );
                    self.metric(json!({"event": "resumed", "reply": rn, "at_token": n, "try": resumes}));
                    if let Some(q) = ps.take() {
                        eprintln!(
                            "torc-server: reply {rn}: pipelined speculation {} of {} drafts right before the drop; \
                            plain decode from here",
                            q.hit, q.tried
                        );
                    }
                    // t is in `seen` already when the step failed after its pass came back (ps_next's send of the
                    // next token): it must not be read twice
                    let committed = self.seen.len() > seen0;
                    self.reply_resume(t, committed, &mut one)?;
                    sp.pick(&one)
                }
                Err(e) => return Err(e),
            };
            n += 1;
        }
        self.timed = (total, prompt_s, ttft_s, n, tg.elapsed().as_secs_f64());
        // a reply that ended on its first token still starts the block, as the state after it expects
        if dr_due {
            dr = self.mtp_start(&ids, mtp_tail.take(), t1)?;
        }
        if let Some(q) = ps.as_mut() {
            self.ps_close(t, q)?;
            eprintln!(
                "torc-server: reply {rn}: {n} tokens, pipelined speculation {} of {} drafts right",
                q.hit, q.tried
            );
        }
        if let Some(d) = dr.as_mut() {
            // the open pass keeps what the reply took of it (a stop token or the length limit can end the reply inside
            // a pass)
            self.mtp_close(d);
            eprintln!(
                "torc-server: reply {rn}: {n} tokens in {} passes, {} of {} drafts kept",
                d.passes, d.accepted, d.drafted
            );
            eprintln!("torc-server: {}", self.pl.report(self.mtp_d + 1));
        }
        self.metric(json!({"event": "reply", "reply": rn, "output_tokens": n}));
        prof_say(&format!("reply {rn}, {n} tokens"), prof_b, self.m.prof, &attn_b, n);
        // the experts used so far go into the machine's usage file (a front end stops a server rather than letting it
        // exit)
        match &self.m.w.cache {
            Some(c) => c.usage_flush(),
            None => crate::npu::usage_flush(&self.m.w, self.m.h.n_layer, self.m.h.n_expert),
        }
        if in_tool {
            // cut off inside a call by max tokens: kept, and the parser decides
            calls.push(String::from_utf8_lossy(&call).into_owned());
        }
        if let Some(tc) = text_calls.as_mut() {
            let rest = tc.finish(&mut calls);
            if !rest.is_empty() {
                emit(false, &rest);
            }
        }
        if !calls.is_empty() && reason == "stop" {
            reason = "tool_calls";
        }
        Ok((ids.len(), n, nthink, reason, calls))
    }
}

/// The chat format of the model loaded, each in its model's folder: Qwen's ChatML (qwen/chat.rs),
/// ERNIE 4.5's (ernie4_5-moe), GLM-4.7's (deepseek2 with a glm4 tokenizer), LFM2's (lfm2moe) and Moonlight's (deepseek2
/// with a kimi-k2 tokenizer). Set once by open, from the file's metadata.
#[derive(Clone, Copy, PartialEq)]
enum Chat {
    Qwen,
    Ernie,
    Glm,
    /// LFM2 / LFM2.5: no reasoning opener (llama-server opens its turn with the plain assistant header, thinking on or
    /// off; the model writes its own <think> when it reasons)
    Lfm2,
    Kimi,
}
static CHAT: std::sync::OnceLock<Chat> = std::sync::OnceLock::new();

fn chat() -> Chat {
    CHAT.get().copied().unwrap_or(Chat::Qwen)
}

/// The markers this family's prompt format writes that its vocabularies keep as added tokens, not control tokens:
/// broken in user and tool text along with the control tokens (Tokenizer::defuse). GLM's <arg_key>/<arg_value> and
/// <|observation|>, and LFM2's call tags, are among them.
const MARKERS: &[&str] = &[
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<|observation|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
];

impl Engine {
    /// The request's prompt, with what the user, the tools and the tool servers wrote made safe to put in it.
    fn prompt_of(&self, c: &api::Chat) -> String {
        let d: &dyn Fn(&str) -> std::borrow::Cow<str> = &|t| self.tok.defuse(t, MARKERS);
        Engine::prompt(&defused(&c.msgs, d), &defused_tools(&c.tools, d))
    }
}

impl Serve for Engine {
    fn name(&self) -> &str {
        &self.name
    }

    fn count(&self, c: &api::Chat) -> usize {
        self.tok.encode(&self.prompt_of(c), true).map_or(0, |v| v.len())
    }

    fn reply(&mut self, c: &api::Chat, emit: &mut dyn FnMut(bool, &str) -> bool) -> format::Result<Reply> {
        self.reply_calls(c, emit, &mut |_| true)
    }

    fn reply_calls(
        &mut self,
        c: &api::Chat,
        emit: &mut dyn FnMut(bool, &str) -> bool,
        on_call: &mut dyn FnMut(CallPiece) -> bool,
    ) -> format::Result<Reply> {
        let seed = c.seed.unwrap_or_else(|| {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64)
        });
        // the generator's state can't be 0, so seed 0 starts from a fixed constant instead
        let rng = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        let mut sp = Sampling::new(c.temperature, c.top_p, c.top_k.unwrap_or(40), rng);
        sp.penalties(c.presence_penalty, c.frequency_penalty);
        let text = self.prompt_of(c);
        let think = c.think.unwrap_or(self.think_default) && self.reasons;
        // a reasoning model's turn is opened the way the template opens it: with the reasoning opened, or with it
        // passed empty; any other model's turn opens on its reply GLM-4.7 opens it with "<think>", or with "</think>"
        // when thinking is off (its template's enable_thinking)
        let opener = if chat() == Chat::Glm {
            if think { "<think>" } else { "</think>" }
        } else if !self.reasons || chat() == Chat::Lfm2 {
            ""
        } else if think {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        };
        let (prompt, output, reasoning, finish, calls) =
            self.run(&text, opener, c.max, think, &mut sp, emit, &c.tools, on_call)?;
        let calls = match chat() {
            Chat::Qwen => super::chat::read(&calls, &c.tools),
            Chat::Glm => crate::models::glm::chat::read(&calls, &c.tools),
            Chat::Lfm2 => crate::models::lfm2::chat::read(&calls),
            Chat::Ernie | Chat::Kimi => json_calls::read(&calls),
        };
        // a reply whose call didn't read as one ended as text, not as a call
        let finish = if finish == "tool_calls" && calls.is_empty() { "stop" } else { finish };
        Ok(Reply { prompt, output, reasoning, finish, calls })
    }

    fn kept(&self) -> usize {
        self.kept
    }

    fn timed(&self) -> (usize, f64, f64, usize, f64) {
        self.timed
    }

    fn replies(&self) -> u64 {
        self.replies
    }

    fn idle(&mut self) {
        self.save_idle();
    }
}

/// Loads a Qwen-family file for torc-server: the worker pool is up already, `ctx` is the context length and `name`
/// the name it is served under (the file's own when None).
pub fn open(path: &str, ctx: usize, name: Option<String>) -> Box<dyn Serve> {
    // TORC_TOKENIZER=<original model> with TORC_VOCAB_MAP=<a u32 per model id: the original's id> runs a
    // vocabulary-sliced model (an English-only Flash Next, for example) through the original model's tokenizer
    let tok = match (crate::switches::get("TORC_TOKENIZER"), crate::switches::get("TORC_VOCAB_MAP")) {
        (Some(t), Some(m)) => crate::tok::Tokenizer::mapped(&t, &m),
        _ => crate::tok::Tokenizer::open(path),
    }
    .unwrap_or_else(|e| {
        eprintln!("torc-server: {e}");
        std::process::exit(1)
    });
    // the chat template family (Engine::prompt), from the file's metadata
    let meta = format::read::Model::open(path).ok().and_then(|f| format::meta::Meta::parse(&f.metadata).ok());
    let _ = CHAT.set(match meta.as_ref().map(|mm| (mm.str("general.architecture"), mm.str("tokenizer.ggml.pre"))) {
        Some((Some("ernie4_5-moe"), _)) => Chat::Ernie,
        Some((Some("deepseek2"), Some("glm4"))) => Chat::Glm,
        Some((Some("deepseek2"), Some("kimi-k2"))) => Chat::Kimi,
        Some((Some("lfm2moe"), _)) => Chat::Lfm2,
        _ => Chat::Qwen,
    });
    // "</s>": a SentencePiece model's end of sequence (ERNIE 4.5); "<|user|>" and "<|observation|>": GLM's turn ends.
    // The Qwen vocabularies have none of these tokens
    let glm_stops: &[&str] = if chat() == Chat::Glm { &["<|user|>", "<|observation|>"] } else { &[] };
    // Moonlight's turns end in <|im_end|>; its end of sequence (tokenizer.ggml.eos_token_id) stops a reply too
    let kimi_eos =
        if chat() == Chat::Kimi { meta.as_ref().and_then(|mm| mm.u32("tokenizer.ggml.eos_token_id")) } else { None };
    let stops = ["<|im_end|>", "<|endoftext|>", "</s>"]
        .iter()
        .chain(glm_stops)
        .filter_map(|s| tok.find(s))
        .chain(kimi_eos)
        .collect();
    let (reason_start, reason_end) = (tok.find("<think>"), tok.find("</think>"));
    // the tokens a tool call is written between. ERNIE's and Moonlight's vocabularies have none: their calls are found
    // in the text (json_calls::TextCalls)
    let (open_tag, close_tag) = match chat() {
        Chat::Lfm2 => crate::models::lfm2::chat::CALL_TAGS,
        Chat::Glm => crate::models::glm::chat::CALL_TAGS,
        _ => super::chat::CALL_TAGS,
    };
    let (e_tok_find_open, e_tok_find_close) = (tok.find(open_tag), tok.find(close_tag));
    // on one machine the NPU's hot experts come from this machine's own usage of the model (npu.rs)
    crate::npu::own_usage(!crate::switches::given("TORC_PIPE"));
    let m = crate::Model::load(path, ctx as u32).unwrap_or_else(|e| {
        eprintln!("torc-server: {e}");
        std::process::exit(1)
    });
    // TORC_KLAY=1: the server runs every layer (alone, or as the main machine, which writes alone), so its experts are
    // laid out now, before the first request; TORC_KLAY_LAZY=1 leaves that to each layer's first use
    if !crate::switches::on("TORC_KLAY_LAZY") {
        match m.w.klay_all_now() {
            Ok(s) if s > 0.0 => eprintln!("klay: every layer laid out at load in {s:.1} s"),
            Ok(_) => {}
            Err(x) => {
                eprintln!("torc-server: {x}");
                std::process::exit(1)
            }
        }
    }
    let nv = m.n_vocab().expect("vocab");
    // whether the model reasons, and whether it does when a request doesn't say: read from how its chat template
    // opens the reply (the part after its last add_generation_prompt). Qwen3-30B-A3B-Instruct-2507 has <think> only
    // to strip it from earlier turns; Qwen3.5 reasons only when asked
    let (reasons, think_default) = format::meta::Meta::parse(&m.w.file.metadata)
        .ok()
        .and_then(|mm| {
            mm.str("tokenizer.chat_template").map(|t| {
                let open = t.rsplit("add_generation_prompt").next().unwrap_or("");
                let reasons = open.contains("<think>");
                (reasons, reasons && !open.contains("enable_thinking is true"))
            })
        })
        .unwrap_or((true, true));
    let name = name.unwrap_or_else(|| {
        std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
    });
    let metrics = crate::switches::get("TORC_METRICS")
        .filter(|p| !p.is_empty())
        .and_then(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).ok());
    let mut e = Engine {
        m,
        tok,
        nv,
        stops,
        seen: Vec::new(),
        name,
        reason_start,
        reason_end,
        reasons,
        think_default,
        ctx,
        metrics,
        tool_start: e_tok_find_open,
        tool_end: e_tok_find_close,
        replies: 0,
        mark: None,
        sysmark: None,
        kept: 0,
        saved: Vec::new(),
        turns: 0,
        touched: false,
        pull: crate::switches::on("TORC_PIPE_PULL"),
        block: crate::tune::block(),
        mtp_d: 0,
        pl: crate::plan::planner(),
        timed: (0, 0.0, 0.0, 0, 0.0),
    };
    // MTP drafts are on by default for the qwen35moe and qwen3next blocks, and off for qwen4exp's (Flash Next)
    e.mtp_d = if e.m.mtp_std() { crate::switches::num("TORC_MTP").unwrap_or(8).min(16) } else { 0 };
    eprintln!(
        "torc-server: MTP drafts {}",
        if e.mtp_d > 0 { format!("on (up to {} a pass, planned)", e.mtp_d) } else { "off".to_string() }
    );
    if e.m.can_pipespec() {
        eprintln!(
            "torc-server: pipelined speculation {}",
            if crate::switches::is("TORC_PIPESPEC", "0") {
                "off (TORC_PIPESPEC=0)"
            } else {
                "on (two boxes, the MTP block's drafts)"
            }
        );
    }
    // the states saved on disk by earlier runs (markfile): a prompt that begins with one of them starts there
    e.saved = crate::markfile::index(crate::pipe::ident());
    if !e.saved.is_empty() {
        eprintln!("torc-server: {} saved state(s) on disk", e.saved.len());
    }
    Box::new(e)
}

// Whatever a model writes, every call reader this engine uses ends, and doesn't panic: each good call cut at every
// byte, and with each byte in turn replaced by one of the characters the readers look for.
#[cfg(test)]
mod readers {
    use crate::chat::format::fixtures::typed_tools;
    use crate::chat::json_calls::{self, TextCalls};
    use crate::models::{glm, lfm2};

    #[test]
    fn readers_end_on_any_input() {
        let good = [
            "[calc(n=42, note='it\'s', opts={\"a\": [1, 2.5]}, on=True), now()]",
            "calc<arg_key>n</arg_key><arg_value>42</arg_value><arg_key>o</arg_key><arg_value>{\"a\": 1}</arg_value>",
            "{\"name\": \"calc\", \"arguments\": {\"n\": 42, \"l\": [1, {\"x\": null}]}}",
            "<function=calc>\n<parameter=n>\n42\n</parameter>\n</function>",
            "<tool_call>{\"name\": \"now\"}</tool_call> text <tool_ca",
        ];
        let mut inputs: Vec<String> = vec!["[search(ids=[1, 2)]".into(), "[f(a={1: 2, ]".into(), "[f(a=[".into()];
        for g in good {
            for k in 0..=g.len() {
                if g.is_char_boundary(k) {
                    inputs.push(g[..k].to_string());
                }
            }
            for k in 0..g.len() {
                for c in ["(", ")", "[", "]", "{", "}", "'", "\"", ",", "=", "<", ">", ":"] {
                    if g.is_char_boundary(k) && g.is_char_boundary(k + 1) {
                        inputs.push(format!("{}{c}{}", &g[..k], &g[k + 1..]));
                    }
                }
            }
        }
        let tools = typed_tools();
        for s in &inputs {
            let one = vec![s.clone()];
            super::super::chat::read(&one, &tools);
            glm::chat::read(&one, &tools);
            json_calls::read(&one);
            lfm2::chat::read(&one);
            let (mut tc, mut calls) = (TextCalls::default(), Vec::new());
            tc.feed(s, &mut calls);
            tc.finish(&mut calls);
        }
    }
}
