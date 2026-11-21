//! serve.rs - gpt-oss for torc-server (chat/serve.rs), one request at a time. Replies are greedy at temperature 0, or
//! sampled the way the Qwen engine samples (temperature, top_p, top_k, seed, penalties). A prompt that extends the last
//! one reads only its new tokens (the cache is positional).
//!
//! The prompt is the harmony format: a system message (the model's identity, its knowledge cutoff, the reasoning effort
//! and the valid channels), the conversation, and the assistant's turn opened. The reasoning effort comes from the
//! request (low, medium or high; medium when it names none, low when it turns thinking off). The model writes its reply
//! as messages on channels, and each goes to the client as what it is: the analysis channel as reasoning, the final
//! channel as the text, and a message addressed to a function as a tool call. It ends at <|return|> or <|call|>. A
//! request with tools gets them listed in the developer message (chat.rs, with the channel reader). Chats are saved to
//! disk and loaded back as the Qwen family's are (saved.rs).

use super::chat::{Channels, prompt};
use crate::api::Chat;
use crate::serve::{Call, Reply, Serve};
use serde_json::Value;

struct Engine {
    m: crate::gptoss::Oss,
    tok: crate::tok::Tokenizer,
    nv: usize,
    ctx: usize,
    /// the tokens the cache holds, in order
    seen: Vec<u32>,
    stops: Vec<u32>,
    name: String,
    /// the prompt is read in blocks of this many tokens (TORC_BLOCK, default 512)
    block: usize,
    /// the last prompt's tokens that were not read again
    kept: usize,
    /// the last reply's timings (serve.rs)
    timed: (usize, f64, f64, usize, f64),
    replies: u64,
    saved: crate::chat::saved::Saved,
}

impl Serve for Engine {
    fn effort_levels(&self) -> Vec<&'static str> {
        crate::api::EFFORT_LEVELS.to_vec()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn count(&self, c: &Chat) -> usize {
        self.tok.encode(&prompt(c, &self.tok), true).map_or(0, |v| v.len())
    }

    fn reply(&mut self, c: &Chat, emit: &mut dyn FnMut(bool, &str) -> bool) -> format::Result<Reply> {
        let ids = self.tok.encode(&prompt(c, &self.tok), true).unwrap_or_default();
        if ids.is_empty() {
            return Err(format::Error("the prompt could not be tokenized".into()));
        }
        // a reply runs to max_tokens or to the end of the context, whichever comes first
        if ids.len() >= self.ctx {
            return Err(crate::chat::serve::too_long(ids.len(), self.ctx));
        }
        let max = c.max.min(self.ctx - ids.len());
        // the engine's own timing: tokens read and seconds to the first token
        let tp = std::time::Instant::now();
        // the longest prefix already read, kept one token short so the last prompt token runs for its logits
        let keep = self.seen.iter().zip(&ids).take_while(|(a, b)| a == b).count().min(ids.len() - 1);
        // or a chat saved on disk that it begins with, when that is further on
        let keep = self.saved.begin(&mut self.m, &ids, keep);
        self.seen = ids[..keep].to_vec();
        let mut last = vec![0.0f32; self.nv];
        let mut p0 = keep;
        while p0 < ids.len() {
            let p1 = (p0 + self.block).min(ids.len());
            self.m.forward(&ids[p0..p1], p0 as u32, &mut last, false)?;
            p0 = p1;
        }
        let read = ids.len() - keep;
        let prompt_s = tp.elapsed().as_secs_f64();
        self.saved.read(&ids);
        self.seen = ids.clone();
        let tg = std::time::Instant::now();
        let (mut pend, mut n_out, mut n_think) = (Vec::<u8>::new(), 0usize, 0usize);
        let mut finish = "length";
        let mut reader = Channels::new();
        let mut pos = ids.len();
        let mut sp = crate::chat::serve::sampling(c);
        loop {
            let t = sp.pick(&last);
            n_out += 1;
            if self.stops.contains(&t) {
                finish = "stop";
                break;
            }
            pend.extend(self.tok.decode(t));
            // send whole UTF-8 characters only; a character split across tokens is held for the next one
            let cut = match std::str::from_utf8(&pend) {
                Ok(_) => pend.len(),
                Err(e) if e.error_len().is_none() => e.valid_up_to(),
                Err(_) => pend.len(),
            };
            let piece = String::from_utf8_lossy(&pend[..cut]).into_owned();
            pend.drain(..cut);
            n_think += reader.thinking() as usize;
            // the client has gone: the reply ends here
            if !reader.push(&piece, emit) {
                break;
            }
            if n_out == max || pos + 1 >= self.ctx {
                break;
            }
            self.m.forward(&[t], pos as u32, &mut last, false)?;
            self.seen.push(t);
            pos += 1;
        }
        self.kept = keep;
        self.timed = (read, prompt_s, prompt_s, n_out, tg.elapsed().as_secs_f64());
        self.replies += 1;
        reader.finish(emit);
        let calls: Vec<Call> = reader
            .calls
            .iter()
            .map(|(name, args)| Call {
                name: name.clone(),
                args: match serde_json::from_str(args) {
                    Ok(Value::Object(m)) => m,
                    _ => Default::default(),
                },
            })
            .collect();
        if !calls.is_empty() && finish == "stop" {
            finish = "tool_calls";
        }
        Ok(Reply { prompt: ids.len(), output: n_out, reasoning: n_think, finish, calls })
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
        self.saved.idle(&mut self.m);
    }
}

/// Loads a gpt-oss file for torc-server: the worker pool is up already, `ctx` is the context length and `name` the
/// name it is served under (the file's own when None).
pub fn open(path: &str, ctx: usize, name: Option<String>) -> Box<dyn Serve> {
    let t0 = std::time::Instant::now();
    let m = crate::gptoss::Oss::load(path, ctx as u32).unwrap_or_else(|e| {
        eprintln!("{path}: {}", e.0);
        std::process::exit(1)
    });
    let tok = crate::tok::Tokenizer::open(path).unwrap_or_else(|e| {
        eprintln!("tokenizer: {}", e.0);
        std::process::exit(1)
    });
    let nv = m.n_vocab().unwrap();
    let stops: Vec<u32> = ["<|return|>", "<|call|>"].iter().filter_map(|s| tok.find(s)).collect();
    let name = name.unwrap_or_else(|| {
        std::path::Path::new(path).file_stem().map_or("gpt-oss".into(), |s| s.to_string_lossy().into_owned())
    });
    let saved = crate::chat::saved::Saved::new(true, &m.w.file.header.model_id, "gpt-oss");
    let mut e = Engine {
        m,
        tok,
        nv,
        ctx,
        seen: Vec::new(),
        stops,
        name,
        block: crate::switches::num::<usize>("TORC_BLOCK").filter(|&b| b > 0).unwrap_or(512),
        kept: 0,
        timed: (0, 0.0, 0.0, 0, 0.0),
        replies: 0,
        saved,
    };
    // a warm-up turn, so the first request does not pay the first-use costs
    {
        let mut last = vec![0.0f32; nv];
        let _ = e.m.forward(&[0u32; 16], 0, &mut last, false);
        let _ = e.m.forward(&[0], 16, &mut last, false);
    }
    eprintln!("torc-server: gpt-oss, ctx {ctx}, loaded in {:.1} s (stops {:?})", t0.elapsed().as_secs_f64(), e.stops);
    Box::new(e)
}

/// What saving a chat needs from the model (chat/saved.rs): its whole state is its attention cache.
impl crate::chat::saved::Positional for crate::gptoss::Oss {
    fn state_bytes(&self, npos: usize) -> Vec<u8> {
        crate::gptoss::Oss::state_bytes(self, npos)
    }
    fn set_state_bytes(&mut self, npos: usize, b: &[u8]) -> format::Result<()> {
        crate::gptoss::Oss::set_state_bytes(self, npos, b)
    }
    fn n_layer(&self) -> usize {
        self.h.n_layer
    }
}
