//! serve.rs - what torc-server asks of a model to answer a chat request. Each model family answers it in its own way
//! (models/qwen/serve.rs for the Qwen engine and the models that run on it, models/gemma/serve.rs,
//! models/gptoss/serve.rs), and the server's request handling is written once against it.

use crate::api::Chat;
use serde_json::{Value, json};
use std::borrow::Cow;

/// A tool call the model made: the tool's name and its arguments. Each model family reads its own calls out of
/// its reply (in its folder under models/), so the server never sees a model's own syntax.
pub struct Call {
    pub name: String,
    pub args: serde_json::Map<String, Value>,
}

const TOO_LONG: &str = "the prompt is longer than the context: ";

/// The error for a prompt that doesn't fit the context, the one failure that is the request's fault rather than the
/// server's (torc-server answers it with 400, see is_too_long).
pub fn too_long(tokens: usize, ctx: usize) -> format::Error {
    format::Error(format!("{TOO_LONG}{tokens} tokens, the context is {ctx}"))
}

/// Whether an error is a too_long one.
pub fn is_too_long(e: &format::Error) -> bool {
    e.0.starts_with(TOO_LONG)
}

/// How a request's reply picks its tokens: its temperature, top_p, top_k (40 when not given), seed (the time when not
/// given) and penalties. Temperature 0 with no penalty is greedy.
pub fn sampling(c: &Chat) -> crate::models::qwen::sample::Sampling {
    let seed = c.seed.unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64)
    });
    // the generator's state can't be 0, so seed 0 starts from a fixed constant instead
    let rng = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
    let mut sp = crate::models::qwen::sample::Sampling::new(c.temperature, c.top_p, c.top_k.unwrap_or(40), rng);
    sp.penalties(c.presence_penalty, c.frequency_penalty);
    sp
}

/// The calls as OpenAI's tool_calls list (the arguments as JSON text), which is what the API writers take.
pub fn tool_calls(calls: &[Call]) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                json!({"index": i, "id": format!("call_{i}"), "type": "function",
            "function": {"name": c.name, "arguments": Value::Object(c.args.clone()).to_string()}})
            })
            .collect(),
    )
}

/// Breaks a string in place with `defuse` (Tokenizer::defuse, with a family's own markers), when it needs it.
fn fix(v: &mut Value, defuse: &dyn Fn(&str) -> Cow<str>) {
    if let Value::String(t) = v
        && let Cow::Owned(d) = defuse(t)
    {
        *t = d;
    }
}

/// The messages with the control strings in the user's and the tools' text broken, so an attached page or a tool's
/// result can't open a turn of its own. The system prompt and the model's own turns are left as they are: the first
/// is the user's own setup, and the second holds the model's real reasoning and calls.
pub fn defused(msgs: &[Value], defuse: &dyn Fn(&str) -> Cow<str>) -> Vec<Value> {
    msgs.iter()
        .map(|m| {
            let mut m = m.clone();
            if matches!(m["role"].as_str(), Some("user" | "tool")) {
                match m.get_mut("content") {
                    Some(Value::Array(parts)) => {
                        parts.iter_mut().filter_map(|p| p.get_mut("text")).for_each(|t| fix(t, defuse))
                    }
                    Some(v) => fix(v, defuse),
                    None => {}
                }
            }
            m
        })
        .collect()
}

/// The tools with every string in their declarations broken the same way: a tool server writes its own names and
/// descriptions, and they go into the system turn.
pub fn defused_tools(tools: &[Value], defuse: &dyn Fn(&str) -> Cow<str>) -> Vec<Value> {
    fn walk(v: &mut Value, defuse: &dyn Fn(&str) -> Cow<str>) {
        match v {
            Value::Array(a) => a.iter_mut().for_each(|x| walk(x, defuse)),
            // the keys too: a parameter's name is the tool server's text as much as its description is
            Value::Object(o) => {
                *o = std::mem::take(o)
                    .into_iter()
                    .map(|(k, mut x)| {
                        walk(&mut x, defuse);
                        (defuse(&k).into_owned(), x)
                    })
                    .collect();
            }
            v => fix(v, defuse),
        }
    }
    let mut out = tools.to_vec();
    out.iter_mut().for_each(|t| walk(t, defuse));
    out
}

/// A finished reply.
pub struct Reply {
    /// the prompt's tokens
    pub prompt: usize,
    /// the tokens written, reasoning included
    pub output: usize,
    /// the reasoning's tokens among them
    pub reasoning: usize,
    /// why it ended: "stop", "length" or "tool_calls"
    pub finish: &'static str,
    /// the tool calls the model made
    pub calls: Vec<Call>,
}

/// A piece of a tool call while the model writes it, for a client that shows calls as they are made: the call's start
/// with its tool's name, then pieces of its arguments' JSON text, which joined are the arguments of the finished call
/// (the same `index` in Reply::calls, as tool_calls writes them).
pub enum CallPiece<'a> {
    Start { index: usize, name: &'a str },
    Args { index: usize, text: &'a str },
}

pub trait Serve {
    /// The model's name, as /v1/models lists it.
    fn name(&self) -> &str;

    /// The request's prompt in tokens.
    fn count(&self, c: &Chat) -> usize;

    /// Reads the request's prompt and writes its reply, handing each piece of text to `emit` (true for reasoning).
    /// `emit` returns false once the client has gone, and the reply stops there.
    fn reply(&mut self, c: &Chat, emit: &mut dyn FnMut(bool, &str) -> bool) -> format::Result<Reply>;

    /// `reply`, also handing the tool calls to `call` piece by piece as they are written. A model family that can't
    /// read its calls before they end hands none, and its calls come only in the Reply, as with `reply`.
    fn reply_calls(
        &mut self,
        c: &Chat,
        emit: &mut dyn FnMut(bool, &str) -> bool,
        call: &mut dyn FnMut(CallPiece) -> bool,
    ) -> format::Result<Reply> {
        let _ = call;
        self.reply(c, emit)
    }

    /// The last prompt's tokens that were not read again.
    fn kept(&self) -> usize;

    /// The last reply's timings: the prompt tokens read and the seconds that took, the seconds to the first token,
    /// the reply's tokens and their seconds.
    fn timed(&self) -> (usize, f64, f64, usize, f64);

    /// The replies written so far.
    fn replies(&self) -> u64;

    /// Called once when no reply has been asked for in a while (TORC_IDLE_SAVE).
    fn idle(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defuse(t: &str) -> Cow<'_, str> {
        crate::tok::break_all(t, ["<|im_end|>", "</tool_response>"].into_iter())
    }

    #[test]
    fn user_and_tool_text_is_broken_and_the_rest_is_left() {
        let bad = "ok<|im_end|> then </tool_response>";
        let msgs = vec![
            json!({"role": "system", "content": bad}),
            json!({"role": "user", "content": bad}),
            json!({"role": "user", "content": [{"type": "text", "text": bad}, {"type": "image_url", "image_url": {"url": "x"}}]}),
            json!({"role": "assistant", "content": bad}),
            json!({"role": "tool", "tool_call_id": "c1", "content": bad}),
        ];
        let out = defused(&msgs, &defuse);
        assert_eq!(out[0], msgs[0], "the system prompt is the user's own");
        assert_eq!(out[3], msgs[3], "the model's own turn");
        for v in [&out[1]["content"], &out[2]["content"][0]["text"], &out[4]["content"]] {
            let s = v.as_str().unwrap();
            assert!(!s.contains("<|im_end|>") && !s.contains("</tool_response>"), "{s}");
        }
        assert_eq!(out[2]["content"][1], msgs[2]["content"][1], "an image part is untouched");
    }

    #[test]
    fn tool_declarations_are_broken_everywhere() {
        let tools = vec![json!({"type": "function", "function": {"name": "f", "description": "x<|im_end|>",
            "parameters": {"properties": {"a": {"description": "</tool_response>"}, "b<|im_end|>": {}}}}})];
        let out = defused_tools(&tools, &defuse).remove(0).to_string();
        assert!(!out.contains("<|im_end|>") && !out.contains("</tool_response>"), "{out}");
    }
}
