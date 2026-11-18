//! anthropic.rs - the Anthropic Messages API (POST /v1/messages): its requests as a [`Chat`], and its replies, whole
//! or as server-sent events.

use super::*;

/// Maps why a reply ended, as the servers say it ("stop", "length" or "tool_calls"), to Anthropic's terms.
fn anthropic_stop(reason: &str) -> &'static str {
    match reason {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
}

/// A request of the Anthropic Messages API, as a [`Chat`].
pub fn from_messages(req: &Value) -> Chat {
    let mut msgs = Vec::new();
    for m in req["messages"].as_array().map(|v| v.as_slice()).unwrap_or(&[]) {
        let role = m["role"].as_str().unwrap_or("user");
        let Value::Array(blocks) = &m["content"] else {
            msgs.push(json!({"role": role, "content": content_text(&m["content"])}));
            continue;
        };
        let mut text = String::new();
        let mut calls = Vec::new();
        // the user's words and images in order, when it has images and a vision tower is loaded
        let mut parts = Vec::new();
        let mut has_image = false;
        for b in blocks {
            match b["type"].as_str().unwrap_or("") {
                "text" => {
                    text += b["text"].as_str().unwrap_or("");
                    parts.push(json!({"type": "text", "text": b["text"]}));
                }
                "image" if crate::vision::enabled() && role != "assistant" => {
                    has_image = true;
                    parts.push(b.clone());
                }
                // a tool's answer becomes its own message, before the user's words that came with it
                "tool_result" => {
                    let mut out = content_text(&b["content"]);
                    if b["is_error"].as_bool() == Some(true) {
                        out = format!("Error: {out}");
                    }
                    msgs.push(json!({"role": "tool", "tool_call_id": b["tool_use_id"], "content": out}));
                }
                "tool_use" => calls.push(json!({"id": b["id"], "type": "function",
                    "function": {"name": b["name"], "arguments": b["input"]}})),
                // thinking, redacted thinking, images and documents are not read
                _ => {}
            }
        }
        if role == "assistant" {
            let mut a = json!({"role": "assistant", "content": text});
            if !calls.is_empty() {
                a["tool_calls"] = Value::Array(calls);
            }
            msgs.push(a);
        } else if has_image {
            msgs.push(json!({"role": role, "content": parts}));
        } else if !text.is_empty() {
            msgs.push(json!({"role": role, "content": text}));
        }
    }
    let system = match &req["system"] {
        Value::Null => vec![],
        v => vec![content_text(v)],
    };
    // server tools (web search and the like) have no input schema, so they are not offered
    let tools = req["tools"]
        .as_array()
        .map(|v| v.as_slice())
        .unwrap_or(&[])
        .iter()
        .filter(|t| t["input_schema"].is_object())
        .map(|t| function_tool(&t["name"], &t["description"], &t["input_schema"]))
        .collect();
    let think = match req["thinking"]["type"].as_str() {
        Some("enabled" | "adaptive") => Some(true),
        Some("disabled") => Some(false),
        _ => None,
    };
    Chat {
        msgs: with_system(system, msgs),
        tools,
        max: req["max_tokens"].as_u64().unwrap_or_else(super::default_max) as usize,
        stream: req["stream"].as_bool().unwrap_or(false),
        think,
        effort: None,
        show_reasoning: think == Some(true),
        temperature: req["temperature"].as_f64().unwrap_or(0.0) as f32,
        top_p: req["top_p"].as_f64().unwrap_or(1.0) as f32,
        top_k: req["top_k"].as_u64().map(|k| k as usize),
        seed: None,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
    }
}

fn messages_usage(u: &Usage) -> Value {
    json!({"input_tokens": u.prompt - u.cached.min(u.prompt), "output_tokens": u.output,
        "cache_read_input_tokens": u.cached, "cache_creation_input_tokens": 0})
}

/// A whole reply of the Messages API.
pub fn messages_reply(
    model: &str,
    reasoning: &str,
    text: &str,
    calls: &Value,
    reason: &str,
    u: &Usage,
    show_reasoning: bool,
) -> Value {
    let mut content = Vec::new();
    if show_reasoning && !reasoning.is_empty() {
        content.push(json!({"type": "thinking", "thinking": reasoning, "signature": ""}));
    }
    if !text.is_empty() {
        content.push(json!({"type": "text", "text": text}));
    }
    for c in calls.as_array().map(|v| v.as_slice()).unwrap_or(&[]) {
        let (name, args) = call_parts(c);
        let input: Value = serde_json::from_str(&args).unwrap_or(json!({}));
        content.push(json!({"type": "tool_use", "id": new_id("toolu"), "name": name, "input": input}));
    }
    json!({"id": new_id("msg"), "type": "message", "role": "assistant", "model": model, "content": content,
        "stop_reason": anthropic_stop(reason), "stop_sequence": null, "usage": messages_usage(u)})
}

/// The Messages API's event stream: [`start`](Self::start), then [`delta`](Self::delta) for each piece of text or
/// reasoning, then [`finish`](Self::finish). Each returns the events to write.
pub struct MessagesStream {
    show_reasoning: bool,
    /// the next content block's index
    index: usize,
    /// the open block: Some(true) for thinking, Some(false) for text
    open: Option<bool>,
}

impl MessagesStream {
    pub fn new(show_reasoning: bool) -> Self {
        MessagesStream { show_reasoning, index: 0, open: None }
    }

    /// message_start, where `prompt` is the prompt's tokens as far as they are known before the reply.
    pub fn start(&self, model: &str, prompt: usize) -> String {
        let msg = json!({"id": new_id("msg"), "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": prompt, "output_tokens": 0, "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0}});
        sse("message_start", &json!({"type": "message_start", "message": msg}))
    }

    fn close(&mut self) -> String {
        match self.open.take() {
            Some(_) => {
                self.index += 1;
                sse("content_block_stop", &json!({"type": "content_block_stop", "index": self.index - 1}))
            }
            None => String::new(),
        }
    }

    /// Streams one fragment of the answer, or of the thinking when `reasoning` is set.
    pub fn delta(&mut self, reasoning: bool, text: &str) -> String {
        if text.is_empty() || reasoning && !self.show_reasoning {
            return String::new();
        }
        let mut out = String::new();
        if self.open != Some(reasoning) {
            out += &self.close();
            let block = if reasoning {
                json!({"type": "thinking", "thinking": "", "signature": ""})
            } else {
                json!({"type": "text", "text": ""})
            };
            out += &sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": self.index,
                "content_block": block}),
            );
            self.open = Some(reasoning);
        }
        let d = if reasoning {
            json!({"type": "thinking_delta", "thinking": text})
        } else {
            json!({"type": "text_delta", "text": text})
        };
        out + &sse("content_block_delta", &json!({"type": "content_block_delta", "index": self.index, "delta": d}))
    }

    /// The tool calls (each one a block), the stop reason and the usage, then message_stop.
    pub fn finish(&mut self, calls: &Value, reason: &str, u: &Usage) -> String {
        let mut out = self.close();
        for c in calls.as_array().map(|v| v.as_slice()).unwrap_or(&[]) {
            let (name, args) = call_parts(c);
            let i = self.index;
            out += &sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": i,
                "content_block": {"type": "tool_use", "id": new_id("toolu"), "name": name, "input": {}}}),
            );
            out += &sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": i,
                "delta": {"type": "input_json_delta", "partial_json": args}}),
            );
            out += &sse("content_block_stop", &json!({"type": "content_block_stop", "index": i}));
            self.index += 1;
        }
        out += &sse(
            "message_delta",
            &json!({"type": "message_delta",
            "delta": {"stop_reason": anthropic_stop(reason), "stop_sequence": null}, "usage": messages_usage(u)}),
        );
        out + &sse("message_stop", &json!({"type": "message_stop"}))
    }

    /// An error as the Messages API streams it.
    pub fn error(message: &str) -> String {
        sse("error", &json!({"type": "error", "error": {"type": "api_error", "message": message}}))
    }
}

/// An error body of the Messages API.
pub fn messages_error(kind: &str, message: &str) -> Value {
    json!({"type": "error", "error": {"type": kind, "message": message}})
}
