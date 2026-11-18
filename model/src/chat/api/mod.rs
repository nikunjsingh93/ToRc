//! api - the chat APIs other than chat completions, on top of the chat completions request the servers run: the
//! Anthropic Messages API (POST /v1/messages, anthropic.rs) and the OpenAI Responses API (POST /v1/responses,
//! responses.rs).
//!
//! Each request is rewritten as a [`Chat`]: OpenAI chat messages and function tools, the shape the servers' prompt
//! builder reads. Each reply (its text, its reasoning and the tool calls it made) is written back in the caller's API,
//! either as one response or as that API's server-sent events: [`MessagesStream`] and [`ResponsesStream`] take the
//! pieces as the model produces them and return the events to write.
//!
//! Tool calls arrive here as OpenAI `tool_calls` entries (with `function.arguments` as a JSON string), as the servers
//! parse them from the model's output.

mod anthropic;
mod responses;

pub use anthropic::{MessagesStream, from_messages, messages_error, messages_reply};
pub use responses::{ResponsesStream, from_responses};

use crate::chat::format::content_text;
use serde_json::{Map, Value, json};
use std::sync::atomic::{AtomicU64, Ordering};

/// The most tokens a reply may take when the request doesn't say, the same for all three APIs. A thinking model can
/// spend a thousand or two before it answers, so a small default ends many replies with nothing but reasoning.
pub const DEFAULT_MAX: usize = 4096;

/// The reply limit for a request that doesn't set one: TORC_MAX_TOKENS when it is set to a number above 0 (clients
/// such as chat apps often send no max_tokens, and long code answers run past DEFAULT_MAX), else DEFAULT_MAX.
pub fn default_max() -> u64 {
    crate::switches::num::<u64>("TORC_MAX_TOKENS").filter(|&n| n > 0).unwrap_or(DEFAULT_MAX as u64)
}

/// A request in the form the chat completions endpoint takes.
#[derive(Debug, Clone, PartialEq)]
pub struct Chat {
    /// OpenAI chat messages: system, user, assistant (with `tool_calls`) and tool (with `tool_call_id`)
    pub msgs: Vec<Value>,
    /// each one `{"type": "function", "function": {"name", "description", "parameters"}}`
    pub tools: Vec<Value>,
    /// the most tokens to generate
    pub max: usize,
    pub stream: bool,
    /// reasoning on or off; None means the server's default
    pub think: Option<bool>,
    /// the reasoning effort the request named ("low", "medium", "high"), for a model that takes one
    pub effort: Option<String>,
    /// whether the reasoning is sent to the client (otherwise it is generated but not sent)
    pub show_reasoning: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
    /// OpenAI's penalties on the reply's own tokens: presence once a token has appeared, frequency per time
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

/// The tokens a reply cost, as both APIs report them.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Usage {
    /// the prompt's tokens, including those not read again (`cached`)
    pub prompt: usize,
    /// the prompt's tokens kept from an earlier request
    pub cached: usize,
    /// the tokens generated, including reasoning
    pub output: usize,
    pub reasoning: usize,
}

/// An id no other in this process has: the prefix, the time in ms, and a counter.
pub fn new_id(prefix: &str) -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
    format!("{prefix}_{ms:x}{:04x}", N.fetch_add(1, Ordering::Relaxed) & 0xffff)
}

/// A function tool in the chat completions shape.
fn function_tool(name: &Value, description: &Value, parameters: &Value) -> Value {
    let mut f = Map::new();
    f.insert("name".into(), name.clone());
    if !description.is_null() {
        f.insert("description".into(), description.clone());
    }
    f.insert(
        "parameters".into(),
        if parameters.is_null() { json!({"type": "object", "properties": {}}) } else { parameters.clone() },
    );
    json!({"type": "function", "function": Value::Object(f)})
}

/// The system texts, joined and given first as one system message (the prompt builder reads a system message first).
fn with_system(system: Vec<String>, mut msgs: Vec<Value>) -> Vec<Value> {
    let system: Vec<String> = system.into_iter().filter(|s| !s.trim().is_empty()).collect();
    if !system.is_empty() {
        msgs.insert(0, json!({"role": "system", "content": system.join("\n\n")}));
    }
    msgs
}

/// A chat completions request as a [`Chat`] (the servers' own endpoint, so all three APIs reach the model the same
/// way).
pub fn from_chat_completions(req: &Value) -> Chat {
    // thinking is off with chat_template_kwargs.enable_thinking false, "think": false, or reasoning_effort "none" / a
    // reasoning budget of 0, as OpenAI-style clients send it
    let effort_off = matches!(req["reasoning_effort"].as_str(), Some("none" | "minimal"))
        || req["reasoning_budget_tokens"].as_i64() == Some(0);
    let think = req["chat_template_kwargs"]["enable_thinking"].as_bool().or(req["think"].as_bool()).or(if effort_off {
        Some(false)
    } else {
        None
    });
    Chat {
        msgs: req["messages"].as_array().cloned().unwrap_or_default(),
        tools: req["tools"].as_array().cloned().unwrap_or_default(),
        max: req["max_tokens"].as_u64().or(req["max_completion_tokens"].as_u64()).unwrap_or_else(default_max)
            as usize,
        stream: req["stream"].as_bool().unwrap_or(false),
        think,
        effort: req["reasoning_effort"]
            .as_str()
            .or(req["chat_template_kwargs"]["reasoning_effort"].as_str())
            .map(str::to_string),
        show_reasoning: true,
        temperature: req["temperature"].as_f64().unwrap_or(0.0) as f32,
        top_p: req["top_p"].as_f64().unwrap_or(1.0) as f32,
        top_k: req["top_k"].as_u64().map(|k| k as usize),
        seed: req["seed"].as_u64(),
        presence_penalty: req["presence_penalty"].as_f64().unwrap_or(0.0) as f32,
        frequency_penalty: req["frequency_penalty"].as_f64().unwrap_or(0.0) as f32,
    }
}

/// Splits a tool call into (function name, arguments serialised as JSON text).
fn call_parts(c: &Value) -> (String, String) {
    let name = c["function"]["name"].as_str().unwrap_or("").to_string();
    let args = match &c["function"]["arguments"] {
        Value::String(a) => a.clone(),
        v => v.to_string(),
    };
    (name, args)
}

/// One server-sent event.
fn sse(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_request() {
        let req = json!({
            "model": "x", "max_tokens": 100, "stream": true, "system": [{"type": "text", "text": "Be brief."}],
            "thinking": {"type": "enabled", "budget_tokens": 1000},
            "tools": [{"name": "Read", "description": "read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
                {"type": "web_search_20250305", "name": "web_search"}],
            "messages": [
                {"role": "user", "content": "Read a.txt"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "s"},
                    {"type": "text", "text": "Reading."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.txt"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "hello"},
                    {"type": "text", "text": "And?"}]}]});
        let c = from_messages(&req);
        assert_eq!(
            c.msgs,
            vec![
                json!({"role": "system", "content": "Be brief."}),
                json!({"role": "user", "content": "Read a.txt"}),
                json!({"role": "assistant", "content": "Reading.", "tool_calls": [{"id": "toolu_1", "type": "function",
                "function": {"name": "Read", "arguments": {"path": "a.txt"}}}]}),
                json!({"role": "tool", "tool_call_id": "toolu_1", "content": "hello"}),
                json!({"role": "user", "content": "And?"}),
            ]
        );
        assert_eq!(c.tools.len(), 1);
        assert_eq!(c.tools[0]["function"]["name"], "Read");
        assert_eq!(c.tools[0]["function"]["parameters"]["properties"]["path"]["type"], "string");
        assert_eq!((c.max, c.stream, c.think, c.show_reasoning), (100, true, Some(true), true));
    }

    #[test]
    fn responses_request() {
        let req = json!({
            "model": "x", "instructions": "You are a coder.", "stream": true, "reasoning": {"effort": "low"},
            "tools": [{"type": "function", "name": "shell", "description": "run",
                "parameters": {"type": "object", "properties": {"command": {"type": "array"}}}},
                {"type": "web_search"}],
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Be careful."}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List files"}]},
                {"type": "reasoning", "summary": []},
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{\"command\":[\"ls\"]}"},
                {"type": "function_call_output", "call_id": "c1", "output": "a.txt"}]});
        let c = from_responses(&req);
        assert_eq!(
            c.msgs,
            vec![
                json!({"role": "system", "content": "You are a coder.\n\nBe careful."}),
                json!({"role": "user", "content": "List files"}),
                json!({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "shell", "arguments": "{\"command\":[\"ls\"]}"}}]}),
                json!({"role": "tool", "tool_call_id": "c1", "content": "a.txt"}),
            ]
        );
        assert_eq!(c.tools.len(), 1);
        assert_eq!((c.think, c.show_reasoning), (Some(true), false));
    }

    fn events(s: &str) -> Vec<Value> {
        s.split("\n\n")
            .filter_map(|e| e.lines().find_map(|l| l.strip_prefix("data: ")))
            .map(|d| serde_json::from_str(d).unwrap())
            .collect()
    }

    #[test]
    fn messages_stream() {
        let mut m = MessagesStream::new(true);
        let mut s = m.start("x", 10);
        s += &m.delta(true, "think");
        s += &m.delta(false, "Hi");
        s += &m.delta(false, " there");
        let calls = json!([{"id": "call_0", "type": "function", "function": {"name": "Read", "arguments": "{\"path\":\"a\"}"}}]);
        s += &m.finish(&calls, "tool_calls", &Usage { prompt: 10, cached: 4, output: 5, reasoning: 1 });
        let ev = events(&s);
        let types: Vec<&str> = ev.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert_eq!(ev[4]["index"], 1);
        assert_eq!(ev[8]["content_block"]["type"], "tool_use");
        assert_eq!(ev[8]["index"], 2);
        assert_eq!(ev[9]["delta"]["partial_json"], "{\"path\":\"a\"}");
        assert_eq!(ev[11]["delta"]["stop_reason"], "tool_use");
        assert_eq!(ev[11]["usage"]["input_tokens"], 6);
        assert_eq!(ev[11]["usage"]["cache_read_input_tokens"], 4);
    }

    #[test]
    fn messages_stream_hides_reasoning() {
        let mut m = MessagesStream::new(false);
        let s = m.delta(true, "secret") + &m.delta(false, "ok") + &m.finish(&json!([]), "stop", &Usage::default());
        let ev = events(&s);
        assert!(ev.iter().all(|e| e["content_block"]["type"] != "thinking"));
        assert_eq!(ev[0]["index"], 0);
        assert_eq!(ev.iter().find(|e| e["type"] == "message_delta").unwrap()["delta"]["stop_reason"], "end_turn");
    }

    #[test]
    fn responses_stream() {
        let mut r = ResponsesStream::new("x", true, true);
        let mut s = r.start();
        s += &r.delta(true, "plan");
        s += &r.delta(false, "Done");
        let calls = json!([{"function": {"name": "shell", "arguments": "{}"}}]);
        s += &r.finish(&calls, "tool_calls", &Usage { prompt: 3, cached: 0, output: 2, reasoning: 1 });
        let ev = events(&s);
        let seqs: Vec<u64> = ev.iter().map(|e| e["sequence_number"].as_u64().unwrap()).collect();
        assert_eq!(seqs, (0..ev.len() as u64).collect::<Vec<_>>());
        let last = ev.last().unwrap();
        assert_eq!(last["type"], "response.completed");
        let out = last["response"]["output"].as_array().unwrap();
        assert_eq!(
            out.iter().map(|o| o["type"].as_str().unwrap()).collect::<Vec<_>>(),
            ["reasoning", "message", "function_call"]
        );
        assert_eq!(out[1]["content"][0]["text"], "Done");
        assert_eq!(out[2]["arguments"], "{}");
        assert_eq!(last["response"]["usage"]["total_tokens"], 5);
    }

    #[test]
    fn responses_whole() {
        let mut r = ResponsesStream::new("x", false, false);
        assert_eq!(r.start() + &r.delta(true, "hidden") + &r.delta(false, "Hello"), "");
        let v: Value = serde_json::from_str(&r.finish(&json!([]), "length", &Usage::default())).unwrap();
        assert_eq!(v["status"], "incomplete");
        assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(v["output"].as_array().unwrap().len(), 1);
        assert_eq!(v["output"][0]["content"][0]["text"], "Hello");
    }
}
