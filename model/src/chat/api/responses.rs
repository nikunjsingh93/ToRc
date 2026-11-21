//! responses.rs - the OpenAI Responses API (POST /v1/responses): its requests as a [`Chat`], and its replies, whole
//! or as server-sent events.

use super::*;

/// A request of the OpenAI Responses API, as a [`Chat`]. `previous_response_id` is not supported: the caller must send
/// the whole conversation in `input`.
pub fn from_responses(req: &Value) -> Chat {
    let mut system: Vec<String> = req["instructions"].as_str().map(|s| vec![s.to_string()]).unwrap_or_default();
    let mut msgs: Vec<Value> = Vec::new();
    let items: Vec<Value> = match &req["input"] {
        Value::String(t) => vec![json!({"type": "message", "role": "user", "content": t})],
        Value::Array(a) => a.clone(),
        _ => vec![],
    };
    for it in &items {
        let ty = it["type"].as_str().unwrap_or(if it["role"].is_string() { "message" } else { "" });
        match ty {
            "message" => {
                let role = it["role"].as_str().unwrap_or("user");
                let text = content_text(&it["content"]);
                match role {
                    // the system's words, wherever they appear, go first with the instructions
                    "system" | "developer" => system.push(text),
                    // a user turn with images keeps its parts, so the server can read the images (vision.rs)
                    _ if crate::vision::enabled()
                        && it["content"].as_array().is_some_and(|a| a.iter().any(crate::vision::is_image_part)) =>
                    {
                        msgs.push(json!({"role": role, "content": it["content"]}))
                    }
                    _ => msgs.push(json!({"role": role, "content": text})),
                }
            }
            "function_call" => {
                let call = json!({"id": it["call_id"], "type": "function",
                    "function": {"name": it["name"], "arguments": it["arguments"]}});
                // the calls of one turn go on one assistant message (its text, if any, came just before)
                match msgs.last_mut() {
                    Some(m) if m["role"] == "assistant" => {
                        if let Some(a) = m["tool_calls"].as_array_mut() {
                            a.push(call);
                        } else {
                            m["tool_calls"] = json!([call]);
                        }
                    }
                    _ => msgs.push(json!({"role": "assistant", "content": "", "tool_calls": [call]})),
                }
            }
            "function_call_output" => {
                let out = match &it["output"] {
                    Value::String(t) => t.clone(),
                    v => content_text(v),
                };
                msgs.push(json!({"role": "tool", "tool_call_id": it["call_id"], "content": out}));
            }
            // reasoning items and anything else are not read
            _ => {}
        }
    }
    let tools = req["tools"]
        .as_array()
        .map(|v| v.as_slice())
        .unwrap_or(&[])
        .iter()
        .filter(|t| t["type"].as_str() == Some("function"))
        .map(|t| function_tool(&t["name"], &t["description"], &t["parameters"]))
        .collect();
    let think = match req["reasoning"]["effort"].as_str() {
        Some("none" | "minimal") => Some(false),
        Some(_) => Some(true),
        None => None,
    };
    Chat {
        msgs: with_system(system, msgs),
        tools,
        max: req["max_output_tokens"].as_u64().unwrap_or_else(super::default_max) as usize,
        stream: req["stream"].as_bool().unwrap_or(false),
        think,
        effort: req["reasoning"]["effort"].as_str().map(str::to_string),
        think_budget: super::think_budget(req["reasoning"]["effort"].as_str(), None),
        show_reasoning: think != Some(false) && req["reasoning"]["summary"].is_string(),
        temperature: req["temperature"].as_f64().unwrap_or(0.0) as f32,
        top_p: req["top_p"].as_f64().unwrap_or(1.0) as f32,
        top_k: None,
        seed: None,
        presence_penalty: req["presence_penalty"].as_f64().unwrap_or(0.0) as f32,
        frequency_penalty: req["frequency_penalty"].as_f64().unwrap_or(0.0) as f32,
    }
}

fn responses_usage(u: &Usage) -> Value {
    json!({"input_tokens": u.prompt, "input_tokens_details": {"cached_tokens": u.cached},
        "output_tokens": u.output, "output_tokens_details": {"reasoning_tokens": u.reasoning},
        "total_tokens": u.prompt + u.output})
}

/// The Responses API's reply, streamed or whole: [`delta`](Self::delta) for each piece of text or reasoning, then
/// [`finish`](Self::finish). With `stream`, each returns the events to write; otherwise they return nothing, and
/// `finish` returns the response object as JSON.
pub struct ResponsesStream {
    id: String,
    model: String,
    created: u64,
    stream: bool,
    show_reasoning: bool,
    seq: u64,
    /// the finished output items
    output: Vec<Value>,
    /// the open item: (whether it is reasoning, its id, its text so far)
    open: Option<(bool, String, String)>,
}

impl ResponsesStream {
    pub fn new(model: &str, stream: bool, show_reasoning: bool) -> Self {
        let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        ResponsesStream {
            id: new_id("resp"),
            model: model.into(),
            created,
            stream,
            show_reasoning,
            seq: 0,
            output: Vec::new(),
            open: None,
        }
    }

    fn ev(&mut self, ty: &str, mut data: Value) -> String {
        if !self.stream {
            return String::new();
        }
        data["type"] = json!(ty);
        data["sequence_number"] = json!(self.seq);
        self.seq += 1;
        sse(ty, &data)
    }

    fn response(&self, status: &str, usage: Option<&Usage>, incomplete: Option<&str>) -> Value {
        json!({"id": self.id, "object": "response", "created_at": self.created, "status": status,
            "model": self.model, "output": self.output,
            "incomplete_details": incomplete.map(|r| json!({"reason": r})),
            "usage": usage.map(responses_usage)})
    }

    /// response.created and response.in_progress.
    pub fn start(&mut self) -> String {
        let r = self.response("in_progress", None, None);
        let a = self.ev("response.created", json!({"response": r}));
        a + &self.ev("response.in_progress", json!({"response": r}))
    }

    fn close(&mut self) -> String {
        let Some((reasoning, id, text)) = self.open.take() else { return String::new() };
        let oi = self.output.len();
        let mut out = String::new();
        let item = if reasoning {
            out += &self.ev(
                "response.reasoning_summary_text.done",
                json!({"item_id": id, "output_index": oi,
                "summary_index": 0, "text": text}),
            );
            let part = json!({"type": "summary_text", "text": text});
            out += &self.ev(
                "response.reasoning_summary_part.done",
                json!({"item_id": id, "output_index": oi,
                "summary_index": 0, "part": part}),
            );
            json!({"type": "reasoning", "id": id, "summary": [part]})
        } else {
            out += &self.ev(
                "response.output_text.done",
                json!({"item_id": id, "output_index": oi,
                "content_index": 0, "text": text}),
            );
            let part = json!({"type": "output_text", "text": text, "annotations": []});
            out += &self.ev(
                "response.content_part.done",
                json!({"item_id": id, "output_index": oi,
                "content_index": 0, "part": part}),
            );
            json!({"type": "message", "id": id, "status": "completed", "role": "assistant", "content": [part]})
        };
        out += &self.ev("response.output_item.done", json!({"output_index": oi, "item": item}));
        self.output.push(item);
        out
    }

    /// Streams one fragment of the answer, or of the thinking when `reasoning` is set.
    pub fn delta(&mut self, reasoning: bool, text: &str) -> String {
        if text.is_empty() || reasoning && !self.show_reasoning {
            return String::new();
        }
        let mut out = String::new();
        if self.open.as_ref().map(|o| o.0) != Some(reasoning) {
            out += &self.close();
            let oi = self.output.len();
            let id = new_id(if reasoning { "rs" } else { "msg" });
            if reasoning {
                out += &self.ev(
                    "response.output_item.added",
                    json!({"output_index": oi,
                    "item": {"type": "reasoning", "id": id, "summary": []}}),
                );
                out += &self.ev(
                    "response.reasoning_summary_part.added",
                    json!({"item_id": id, "output_index": oi,
                    "summary_index": 0, "part": {"type": "summary_text", "text": ""}}),
                );
            } else {
                out += &self.ev("response.output_item.added", json!({"output_index": oi,
                    "item": {"type": "message", "id": id, "status": "in_progress", "role": "assistant", "content": []}}));
                out += &self.ev(
                    "response.content_part.added",
                    json!({"item_id": id, "output_index": oi,
                    "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
                );
            }
            self.open = Some((reasoning, id, String::new()));
        }
        let oi = self.output.len();
        let o = self.open.as_mut().unwrap();
        o.2 += text;
        let id = o.1.clone();
        out + &if reasoning {
            self.ev(
                "response.reasoning_summary_text.delta",
                json!({"item_id": id, "output_index": oi,
                "summary_index": 0, "delta": text}),
            )
        } else {
            self.ev(
                "response.output_text.delta",
                json!({"item_id": id, "output_index": oi, "content_index": 0,
                "delta": text}),
            )
        }
    }

    /// The tool calls (each one a function_call item) and the end: the events of a stream, or else the response object.
    pub fn finish(&mut self, calls: &Value, reason: &str, u: &Usage) -> String {
        let mut out = self.close();
        for c in calls.as_array().map(|v| v.as_slice()).unwrap_or(&[]) {
            let (name, args) = call_parts(c);
            let (id, call_id) = (new_id("fc"), new_id("call"));
            let oi = self.output.len();
            out += &self.ev(
                "response.output_item.added",
                json!({"output_index": oi, "item": {"type": "function_call",
                "id": id, "call_id": call_id, "name": name, "arguments": "", "status": "in_progress"}}),
            );
            out += &self.ev(
                "response.function_call_arguments.delta",
                json!({"item_id": id, "output_index": oi,
                "delta": args}),
            );
            out += &self.ev(
                "response.function_call_arguments.done",
                json!({"item_id": id, "output_index": oi,
                "arguments": args}),
            );
            let item = json!({"type": "function_call", "id": id, "call_id": call_id, "name": name, "arguments": args,
                "status": "completed"});
            out += &self.ev("response.output_item.done", json!({"output_index": oi, "item": item}));
            self.output.push(item);
        }
        let (status, incomplete) =
            if reason == "length" { ("incomplete", Some("max_output_tokens")) } else { ("completed", None) };
        let r = self.response(status, Some(u), incomplete);
        if !self.stream {
            return r.to_string();
        }
        let ty = if incomplete.is_some() { "response.incomplete" } else { "response.completed" };
        out + &self.ev(ty, json!({"response": r}))
    }

    /// An error as the Responses API streams it.
    pub fn error(&mut self, message: &str) -> String {
        self.ev("error", json!({"code": "server_error", "message": message}))
    }
}
