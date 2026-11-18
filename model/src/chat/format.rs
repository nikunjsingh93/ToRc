//! format.rs - what every chat format's writer and reader needs: a message's text and calls, the system message, the
//! tools' answers, JSON as the chat templates write it, and a call's arguments read back by their schema. The formats
//! themselves are in each model's folder under models/ (chat.rs).

use serde_json::{Map, Value};

/// A message's text: its content, or the text parts of it.
pub fn text_of(m: &Value) -> String {
    content_text(&m["content"])
}

/// The text of a content value: a string, or the `text` of each part of an array, joined.
pub fn content_text(c: &Value) -> String {
    match c {
        Value::String(t) => t.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

/// An assistant message's calls as name and arguments (OpenAI sends the arguments as JSON text; an object is taken
/// too).
pub fn calls_of(m: &Value) -> Vec<(String, Map<String, Value>)> {
    m["tool_calls"]
        .as_array()
        .map(|v| v.as_slice())
        .unwrap_or(&[])
        .iter()
        .map(|c| {
            let args = match &c["function"]["arguments"] {
                Value::String(a) => serde_json::from_str(a).unwrap_or_default(),
                Value::Object(o) => o.clone(),
                _ => Map::new(),
            };
            (c["function"]["name"].as_str().unwrap_or("").to_string(), args)
        })
        .collect()
}

/// The first message's text when it is a system message, and the messages after it.
pub fn split_system(msgs: &[Value]) -> (Option<String>, &[Value]) {
    match msgs.first() {
        Some(m) if m["role"].as_str() == Some("system") => (Some(text_of(m)), &msgs[1..]),
        _ => (None, msgs),
    }
}

/// The tool answers from msgs[*i] on, as one turn of <tool_response> blocks; *i moves past them.
pub fn tool_answers(msgs: &[Value], i: &mut usize) -> String {
    let mut out = Vec::new();
    while *i < msgs.len() && msgs[*i]["role"].as_str() == Some("tool") {
        out.push(format!("<tool_response>\n{}\n</tool_response>", text_of(&msgs[*i])));
        *i += 1;
    }
    out.join("\n")
}

/// A value as Jinja's tojson writes it, i.e. Python's json.dumps: ", " and ": " separators, keys in their order,
/// non-ASCII left as it is.
pub fn tojson(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let kv: Vec<String> =
                m.iter().map(|(k, x)| format!("{}: {}", Value::String(k.clone()), tojson(x))).collect();
            format!("{{{}}}", kv.join(", "))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(tojson).collect::<Vec<_>>().join(", ")),
        _ => v.to_string(),
    }
}

/// A tool's parameters in the request, if the tool is declared.
pub fn props<'a>(tools: &'a [Value], name: &str) -> Option<&'a Value> {
    tools
        .iter()
        .map(|t| if t["function"].is_object() { &t["function"] } else { t })
        .find(|f| f["name"].as_str() == Some(name))
        .map(|f| &f["parameters"]["properties"])
}

/// A call's argument written as text: kept as text when its schema says string (or gives no type), otherwise parsed as
/// JSON when it parses.
pub fn value(props: Option<&Value>, key: &str, v: &str) -> Value {
    if is_text(props, key) {
        Value::String(v.into())
    } else {
        serde_json::from_str(v).unwrap_or(Value::String(v.into()))
    }
}

/// Whether `value` keeps a call's argument as text: its schema says string (or gives no type).
pub fn is_text(props: Option<&Value>, key: &str) -> bool {
    let schema = props.map(|ps| &ps[key]);
    match schema.map(|sc| &sc["type"]) {
        Some(Value::String(t)) => t == "string",
        Some(Value::Array(ts)) => ts.iter().any(|t| t == "string"),
        _ => {
            schema.is_none_or(|sc| !sc["properties"].is_object() && !sc["items"].is_object() && !sc["enum"].is_array())
        }
    }
}

/// The chat and the tools every format's pinned prompt test writes, so the formats are tested on the same thing.
#[cfg(test)]
pub mod fixtures {
    use serde_json::{Value, json};

    /// A system message, a question, a call and its answer, the reply, and a last user turn.
    pub fn chat() -> Vec<Value> {
        vec![
            json!({"role": "system", "content": "Be brief."}),
            json!({"role": "user", "content": "What is the weather in Paris?"}),
            json!({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "17C, light rain"}),
            json!({"role": "assistant", "content": "It is 17C with light rain."}),
            json!({"role": "user", "content": "Thanks"}),
        ]
    }

    /// The same chat without the call and its answer.
    pub fn plain() -> Vec<Value> {
        chat().into_iter().filter(|m| m["role"] != "tool" && m["tool_calls"].is_null()).collect()
    }

    pub fn tools() -> Vec<Value> {
        vec![json!({"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})]
    }

    /// The text a pinned prompt must be, from its JSON form.
    pub fn want(json: &str) -> String {
        serde_json::from_str(json).unwrap()
    }

    /// Tools with typed parameters, for the call readers.
    pub fn typed_tools() -> Vec<Value> {
        vec![
            json!({"type": "function", "function": {"name": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}),
            json!({"type": "function", "function": {"name": "calc", "parameters": {"type": "object", "properties": {
                "n": {"type": "integer"}, "opts": {"type": "object", "properties": {}}, "tags": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string"}, "any": {}, "mode": {"enum": ["fast", "slow"]}, "bad": {"type": ["number", "null"]}}}}}),
            json!({"name": "Write", "parameters": {"properties": {"text": {"type": "string"}, "path": {"type": ["string", "null"]}}}}),
            json!({"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}}),
        ]
    }

    /// Calls as name and arguments, for comparing.
    pub fn named(calls: &[crate::serve::Call]) -> Vec<(String, Value)> {
        calls.iter().map(|c| (c.name.clone(), Value::Object(c.args.clone()))).collect()
    }
}
