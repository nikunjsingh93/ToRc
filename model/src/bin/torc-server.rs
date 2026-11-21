/*
 * ToRc - torc-server, the chat server
 *
 *                                                                                    ,---,_          ,
 *     Author: Rendeverance                                                           _>   `'-.  .--'/
 *                                                                               .--'` ._      `/   <_
 *                                                                                 >,-' ._'.. ..__ . ' '-.
 *                                                                             .-'   .'`         `'.     '.
 *                                                                              >   / >`-.     .-'< \ , '._\
 *                                                                             /    ; '-._>   <_.-' ;  '._>
 *                                                                            `>  ,/  /___\ /___\  \_  /
 *                                                                             `.-|(|  \o_/  \o_/   |)|`
 *                                                                                 \;        \      ;/
 *                                                                                   \  .-,   )-.  /
 *                                                                                    /`  .'-'.  `\
 *                                                                                   ;_.-`.___.'-.;
 *
 *      torc-server.rs
 *      Big models, small machines.
 */
//! torc-server <model> [port] [threads] [ctx] [name]: a chat server for every model ToRc runs.
//!
//! One conversation at a time, over HTTP: GET /health and /v1/models (also /v1/models/<name>), and POST
//! /v1/chat/completions (OpenAI), /v1/messages (Anthropic) and /v1/responses (OpenAI Responses), each as one JSON reply
//! or as that API's stream of events. GET /torc/settings gives the settings it runs with as a profile, and POST
//! /torc/settings reads one in for its next start (runtime/profile.rs, AUTOTUNE.md). This file is the HTTP side. The
//! model answers through chat/serve.rs, each family from its own folder under models/, picked by the file's
//! architecture.

use model::api;
use model::chat::serve::{CallPiece, Reply, Serve, tool_calls};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

/// OpenAI's usage block. cached_tokens is how many prompt tokens were not read again (from the state kept from the last
/// request, its mark or the system prompt's).
fn usage(np: usize, nr: usize, nt: usize, kept: usize) -> Value {
    json!({"prompt_tokens": np, "completion_tokens": nr, "total_tokens": np + nr,
        "prompt_tokens_details": {"cached_tokens": kept}, "completion_tokens_details": {"reasoning_tokens": nt}})
}

/// The engine's own timing of a reply, in llama-server's form: the prompt tokens read and the time reading them, the
/// reply tokens and the time from the first token going out to the end, and the time to the first token (ttft_ms).
/// No HTTP or client time is in it.
fn timings((pn, ps, ft, rn, rs): (usize, f64, f64, usize, f64)) -> Value {
    let per = |n: usize, s: f64| if s > 0.0 { n as f64 / s } else { 0.0 };
    json!({"prompt_n": pn, "prompt_ms": ps * 1e3, "prompt_per_second": per(pn, ps), "ttft_ms": ft * 1e3,
        "predicted_n": rn, "predicted_ms": rs * 1e3, "predicted_per_second": per(rn, rs)})
}

fn respond(s: &mut TcpStream, status: &str, ctype: &str, body: &str) {
    // the charset named: a client that guesses (Windows PowerShell 5.1 reads ISO-8859-1) shows UTF-8 text wrongly
    let _ = write!(
        s,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
}

/// The largest request body read (a chat with long attachments runs to a few MB).
const MAX_BODY: usize = 16 << 20;

/// A request's first line and headers together may take this many bytes.
const MAX_HEAD: usize = 64 << 10;

/// A whole request must arrive within this long.
const REQUEST_DEADLINE: std::time::Duration = std::time::Duration::from_secs(120);

/// The connection while a request is read. Each read waits at most until the request's deadline (and never more than
/// 10 s), so a client that sends its head a byte at a time can't hold the server past REQUEST_DEADLINE: read_line
/// would otherwise keep waiting as long as each byte comes inside the socket's timeout.
struct Deadline {
    s: TcpStream,
    end: std::time::Instant,
}

impl Read for Deadline {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.end.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            return Err(std::io::ErrorKind::TimedOut.into());
        }
        self.s.set_read_timeout(Some(left.min(std::time::Duration::from_secs(10))))?;
        self.s.read(buf)
    }
}

/// One line of a request's head, within what is left of MAX_HEAD. None when the head is too long, the line is cut
/// short or the read fails.
fn head_line(r: &mut BufReader<Deadline>, left: &mut usize) -> Option<String> {
    // nothing left: a head that has used all of MAX_HEAD without ending is too long, not finished
    if *left == 0 {
        return None;
    }
    let mut h = String::new();
    let n = r.by_ref().take(*left as u64).read_line(&mut h).ok()?;
    *left -= n;
    if n > 0 && !h.ends_with('\n') {
        return None;
    }
    Some(h)
}

/// Whether an Origin header's value names this machine (a page served from it), as against another site's page.
fn local_origin(v: &str) -> bool {
    let host = v.trim().split("://").nth(1).unwrap_or("");
    let host = host.split('/').next().unwrap_or("");
    let name =
        if host.starts_with('[') { host.split(']').next().unwrap_or("") } else { host.split(':').next().unwrap_or("") };
    matches!(name, "localhost" | "127.0.0.1" | "[::1")
}

/// The key another machine's requests must carry (TORC_API_KEY), read once at start. None: no key is asked for.
static KEY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

/// The "created" time /v1/models gives the model, in seconds since 1970: when its file was last written.
static CREATED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// With a key set, how long another machine has to send its request's head: every client sends it at once, and the
/// server takes one connection at a time, so a machine without the key must not hold it for REQUEST_DEADLINE.
const KEYED_HEAD_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

/// A key one header line carries: "Authorization: Bearer <key>" (OpenAI's clients) or "x-api-key: <key>"
/// (Anthropic's). The header names and "Bearer" in any case; the key as written.
fn carried_key(line: &str) -> Option<&str> {
    let (name, value) = line.split_once(':')?;
    let value = value.trim();
    match name.trim().to_ascii_lowercase().as_str() {
        "x-api-key" => Some(value),
        "authorization" => {
            let (scheme, key) = value.split_once(' ')?;
            scheme.eq_ignore_ascii_case("bearer").then(|| key.trim())
        }
        _ => None,
    }
    .filter(|k| !k.is_empty())
}

/// Whether two keys are the same, in a time that doesn't depend on where they first differ.
fn same_key(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

/// Whether a request may be served: always without a key set, and always from this machine; from another machine only
/// when one of the keys it carries is the key.
fn allowed(key: Option<&str>, local: bool, given: &[String]) -> bool {
    match key {
        None => true,
        Some(_) if local => true,
        Some(k) => given.iter().any(|g| same_key(g.as_bytes(), k.as_bytes())),
    }
}

/// A connection from this machine: 127.0.0.1 and the rest of 127/8, ::1, and 127.0.0.1 written as IPv6.
fn local_peer(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => v4.is_loopback(),
        std::net::IpAddr::V6(v6) => v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback()),
    }
}

/// The refusal of a request without the key, in the error shape of the API it came by, so a client shows it.
fn key_refused(path: &str) -> String {
    let text = "this server needs its API key from other machines: send it as \"Authorization: Bearer <key>\" or \
                \"x-api-key: <key>\"";
    if route(path).starts_with("/v1/messages") {
        json!({"type": "error", "error": {"type": "authentication_error", "message": text}}).to_string()
    } else {
        json!({"error": {"message": text, "type": "invalid_request_error", "code": "invalid_api_key"}}).to_string()
    }
}

/// A request's path as the routes name it: without its query (some clients ask for /v1/messages?beta=true) or a
/// trailing slash, and with the /v1 that clients given a base address without it leave out.
fn route(path: &str) -> String {
    let p = path.split('?').next().unwrap_or("").trim_end_matches('/');
    if p.is_empty() || p == "/v1" || p.starts_with("/v1/") || p.starts_with("/torc/") {
        p.to_string()
    } else {
        format!("/v1{p}")
    }
}

/// The reasoning effort levels the model takes (set once, from the engine or, in the supervisor, from the file).
static LEVELS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();

/// The loaded model as /v1/models lists it. A model that reasons also lists its effort levels, which is how apps
/// that look for them (JoltDesk reads `reasoning_effort_levels`) know to offer them.
fn model_entry(name: &str) -> Value {
    let mut m =
        json!({"id": name, "object": "model", "created": CREATED.get().copied().unwrap_or(0), "owned_by": "torc"});
    if let Some(levels) = LEVELS.get().filter(|l| !l.is_empty()) {
        m["supports_reasoning"] = json!(true);
        m["reasoning_effort_levels"] = json!(levels);
    }
    m
}

/// A path segment with its %XX escapes undone ("Llama%203" is "Llama 3"); a bad escape is kept as written.
fn unescape(segment: &str) -> String {
    let b = segment.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        if b[i] == b'%'
            && i + 2 < b.len()
            && let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2]))
        {
            out.push((h * 16 + l) as u8);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap_or_else(|_| segment.to_string())
}

/// Reads and drops a served request's body that the route doesn't use (GET /health with one).
fn drain(r: &mut BufReader<Deadline>, len: usize) {
    let _ = std::io::copy(&mut r.by_ref().take(len.min(MAX_BODY) as u64), &mut std::io::sink());
}

/// How long a refused request's client is read after the refusal is sent.
const LINGER: std::time::Duration = std::time::Duration::from_secs(5);

/// Refuses a request so that the client gets the reason: the reply goes first, then the sending side is closed, and
/// what the client still sends (a body of any length, by Content-Length or chunked) is read and dropped until it
/// closes, for LINGER at most. Closing a socket with data still unread resets the connection on Windows, and a reset
/// can reach the client before the reply does, which then reports "connection reset" instead of the reason.
fn refuse(s: &mut TcpStream, r: &mut BufReader<Deadline>, status: &str, body: &str) {
    respond(s, status, "application/json", body);
    let _ = s.flush();
    let _ = s.shutdown(std::net::Shutdown::Write);
    r.get_mut().end = std::time::Instant::now() + LINGER;
    let _ = std::io::copy(&mut r.by_ref().take(MAX_BODY as u64), &mut std::io::sink());
}

fn handle(e: &mut dyn Serve, mut s: TcpStream) {
    // one connection at a time, so a client that stalls before finishing its request must not hold the server (the
    // whole request has REQUEST_DEADLINE, see Deadline), and nor must one that stops reading the reply without closing
    // (a write that can't go out in a minute ends it)
    let _ = s.set_write_timeout(Some(std::time::Duration::from_secs(60)));
    let Ok(rs) = s.try_clone() else { return };
    // with TORC_API_KEY set, another machine's request needs the key, and gets only a few seconds to send its head;
    // this machine's never needs it (a failed address lookup counts as another machine)
    let key = KEY.get().and_then(|k| k.as_deref());
    let local = s.peer_addr().map(|a| local_peer(a.ip())).unwrap_or(false);
    let start = std::time::Instant::now();
    let head_end = if key.is_some() && !local { KEYED_HEAD_DEADLINE } else { REQUEST_DEADLINE };
    let mut r = BufReader::new(Deadline { s: rs, end: start + head_end });
    let mut left = MAX_HEAD;
    let Some(line) = head_line(&mut r, &mut left) else { return };
    let (method, path) = {
        let mut it = line.split_whitespace();
        (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string())
    };
    let (mut len, mut foreign) = (0usize, false);
    let mut given = Vec::new();
    loop {
        let Some(h) = head_line(&mut r, &mut left) else { return };
        if h == "\r\n" || h.is_empty() {
            break;
        }
        let low = h.to_ascii_lowercase();
        if let Some(v) = low.strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
        if let Some(v) = low.strip_prefix("origin:") {
            foreign = !local_origin(v);
        }
        if let Some(k) = carried_key(&h) {
            given.push(k.to_string());
        }
    }
    // a browser names the page a request comes from, and another site's page is refused, whatever address the server
    // listens on: it could otherwise run the model from any open tab. Clients that aren't web pages send no Origin
    if foreign {
        let why = json!({"error": "requests from another site's web page are refused"}).to_string();
        return refuse(&mut s, &mut r, "403 Forbidden", &why);
    }
    // whether the server is up, for monitors and frameworks that probe before they start: no key needed (it says
    // nothing but that), as llama-server's /health. It is answered between requests, like everything else
    if method == "GET" && route(&path) == "/v1/health" {
        drain(&mut r, len);
        respond(&mut s, "200 OK", "application/json", r#"{"status":"ok"}"#);
        return;
    }
    if !allowed(key, local, &given) {
        return refuse(&mut s, &mut r, "401 Unauthorized", &key_refused(&path));
    }
    // let in: the rest of the request has what is left of the whole request's time
    r.get_mut().end = start + REQUEST_DEADLINE;
    if len > MAX_BODY {
        let why = json!({"error": format!("request body over {} MB", MAX_BODY >> 20)}).to_string();
        return refuse(&mut s, &mut r, "413 Payload Too Large", &why);
    }
    let mut body = vec![0u8; len];
    let mut got = 0;
    while got < len {
        match r.read(&mut body[got..]) {
            Ok(n) if n > 0 => got += n,
            _ => return,
        }
    }
    let path = route(&path);
    let path = path.as_str();
    match (method.as_str(), path) {
        ("GET", "/v1/models") => {
            let j = json!({"object": "list", "data": [model_entry(e.name())]});
            respond(&mut s, "200 OK", "application/json", &j.to_string());
        }
        // one model by its name, as OpenAI's clients ask before they use it
        ("GET", p) if p.starts_with("/v1/models/") => {
            let asked = unescape(&p["/v1/models/".len()..]);
            if asked == e.name() {
                respond(&mut s, "200 OK", "application/json", &model_entry(e.name()).to_string());
            } else {
                let text = format!("no model {asked} here: this server runs {}", e.name());
                let j = json!({"error": {"message": text, "type": "invalid_request_error", "code": "model_not_found"}});
                respond(&mut s, "404 Not Found", "application/json", &j.to_string());
            }
        }
        ("GET", "/torc/settings") => {
            respond(&mut s, "200 OK", "application/json", &model::profile::export(e.name()).to_string());
        }
        ("POST", "/torc/settings") => {
            let Ok(p) = serde_json::from_slice::<Value>(&body) else {
                return respond(&mut s, "400 Bad Request", "application/json", r#"{"error":"bad json"}"#);
            };
            match model::profile::import(&p) {
                Ok((layout, done)) => {
                    let j = json!({"ok": true, "written": done, "layout": layout, "running": model::tune::layout(),
                        "note": "used from the server's next start in that layout"});
                    respond(&mut s, "200 OK", "application/json", &j.to_string());
                }
                Err(err) => {
                    respond(&mut s, "400 Bad Request", "application/json", &json!({"error": err}).to_string());
                }
            }
        }
        ("POST", "/v1/chat/completions" | "/v1/messages" | "/v1/responses" | "/v1/messages/count_tokens") => {
            let Ok(req) = serde_json::from_slice::<Value>(&body) else {
                return respond(&mut s, "400 Bad Request", "application/json", r#"{"error":"bad json"}"#);
            };
            let (a, c) = match path {
                "/v1/chat/completions" => (Api::Chat, api::from_chat_completions(&req)),
                "/v1/responses" => (Api::Responses, api::from_responses(&req)),
                _ => (Api::Messages, api::from_messages(&req)),
            };
            if path == "/v1/messages/count_tokens" {
                let n = e.count(&c);
                return respond(&mut s, "200 OK", "application/json", &json!({"input_tokens": n}).to_string());
            }
            complete(e, &mut s, a, c);
        }
        _ => respond(&mut s, "404 Not Found", "application/json", r#"{"error":"not found"}"#),
    }
}

/// The API a request came in by, which decides the shape of its reply.
#[derive(Clone, Copy, PartialEq)]
enum Api {
    /// OpenAI chat completions
    Chat,
    /// Anthropic Messages
    Messages,
    /// OpenAI Responses
    Responses,
}

/// The HTTP status for a reply that failed: a prompt that does not fit the context is the request's fault, anything
/// else the server's.
fn status(e: &format::Error) -> &'static str {
    if model::chat::serve::is_too_long(e) { "400 Bad Request" } else { "500 Internal Server Error" }
}

/// Runs a request and writes its reply in the shape of the API it came in by: one JSON body, or that API's stream of
/// events.
fn complete(e: &mut dyn Serve, s: &mut TcpStream, a: Api, c: api::Chat) {
    let name = e.name().to_string();
    if c.stream {
        let _ = write!(
            s,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        );
    }
    let Ok(out) = s.try_clone() else { return };
    // a failed write means the client closed the stream, and run then stops the reply. Shared, so the reply's text
    // and its tool calls can both be sent while it is written
    let out = std::cell::RefCell::new(out);
    let send = |t: &str| {
        t.is_empty() || {
            let mut o = out.borrow_mut();
            write!(o, "{t}").is_ok() && o.flush().is_ok()
        }
    };
    match a {
        Api::Chat => {
            let id = format!(
                "chatcmpl-{}",
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis())
            );
            let chunk = |delta: Value, fin: Value| {
                json!({"id": id, "object": "chat.completion.chunk", "model": name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]})
            };
            if c.stream {
                send(&format!("data: {}\n\n", chunk(json!({"role": "assistant"}), Value::Null)));
                // each call goes out as the model writes it, as llama-server streams them: its id, type and name
                // first, then its arguments' JSON text in pieces; what each call has sent so far is kept here
                let sent = std::cell::RefCell::new(Vec::<String>::new());
                let r = e.reply_calls(
                    &c,
                    &mut |rs, t| {
                        let d = if rs { json!({"reasoning_content": t}) } else { json!({"content": t}) };
                        send(&format!("data: {}\n\n", chunk(d, Value::Null)))
                    },
                    &mut |p| {
                        let d = match p {
                            CallPiece::Start { index, name } => {
                                let mut sent = sent.borrow_mut();
                                let n = sent.len().max(index + 1);
                                sent.resize(n, String::new());
                                json!([{"index": index, "id": format!("call_{index}"), "type": "function",
                                    "function": {"name": name, "arguments": ""}}])
                            }
                            CallPiece::Args { index, text } => {
                                if let Some(a) = sent.borrow_mut().get_mut(index) {
                                    a.push_str(text);
                                }
                                json!([{"index": index, "function": {"arguments": text}}])
                            }
                        };
                        send(&format!("data: {}\n\n", chunk(json!({"tool_calls": d}), Value::Null)))
                    },
                );
                match r {
                    Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                        // what the finished calls add to what went out while they were written: the rest of a call's
                        // arguments, or a whole call that was not streamed
                        let sent = sent.into_inner();
                        let all = tool_calls(&calls);
                        for (i, tc) in all.as_array().into_iter().flatten().enumerate() {
                            let args = tc["function"]["arguments"].as_str().unwrap_or("");
                            let d = match sent.get(i) {
                                None => tc.clone(),
                                Some(so_far) if args.starts_with(so_far.as_str()) => {
                                    if args.len() == so_far.len() {
                                        continue;
                                    }
                                    json!({"index": i, "function": {"arguments": &args[so_far.len()..]}})
                                }
                                Some(so_far) => {
                                    eprintln!(
                                        "torc-server: call {i}'s streamed arguments ({} bytes) are not the start of its \
                                         finished ones; the client has the streamed ones",
                                        so_far.len()
                                    );
                                    continue;
                                }
                            };
                            send(&format!("data: {}\n\n", chunk(json!({"tool_calls": [d]}), Value::Null)));
                        }
                        send(&format!("data: {}\n\n", chunk(json!({}), json!(fin))));
                        // the usage chunk (stream_options.include_usage) has no choices, as OpenAI sends it
                        let u = json!({"id": id, "object": "chat.completion.chunk", "model": name, "choices": [],
                            "usage": usage(np, nr, nt, e.kept()), "timings": timings(e.timed())});
                        send(&format!("data: {u}\n\n"));
                    }
                    Err(err) => {
                        send(&format!(
                            "data: {}\n\n",
                            json!({"error": {"message": err.to_string(), "type": "server_error"}})
                        ));
                    }
                }
                send("data: [DONE]\n\n");
            } else {
                let (mut reply, mut reasoning) = (String::new(), String::new());
                match e.reply(&c, &mut |rs, t| {
                    if rs {
                        reasoning.push_str(t)
                    } else {
                        reply.push_str(t)
                    }
                    true
                }) {
                    Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                        let mut msg = json!({"role": "assistant", "content": reply});
                        if !reasoning.is_empty() {
                            msg["reasoning_content"] = json!(reasoning);
                        }
                        if !calls.is_empty() {
                            msg["tool_calls"] = tool_calls(&calls);
                        }
                        let j = json!({"id": id, "object": "chat.completion", "model": name,
                            "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                            "usage": usage(np, nr, nt, e.kept()), "timings": timings(e.timed())});
                        respond(s, "200 OK", "application/json", &j.to_string());
                    }
                    Err(err) => {
                        respond(s, status(&err), "application/json", &json!({"error": err.to_string()}).to_string())
                    }
                }
            }
        }
        Api::Messages => {
            let u = |np, nr, nt, kept| api::Usage { prompt: np, cached: kept, output: nr, reasoning: nt };
            if c.stream {
                let mut ms = api::MessagesStream::new(c.show_reasoning);
                let np0 = e.count(&c);
                send(&ms.start(&name, np0));
                let r = e.reply(&c, &mut |rs, t| send(&ms.delta(rs, t)));
                match r {
                    Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                        send(&ms.finish(&tool_calls(&calls), fin, &u(np, nr, nt, e.kept())));
                    }
                    Err(err) => {
                        send(&api::MessagesStream::error(&err.to_string()));
                    }
                }
            } else {
                let (mut reply, mut reasoning) = (String::new(), String::new());
                match e.reply(&c, &mut |rs, t| {
                    if rs {
                        reasoning.push_str(t)
                    } else {
                        reply.push_str(t)
                    }
                    true
                }) {
                    Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                        let j = api::messages_reply(
                            &name,
                            &reasoning,
                            &reply,
                            &tool_calls(&calls),
                            fin,
                            &u(np, nr, nt, e.kept()),
                            c.show_reasoning,
                        );
                        respond(s, "200 OK", "application/json", &j.to_string());
                    }
                    Err(err) => respond(
                        s,
                        status(&err),
                        "application/json",
                        &api::messages_error("api_error", &err.to_string()).to_string(),
                    ),
                }
            }
        }
        Api::Responses => {
            let mut rs = api::ResponsesStream::new(&name, c.stream, c.show_reasoning);
            send(&rs.start());
            let r = e.reply(&c, &mut |reasoning, t| send(&rs.delta(reasoning, t)));
            match r {
                Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                    let u = api::Usage { prompt: np, cached: e.kept(), output: nr, reasoning: nt };
                    let body = rs.finish(&tool_calls(&calls), fin, &u);
                    if c.stream {
                        send(&body);
                    } else {
                        respond(s, "200 OK", "application/json", &body);
                    }
                }
                Err(err) if c.stream => {
                    send(&rs.error(&err.to_string()));
                }
                Err(err) => respond(
                    s,
                    status(&err),
                    "application/json",
                    &json!({"error": {"message": err.to_string(), "type": "server_error"}}).to_string(),
                ),
            }
        }
    }
}

fn main() {
    model::set_npu_env();
    // every panic's message and backtrace is appended to panic.log next to the executable, and a panic on the main
    // thread (a request's) ends the server there and then, rather than leaving the process alive with its listener
    // closed and nothing in its log. A helper thread's panic carries on as before (the NPU helper's ends only its
    // request)
    let dflt = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        dflt(info);
        let bt = std::backtrace::Backtrace::force_capture();
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let name = std::thread::current().name().unwrap_or("?").to_string();
        if let Some(p) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("panic.log")))
            && let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p)
        {
            let _ = writeln!(f, "{t} {name}: {info}\n{bt}");
        }
        if name == "main" {
            std::process::exit(101);
        }
    }));
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: torc-server <model.torc> [port] [threads|auto] [ctx] [name]");
        eprintln!(
            "       torc-server <model.torc> --tokenize <text>   (the prompt's ids as the server makes them, BOS first)"
        );
        eprintln!("       torc-server --archs                          (the architectures it runs, one line)");
        std::process::exit(2);
    }
    if a[1] == "--archs" {
        println!("{}", model::models::ARCHS.join(" "));
        return;
    }
    // --tokenize: the ids of a text, special tokens matched, with the model's BOS when it has one (for checking the
    // tokenizer against llama-tokenize)
    if a.get(2).map(String::as_str) == Some("--tokenize") {
        let tok = model::tok::Tokenizer::open(&a[1]).unwrap_or_else(|e| {
            eprintln!("torc-server: {e}");
            std::process::exit(1)
        });
        let text = a.get(3).cloned().unwrap_or_default().replace("\\n", "\n");
        match tok.encode(&text, true) {
            Ok(ids) => println!("{:?}", tok.bos().into_iter().chain(ids).collect::<Vec<u32>>()),
            Err(e) => {
                eprintln!("torc-server: {e}");
                std::process::exit(1)
            }
        }
        return;
    }
    let port: u16 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(8080);
    // TORC_IDLE_UNLOAD: the model runs in a child process that this one starts on the first request that needs it and
    // ends after that many seconds without one, so an unused model gives back all its memory (the NPU's buffers too)
    if let Some(idle) = model::switches::num::<u64>("TORC_IDLE_UNLOAD").filter(|&s| s > 0)
        && !model::switches::given("TORC_SUPERVISED")
    {
        return supervise(&a, port, idle);
    }
    if !cfg!(feature = "gpu") && model::switches::on("TORC_GPU_ATTN") {
        eprintln!("torc-server: TORC_GPU_ATTN is set, but this build has no GPU tier (--features torc-model/gpu)");
    }
    // threads: a number, or auto (or nothing) for every CPU of the fastest core class (pool.rs), never more than there
    // are CPUs
    let asked: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
    let threads = model::pool::init(asked).threads as i32;
    eprintln!("torc-server: threads {threads}{}", if asked == 0 { " (auto: the fastest cores)" } else { "" });
    let ctx: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(8192);
    // each family answers through its own module (serve.rs), chosen by the file's architecture
    let meta = format::read::Model::open(&a[1]).ok().and_then(|f| format::meta::Meta::parse(&f.metadata).ok());
    let arch = meta.as_ref().and_then(|mm| mm.str("general.architecture").map(str::to_string));
    // a context past the one the model was made for runs, but its replies get worse past that point (llama.cpp refuses
    // it): say so, once, where the log shows it
    if let Some(trained) = meta.as_ref().and_then(|mm| mm.arch_u32("context_length"))
        && ctx > trained as usize
    {
        eprintln!(
            "torc-server: context {ctx} is past the {trained} tokens this model was made for: past that, replies get worse"
        );
    }
    let (path, name) = (a[1].as_str(), a.get(5).cloned());
    // the model file's own date, so the same model keeps the same "created" from one start to the next
    let written = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    let since = written.and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
    let _ = CREATED.set(since);
    let mut e = match arch.as_deref() {
        Some("gemma4") => model::models::gemma::serve::open(path, ctx, name),
        Some("gpt-oss") => model::models::gptoss::serve::open(path, ctx, name),
        // every other architecture is one of Model's (models/qwen), which refuses a file it can't run
        _ => model::models::qwen::serve::open(path, ctx, name),
    };
    // the pinning torc-tune chose for this model on this machine, now that its tune file can be found
    model::pool::apply_tuned_pin();
    let _ = LEVELS.set(e.effort_levels());
    listen(&mut *e, port);
}

/// Serves requests on the port, one at a time, until the process ends.
fn listen(e: &mut dyn Serve, port: u16) {
    // this machine only, unless TORC_HOST says otherwise; then TORC_API_KEY, when set, is what another machine's
    // requests must carry (this machine's never need it)
    let host = model::switches::get("TORC_HOST").unwrap_or_else(|| "127.0.0.1".into());
    let key = model::switches::get("TORC_API_KEY").map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
    let keyed = key.is_some();
    let _ = KEY.set(key);
    let l = TcpListener::bind((host.as_str(), port)).unwrap_or_else(|err| {
        eprintln!("torc-server: cannot listen on {host}:{port}: {err}");
        std::process::exit(1)
    });
    let needs = if keyed { ", other machines need the API key" } else { "" };
    eprintln!("torc-server: {} on {host}:{port}{needs}", e.name());
    // between requests: after TORC_IDLE_SAVE seconds without one (default 60, 0 = never) the current chat goes to disk
    // as well (save_leaving), so a reload or a restart comes back to it warm
    let idle = model::switches::num::<u64>("TORC_IDLE_SAVE").unwrap_or(60);
    let _ = l.set_nonblocking(true);
    let mut last = std::time::Instant::now();
    loop {
        match l.accept() {
            Ok((c, _)) => {
                // on Windows an accepted socket inherits the listener's mode
                let _ = c.set_nonblocking(false);
                // only a reply counts: a client's model polls between chats do not keep the chat off the disk
                let n = e.replies();
                handle(e, c);
                if e.replies() != n {
                    last = std::time::Instant::now();
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                if idle > 0 && last.elapsed().as_secs() >= idle {
                    e.idle();
                    last = std::time::Instant::now() + std::time::Duration::from_secs(86400 * 365);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(err) => {
                eprintln!("torc-server: accept: {err}");
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    }
}

/// The model's process under a supervisor, and the local port it serves on.
type ModelProcess = Option<(std::process::Child, u16)>;

/// The server as a supervisor (TORC_IDLE_UNLOAD): it listens on the port, answers /health and /v1/models itself while
/// no model is loaded, and runs the model in a child torc-server on a private local port, started by the first request
/// that needs it and ended after `idle` seconds without one. Ending the process is what gives back everything the model
/// holds, the NPU's buffers included. Requests go through unchanged, one at a time, as the server takes them.
/// The effort levels a model file offers, read from its metadata (the supervisor lists the model before it is loaded,
/// and says the same as the loaded engine does).
fn levels_of_file(path: &str) -> Vec<&'static str> {
    let arch = format::read::Model::open(path)
        .ok()
        .and_then(|f| format::meta::Meta::parse(&f.metadata).ok())
        .and_then(|mm| mm.str("general.architecture").map(str::to_string));
    match arch.as_deref() {
        Some("gpt-oss") => model::api::EFFORT_LEVELS.to_vec(),
        Some("gemma4") => Vec::new(),
        _ => model::models::qwen::serve::effort_levels_of(path),
    }
}
fn supervise(a: &[String], port: u16, idle: u64) {
    let host = model::switches::get("TORC_HOST").unwrap_or_else(|| "127.0.0.1".into());
    let key = model::switches::get("TORC_API_KEY").map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
    let l = TcpListener::bind((host.as_str(), port)).unwrap_or_else(|err| {
        eprintln!("torc-server: cannot listen on {host}:{port}: {err}");
        std::process::exit(1)
    });
    // the name and date the child will give the model, so /v1/models reads the same with it loaded or not
    let name = a.get(5).cloned().unwrap_or_else(|| {
        std::path::Path::new(&a[1]).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
    });
    let written = std::fs::metadata(&a[1]).and_then(|m| m.modified()).ok();
    let since = written.and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
    let _ = CREATED.set(since);
    let _ = LEVELS.set(levels_of_file(&a[1]));
    let needs = if key.is_some() { ", other machines need the API key" } else { "" };
    eprintln!("torc-server: {name} on {host}:{port}{needs}");
    eprintln!("torc-server: the model loads on the first request and unloads after {idle} s without one");
    let _ = l.set_nonblocking(true);
    let mut child: ModelProcess = None;
    let mut last = std::time::Instant::now();
    loop {
        // a child that ended on its own (a crash, or its window closed) is started again by the next request
        if let Some((c, _)) = child.as_mut()
            && !matches!(c.try_wait(), Ok(None))
        {
            eprintln!("torc-server: the model's process ended");
            child = None;
        }
        match l.accept() {
            Ok((c, _)) => {
                // on Windows an accepted socket inherits the listener's mode
                let _ = c.set_nonblocking(false);
                if pass_on(c, &name, key.as_deref(), a, &mut child) {
                    last = std::time::Instant::now();
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                if child.is_some() && last.elapsed().as_secs() >= idle {
                    if let Some((mut c, _)) = child.take() {
                        let _ = c.kill();
                        let _ = c.wait();
                    }
                    eprintln!(
                        "torc-server: no request for {idle} s: the model is unloaded (the next request loads it)"
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(err) => {
                eprintln!("torc-server: accept: {err}");
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    }
}

/// One request to the supervisor. Its head is read and checked as `handle` checks it (another site's page refused,
/// another machine's request without the key refused: the child sees every request as this machine's), /health and
/// /v1/models are answered here while no model is loaded, and everything else goes to the child, started first when
/// need be, head and body as they came, with the reply passed back the same way. True when the request used the model.
fn pass_on(mut s: TcpStream, name: &str, key: Option<&str>, a: &[String], child: &mut ModelProcess) -> bool {
    let _ = s.set_write_timeout(Some(std::time::Duration::from_secs(60)));
    let local = s.peer_addr().map(|p| local_peer(p.ip())).unwrap_or(false);
    let deadline = if key.is_some() && !local { KEYED_HEAD_DEADLINE } else { REQUEST_DEADLINE };
    let _ = s.set_read_timeout(Some(deadline.min(std::time::Duration::from_secs(10))));
    // the head, and whatever of the body came with it, kept as read to pass on unchanged
    let start = std::time::Instant::now();
    let (mut buf, mut chunk) = (Vec::new(), [0u8; 8192]);
    let head_len = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        if buf.len() > MAX_HEAD || start.elapsed() > deadline {
            return false;
        }
        match s.read(&mut chunk) {
            Ok(n) if n > 0 => buf.extend_from_slice(&chunk[..n]),
            _ => return false,
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_len]).into_owned();
    let mut lines = head.split("\r\n");
    let (method, path) = {
        let mut it = lines.next().unwrap_or("").split_whitespace();
        (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string())
    };
    let (mut given, mut foreign) = (Vec::new(), false);
    for h in lines {
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("origin:") {
            foreign = !local_origin(v);
        }
        if let Some(k) = carried_key(h) {
            given.push(k.to_string());
        }
    }
    let r = route(&path);
    if foreign {
        let why = json!({"error": "requests from another site's web page are refused"}).to_string();
        respond(&mut s, "403 Forbidden", "application/json", &why);
        return false;
    }
    if method == "GET" && r == "/v1/health" {
        respond(&mut s, "200 OK", "application/json", r#"{"status":"ok"}"#);
        return false;
    }
    if !allowed(key, local, &given) {
        respond(&mut s, "401 Unauthorized", "application/json", &key_refused(&path));
        return false;
    }
    // a client's model list (apps poll it) neither loads the model nor counts as use
    let listing = method == "GET" && (r == "/v1/models" || r.starts_with("/v1/models/"));
    if listing && child.is_none() {
        let asked = r.strip_prefix("/v1/models/").map(unescape);
        match asked {
            None => {
                let j = json!({"object": "list", "data": [model_entry(name)]});
                respond(&mut s, "200 OK", "application/json", &j.to_string());
            }
            Some(n) if n == name => respond(&mut s, "200 OK", "application/json", &model_entry(name).to_string()),
            Some(n) => {
                let text = format!("no model {n} here: this server runs {name}");
                let j = json!({"error": {"message": text, "type": "invalid_request_error", "code": "model_not_found"}});
                respond(&mut s, "404 Not Found", "application/json", &j.to_string());
            }
        }
        return false;
    }
    let up = start_model(a, child).and_then(|port| TcpStream::connect(("127.0.0.1", port)).ok());
    let Some(mut up) = up else {
        let why = json!({"error": {"message": "the model could not be loaded: see the server's window",
            "type": "server_error"}});
        respond(&mut s, "503 Service Unavailable", "application/json", &why.to_string());
        return false;
    };
    if up.write_all(&buf).is_err() {
        return false;
    }
    // the rest of the request (a body longer than what came with the head) goes up as it arrives, on its own thread,
    // and the reply comes back here until the child closes the connection, as it does after every request
    let _ = s.set_read_timeout(None);
    if let (Ok(mut from), Ok(mut to)) = (s.try_clone(), up.try_clone()) {
        std::thread::spawn(move || {
            let _ = std::io::copy(&mut from, &mut to);
            let _ = to.shutdown(std::net::Shutdown::Write);
        });
    }
    let _ = std::io::copy(&mut up, &mut s);
    let _ = s.flush();
    let _ = s.shutdown(std::net::Shutdown::Both);
    let _ = up.shutdown(std::net::Shutdown::Both);
    !listing
}

/// The model's process, started when there is none: this program again, with the same model and settings on a free
/// local port, marked TORC_SUPERVISED so it serves rather than supervises. It is ready when it answers /health, which
/// it does once the model is loaded. Its log goes to this window. None when it could not be started or ended loading.
fn start_model(a: &[String], child: &mut ModelProcess) -> Option<u16> {
    if let Some((c, port)) = child.as_mut()
        && matches!(c.try_wait(), Ok(None))
    {
        return Some(*port);
    }
    *child = None;
    let port = TcpListener::bind(("127.0.0.1", 0)).ok()?.local_addr().ok()?.port();
    let mut args = vec![a[1].clone(), port.to_string()];
    args.extend(a.iter().skip(3).cloned());
    eprintln!("torc-server: loading the model for a request");
    let mut c = std::process::Command::new(std::env::current_exe().ok()?)
        .args(&args)
        .env("TORC_SUPERVISED", "1")
        .env("TORC_HOST", "127.0.0.1")
        .env_remove("TORC_API_KEY")
        .stdin(std::process::Stdio::null())
        .spawn()
        .map_err(|e| eprintln!("torc-server: cannot start the model's process: {e}"))
        .ok()?;
    #[cfg(windows)]
    job::tie(&c);
    let t = std::time::Instant::now();
    loop {
        if !matches!(c.try_wait(), Ok(None)) {
            eprintln!("torc-server: the model's process ended while loading");
            return None;
        }
        if healthy(port) {
            break;
        }
        if t.elapsed() > std::time::Duration::from_secs(1800) {
            eprintln!("torc-server: the model did not load in 30 minutes");
            let _ = c.kill();
            let _ = c.wait();
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    eprintln!("torc-server: the model is loaded ({} s)", t.elapsed().as_secs());
    *child = Some((c, port));
    Some(port)
}

/// The model's process tied to the supervisor's on Windows: both in a job that ends its processes when the job's last
/// handle closes, which the supervisor holds until it ends, however it ends (its window closed, a crash, Task Manager).
/// Without it, a model left running after its supervisor would keep all its memory with nothing to end it.
#[cfg(windows)]
mod job {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;

    #[repr(C)]
    #[derive(Default)]
    struct BasicLimits {
        per_process_user_time: i64,
        per_job_user_time: i64,
        limit_flags: u32,
        minimum_working_set: usize,
        maximum_working_set: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    /// JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    #[repr(C)]
    #[derive(Default)]
    struct ExtendedLimits {
        basic: BasicLimits,
        io_counters: [u64; 6],
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    const EXTENDED_LIMIT_INFORMATION: i32 = 9;
    const KILL_ON_JOB_CLOSE: u32 = 0x2000;

    unsafe extern "system" {
        fn CreateJobObjectW(attributes: *mut c_void, name: *const u16) -> *mut c_void;
        fn SetInformationJobObject(job: *mut c_void, class: i32, info: *const c_void, len: u32) -> i32;
        fn AssignProcessToJobObject(job: *mut c_void, process: *mut c_void) -> i32;
    }

    /// The job, made once and never closed (it closes as this process ends). None when it could not be made.
    fn the_job() -> Option<usize> {
        static JOB: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        *JOB.get_or_init(|| {
            // SAFETY: plain calls with a valid, sized limits block; the handle is checked before it is used
            unsafe {
                let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
                if job.is_null() {
                    return None;
                }
                let mut l = ExtendedLimits::default();
                l.basic.limit_flags = KILL_ON_JOB_CLOSE;
                let size = std::mem::size_of::<ExtendedLimits>() as u32;
                let ok = SetInformationJobObject(job, EXTENDED_LIMIT_INFORMATION, (&raw const l).cast(), size);
                (ok != 0).then_some(job as usize)
            }
        })
    }

    pub fn tie(child: &std::process::Child) {
        let tied = the_job().is_some_and(|job| {
            // SAFETY: the job handle is a live one this process made, and the child's handle is open while it runs
            unsafe { AssignProcessToJobObject(job as *mut c_void, child.as_raw_handle()) != 0 }
        });
        if !tied {
            eprintln!("torc-server: the model's process could not be tied to this one; end it yourself if this ends");
        }
    }
}

/// Whether a server on this local port answers /health: a whole request, so it is not held waiting for one.
fn healthy(port: u16) -> bool {
    let Ok(mut c) = TcpStream::connect(("127.0.0.1", port)) else { return false };
    let _ = c.set_read_timeout(Some(std::time::Duration::from_secs(5)));
    if c.write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n").is_err() {
        return false;
    }
    let mut reply = String::new();
    let _ = c.read_to_string(&mut reply);
    reply.starts_with("HTTP/1.1 200")
}

// The server's guards that need no socket: which pages count as this machine's, and which errors are the caller's.
#[cfg(test)]
mod guards {
    use super::*;

    #[test]
    fn only_this_machine_counts_as_local() {
        for o in [
            "http://localhost",
            "http://localhost:8080",
            "http://127.0.0.1:8080/chat",
            "http://[::1]:8080",
            " http://localhost ",
        ] {
            assert!(local_origin(o), "{o}");
        }
        for o in [
            "https://example.com",
            "http://localhost.example.com",
            "http://127.0.0.1.example.com:8080",
            "null",
            "",
            "file://",
            "http://[::2]:80",
        ] {
            assert!(!local_origin(o), "{o}");
        }
    }

    #[test]
    fn a_prompt_over_the_context_is_the_callers_error() {
        assert_eq!(status(&model::chat::serve::too_long(9000, 8192)), "400 Bad Request");
        assert_eq!(status(&format::Error("the NPU went away".into())), "500 Internal Server Error");
        // an engine fault that happens to mention the context is still the server's
        assert_eq!(status(&format::Error("position past the context".into())), "500 Internal Server Error");
    }

    #[test]
    fn keys_are_read_the_two_standard_ways() {
        assert_eq!(carried_key("Authorization: Bearer abc-123\r\n"), Some("abc-123"));
        assert_eq!(carried_key("authorization: bearer abc-123"), Some("abc-123"));
        assert_eq!(carried_key("x-api-key: Abc-123\r\n"), Some("Abc-123"));
        assert_eq!(carried_key("X-Api-Key:abc"), Some("abc"));
        for h in ["Authorization: Basic abc", "Authorization: Bearer ", "x-api-key: ", "Content-Length: 5", "Bearer x"]
        {
            assert_eq!(carried_key(h), None, "{h}");
        }
    }

    #[test]
    fn only_another_machine_needs_the_key() {
        let k = |s: &str| vec![s.to_string()];
        // no key set: everyone
        assert!(allowed(None, false, &[]));
        // this machine: never asked
        assert!(allowed(Some("s3cret"), true, &[]));
        // another machine: only with the key, exactly
        assert!(!allowed(Some("s3cret"), false, &[]));
        assert!(!allowed(Some("s3cret"), false, &k("s3cre")));
        assert!(!allowed(Some("s3cret"), false, &k("S3CRET")));
        assert!(!allowed(Some("s3cret"), false, &k("s3cret ")));
        assert!(allowed(Some("s3cret"), false, &k("s3cret")));
        assert!(allowed(Some("s3cret"), false, &["wrong".into(), "s3cret".into()]));
    }

    #[test]
    fn this_machine_is_loopback_only() {
        for ip in ["127.0.0.1", "127.8.0.2", "::1", "::ffff:127.0.0.1"] {
            assert!(local_peer(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["192.168.1.20", "10.0.0.1", "100.80.236.60", "::ffff:192.168.1.20", "fe80::1", "0.0.0.0"] {
            assert!(!local_peer(ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn paths_fold_the_usual_variations() {
        assert_eq!(route("/v1/models"), "/v1/models");
        assert_eq!(route("/v1/models/"), "/v1/models");
        assert_eq!(route("/models"), "/v1/models");
        assert_eq!(route("/models/qwen?x=1"), "/v1/models/qwen");
        assert_eq!(route("/v1/messages?beta=true"), "/v1/messages");
        assert_eq!(route("/chat/completions"), "/v1/chat/completions");
        assert_eq!(route("/torc/settings"), "/torc/settings");
        assert_eq!(route("/"), "");
        assert_eq!(route("/health"), "/v1/health");
        assert_eq!(route("/v1/health/"), "/v1/health");
        // the refusal comes in the shape of the API asked
        assert!(key_refused("/v1/messages").contains("authentication_error"));
        assert!(key_refused("/v1/chat/completions").contains("invalid_api_key"));
        assert!(key_refused("/messages?beta=true").contains("authentication_error"));
    }

    #[test]
    fn model_names_lose_their_escapes() {
        assert_eq!(unescape("Llama%203%208B"), "Llama 3 8B");
        assert_eq!(unescape("qwen3.6-35b-a3b-q4_0"), "qwen3.6-35b-a3b-q4_0");
        assert_eq!(unescape("a%2Fb%2fc"), "a/b/c");
        // a bad or cut-off escape stays as written
        for s in ["100%", "%zz", "a%2", "%"] {
            assert_eq!(unescape(s), s, "{s}");
        }
    }
}
