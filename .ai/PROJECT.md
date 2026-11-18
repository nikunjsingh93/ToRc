# Project Knowledge

Lasting knowledge about this fork. Current progress lives in [HANDOFF.md](HANDOFF.md).

## Overview

ToRc is a CPU-first inference engine for mixture-of-experts language models, written in Rust by Rendeverance
(upstream: https://github.com/Rendeverance/ToRc). It serves an OpenAI-style API (`torc-server`) and is tuned for
Snapdragon X machines, where it also uses the Hexagon NPU for reading prompts and the Adreno GPU for part of long
prompts' attention. Upstream's own docs (README.md, GUIDE.md, SPEC.md, SWITCHES.md, CONTRIBUTING.md) describe the
engine in depth.

This fork (https://github.com/nikunjsingh93/ToRc) exists to run ToRc as a local coding backend for chat/agent apps
(JoltDesk) on a Snapdragon X2 Elite Extreme laptop (Windows 11 ARM64, 48 GB), and adds what that needed:

- tool calls streamed while the model writes them (chat completions API, Qwen-format models);
- the model unloaded after an idle time and loaded again on the next request (`TORC_IDLE_UNLOAD`);
- a configurable reply limit for requests that send no `max_tokens` (`TORC_MAX_TOKENS`).
- images in chat requests for Qwen3.6-35B-A3B (`TORC_MMPROJ`; see VISION.md).

## Technology Stack

- Language: Rust, edition 2024, Cargo workspace (`hwscan`, `format`, `kernels`, `model`). The desktop app
  (`desktop/`, Tauri + `ui/renderer/` web page) is a separate workspace and is not built or changed by this fork.
- Windows ARM64 build: MSVC toolchain (`stable-aarch64-pc-windows-msvc`) with Visual Studio Build Tools and their
  ARM64 C++ component (`Microsoft.VisualStudio.Component.VC.Tools.ARM64`).
- GPU tier: `wgpu` (Vulkan), behind the `torc-model/gpu` feature (the release zips are built with it).
- NPU tier: loads a ggml build with the Hexagon backend at run time (from Qualcomm's GenieX CLI); not linked.
- Key crates: image (png + jpeg only, for vision), serde_json (with `preserve_order`: tool-call argument objects keep the model's order), rayon, memmap2,
  blake3, wgpu.

## Project Structure

- `model/src/bin/torc-server.rs` - the HTTP server (one request at a time) and, in this fork, the idle-unload
  supervisor and the streamed tool-call chunks.
- `model/src/chat/serve.rs` - the `Serve` trait every model family implements; this fork adds `CallPiece` and
  `Serve::reply_calls`.
- `model/src/chat/api/` - request parsing for the three APIs (chat completions in `mod.rs`, Anthropic messages,
  OpenAI Responses) and their stream writers.
- `model/src/chat/format.rs` - shared prompt/tool helpers (`props`, `value`, and this fork's `is_text`).
- `model/src/models/qwen/` - the engine most models run on: `serve.rs` (the generation loop, `run_seq`) and
  `chat.rs` (Qwen3.5/3.6 prompt format and tool-call reader `read`; this fork's `args_so_far` / `CallStream`).
- `model/src/models/{gemma,gptoss,glm,lfm2,ernie,moonlight}` - other families.
- `model/src/models/qwen/vision.rs` - this fork's vision tower, image ids and M-RoPE positions (see Architecture).
- `kernels/src/switches.rs` - the only way to read `TORC_*` switches; `model/tests/switches.rs` fails if
  `SWITCHES.md` and the code disagree.

## Architecture

- `torc-server <model.torc> [port] [threads|auto] [ctx] [name]` loads one model and serves it on 127.0.0.1 (unless
  `TORC_HOST`), one connection at a time. Each family answers through `Serve`; the server writes the replies in the
  shape of the API they came by.
- Generation (qwen `run_seq`): tokens are decoded one at a time (with MTP drafting); text goes out through
  `emit(reasoning, text)`; tokens between `<tool_call>` and `</tool_call>` are collected as a call body and parsed
  after the reply by the family's `read`.
- Fork addition - streamed tool calls: on `<tool_call>` the loop starts a `CallStream`; after each token it re-reads
  the body so far with `args_so_far`, which builds the longest prefix of the finished arguments' JSON
  (`Value::Object(args).to_string()`) that is certain, and hands the new part to `Serve::reply_calls`' callback.
  torc-server sends llama-server-style chunks: `{"index","id":"call_N","type":"function","function":{"name",
  "arguments":""}}`, then `{"index","function":{"arguments":piece}}`. At the end it sends only what the finished calls
  add to what was streamed (normally nothing), and a whole call that was not streamed.
- Fork addition - idle unload: with `TORC_IDLE_UNLOAD=N`, torc-server becomes a supervisor. It listens on the port,
  answers `/health` and `/v1/models` itself while no model is loaded, and on a request that needs the model starts a
  child `torc-server` (same args, a free local port, `TORC_SUPERVISED=1`, `TORC_HOST=127.0.0.1`, no API key), waits
  for its `/health`, and relays the raw request and reply. After N seconds without a model request it kills the child.
  On Windows the child is assigned to a kill-on-job-close job object, so it dies with the supervisor however that ends.
- Fork addition - `TORC_MAX_TOKENS`: `api::default_max()` replaces the fixed 4096 for requests without a limit, in
  all three APIs.
- Fork addition - vision (`TORC_MMPROJ=<mmproj.gguf>`, `model/src/models/qwen/vision.rs`): the tower is the model's
  `qwen3vl_merger` projector (27-block ViT, 2-D RoPE, 2 x 2 merger, outputs d_model-wide rows). An image part in a
  request becomes the chat template's `<|vision_start|><|image_pad|><|vision_end|>` in the prompt text (only while the
  tower is loaded: `content_text` checks `vision::enabled()`); `run_seq` expands the pad token into one id per row. Row
  ids have the top bit set and carry a hash of the image plus the row number, so prompt reuse (which compares ids) tells
  images apart; `std_embed` swaps such an id for its encoded row (registry of the last 32 images; `embed` encodes each
  distinct image once). The language model's attention layers get interleaved M-RoPE positions through
  `vision::rope_slot(slot)`: a slot map built by `set_positions(ids)` at the start of each request gives each slot a
  (time, height, width) triple (text: all equal to slot + a running delta; image rows: grid; after an image the text
  continues from base + max(rows, cols)); with no image in the sequence it is exactly the old `rope_row`. The MTP block
  sees row ids as the pad token. Images cost about 1.2 s (small) to 3.9 s (558 rows) to encode, once per image.

## Development Commands

Run from the repository root (PowerShell; `$env:USERPROFILE\.cargo\bin` must be on PATH):

- Build (as the launchers use it): `cargo build --release --workspace --features torc-model/gpu`
  (output: `target\release\torc-server.exe` and the other tools).
- Tests: `cargo test --release -p torc-model --lib --features gpu`, `cargo test --release -p torc-model --test
  switches`, `cargo test --release -p torc-model --bin torc-server --features gpu`. The full upstream suite is
  `cargo test --release`.
- Format: `cargo fmt --all` (CI runs `cargo fmt --all -- --check`).
- Lint: `cargo clippy --release --workspace --features torc-model/gpu --all-targets -- -D warnings`
  (see the pitfall about the toolchain below).
- Updating from upstream: `git fetch upstream`, fast-forward or merge `upstream/main` into `main`, then rebase
  `custom` onto `main` and rebuild.

## Important Decisions

- Branches: `main` mirrors upstream exactly; all fork work is on `custom`. Keeps upstream merges clean.
- Every addition is off by default behind a documented switch, as upstream's CONTRIBUTING.md requires, so the fork's
  build behaves like upstream unless the switches are set. `TORC_MAX_TOKENS` keeps upstream's 4096 default.
- Idle unload ends a whole process instead of dropping the model inside the server: only a process exit is sure to
  give back everything, including NPU driver buffers and large-page memory held in statics.
- Streaming re-parses the whole body after each token (quadratic in the body's length) instead of an incremental
  parser: it reuses the exact rules of `read`/`value`, so the streamed pieces always join to the finished arguments.
  Measured fast enough (a ~2 KB call streamed in 552 pieces with no visible slowdown).
- Text arguments (schema type string, or no schema) stream character by character; other types are sent whole when
  their `</parameter>` arrives, because `value` parses them as JSON only then.
- Only the chat completions API streams calls; Anthropic messages and OpenAI Responses still send calls at the end
  (`reply_calls` has a default that ignores the callback). Only the Qwen chat format streams; GLM, LFM2, ERNIE and
  Moonlight formats do not.
- The fork's agent instructions are appended to upstream's `AGENTS.md` (not a rewrite) to keep merges simple.

## Known Issues and Pitfalls

- Clippy with `-D warnings` fails on this machine's newer toolchain (rustc 1.100) because of lints in upstream code
  (`manual_is_multiple_of` in kernels; collapsible-if, `map_or`, range patterns and a redundant guard in `moe.rs`,
  `repack.rs`, `weights.rs`, `gpu.rs`). None are in files the fork changed. Upstream's CI presumably uses an older
  clippy. Don't "fix" upstream files for this in the fork unless upstreaming the fix.
- A duplicate parameter name in one call, or a call cut off by `max_tokens` inside a parameter, can make the streamed
  arguments differ from the finished ones; the server then logs a warning and sends nothing more for that call.
- The NPU tier fails with `cannot load ...\GenieX CLI\llama_cpp\ggml.dll (error 126)` unless Qualcomm's GenieX CLI is
  installed with its installer (`geniex-cli-setup-windows-arm64-*.exe`). Copying the libraries from the GenieX
  "bench" zip into that folder and adding it to PATH was not enough (ToRc loads with LOAD_LIBRARY_SEARCH flags, which
  ignore PATH); the cause was not pinned down. The installer made it work.
- `TORC_MTP_FILE` must name a file, not a folder, and is only read when the model file has no MTP layer of its own.
  bartowski's Qwen3.6-35B Q4_0 GGUF already contains it (`blk.40.nextn.*`).
- Upstream torc-server's default reply limit is 4096 tokens including thinking; clients that send no `max_tokens`
  get cut off. Set `TORC_MAX_TOKENS`.
- torc-server serves one conversation at a time.
- Vision limits: data: URIs only (no http(s) image URLs), PNG and JPEG only, no two-machine pipeline, chats with images
  are not saved to disk, the registry keeps the last 32 images (see HANDOFF.md, Known Issues).
- Two copies of the 35B (about 22-26 GB each with the NPU) do not fit in 48 GB with Windows; run one ToRc server at a
  time and close the desktop app's model before starting a launcher.
- PowerShell 5.1 pipes add a BOM when piping text to native commands such as `ssh`; write scripts to files instead.

## Conventions

- Read switches only through `crate::switches`; list every new one in `SWITCHES.md` (the test enforces it).
- Comments explain what the code does and why; no dates, benchmark numbers or bug stories in comments.
- `cargo fmt` with the repo's `rustfmt.toml` (max width 120); keep comment lines within 120 columns.
- Tests next to the code (`#[cfg(test)]`); pin prompt formats byte for byte.
- Commit messages: a short summary line, then what changed and why.

## Local Setup (this machine)

- Launchers outside the repo (`%USERPROFILE%\Documents\Models\Start ToRc Qwen3.6-35B (fast).bat` on port 8081,
  `Start ToRc Qwen3.8-27B.bat` on port 8082) run `target\release\torc-server.exe` from this repo with
  `TORC_NPU=1`, `TORC_GPU_ATTN=1`, `TORC_WARM=wait`, `TORC_MAX_TOKENS=32768`, `TORC_IDLE_UNLOAD=600`. Rebuilding
  updates what they run. To turn vision on, add to the 35B launcher:
  `set "TORC_MMPROJ=%USERPROFILE%\Documents\Models\Qwen3.6-35B\mmproj-Qwen3.6-35B-A3B-BF16.gguf"` (the mmproj is from
  lmstudio-community/Qwen3.6-35B-A3B-GGUF). Both launchers now set `TORC_MMPROJ` (the 27B uses
  `Qwen3.8-27Bmmproj-Qwen3.8-27B-F16.gguf`; remove the line to turn images off).
- Models are converted `.torc` files in the upstream release zip's `models` folder (Qwen_Qwen3.6-35B-A3B-Q4_0.torc,
  Qwen3.8-27B-Q4_0.torc). Auto-tune results live in `%LOCALAPPDATA%\torc\tune`.
- Measured on this machine (8.4K-token coding prompt, server timings): 35B Q4_0 with NPU + GPU + MTP read about
  418 tok/s and wrote 69-82 tok/s, against llama.cpp b12119 CPU at 148.5 / 26.8 on the Q4_K_M file. Q4_0 lets the
  NPU take far more work than Q4_K_M (896 MiB vs 205 MiB of dense products).
