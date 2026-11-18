# Development Handoff

## Last Updated

- Date: 2026-11-18
- Agent: Claude Code (Claude Opus 5.5)

## Current State

Branch `custom` (based on upstream v0.1.5, commit e93e4c0) builds on Windows ARM64 and runs the fork's three
additions, all tested live with Qwen3.6-35B-A3B Q4_0 (NPU + GPU): streamed tool calls, idle unload, and
`TORC_MAX_TOKENS`. The user's two ToRc launchers use this build. `custom` is pushed to origin
(github.com/nikunjsingh93/ToRc). The desktop app is untouched.

Native vision for the 35B is implemented on branch `vision` (off unless `TORC_MMPROJ` names the mmproj file) and works on
the Q4_0 35B with the NPU + GPU: see [VISION.md](VISION.md). Not merged into `custom` yet.

## Recently Completed

- Forked upstream to nikunjsingh93/ToRc (public), cloned to `Documents\Projects\ToRc`, `upstream` remote added,
  work on branch `custom`.
- Safety review of upstream source before building (targeted: network, process spawning, downloads, deletions,
  registry, build scripts, dependencies). Nothing harmful found; the desktop app's only downloads are Node.js
  (nodejs.org, SHA-256 checked) and ToolFunnel (npm, sha512 checked), on demand.
- Installed the toolchain: rustup stable aarch64-pc-windows-msvc (rustc 1.100.0) and the ARM64 C++ component of
  Visual Studio Build Tools 2026.
- `TORC_MAX_TOKENS` switch (default reply limit for requests without one).
- `TORC_IDLE_UNLOAD` supervisor mode with a Windows kill-on-close job object for the child.
- Streamed tool-call arguments for Qwen-format models on the chat completions API.
- Agent docs: fork section at the end of `AGENTS.md`, `CLAUDE.md`, `.ai/PROJECT.md`, `.ai/HANDOFF.md`.
- Pushed `custom` to origin.
- Native-vision research, written to `.ai/VISION.md`: read the 35B's mmproj metadata and tensor layout and the
  model's rope/token metadata with gguf-py (from `%TEMP%\lcpp\llama.cpp-b12119\gguf-py`), read llama.cpp's
  `conversion/qwen3vl.py`, and located the ToRc code a vision path would touch. No code changed for it.

## Files Changed

- `model/src/bin/torc-server.rs` - supervisor (`supervise`, `pass_on`, `start_model`, `healthy`, `job` module);
  chat-completions streaming of `CallPiece`s and the end-of-reply reconciliation; `send` now shares the socket
  through a RefCell.
- `model/src/chat/serve.rs` - `CallPiece`, `Serve::reply_calls` (default ignores the callback).
- `model/src/models/qwen/serve.rs` - `reply` delegates to `reply_calls`; `run`/`run_seq` take the tools and the call
  callback; the generation loop feeds a `CallStream` while inside `<tool_call>`.
- `model/src/models/qwen/chat.rs` - `args_so_far`, `CallStream`, test `pieces_join_to_the_finished_arguments`.
- `model/src/chat/format.rs` - `is_text` split out of `value`.
- `model/src/chat/api/{mod,anthropic,responses}.rs` - `default_max()` from `TORC_MAX_TOKENS`.
- `SWITCHES.md` - `TORC_MAX_TOKENS`, `TORC_IDLE_UNLOAD`, `TORC_SUPERVISED`.
- `AGENTS.md` (appended), `CLAUDE.md`, `.ai/PROJECT.md`, `.ai/HANDOFF.md` - agent docs.

## Work in Progress

None open in the code.

## Known Issues

- `cargo clippy ... -D warnings` fails on rustc 1.100 with lints in upstream files only (see PROJECT.md). Our files
  have no clippy warnings (checked by running clippy without `-D` and filtering for the changed files).
- Streaming covers chat completions + Qwen format only (by design for now).
- No automated test for the supervisor (it needs a real model); it was tested by hand, see Verification.

## Verification

All on this machine (Snapdragon X2 Elite Extreme, Windows 11 ARM64, 48 GB):

- `cargo build --release --workspace --features torc-model/gpu` - passes.
- `cargo test --release -p torc-model --lib --features gpu` - 64 passed, 6 ignored (includes the new streaming test).
- `cargo test --release -p torc-model --test switches` - passed.
- `cargo test --release -p torc-model --bin torc-server --features gpu` - 7 passed.
- `cargo fmt --all -- --check` - passes (after the formatting commit).
- Live test, 35B Q4_0, port 8091, `TORC_IDLE_UNLOAD=45`, `TORC_MAX_TOKENS=40`:
  - `/v1/models` answered with no model process started;
  - request without `max_tokens` stopped at 40 tokens, `finish_reason` length;
  - streamed `write_file` call (about 80-line snake game): name and first argument piece at 1.2 s, 552 argument
    pieces, done at 9.6 s; joined arguments valid JSON with `overwrite` as a real boolean;
  - model process (22.8 GB) ended after 45 s idle; next request reloaded it in 11 s;
  - force-killing only the supervisor ended the model process within a few seconds (job object).
- The 35B launcher starts the supervisor and lists its model.

## Next Steps

1. **Native vision for Qwen3.6-35B: finish.** Done on `vision` (see VISION.md): tower, preprocessing, M-RoPE, reuse, three APIs. Still open: compare the image rows with llama.cpp numerically; the Anthropic and Responses image paths and a JPEG are untested; a large screenshot (576 rows) timing and quality check; NPU/GPU tiers were on in the test and worked; pipeline (two machines) is not supported with images; images are not saved in on-disk chat states; then merge `vision` into `custom`.
2. Stream tool calls on the Anthropic messages and OpenAI Responses APIs too (MessagesStream / ResponsesStream need
   incremental tool-call events), and consider the GLM and LFM2 formats.
3. Consider offering the changes upstream as separate PRs (one change per PR, per CONTRIBUTING.md), with the
   hardware, models and before/after numbers it asks for.

## Additional Context

- The user's other inference setups (llama.cpp launchers and a model switcher on port 8080, a Linux server, an M4
  Air and a ThinkPad) live outside this repo; this repo only provides the ToRc engine they call on ports 8081/8082.
- Upstream releases often (several a week). Merge `upstream/main` into `main`, rebase `custom`, rebuild, rerun the
  tests above, and retest streaming and idle unload with a real model.
