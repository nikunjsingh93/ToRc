# Development Handoff

## Last Updated

- Date: 2026-11-18
- Agent: Claude Code (Claude Sonnet 5.5)

## Current State

Branch `custom` (based on upstream v0.1.5, commit e93e4c0) builds on Windows ARM64 and runs the fork's four additions,
image input (`TORC_MMPROJ`, merged from branch `vision`). `custom` and `vision` are pushed to origin
(github.com/nikunjsingh93/ToRc). The user's two ToRc launchers (35B and 27B) run this build with vision on
(`TORC_MMPROJ` is set in both; the 27B uses its own F16 mmproj). The desktop app is untouched.
own F16 mmproj). The desktop app is untouched.

## Recently Completed

- Forked upstream, safety review of upstream source, toolchain install (see PROJECT.md), `TORC_MAX_TOKENS`,
  `TORC_IDLE_UNLOAD` supervisor with a Windows kill-on-close job object, streamed tool-call arguments (chat completions,
  Qwen format), agent docs, pushed `custom`.
- Native vision for the 35B (research in VISION.md, then implementation):
  - `model/src/models/qwen/vision.rs`: mmproj loader (BF16 via mmap), PNG/JPEG preprocessing (`image` crate, PNG + JPEG
    only), the 27-block ViT with 2-D RoPE, the 2 x 2 merger, image ids (top bit set, hash of the image + row number) and
    a registry of encoded images, the M-RoPE position map (`set_positions`, `position`, `rope_slot`), base64 and image
    part parsing, the `vision_probe` timing/regression test.
  - hooks: `forward.rs` `std_embed` swaps an image id for its row; `layers/attn.rs` and `forward.rs` `decode` call
    `vision::rope_slot` (plain `rope_row` while no image is in the sequence, so text is unchanged); `qwen/serve.rs`
    loads the tower (`TORC_MMPROJ`), reads the request's images, expands each `<|image_pad|>` into row ids, calls
    `set_positions`, maps row ids to the pad token for the MTP block, skips disk-saving chats that hold images;
    `chat/format.rs` `content_text` writes the image marker while vision is on; the Anthropic and Responses parsers keep
    image parts.
  - speed: encoder optimised from 16.4 s to 4.7 s on a 558-row image (NEON 4 x 4 tile kernel for the products and for
    attention scores/values, values stored transposed, elementwise ops on the pool). Through the server: 3.9 s for 558
    rows, 1.2 s for the 800 x 300 test image.

## Files Changed

Streaming / idle / reply limit (earlier work):

- `model/src/bin/torc-server.rs` - supervisor (`supervise`, `pass_on`, `start_model`, `healthy`, `job` module);
  chat-completions streaming of `CallPiece`s and the end-of-reply reconciliation.
- `model/src/chat/serve.rs` - `CallPiece`, `Serve::reply_calls`.
- `model/src/models/qwen/chat.rs` - `args_so_far`, `CallStream`.
- `model/src/chat/api/{mod,anthropic,responses}.rs` - `default_max()` from `TORC_MAX_TOKENS`.

Vision (this work):

- `model/src/models/qwen/vision.rs` (new), `model/src/models/qwen/mod.rs`, `model/src/lib.rs` (`pub use ... vision`).
- `model/src/models/qwen/forward.rs`, `model/src/layers/attn.rs`, `model/src/models/qwen/serve.rs` - hooks above.
- `model/src/chat/format.rs`, `model/src/chat/api/{anthropic,responses}.rs` - image parts.
- `model/Cargo.toml`, `Cargo.lock` - the `image` crate (png, jpeg; default features off).
- `SWITCHES.md` - `TORC_MAX_TOKENS`, `TORC_IDLE_UNLOAD`, `TORC_SUPERVISED`, `TORC_MMPROJ`, `TORC_VISION_TOKENS`,
  `TORC_VISION_PROF`, `TORC_TEST_IMAGE`, `TORC_VISION_OUT`.
- `AGENTS.md` (appended section), `CLAUDE.md`, `.ai/*` - agent docs.

## Work in Progress

None open in the code.

## Known Issues

- `cargo clippy ... -D warnings` fails on rustc 1.100 with lints in upstream files only (see PROJECT.md). Not checked
  for `vision.rs`: run clippy without `-D` and filter for the changed files before upstreaming.
- Streaming covers chat completions + Qwen format only (by design for now).
- No automated test for the supervisor (needs a real model).
- Vision limits: the image rows were not compared numerically with llama.cpp (only by answers); the Anthropic and
  Responses image paths and http(s) image URLs are untested/unsupported (only `data:` URIs); two-machine pipeline mode
  does not support images; chats holding images are not saved to disk; the registry keeps the last 32 images, and an
  evicted image still referenced by a kept prompt makes a later re-read fail with an error; the preprocessing resize
  (CatmullRom) is not llama.cpp's, and the size budget (`TORC_VISION_TOKENS`, default 576 rows) is our own choice.
- `torc-golden` (upstream's text-output check) was not run after the vision hooks; text requests were checked by hand.

## Verification

All on this machine (Snapdragon X2 Elite Extreme, Windows 11 ARM64, 48 GB):

- `cargo build --release --workspace --features torc-model/gpu` - passes.
- `cargo test --release -p torc-model --lib --features gpu` - 69 passed, 7 ignored (new: 5 vision tests; ignored:
  `vision_probe`).
- `cargo test --release -p torc-model --test switches` and `--bin torc-server --features gpu` - pass.
- `cargo fmt --all` - clean.
- Streaming / idle unload: tested live earlier (see git history of this file: 552 argument pieces for a ~2 KB call;
  model unloaded after 45 s idle and reloaded in 11 s; the job object ended the child with the supervisor).
- Vision on the 27B (Qwen3.8-27B Q4_0, its F16 mmproj; F16 matrices and the M-RoPE sections from the model file were added for it): the invoice image and the code screenshot were read correctly (request 20 s and 12 s incl. first-use warm-up).
- Vision, live, 35B Q4_0 + NPU + GPU + MTP, port 8093, thinking off, greedy:
  - 800 x 300 PNG (invoice): all texts and colours read correctly; turn 2 reused 241 of 279 prompt tokens; text-only
    requests answered correctly afterwards (17 x 23 = 391).
  - generated 1920 x 1080 code screenshot, PNG and JPEG (18 x 31 = 558 rows): the code transcribed exactly, the red
    box and the TODO comment read correctly; request 7.2 s of which 3.9 s encoding.
- `vision_probe` (ignored test): `TORC_MMPROJ=<mmproj> TORC_TEST_IMAGE=<png> TORC_VISION_OUT=<file> TORC_VISION_PROF=1
  cargo test --release -p torc-model --lib vision_probe -- --ignored --nocapture` prints the tower's time per phase and,
  on a second run with the same file, the relative difference from the saved rows (use it after any kernel change).

## Next Steps

1. Compare the image rows with llama.cpp numerically (VISION.md plan step 1), and run `torc-golden` to prove text output
   is bit-identical.
2. Test the Anthropic image block and Responses `input_image` paths; consider http(s) image URLs (needs a fetch, so
   think about the safety side first).
3. Vision speed: the products (about 2.7 s of 4.7) and softmax exp are what is left; cache blocking was tried and was
   slower; int8 weights would change the numbers.
4. Stream tool calls on the Anthropic messages and OpenAI Responses APIs too, and consider the GLM and LFM2 formats.
5. Consider offering the changes upstream as separate PRs (one change per PR, per CONTRIBUTING.md) with the hardware,
   models and before/after numbers it asks for.

## Additional Context

- The user's other inference setups (llama.cpp launchers and a model switcher on port 8080, a Linux server, an M4
  Air and a ThinkPad) live outside this repo; this repo only provides the ToRc engine they call on ports 8081/8082.
- Upstream releases often (several a week). Merge `upstream/main` into `main`, rebase `custom`, rebuild, rerun the
  tests above, and retest streaming, idle unload and a vision request with a real model. The vision hooks touch
  `attn.rs`, `forward.rs` and `serve.rs`, which upstream edits often: expect conflicts there.
