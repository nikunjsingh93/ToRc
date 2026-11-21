# Development Handoff

## Last Updated

- Date: 2026-11-18
- Agent: Claude Code (Claude Sonnet 5.5)

## Current State
Branch `custom` (based on upstream v0.1.5, commit e93e4c0) builds on Windows ARM64 and runs the fork's four additions,
all tested live with Qwen3.6-35B-A3B Q4_0 (NPU + GPU): streamed tool calls, idle unload, `TORC_MAX_TOKENS`, and native
image input (`TORC_MMPROJ`, merged from branch `vision`; also tested on Qwen3.8-27B). `custom` and `vision` are pushed
to origin (github.com/nikunjsingh93/ToRc). The user's two ToRc launchers (35B and 27B) run this build with vision on
(`TORC_MMPROJ` is set in both; the 27B uses its own F16 mmproj). The desktop app is untouched. Flash Next UD-IQ3_XXS runs too (three new Arm64 quant kernels and opt-in one-machine MTP, see the section below); it has its own launcher on port 8083.

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

## Thinking levels (2026-11-20, uncommitted until asked)

`torc-server` now has low / medium / high reasoning levels for the Qwen format (Qwen3.x, Ornith). Low and medium are caps on the
reasoning's tokens (`TORC_THINK_LOW` 256, `TORC_THINK_MEDIUM` 1536; high or no level is uncapped), and a request can name its
own budget (`thinking_budget_tokens` and aliases, Anthropic `thinking.budget_tokens`). `api::think_budget` turns a request into
`Chat::think_budget`; `qwen/serve.rs` `run_seq` ends the reply with the finish reason "think_cap" at the budget (the token that
would be next is not run, as at a stop token), and `reply_calls` then runs a second `run` whose prompt is the same sequence plus
the reasoning's own token ids plus `\n</think>\n\n`, so the kept sequence is still a prefix and nothing is read twice (the
reply's kept tokens, turns, replies and timings are restored/summed). `/v1/models` lists `supports_reasoning` and
`reasoning_effort_levels` (`Serve::effort_levels`, and `effort_levels_of(path)` for the supervisor while unloaded) so apps like
JoltDesk can offer the levels. Tested live on Ornith (MTP on): same prompt, no budget: 5,284 chars of reasoning, first answer at
29 s; budget 60: answer at 1.0 s; low: 3.4 s; medium capped at 1536 tokens (4,137 chars, 22 s); high identical to no budget;
non-streaming reports exactly 256 reasoning tokens for low; a follow-up turn after a capped reply works. Not done: caps for the
GLM, LFM2, ERNIE and Moonlight formats (Qwen format only), a test of the cap with tool calls in the same reply, and a real
JoltDesk session with the levels showing.
## Flash Next UD-IQ3_XXS (committed: de734a8 kernels, ed31dcc MTP)

Update 2026-11-20: the user removed the model, the MTP file and the launcher from this machine (too heavy for 48 GB; the GGUFs were deleted earlier). Everything below is history plus how to redo it; the kernels and the opt-in MTP stay in the code. Also tried: TORC_TAIL_SKIP 8 and 4 (4 read a 6K prompt 47% faster but changed the answers; 8 did nothing), not kept.

- The user's `Qwen3.8-Flash-Next\UD-IQ3_XXS` GGUF (82 GiB, 4 parts) converts and checks fine
  (`Qwen3.8-Flash-Next-UD-IQ3_XXS.torc` next to the folder), but the server panicked with "no kernel" for three of its
  tensor types, so the fork got Arm64 NEON kernels for them: IQ3_XXS (ggml 18, 42 tensors), IQ2_S (22, 52 tensors) and
  Q2_0 (42, 3 tensors: down of layers 8, 10, 13; 64 weights per block, levels -1..+2 times d). Tallied with a small node
  script over the GGUF headers; the converter already knew the sizes of all three.
- Files: `kernels/src/quant/{iq2_s,iq2s_grid,iq3_xxs,iq3xxs_grid,q2_0}.rs` (tables copied mechanically from llama.cpp's
  `ggml-common.h`), `dequant.rs` (reference arms + geometry), `platform/cpu/neon.rs` (`iq3xxs_matmul`, `iq2s_matmul`,
  `q2_0_matmul`), `model/src/layers/moe.rs` (dispatch), `kernels/tests/parity.rs` (the three types in `TYPES`, plus three
  tests that compare the dequantiser with a line-by-line copy of ggml's loop).
- Verified: `cargo test --release --workspace --features torc-model/gpu` passes; live on the model (port 8083, NPU, no MTP,
  thinking off): "391; Paris" and a correct Fibonacci function, about 7.7 tokens/s decode, first request 9 s.
- Not done: x86 kernels (the types panic with "no kernel" there; the parity table gates them on aarch64), a numeric
  comparison with llama.cpp's output on this model, MTP drafts (`mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` in Draft Models,
  untested with ToRc), vision. The launcher is `Documents\Models\Start ToRc Qwen3.8-Flash-Next.bat` (port 8083, 32K
  context). The model needs 31 GiB resident plus 51 GiB of experts on a 48 GB machine, so it pages.
- MTP on one machine (this fork): upstream only drafts for qwen4exp over two machines. `TORC_MTP=<n>` with
  `TORC_MTP_FILE=<mtp .torc>` now turns drafting on for Flash Next on one machine (`serve.rs`, `Model::mtp_hc_loaded`);
  the draft GGUF (`Draft Models\mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`) is converted next to it as `.torc` and passes both
  checks. `hc.rs` `Bf16::load` also accepts Q5_0 mixers (the MTP file has some). Measured, same Fibonacci prompt, greedy,
  warm: 7.7 tokens/s without, 11.7 with `TORC_MTP=3`, 9.5 with 6 (more drafts cost more expert reads on this paging
  model). The text matched the non-MTP run's start and length; a full bit comparison of the two outputs was not done.
  The launcher sets both variables.
- Found on the way: a `torc-server.exe` built without `--features torc-model/gpu` prints "TORC_GPU_ATTN is set, but this
  build has no GPU tier"; always build the workspace with the feature (the launchers set `TORC_GPU_ATTN`).

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

0. Flash Next: x86 kernels for IQ3_XXS / IQ2_S / Q2_0 (only Arm64 NEON exists), compare its output with llama.cpp numerically, try `TORC_MTP` 2 and 4, and a full bit comparison of MTP and non-MTP output. Offer the kernels upstream as separate PRs.
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
