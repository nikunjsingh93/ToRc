# Native vision for Qwen3.6-35B-A3B in ToRc - research notes

Status: not started. No vision code exists in ToRc (upstream lists "Vision" under future work in README.md). This
file collects what was found before starting, split into what was verified and what still has to be checked.

Goal: images (screenshots first) in chat requests to `torc-server` for the Qwen3.6-35B-A3B model, giving the same
answers llama.cpp gives with the model's mmproj file, without changing anything for text-only requests.

## 1. Verified facts

### The vision file (mmproj)

File: `%USERPROFILE%\Documents\Models\Qwen3.6-35B\mmproj-Qwen3.6-35B-A3B-BF16.gguf` (902,822,016 bytes,
from lmstudio-community/Qwen3.6-35B-A3B-GGUF). Read with gguf-py; 334 tensors. Metadata:

| Key | Value |
|---|---|
| general.architecture / general.type | `clip` / `mmproj` |
| clip.projector_type | `qwen3vl_merger` |
| clip.vision.block_count | 27 |
| clip.vision.embedding_length | 1152 (16 heads, so head dim 72) |
| clip.vision.feed_forward_length | 4304 |
| clip.vision.attention.head_count | 16 |
| clip.vision.attention.layer_norm_epsilon | 1e-6 |
| clip.vision.patch_size | 16 |
| clip.vision.image_size | 768 |
| clip.vision.spatial_merge_size | 2 |
| clip.vision.projection_dim | 2048 (= the 35B's d_model) |
| clip.vision.image_mean / image_std | [0.5, 0.5, 0.5] / [0.5, 0.5, 0.5] |
| clip.use_gelu | true |
| clip.vision.is_deepstack_layers | 27 x false: **no DeepStack** (no `v.deepstack.*` tensors either) |

Tensors (BF16 matrices, F32 vectors):

| Tensor | Shape (gguf order) | Role |
|---|---|---|
| `v.patch_embd.weight`, `v.patch_embd.weight.1` | [16, 16, 3, 1152] each, F32 | the Conv3D patch embedding split along time (temporal patch size 2): frame 0 and frame 1 kernels |
| `v.patch_embd.bias` | [1152] | |
| `v.position_embd.weight` | [1152, 2304] F32 | learned position embedding for a 48 x 48 grid (2304 = 48^2; 48 x 16 = 768 = image_size) |
| `v.blk.N.ln1.{weight,bias}`, `v.blk.N.ln2.{weight,bias}` | [1152] | LayerNorm (with bias) before attention / FFN |
| `v.blk.N.attn_qkv.{weight,bias}` | [1152, 3456] + [3456] | fused Q, K, V with bias |
| `v.blk.N.attn_out.{weight,bias}` | [1152, 1152] + [1152] | |
| `v.blk.N.ffn_up.{weight,bias}` | [1152, 4304] + [4304] | |
| `v.blk.N.ffn_down.{weight,bias}` | [4304, 1152] + [1152] | |
| `v.post_ln.{weight,bias}` | [1152] | the merger's norm, per patch before the 2x2 merge |
| `mm.0.{weight,bias}` | [4608, 4608] + [4608] | merger linear_fc1 (4608 = 4 x 1152, the 2x2 patches concatenated) |
| `mm.2.{weight,bias}` | [4608, 2048] + [2048] | merger linear_fc2, out to the language model's width |

From llama.cpp's converter (`conversion/qwen3vl.py`, b12119, read locally): `v.post_ln` is HF's
`visual.merger.norm`; `mm.0` / `mm.2` are `visual.merger.linear_fc1` / `linear_fc2`; the patch Conv3D (time kernel 2)
is split into the two 2-D kernels above; the class covers `Qwen3_5MoeForConditionalGeneration` (this model's family).

### The language model (Qwen3.6-35B-A3B, from its GGUF metadata)

- `general.architecture = qwen35moe`, 40 layers, d_model 2048, 16 query heads, 2 KV heads, head dim 256,
  `full_attention_interval = 4` (every 4th layer is full attention; the others are Gated DeltaNet with no positions).
- `rope.dimension_count = 64`, `rope.freq_base = 1e7`, **`rope.dimension_sections = [11, 11, 10, 0]`**: M-RoPE
  sections in rotation pairs (11 + 11 + 10 = 32 pairs = 64 dims), i.e. temporal / height / width.
- Special tokens: `<|vision_start|>` 248053, `<|vision_end|>` 248054, `<|vision_pad|>` 248055, `<|image_pad|>` 248056,
  `<|video_pad|>` 248057 (`<|im_start|>` 248045, `<|im_end|>` 248046).
- The chat template renders each image part as `<|vision_start|><|image_pad|><|vision_end|>` (seen in the template
  llama-server returns from `/props`); llama.cpp expands the single `<|image_pad|>` into one embedding row per merged
  patch.

### ToRc today (code read on branch `custom`)

- No vision code anywhere in `model/src` (no mmproj, image, M-RoPE handling).
- `chat/format.rs::content_text` keeps only `text` parts of a content array: image parts are silently dropped.
- Positions are 1-D: `layers/ops.rs::rope_row(cs, sn, pos, n_rot, base)` fills one cos/sin table from a single
  position (theta = pos * base^(-2i/n_rot)); `rope_neox` applies it; used from `layers/attn.rs` and
  `models/qwen/forward.rs`. `dimension_sections` is not read (`models/qwen/hparams.rs` reads only `n_rot` and
  `rope_base`). For text this is exactly right, because M-RoPE with t = h = w = pos reduces to 1-D RoPE.
- Embeddings come only from token ids: `forward.rs` `decode` (line ~556) and `prefill` (line ~673) look rows up in
  `token_embd.weight`; there is no way to feed precomputed rows.
- Prompt reuse compares token ids (`models/qwen/serve.rs`: `seen`, marks, saved states, around lines 345-560 and 745).
  Two different images would expand to identical `<|image_pad|>` id runs, so reuse would wrongly treat them as the
  same prompt.
- The `format` crate reads GGUF (`format/src/gguf.rs`) but has no BF16 tensor type; ToRc's kernels cover Q4_0, Q4_1,
  Q5_1, Q8_0, IQ4_NL, IQ4_XS, IQ3_S, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4 and F32 rows, not BF16/F16 matrices.
- The MTP block's prompt fill (`mtp.rs`, `mtp_fill`) also embeds prompt rows from token ids.

## 2. Not yet verified (check against a reference before relying on it)

These come from knowledge of Qwen3-VL and llama.cpp, not from code read here:

1. **Which position each rotation pair uses.** Qwen3-VL uses *interleaved* M-RoPE (llama.cpp `LLAMA_ROPE_TYPE_IMROPE`).
   Believed rule per pair index s (0..31): s % 3 == 1 and s < 3 * sections[1] -> height; s % 3 == 2 and
   s < 3 * sections[2] -> width; s % 3 == 0 and s < 3 * sections[0] -> time; otherwise time. Confirm in ggml's CPU
   rope (`ggml/src/ggml-cpu/ops.cpp`, the mrope/imrope cache init) and that `qwen35moe` uses IMROPE in
   `src/llama-model.cpp`.
2. **Image token positions.** Believed (Qwen2/3-VL `get_rope_index`): the image's rows all take time = the position
   after `<|vision_start|>`; height = that + row, width = that + column, on the merged grid (grid / 2); the text after
   the image continues from that start + max(grid_h, grid_w) (llama.cpp: `mtmd_image_tokens_get_n_pos`). Confirm.
3. **Vision encoder details:** 2-D RoPE inside the ViT (head dim 72 split between row and column, theta 10000?), full
   attention in every block (Qwen2.5-VL used windows; Qwen3-VL is believed not to), GELU variant (tanh approximation
   or erf), and how `v.position_embd` (48 x 48) is resized to the image's grid (bilinear interpolation in Qwen3-VL).
4. **Patch order:** Qwen's processor orders patches so each 2 x 2 merge window is consecutive; the merger then
   concatenates 4 consecutive 1152-d rows. Confirm the exact order.
5. **Still images through a time-2 patch kernel:** an image is believed to be fed as 2 identical frames, which equals one
   2-D convolution with kernel `weight + weight.1`.
6. **Image size budget:** the mmproj has no min/max pixel keys, so llama.cpp uses its Qwen-VL defaults; sides are
   rounded to multiples of 32 (patch 16 x merge 2). Tokens per image = (H / 32) x (W / 32). Confirm the defaults.

References to read: llama.cpp `tools/mtmd/clip.cpp` (the `PROJECTOR_TYPE_QWEN3VL` graph), `tools/mtmd/mtmd.cpp`
(image tokens and positions), `ggml/src/ggml-cpu/ops.cpp` (rope), `src/llama-model.cpp` (qwen35moe); Hugging Face
transformers `models/qwen3_vl` (vision model, `get_rope_index`) and the Qwen3.5/3.6 processor config. Only the Python
converter of llama.cpp b12119 is on this machine (`%TEMP%\lcpp\llama.cpp-b12119`), not the C++ sources.

## 3. Proposed plan (each step testable on its own)

1. **Reference outputs first.** Get numbers to compare against before writing the encoder: llama.cpp with the same
   model and mmproj (the user's llama.cpp launcher `Start Qwen3.6-35B.bat` runs Q4_K_M + this mmproj), and ideally the
   image embeddings themselves (llama.cpp's mtmd debug output, or transformers on CPU from the HF weights). Note: the
   ToRc model is the Q4_0 `.torc`; its source GGUF (bartowski Qwen_Qwen3.6-35B-A3B-Q4_0.gguf) was deleted, so download
   it again if llama.cpp must run the identical quant.
2. **Load the mmproj** from GGUF with `format::gguf` (add BF16 -> f32, or quantize the matrices to Q8_0 at load and
   check the drift). Keep it a separate file passed to the server (e.g. a `--mmproj` argument or `TORC_MMPROJ`), off
   by default, as upstream's CONTRIBUTING.md requires for new paths.
3. **Image preprocessing:** decode PNG/JPEG (a new dependency; keep it small, PNG + JPEG only), resize to multiples of
   32 within the pixel budget, normalize with mean/std 0.5, cut 16 x 16 patches in Qwen's merge-window order. Unit-test
   the shapes and the order.
4. **Vision encoder forward:** patch embedding, + interpolated position embedding, 27 blocks (LayerNorm, fused QKV
   with bias, 2-D RoPE, attention, out, LayerNorm, GELU FFN), `post_ln`, 2 x 2 merge, `mm.0`, GELU, `mm.2` -> rows of
   2048. Use ToRc's thread pool and existing matmul kernels where they fit. Compare the rows with the reference.
5. **Engine input:** a prefill variant that takes embedding rows for the image positions (and token ids elsewhere), and
   per-row 3-component positions for the full-attention layers (DeltaNet layers need no positions). Add an M-RoPE
   table builder next to `rope_row`, used only when a prompt contains an image, so text stays bit-identical.
6. **Prompt and APIs:** write `<|vision_start|>` + N x `<|image_pad|>` + `<|vision_end|>` per image in
   `models/qwen/chat.rs`; parse image parts in all three APIs (OpenAI `image_url` with a data URI, Anthropic `image`
   with a base64 source, Responses `input_image`); stop `content_text` from dropping them silently.
7. **Prompt reuse and saved states:** make an image part of the sequence identity (e.g. hash the image into the ids
   compared in `seen`, or end reuse at the first image), including the saved-state files.
8. **MTP drafting:** feed image rows to the MTP block's prompt fill too, or drafts after an image get worse (speed
   only; replies are verified either way).
9. **Tiers:** check that the NPU and GPU attention tiers accept the embedding-row prefill and the M-RoPE positions (or
   fall back to the CPU for image prompts at first).
10. **End-to-end test:** a screenshot reading test (see below) against llama.cpp's answer, plus text-only requests
    proving outputs are unchanged (upstream's `torc-golden` check).

Expected effort: several focused sessions. The riskiest parts are 2-D RoPE inside the encoder and M-RoPE positions in
the language model, because a mistake there degrades answers without crashing; compare against a reference at every
step.

## 4. Test material used so far

- A generated 800 x 300 PNG: blue header "Order Summary", the line "Invoice #48213   Total: $129.50", and a green
  "Pay Now" button (drawn with System.Drawing in PowerShell). llama.cpp with this mmproj read all of it correctly on the
  35B and on Ornith-1.5-35B.
- Image prompt timing on llama.cpp (CPU, this laptop): the first screenshot reply ran at about 16-19 tok/s writing.

## 5. Fallback if native vision is shelved

Route by content in the supervisor (`torc-server.rs`, `pass_on`): requests with image parts go to a llama.cpp server
that has the mmproj (unloading the ToRc child first, since two 35B copies do not fit in 48 GB), text goes to ToRc.
Same endpoint and model name for the client; costs a model reload when switching between image and text requests.
