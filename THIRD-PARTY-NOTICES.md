# Third-party notices

ToRc is licensed under either of the MIT licence (LICENSE-MIT) or the Apache License 2.0 (LICENSE-APACHE), at your option. A few files carry data or definitions taken from other projects, under their own licences. The crates ToRc builds with, and their licence texts, are in [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md) (regenerate it with `tools/third_party.py`).

## ggml and llama.cpp (MIT)

Copyright (c) 2023-2026 The ggml authors. MIT licence: https://github.com/ggml-org/ggml/blob/master/LICENSE and https://github.com/ggml-org/llama.cpp/blob/master/LICENSE

- `kernels/src/quant/iq3s_grid.rs` - IQ3_S's grid (`iq3s_grid`, from ggml's `ggml-common.h`), copied as data, since it defines the format.
- `kernels/src/quant/iq2s_grid.rs` - IQ2_S's grid (`iq2s_grid`, from ggml's `ggml-common.h`), copied as data, since it defines the format.
- `kernels/src/quant/iq3xxs_grid.rs` - IQ3_XXS's grid and sign table (`iq3xxs_grid`, `ksigns_iq2xs`, from ggml's `ggml-common.h`), copied as data, since they define the format.
- `model/src/chat/uniflags.rs` - the Unicode category ranges, generated from llama.cpp's `src/unicode-data.cpp` (itself generated from the Unicode Character Database).
- The quantised block layouts, their reference dequantisation and dot-product orders (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S, Q2_0, Q8_K, MXFP4) and the GGUF file format are implemented from ggml's definitions, which the comments cite where they are used.
- `kernels/src/platform/npu/ggml.rs` - declarations of ggml's C API (function signatures and enum values from `ggml.h`), used to call a ggml build at run time.
- `model/src/models/gemma/npu_gemma.rs` - Gemma 4's graph for the NPU follows llama.cpp's `src/models/gemma4.cpp` and its `build_moe_ffn`. `model/src/models/gemma/gemma_mtp.rs` - Gemma 4's draft model follows llama.cpp's `src/models/gemma4-assistant.cpp` and `common/speculative.cpp`.
- `model/src/models/qwen/hc.rs` `to_bf16` rounds as ggml's `ggml_compute_fp32_to_bf16`. The eight-lane horizontal sums in `kernels/src/platform/cpu/x86.rs`, `x86dot.rs` and `model/src/layers/moe.rs` add the lanes in the order of ggml's `hsum_float_8`.
- Where a model's maths has to match llama.cpp's to give the same answers, the code follows llama.cpp's and says so
  in its comments: MLA's absorbed attention (`model/src/layers/mla.rs`), LFM2's short convolution
  (`model/src/layers/rec.rs`, `build_shortconv_block`), the MoE's sigmoid gating and weight sums
  (`model/src/layers/moe.rs`, `build_moe_ffn`), ERNIE's rotary pairs (`model/src/layers/ops.rs`), gpt-oss's YaRN
  (`model/src/models/gptoss/gptoss.rs`, `rope_yarn`), and the tokenizer's SentencePiece rules, space prefix, BOS and Han
  ranges (`model/src/chat/tok.rs`, `llm_tokenizer_spm`). The code is ToRc's own.

ggml's MIT licence text:

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Unicode data

`model/src/chat/uniflags.rs` holds Unicode general-category ranges, derived (through llama.cpp's tables, above) from the Unicode Character Database, under the Unicode License V3:

```
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
```

## LLVM runtime (Windows on ARM builds)

A Windows on ARM build made with llvm-mingw needs its `libunwind.dll` beside the executables. libunwind is part of the LLVM project, under the Apache License 2.0 with LLVM Exceptions (https://llvm.org/LICENSE.txt): if you pass your build on to others with that DLL, include that licence with it. The Windows on ARM zip does, in THIRD-PARTY-DLLS.md, with the licence of the WebView2Loader.dll beside it.

## Cephes

The `exp` used by attention and the MoE uses the polynomial coefficients of the Cephes `expf` with Cody-Waite range reduction (Stephen L. Moshier's Cephes Mathematical Library, distributed on Netlib), written afresh in Rust.

## Benchmark text

`tools/bench/lb_alice.txt`, `tools/bench/lb_p8192.txt` and `model/src/bin/tune_story.txt` are the opening of *Alice's Adventures in Wonderland* by Lewis Carroll, from Project Gutenberg eBook #11. `tools/bench/lb_p1k.txt`, `lb_p4k.txt`, `lb_p16k.txt` and `lb_p32k.txt` are the opening of *Pride and Prejudice* by Jane Austen, from Project Gutenberg eBook #1342. Both are in the public domain.

## Strata (MIT)

https://github.com/Niko1221/Strata

- `model/src/layers/moe.rs` `predict_block`: reading a prompt block's likely experts ahead by the next layer's router votes
  follows Strata's idea. The code is ToRc's own.

## Research credits

- `model/src/layers/moe.rs` `tail_skip` (fast mode, off by default): letting the experts already in RAM shape the routed
  set comes from A. Skliar et al., "Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference",
  Transactions on Machine Learning Research, 2025 (arXiv:2412.00099). Their method reranks the router with a cache
  prior; ToRc's fast mode only drops low-ranked experts that would have to be read from disk.
- `model/src/models/qwen/forward.rs` `read_ahead` (TORC_READ_AHEAD, off by default): guessing a layer's experts by applying its
  gate to the previous layer's state is from A. Eliseev and D. Mazur, "Fast Inference of Mixture-of-Experts Language
  Models with Offloading", 2023 (arXiv:2312.17238).

## Qwen chat template

`model/src/models/qwen/chat.rs` `TOOL_CALL_RULES` is the tool-calling instruction text of the Qwen3.5 / Qwen3.6
chat template (the `tokenizer.chat_template` the Qwen models ship with), so that a request with tools is laid out as
the model was trained to see it. The same text is in Qwen's published `chat_template.jinja` for Qwen3-Coder
(huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct, Apache-2.0).

ERNIE 4.5's and Moonlight's chat templates have no tool format, so `model/src/chat/json_calls.rs` gives them the
tool instructions and the JSON call format of Qwen2.5's chat template (huggingface.co/Qwen/Qwen2.5-7B-Instruct,
Apache-2.0), which Qwen3-30B-A3B-Instruct-2507's template uses too.

## Other models' prompt formats

`model/src/models/gptoss/chat.rs` writes OpenAI's harmony format, its system line and its tool declarations as
gpt-oss's chat template does (github.com/openai/harmony and huggingface.co/openai/gpt-oss-20b, Apache-2.0).
`model/src/models/gemma/chat.rs` writes Gemma 4's turns, thinking channel and tool format as the chat template that
ships with Gemma 4 does. `model/src/models/glm/chat.rs` and `model/src/models/lfm2/chat.rs` write GLM-4.7's and
LFM2's turns and tool formats as the chat templates that ship with those models do. In each, the template is the
reference for every string, so the prompt is the one the model was trained on.

## At run time

The optional NPU tier loads a ggml build with the Hexagon backend (for example the one GenieX CLI installs) from the user's machine at run time. ToRc does not ship or link it.
