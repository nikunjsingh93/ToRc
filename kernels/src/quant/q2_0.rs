//! q2_0.rs - Q2_0 (ggml): 64 weights of 2 bits each under one fp16 scale; the levels are -1, 0, +1 and +2 times it.

/// Q2_0's block: fp16 d, then 16 bytes of 2-bit fields (weight j is bits 2 (j % 4) .. 2 (j % 4) + 1 of byte j / 4).
pub const Q2_0_BLOCK: usize = 18;
