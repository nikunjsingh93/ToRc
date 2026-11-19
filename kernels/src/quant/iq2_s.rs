//! iq2_s.rs - IQ2_S (ggml): grid-indexed 2-bit weights with signs and 4-bit scales for runs of 16.

use super::iq2s_grid;

/// IQ2_S's block: fp16 d, 32 bytes of index low bits (qs), 32 of signs, 8 of index high bits (qh) and 8 of scales.
pub const IQ2_S_BLOCK: usize = 82;

/// The integer scale of the 16 weights c (0..16) in an IQ2_S block, from its 4-bit field s: 2s + 1. ggml's weight is
/// d x (0.5 + s) x 0.25 x level, which is (d / 8) x (2s + 1) x level, so the kernels carry d / 8 as the block's scale.
#[inline(always)]
pub fn iq2s_chunk_scale(blk: &[u8], c: usize) -> i32 {
    1 + 2 * ((blk[74 + c / 2] >> (4 * (c % 2))) & 15) as i32
}

/// Group i's (0..32) eight unsigned levels in an IQ2_S block (a grid entry, with index = qs | two qh bits << 8) and its
/// sign byte (bit e set means weight e is negative).
#[inline(always)]
pub fn iq2s_group(blk: &[u8], i: usize) -> ([u8; 8], u8) {
    let idx = blk[2 + i] as usize | (((blk[66 + i / 4] as usize) << (8 - 2 * (i % 4))) & 0x300);
    (iq2s_grid::IQ2S_GRID[idx].to_le_bytes(), blk[34 + i])
}
