//! iq3_xxs.rs - IQ3_XXS (ggml): grid-indexed 3-bit weights; each group of 32 carries its 4-bit scale and its signs
//! (seven bits for each run of 8) in one u32.

use super::iq3xxs_grid;

/// IQ3_XXS's block: fp16 d, 64 grid indices (qs) and 8 x 4 bytes of signs and scales.
pub const IQ3_XXS_BLOCK: usize = 98;

/// Group g's (0..8) 32 bits of signs and scale in an IQ3_XXS block.
#[inline(always)]
fn iq3xxs_aux(blk: &[u8], g: usize) -> u32 {
    u32::from_le_bytes([blk[66 + 4 * g], blk[67 + 4 * g], blk[68 + 4 * g], blk[69 + 4 * g]])
}

/// The integer scale of group g in an IQ3_XXS block, from its 4-bit field s (the top of its u32): 2s + 1. ggml's weight
/// is d x (0.5 + s) x 0.5 x level, which is (d / 4) x (2s + 1) x level, so the kernels carry d / 4 as the block's scale.
#[inline(always)]
pub fn iq3xxs_group_scale(blk: &[u8], g: usize) -> i32 {
    1 + 2 * (iq3xxs_aux(blk, g) >> 28) as i32
}

/// Group g's 32 unsigned levels in an IQ3_XXS block (eight grid entries, one per byte of qs) and its 32 sign bits (bit
/// e set means weight e is negative).
#[inline(always)]
pub fn iq3xxs_group(blk: &[u8], g: usize) -> ([u8; 32], u32) {
    let aux = iq3xxs_aux(blk, g);
    let mut l = [0u8; 32];
    for k in 0..8 {
        l[4 * k..4 * k + 4].copy_from_slice(&iq3xxs_grid::IQ3XXS_GRID[blk[2 + 8 * g + k] as usize].to_le_bytes());
    }
    let mut s = 0u32;
    for q in 0..4 {
        s |= (iq3xxs_grid::KSIGNS_IQ2XS[((aux >> (7 * q)) & 127) as usize] as u32) << (8 * q);
    }
    (l, s)
}
