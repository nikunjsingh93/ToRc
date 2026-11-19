//! dequant.rs - weights back to f32 (SPEC.md section 4).
//!
//! ggml's block formats as ggml defines them (ggml-quants.c, MIT), with one multiply per number in the order written
//! there. The embeddings' rows and the requantised heads go through here.

use crate::f16_to_f32;

/// IQ4_NL's sixteen levels (ggml-common.h).
const IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

fn h(b: &[u8], o: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[o], b[o + 1]]))
}

/// The scale and minimum of sub-block j, from a Q4_K / Q5_K block's 12 scale bytes (six bits each).
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// The bytes in a block and the numbers it holds, for the types here.
pub fn block_geometry(ty: u32) -> Option<(usize, usize)> {
    Some(match ty {
        0 => (4, 1),
        1 => (2, 1),
        30 => (2, 1),
        2 => (18, 32),
        3 => (20, 32),
        6 => (22, 32),
        7 => (24, 32),
        8 => (34, 32),
        12 => (144, 256),
        13 => (176, 256),
        14 => (210, 256),
        11 => (110, 256),
        23 => (136, 256),
        21 => (110, 256),
        22 => (82, 256),
        18 => (98, 256),
        42 => (18, 64),
        20 => (18, 32),
        39 => (17, 32),
        _ => return None,
    })
}

/// y.len() numbers of type `ty` from w; returns false where the type is not one of these (the caller then has no
/// values).
pub fn dequantize(ty: u32, w: &[u8], y: &mut [f32]) -> bool {
    let Some((bs, qk)) = block_geometry(ty) else { return false };
    assert!(y.len().is_multiple_of(qk) && w.len() >= y.len() / qk * bs);
    for (b, y) in w.chunks_exact(bs).zip(y.chunks_exact_mut(qk)) {
        match ty {
            0 => y[0] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            1 => y[0] = h(b, 0),
            30 => y[0] = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16),
            2 => {
                let d = h(b, 0);
                for j in 0..16 {
                    y[j] = ((b[2 + j] & 15) as i32 - 8) as f32 * d;
                    y[j + 16] = ((b[2 + j] >> 4) as i32 - 8) as f32 * d;
                }
            }
            3 => {
                let (d, m) = (h(b, 0), h(b, 2));
                for j in 0..16 {
                    y[j] = (b[4 + j] & 15) as f32 * d + m;
                    y[j + 16] = (b[4 + j] >> 4) as f32 * d + m;
                }
            }
            6 | 7 => {
                let one = ty == 7;
                let d = h(b, 0);
                let (m, o) = if one { (h(b, 2), 4) } else { (0.0, 2) };
                let qh = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
                let qs = &b[o + 4..o + 20];
                for j in 0..16 {
                    let h0 = ((qh >> j) << 4) as u8 & 0x10;
                    let h1 = (qh >> (j + 12)) as u8 & 0x10;
                    let (x0, x1) = ((qs[j] & 15) | h0, (qs[j] >> 4) | h1);
                    if one {
                        y[j] = x0 as f32 * d + m;
                        y[j + 16] = x1 as f32 * d + m;
                    } else {
                        y[j] = (x0 as i32 - 16) as f32 * d;
                        y[j + 16] = (x1 as i32 - 16) as f32 * d;
                    }
                }
            }
            8 => {
                let d = h(b, 0);
                for j in 0..32 {
                    y[j] = b[2 + j] as i8 as f32 * d;
                }
            }
            12 | 13 => {
                let five = ty == 13;
                let (d, mn) = (h(b, 0), h(b, 2));
                let sc = &b[4..16];
                let (qh, ql) = if five { (&b[16..48], &b[48..176]) } else { (&b[0..0], &b[16..144]) };
                for g in 0..4 {
                    let (s1, m1) = scale_min_k4(2 * g, sc);
                    let (s2, m2) = scale_min_k4(2 * g + 1, sc);
                    let (d1, m1) = (d * s1 as f32, mn * m1 as f32);
                    let (d2, m2) = (d * s2 as f32, mn * m2 as f32);
                    let q = &ql[32 * g..32 * g + 32];
                    for l in 0..32 {
                        let (mut lo, mut hi) = ((q[l] & 15) as u32, (q[l] >> 4) as u32);
                        if five {
                            lo += if qh[l] & (1 << (2 * g)) != 0 { 16 } else { 0 };
                            hi += if qh[l] & (2 << (2 * g)) != 0 { 16 } else { 0 };
                        }
                        y[64 * g + l] = d1 * lo as f32 - m1;
                        y[64 * g + 32 + l] = d2 * hi as f32 - m2;
                    }
                }
            }
            14 => {
                let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
                let d = h(b, 208);
                for n in 0..2 {
                    let (ql, qh, sc) = (&ql[64 * n..], &qh[32 * n..], &sc[8 * n..]);
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 15) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        let y = &mut y[128 * n..];
                        y[l] = d * sc[is] as i8 as f32 * q1 as f32;
                        y[l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                        y[l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                        y[l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
                    }
                }
            }
            21 => {
                // (d x scale_g) x the signed level
                let d = h(b, 0);
                for g in 0..8 {
                    let db = d * crate::iq3s_group_scale(b, g) as f32;
                    let (l, s) = crate::iq3s_group(b, g);
                    for e in 0..32 {
                        let v = l[e] as i32;
                        y[32 * g + e] = db * (if (s >> e) & 1 == 1 { -v } else { v }) as f32;
                    }
                }
            }
            42 => {
                // d x (q - 1) for each 2-bit field q, four fields a byte, lowest first
                let d = h(b, 0);
                for (j, yj) in y.iter_mut().enumerate().take(64) {
                    let q = (b[2 + j / 4] >> (2 * (j % 4))) & 3;
                    *yj = (q as i32 - 1) as f32 * d;
                }
            }
            18 => {
                // (d x (2s + 1) / 4 for the group) x the signed level
                let d = h(b, 0);
                for g in 0..8 {
                    let db = d * crate::iq3xxs_group_scale(b, g) as f32 * 0.25;
                    let (l, s) = crate::iq3xxs_group(b, g);
                    for e in 0..32 {
                        let v = l[e] as i32;
                        y[32 * g + e] = db * (if (s >> e) & 1 == 1 { -v } else { v }) as f32;
                    }
                }
            }
            22 => {
                // (d x (2s + 1) / 8) x the signed level, for each run of 16
                let d = h(b, 0);
                for c in 0..16 {
                    let db = d * crate::iq2s_chunk_scale(b, c) as f32 * 0.125;
                    for i in 2 * c..2 * c + 2 {
                        let (l, s) = crate::iq2s_group(b, i);
                        for e in 0..8 {
                            let v = l[e] as i32;
                            y[8 * i + e] = db * (if (s >> e) & 1 == 1 { -v } else { v }) as f32;
                        }
                    }
                }
            }
            23 => {
                // (d x scale_g) x the level
                let sc = crate::iq4xs_scales(b);
                let d = h(b, 0);
                for g in 0..8 {
                    let dl = d * sc[g] as f32;
                    for l in 0..16 {
                        let q = b[8 + 16 * g + l];
                        y[32 * g + l] = dl * crate::IQ4_LEVELS[(q & 15) as usize] as f32;
                        y[32 * g + l + 16] = dl * crate::IQ4_LEVELS[(q >> 4) as usize] as f32;
                    }
                }
            }
            11 => {
                // d x scale x value, with the value (-4..3) from two qs bits and an hmask bit
                let sc = crate::q3k_unpack_scales(b[96..108].try_into().unwrap());
                let d = h(b, 108);
                for (e, ye) in y.iter_mut().enumerate() {
                    let (n, j, hh, l) = (e / 128, e % 128 / 32, e % 32 / 16, e % 16);
                    let q = (b[32 + 32 * n + 16 * hh + l] >> (2 * j)) & 3;
                    let hi = (b[16 * hh + l] >> (j + 4 * n)) & 1;
                    *ye = d * sc[e / 16] as f32 * (q as i32 - if hi == 1 { 0 } else { 4 }) as f32;
                }
            }
            20 => {
                let d = h(b, 0);
                for j in 0..16 {
                    y[j] = d * IQ4NL[(b[2 + j] & 15) as usize] as f32;
                    y[j + 16] = d * IQ4NL[(b[2 + j] >> 4) as usize] as f32;
                }
            }
            39 => {
                let d = crate::e8m0_half(b[0]);
                for j in 0..16 {
                    y[j] = d * crate::MXFP4_LEVELS[(b[1 + j] & 15) as usize] as f32;
                    y[j + 16] = d * crate::MXFP4_LEVELS[(b[1 + j] >> 4) as usize] as f32;
                }
            }
            _ => unreachable!(),
        }
    }
    true
}
