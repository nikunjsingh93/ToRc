//! vision.rs - native image input for Qwen3.6-35B-A3B (and its family): the vision tower from the model's mmproj GGUF
//! (`qwen3vl_merger`: a 27-block ViT with a 2 x 2 patch merger, no DeepStack), the image preprocessing, and the
//! bookkeeping the language model needs to read the resulting rows.
//!
//! An image becomes `mh * mw` rows of the language model's width. They enter the prompt as ids with the top bit set
//! (`ROW_BIT`): the prefix names the image (a hash of its bytes), the low 12 bits the row. Prompt reuse compares ids, so
//! two different images never look alike, and the same image looks the same in the next turn. The forward pass swaps
//! such an id for its row (`copy_row`); everything else about the sequence is as for text.
//!
//! The language model reads positions as (time, height, width), interleaved over the rotation pairs (M-RoPE as Qwen3-VL
//! writes it). `set_positions` records where each image sits in the sequence, and `position` answers a slot's three
//! positions: text keeps t = h = w (plain RoPE), an image's rows take the grid, and the text after an image continues
//! from the larger of its two sides.

use format::gguf::Gguf;
use format::meta::Meta;
use format::{Result, err};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// The top bit marks an id as an image row.
pub const ROW_BIT: u32 = 0x8000_0000;
/// The most rows one image may have (the row number is 12 bits).
pub const MAX_ROWS: usize = 4095;

pub fn is_row(id: u32) -> bool {
    id & ROW_BIT != 0
}

/// The id a row stands for in places that only know text (the MTP block's token embeddings).
pub fn text_id(id: u32, pad: u32) -> u32 {
    if is_row(id) { pad } else { id }
}

// ---------------------------------------------------------------------------------------------------------------------
// the encoded image and the registry of what the sequence refers to

/// One image as the language model reads it: `mh * mw` rows of `dim` floats (the merged patch grid, row by row).
pub struct ImageRows {
    pub mh: usize,
    pub mw: usize,
    pub dim: usize,
    pub rows: Vec<f32>,
}

struct Entry {
    prefix: u32,
    hash: [u8; 32],
    img: Arc<ImageRows>,
}

const KEEP: usize = 32;
static REG: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

/// The image's ids: its rows, as the prompt holds them. The same bytes always give the same ids.
pub fn register(hash: [u8; 32], img: Arc<ImageRows>) -> Vec<u32> {
    let mut reg = REG.lock().unwrap_or_else(|p| p.into_inner());
    let n = img.mh * img.mw;
    let prefix = match reg.iter().position(|e| e.hash == hash) {
        Some(i) => {
            // the newest last: the oldest go first
            let e = reg.remove(i);
            let p = e.prefix;
            reg.push(e);
            p
        }
        None => {
            let mut p = u32::from_le_bytes(hash[..4].try_into().unwrap()) & 0x7_ffff;
            while reg.iter().any(|e| e.prefix == p) {
                p = (p + 1) & 0x7_ffff;
            }
            reg.push(Entry { prefix: p, hash, img });
            if reg.len() > KEEP {
                reg.remove(0);
            }
            p
        }
    };
    (0..n as u32).map(|k| ROW_BIT | prefix << 12 | k).collect()
}

fn lookup(id: u32) -> Option<(Arc<ImageRows>, usize)> {
    let reg = REG.lock().unwrap_or_else(|p| p.into_inner());
    let prefix = (id & !ROW_BIT) >> 12;
    reg.iter().find(|e| e.prefix == prefix).map(|e| (e.img.clone(), (id & 0xfff) as usize))
}

/// Writes the row an image id stands for into dst (the language model's width). False when the image is not known.
pub fn copy_row(id: u32, dst: &mut [f32]) -> bool {
    match lookup(id) {
        Some((img, k)) if k < img.mh * img.mw && dst.len() == img.dim => {
            dst.copy_from_slice(&img.rows[k * img.dim..(k + 1) * img.dim]);
            true
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// positions

struct Span {
    /// the slot of the image's first row
    start: usize,
    mh: usize,
    mw: usize,
    /// the position of the first row (all rows take it as their time, and rows and columns count up from it)
    base: i64,
    /// what a slot after the image adds to get its position: base + max(mh, mw) - (start + mh * mw)
    delta: i64,
}

static ACTIVE: AtomicBool = AtomicBool::new(false);
static SPANS: RwLock<Vec<Span>> = RwLock::new(Vec::new());

/// Records where the images in `ids` sit. With none, positions are the slots (plain RoPE) as before.
pub fn set_positions(ids: &[u32]) {
    let mut spans: Vec<Span> = Vec::new();
    let mut delta = 0i64;
    let mut i = 0;
    while i < ids.len() {
        if !is_row(ids[i]) {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < ids.len() && is_row(ids[j]) && (ids[j] & 0xfff) as usize == j - i {
            j += 1;
        }
        let (mh, mw) = lookup(ids[i]).map_or((1, j - i), |(img, _)| (img.mh, img.mw));
        // a run that is not a whole image (cut by the context) is laid out as a single row of its own length
        let (mh, mw) = if mh * mw == j - i { (mh, mw) } else { (1, j - i) };
        let base = i as i64 + delta;
        delta = base + mh.max(mw) as i64 - (i + mh * mw) as i64;
        spans.push(Span { start: i, mh, mw, base, delta });
        i = j;
    }
    ACTIVE.store(!spans.is_empty(), Ordering::SeqCst);
    *SPANS.write().unwrap_or_else(|p| p.into_inner()) = spans;
}

/// A slot's (time, height, width) positions.
pub fn position(slot: usize) -> [i64; 3] {
    let spans = SPANS.read().unwrap_or_else(|p| p.into_inner());
    let Some(s) = spans.iter().rev().find(|s| s.start <= slot) else {
        return [slot as i64; 3];
    };
    let k = slot - s.start;
    if k < s.mh * s.mw {
        [s.base, s.base + (k / s.mw) as i64, s.base + (k % s.mw) as i64]
    } else {
        [slot as i64 + s.delta; 3]
    }
}

/// The RoPE table for a slot: plain RoPE (rope_row) while no image is in the sequence, else the interleaved M-RoPE of
/// Qwen3-VL: rotation pair s reads the height for s % 3 == 1, the width for s % 3 == 2 and the time for s % 3 == 0.
pub fn rope_slot(cs: &mut [f32], sn: &mut [f32], slot: u32, n_rot: u32, base: f32) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return crate::ops::rope_row(cs, sn, slot, n_rot, base);
    }
    let p = position(slot as usize);
    for i in 0..(n_rot / 2) as usize {
        let pos = p[i % 3] as f32;
        let theta = pos * base.powf(-2.0 * i as f32 / n_rot as f32);
        cs[i] = theta.cos();
        sn[i] = theta.sin();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// the vision tower's weights

/// A BF16 matrix in the mmap: `rows` rows of `cols` weights.
struct Mat {
    off: usize,
    rows: usize,
    cols: usize,
}

struct Block {
    ln1: (Vec<f32>, Vec<f32>),
    ln2: (Vec<f32>, Vec<f32>),
    qkv: (Mat, Vec<f32>),
    out: (Mat, Vec<f32>),
    up: (Mat, Vec<f32>),
    down: (Mat, Vec<f32>),
}

pub struct Mmproj {
    map: memmap2::Mmap,
    dim: usize,
    heads: usize,
    patch: usize,
    merge: usize,
    grid: usize,
    out_dim: usize,
    /// the patch embedding as one matrix over a flattened patch (the two time kernels added: a still image is two
    /// equal frames)
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    /// the learned position table, grid x grid rows of dim, row-major by (y, x)
    pos: Vec<f32>,
    blocks: Vec<Block>,
    post_ln: (Vec<f32>, Vec<f32>),
    fc1: (Mat, Vec<f32>),
    fc2: (Mat, Vec<f32>),
}

const LN_EPS: f32 = 1e-6;

impl Mmproj {
    pub fn open(path: &str) -> Result<Mmproj> {
        let g = Gguf::open(path)?;
        let m = Meta::parse(&g.metadata)?;
        if m.str("clip.projector_type") != Some("qwen3vl_merger") {
            return err(format!("{path}: not a qwen3vl_merger projector"));
        }
        let need = |k: &str| m.u32(k).ok_or_else(|| format::Error(format!("{path}: missing {k}")));
        let (dim, nb) = (need("clip.vision.embedding_length")? as usize, need("clip.vision.block_count")? as usize);
        let heads = need("clip.vision.attention.head_count")? as usize;
        let patch = need("clip.vision.patch_size")? as usize;
        let size = need("clip.vision.image_size")? as usize;
        let merge = need("clip.vision.spatial_merge_size")? as usize;
        let out_dim = need("clip.vision.projection_dim")? as usize;
        if merge != 2 || dim % heads != 0 || (dim / heads) % 2 != 0 || size % patch != 0 {
            return err(format!("{path}: unexpected vision geometry"));
        }
        let file = std::fs::File::open(path).map_err(|e| format::Error(format!("{path}: {e}")))?;
        // SAFETY: the file is only read, and the map is kept for as long as the matrices that point into it
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format::Error(format!("{path}: {e}")))?;
        let t = |name: &str| g.tensor(name).ok_or_else(|| format::Error(format!("{path}: missing tensor {name}")));
        let f32s = |name: &str| -> Result<Vec<f32>> {
            let t = t(name)?;
            let n: u64 = t.dims.iter().product();
            let b = &map[t.offset as usize..(t.offset + t.bytes) as usize];
            match t.ty {
                0 => Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
                30 => Ok(b.chunks_exact(2).map(|c| bf16(u16::from_le_bytes(c.try_into().unwrap()))).collect()),
                1 => Ok(b.chunks_exact(2).map(|c| f16(u16::from_le_bytes(c.try_into().unwrap()))).collect()),
                _ => err(format!("{path}: tensor {name}: type {} ({n} values)", t.ty)),
            }
        };
        let mat = |name: &str, rows: usize, cols: usize| -> Result<Mat> {
            let t = t(name)?;
            if t.ty != 30 || t.dims[0] as usize != cols || t.dims[1] as usize != rows || t.offset % 2 != 0 {
                return err(format!("{path}: tensor {name}: expected a BF16 matrix of {rows} x {cols}"));
            }
            Ok(Mat { off: t.offset as usize, rows, cols })
        };
        let pair = |a: &str, b: &str| -> Result<(Vec<f32>, Vec<f32>)> { Ok((f32s(a)?, f32s(b)?)) };
        let lin = |name: &str, rows: usize, cols: usize| -> Result<(Mat, Vec<f32>)> {
            Ok((mat(&format!("{name}.weight"), rows, cols)?, f32s(&format!("{name}.bias"))?))
        };
        let (w0, w1) = (f32s("v.patch_embd.weight")?, f32s("v.patch_embd.weight.1")?);
        let patch_w: Vec<f32> = w0.iter().zip(&w1).map(|(a, b)| a + b).collect();
        let k = 3 * patch * patch;
        if patch_w.len() != dim * k {
            return err(format!("{path}: patch embedding of {} values", patch_w.len()));
        }
        let ffn = need("clip.vision.feed_forward_length")? as usize;
        let mut blocks = Vec::with_capacity(nb);
        for i in 0..nb {
            let p = format!("v.blk.{i}");
            blocks.push(Block {
                ln1: pair(&format!("{p}.ln1.weight"), &format!("{p}.ln1.bias"))?,
                ln2: pair(&format!("{p}.ln2.weight"), &format!("{p}.ln2.bias"))?,
                qkv: lin(&format!("{p}.attn_qkv"), 3 * dim, dim)?,
                out: lin(&format!("{p}.attn_out"), dim, dim)?,
                up: lin(&format!("{p}.ffn_up"), ffn, dim)?,
                down: lin(&format!("{p}.ffn_down"), dim, ffn)?,
            });
        }
        let grid = size / patch;
        let pos = f32s("v.position_embd.weight")?;
        if pos.len() != grid * grid * dim {
            return err(format!("{path}: position table of {} values", pos.len()));
        }
        let md = dim * merge * merge;
        Ok(Mmproj {
            dim,
            heads,
            patch,
            merge,
            grid,
            out_dim,
            patch_w,
            patch_b: f32s("v.patch_embd.bias")?,
            pos,
            post_ln: pair("v.post_ln.weight", "v.post_ln.bias")?,
            fc1: lin("mm.0", md, md)?,
            fc2: lin("mm.2", out_dim, md)?,
            blocks,
            map,
        })
    }

    /// The width of the rows it produces (the language model's d_model).
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    fn w16(&self, m: &Mat) -> &[u16] {
        let b = &self.map[m.off..m.off + m.rows * m.cols * 2];
        // SAFETY: the offset is even and the map is page aligned, so the bytes are a whole, aligned [u16]; BF16 is
        // stored little-endian and every target here is little-endian
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, m.rows * m.cols) }
    }

    /// The image's rows. `gh` x `gw` is the patch grid (both multiples of the merge size) and `patches` the flattened
    /// patches in merge-window order (see `patches_of`).
    pub fn encode(&self, patches: &[f32], gh: usize, gw: usize) -> Result<ImageRows> {
        let (d, n) = (self.dim, gh * gw);
        let k = 3 * self.patch * self.patch;
        if gh % self.merge != 0 || gw % self.merge != 0 || patches.len() != n * k {
            return err("vision: bad patch grid");
        }
        // the patch embedding and the interpolated position table
        let mut x = vec![0.0f32; n * d];
        linear_f32(patches, n, k, &self.patch_w, d, &self.patch_b, &mut x);
        let (rows, cols) = merge_order(gh, gw, self.merge);
        let posr = self.pos_rows(&rows, &cols, gh, gw);
        for (a, b) in x.iter_mut().zip(&posr) {
            *a += b;
        }
        let rope = self.rope_table(&rows, &cols);
        let (hd, nh) = (d / self.heads, self.heads);
        let mut xn = vec![0.0f32; n * d];
        let mut qkv = vec![0.0f32; n * 3 * d];
        let mut att = vec![0.0f32; n * d];
        let mut tmp = vec![0.0f32; n * d];
        let ffn = self.blocks[0].up.0.rows;
        let mut mid = vec![0.0f32; n * ffn];
        let (mut t_lin, mut t_att, mut t_oth) = (0.0f64, 0.0f64, 0.0f64);
        let mut tm = std::time::Instant::now();
        let mut lap = |acc: &mut f64| {
            let now = std::time::Instant::now();
            *acc += (now - tm).as_secs_f64();
            tm = now;
        };
        for b in &self.blocks {
            layer_norm(&x, d, &b.ln1.0, &b.ln1.1, &mut xn);
            lap(&mut t_oth);
            self.linear(&xn, n, &b.qkv, &mut qkv);
            lap(&mut t_lin);
            attention(&qkv, n, nh, hd, &rope, &mut att);
            lap(&mut t_att);
            self.linear(&att, n, &b.out, &mut tmp);
            lap(&mut t_lin);
            add_into(&mut x, &tmp);
            layer_norm(&x, d, &b.ln2.0, &b.ln2.1, &mut xn);
            lap(&mut t_oth);
            self.linear(&xn, n, &b.up, &mut mid);
            lap(&mut t_lin);
            gelu_all(&mut mid);
            lap(&mut t_oth);
            self.linear(&mid, n, &b.down, &mut tmp);
            lap(&mut t_lin);
            add_into(&mut x, &tmp);
            lap(&mut t_oth);
        }
        if crate::switches::on("TORC_VISION_PROF") {
            let a = |i: usize| ATT_NS[i].load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
            eprintln!("vision: attention thread-seconds: scores {:.1}, softmax {:.1}, values {:.1}", a(0), a(1), a(2));
            eprintln!("vision: {n} patches: products {t_lin:.2} s, attention {t_att:.2} s, other {t_oth:.2} s");
        }
        // the merger: a norm per patch, then four neighbours side by side (they are consecutive rows), two products
        layer_norm(&x, d, &self.post_ln.0, &self.post_ln.1, &mut xn);
        let (nm, md) = (n / (self.merge * self.merge), d * self.merge * self.merge);
        let mut h1 = vec![0.0f32; nm * md];
        self.linear(&xn, nm, &self.fc1, &mut h1);
        gelu_all(&mut h1);
        let mut out = vec![0.0f32; nm * self.out_dim];
        self.linear(&h1, nm, &self.fc2, &mut out);
        Ok(ImageRows { mh: gh / self.merge, mw: gw / self.merge, dim: self.out_dim, rows: out })
    }

    fn linear(&self, x: &[f32], n: usize, w: &(Mat, Vec<f32>), out: &mut [f32]) {
        linear_bf16(x, n, w.0.cols, self.w16(&w.0), w.0.rows, &w.1, out);
    }

    /// The position table interpolated (bilinear, corners aligned) to the patch grid, for each patch in merge order.
    fn pos_rows(&self, rows: &[usize], cols: &[usize], gh: usize, gw: usize) -> Vec<f32> {
        let (d, g) = (self.dim, self.grid);
        let at = |i: usize, n: usize| if n > 1 { i as f32 * (g - 1) as f32 / (n - 1) as f32 } else { 0.0 };
        let mut out = vec![0.0f32; rows.len() * d];
        for (p, (&r, &c)) in rows.iter().zip(cols).enumerate() {
            let (fy, fx) = (at(r, gh), at(c, gw));
            let (y0, x0) = (fy.floor() as usize, fx.floor() as usize);
            let (y1, x1) = ((y0 + 1).min(g - 1), (x0 + 1).min(g - 1));
            let (dy, dx) = (fy - y0 as f32, fx - x0 as f32);
            let row = |y: usize, x: usize| &self.pos[(y * g + x) * d..(y * g + x + 1) * d];
            let (a, b, cc, dd) = (row(y0, x0), row(y0, x1), row(y1, x0), row(y1, x1));
            let o = &mut out[p * d..(p + 1) * d];
            for i in 0..d {
                o[i] =
                    a[i] * (1.0 - dy) * (1.0 - dx) + b[i] * (1.0 - dy) * dx + cc[i] * dy * (1.0 - dx) + dd[i] * dy * dx;
            }
        }
        out
    }

    /// The rotary table of the tower, per patch: half the head's pairs take the patch's row and half its column, both
    /// with the frequencies of a half-width head (theta 10000).
    fn rope_table(&self, rows: &[usize], cols: &[usize]) -> Vec<(Vec<f32>, Vec<f32>)> {
        let hd = self.dim / self.heads;
        let (half, quarter) = (hd / 2, hd / 4);
        let inv: Vec<f32> = (0..quarter).map(|j| 10000f32.powf(-(j as f32) / quarter as f32)).collect();
        rows.iter()
            .zip(cols)
            .map(|(&r, &c)| {
                let (mut cs, mut sn) = (vec![0.0f32; half], vec![0.0f32; half]);
                for i in 0..half {
                    let pos = if i < quarter { r } else { c } as f32;
                    let a = pos * inv[i % quarter];
                    cs[i] = a.cos();
                    sn[i] = a.sin();
                }
                (cs, sn)
            })
            .collect()
    }
}

/// The patches' (row, column) in the grid, in merge-window order: each 2 x 2 window's four patches are consecutive.
fn merge_order(gh: usize, gw: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    let (mut rows, mut cols) = (Vec::with_capacity(gh * gw), Vec::with_capacity(gh * gw));
    for bi in 0..gh / m {
        for bj in 0..gw / m {
            for di in 0..m {
                for dj in 0..m {
                    rows.push(bi * m + di);
                    cols.push(bj * m + dj);
                }
            }
        }
    }
    (rows, cols)
}

fn bf16(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn f16(x: u16) -> f32 {
    let (s, e, m) = ((x >> 15) as u32, ((x >> 10) & 0x1f) as i32, (x & 0x3ff) as u32);
    let v = match e {
        0 => m as f32 * 2f32.powi(-24),
        31 => {
            if m == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + m as f32 / 1024.0) * 2f32.powi(e - 15),
    };
    if s == 1 { -v } else { v }
}

/// GELU, tanh form (ggml_gelu, which llama.cpp uses for this tower).
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

fn layer_norm(x: &[f32], d: usize, w: &[f32], b: &[f32], out: &mut [f32]) {
    par_pieces(out, 16 * d, &|off, piece| {
        for (xr, or) in x[off..off + piece.len()].chunks_exact(d).zip(piece.chunks_exact_mut(d)) {
            let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = xr.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let s = 1.0 / (var as f32 + LN_EPS).sqrt();
            for i in 0..d {
                or[i] = (xr[i] - mean as f32) * s * w[i] + b[i];
            }
        }
    });
}

/// a += b, across the pool.
fn add_into(a: &mut [f32], b: &[f32]) {
    par_pieces(a, 1 << 14, &|off, piece| {
        for (x, y) in piece.iter_mut().zip(&b[off..]) {
            *x += y;
        }
    });
}

/// GELU in place, across the pool.
fn gelu_all(a: &mut [f32]) {
    par_pieces(a, 1 << 14, &|_, piece| piece.iter_mut().for_each(|v| *v = gelu(*v)));
}

// ---------------------------------------------------------------------------------------------------------------------
// products

/// out (n x m) = x (n x k) * w^T + bias, with w as f32 rows.
fn linear_f32(x: &[f32], n: usize, k: usize, w: &[f32], m: usize, bias: &[f32], out: &mut [f32]) {
    gemm(x, n, k, m, bias, out, &|r0, r1, buf: &mut [f32]| buf[..(r1 - r0) * k].copy_from_slice(&w[r0 * k..r1 * k]));
}

/// out (n x m) = x (n x k) * w^T + bias, with w as BF16 rows.
fn linear_bf16(x: &[f32], n: usize, k: usize, w: &[u16], m: usize, bias: &[f32], out: &mut [f32]) {
    gemm(x, n, k, m, bias, out, &|r0, r1, buf: &mut [f32]| {
        for (d, s) in buf.iter_mut().zip(&w[r0 * k..r1 * k]) {
            *d = bf16(*s);
        }
    });
}

/// Weight rows a task takes, and tokens: a task's working set (its rows as f32 and its tokens) stays in the L2 cache,
/// and the tokens are read once per block of rows rather than once per few rows.
const RB: usize = 16;
const TB: usize = 1 << 20;

/// The product in blocks of RB weight rows by TB tokens: a block's rows are made f32 (`fetch`), then four weight rows
/// at a time take their dot products against each group of four tokens. k must be a multiple of 4 and m of 4.
fn gemm(
    x: &[f32],
    n: usize,
    k: usize,
    m: usize,
    bias: &[f32],
    out: &mut [f32],
    fetch: &(dyn Fn(usize, usize, &mut [f32]) + Sync),
) {
    assert!(k % 4 == 0 && m % 4 == 0 && x.len() == n * k && out.len() == n * m);
    let op = crate::pool::SyncPtr::new(out);
    let (nr, nt) = (m.div_ceil(RB), n.div_ceil(TB));
    crate::pool::par_for(nr * nt, &|task| {
        let (rb, tb) = (task / nt, task % nt);
        let (r0, r1) = (rb * RB, ((rb + 1) * RB).min(m));
        let (t0, t1) = (tb * TB, ((tb + 1) * TB).min(n));
        let mut wf = vec![0.0f32; (r1 - r0) * k];
        fetch(r0, r1, &mut wf);
        for rr in (0..r1 - r0).step_by(4) {
            let ws: [&[f32]; 4] = std::array::from_fn(|i| &wf[(rr + i) * k..(rr + i + 1) * k]);
            let mut t = t0;
            while t < t1 {
                // a last group short of four tokens repeats its final one, and only the real ones are written
                let xs: [&[f32]; 4] = std::array::from_fn(|i| {
                    let r = (t + i).min(n - 1);
                    &x[r * k..(r + 1) * k]
                });
                let acc = tile(&xs, &ws);
                for i in 0..4 {
                    if t + i < t1 {
                        for j in 0..4 {
                            // SAFETY: this task alone writes these columns of these rows
                            unsafe { *op.at((t + i) * m + r0 + rr + j) = acc[i][j] + bias[r0 + rr + j] };
                        }
                    }
                }
                t += 4;
            }
        }
    });
}

/// Four tokens against four weight rows: the sixteen dot products, each as four lanes summed at the end.
#[inline(always)]
#[cfg(target_arch = "aarch64")]
fn tile(xs: &[&[f32]; 4], ws: &[&[f32]; 4]) -> [[f32; 4]; 4] {
    use std::arch::aarch64::*;
    let k = xs[0].len();
    assert!(k % 4 == 0 && xs.iter().all(|c| c.len() == k) && ws.iter().all(|c| c.len() == k));
    // SAFETY: every load is at an index below k, which is the length of each slice
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
        let mut i = 0;
        while i < k {
            let x = [
                vld1q_f32(xs[0].as_ptr().add(i)),
                vld1q_f32(xs[1].as_ptr().add(i)),
                vld1q_f32(xs[2].as_ptr().add(i)),
                vld1q_f32(xs[3].as_ptr().add(i)),
            ];
            let w = [
                vld1q_f32(ws[0].as_ptr().add(i)),
                vld1q_f32(ws[1].as_ptr().add(i)),
                vld1q_f32(ws[2].as_ptr().add(i)),
                vld1q_f32(ws[3].as_ptr().add(i)),
            ];
            for t in 0..4 {
                for r in 0..4 {
                    acc[t][r] = vfmaq_f32(acc[t][r], x[t], w[r]);
                }
            }
            i += 4;
        }
        std::array::from_fn(|t| std::array::from_fn(|r| vaddvq_f32(acc[t][r])))
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn tile(xs: &[&[f32]; 4], ws: &[&[f32]; 4]) -> [[f32; 4]; 4] {
    let xc: [&[[f32; 4]]; 4] = std::array::from_fn(|t| xs[t].as_chunks::<4>().0);
    let wc: [&[[f32; 4]]; 4] = std::array::from_fn(|r| ws[r].as_chunks::<4>().0);
    let n4 = xc[0].len();
    // equal lengths up front, so the indexing below needs no checks
    assert!(xc.iter().all(|c| c.len() == n4) && wc.iter().all(|c| c.len() == n4));
    let mut acc = [[[0.0f32; 4]; 4]; 4];
    for i in 0..n4 {
        let xv: [[f32; 4]; 4] = std::array::from_fn(|t| xc[t][i]);
        let wv: [[f32; 4]; 4] = std::array::from_fn(|r| wc[r][i]);
        for t in 0..4 {
            for r in 0..4 {
                for l in 0..4 {
                    acc[t][r][l] = xv[t][l].mul_add(wv[r][l], acc[t][r][l]);
                }
            }
        }
    }
    std::array::from_fn(|t| std::array::from_fn(|r| (acc[t][r][0] + acc[t][r][1]) + (acc[t][r][2] + acc[t][r][3])))
}

/// Full attention over all n patches, per head, from the fused qkv rows (q, k, v each nh * hd wide). q and k are rotated
/// first (pairs i, i + hd / 2, with each patch's table).
fn attention(qkv: &[f32], n: usize, nh: usize, hd: usize, rope: &[(Vec<f32>, Vec<f32>)], out: &mut [f32]) {
    let d = nh * hd;
    let half = hd / 2;
    // head-major copies, rotated: [head][patch][hd]
    let mut q = vec![0.0f32; n * d];
    let mut k = vec![0.0f32; n * d];
    let mut v = vec![0.0f32; n * d];
    for p in 0..n {
        let row = &qkv[p * 3 * d..(p + 1) * 3 * d];
        let (cs, sn) = &rope[p];
        for h in 0..nh {
            let dst = (h * n + p) * hd;
            for (src, tgt, rot) in [(0, &mut q, true), (d, &mut k, true), (2 * d, &mut v, false)] {
                let s = &row[src + h * hd..src + (h + 1) * hd];
                let t = &mut tgt[dst..dst + hd];
                if rot {
                    for i in 0..half {
                        t[i] = s[i] * cs[i] - s[i + half] * sn[i];
                        t[i + half] = s[i + half] * cs[i] + s[i] * sn[i];
                    }
                } else {
                    // the values go transposed: [head][i][patch]
                    for i in 0..hd {
                        tgt[h * hd * n + i * n + p] = s[i];
                    }
                }
            }
        }
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let op = crate::pool::SyncPtr::new(out);
    // eight queries a task: their scores against every key (the four-by-four tile of the products), the softmax, then
    // the weighted values. n is a multiple of 4 (whole 2 x 2 windows); the last block's missing queries repeat the
    // final one and are not written
    const QB: usize = 8;
    let nq = n.div_ceil(QB);
    crate::pool::par_for(nh * nq, &|c| {
        let (h, qb) = (c / nq, c % nq);
        let (qh, kh, vh) =
            (&q[h * n * hd..(h + 1) * n * hd], &k[h * n * hd..(h + 1) * n * hd], &v[h * n * hd..(h + 1) * n * hd]);
        let t0 = std::time::Instant::now();
        let mut sc = vec![0.0f32; QB * n];
        let qrow = |i: usize| {
            let r = (qb * QB + i).min(n - 1);
            &qh[r * hd..(r + 1) * hd]
        };
        for g in 0..QB / 4 {
            let xs: [&[f32]; 4] = std::array::from_fn(|i| qrow(g * 4 + i));
            for j in (0..n).step_by(4) {
                let ws: [&[f32]; 4] = std::array::from_fn(|i| &kh[(j + i) * hd..(j + i + 1) * hd]);
                let t = tile(&xs, &ws);
                for i in 0..4 {
                    for l in 0..4 {
                        sc[(g * 4 + i) * n + j + l] = t[i][l] * scale;
                    }
                }
            }
        }
        let t1 = std::time::Instant::now();
        let mut sum = [0.0f32; QB];
        for i in 0..QB {
            let row = &mut sc[i * n..(i + 1) * n];
            let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut s = 0.0f32;
            for x in row.iter_mut() {
                *x = (*x - mx).exp();
                s += *x;
            }
            sum[i] = s;
        }
        let t2 = std::time::Instant::now();
        // the values, transposed per head (vt[head][i][key]): each output is a dot over the keys, the same tile
        for g in 0..QB / 4 {
            let xs: [&[f32]; 4] = std::array::from_fn(|i| &sc[(g * 4 + i) * n..(g * 4 + i + 1) * n]);
            for i0 in (0..hd).step_by(4) {
                let ws: [&[f32]; 4] = std::array::from_fn(|i| &vh[(i0 + i) * n..(i0 + i + 1) * n]);
                let t = tile(&xs, &ws);
                for qi in 0..4 {
                    let r = qb * QB + g * 4 + qi;
                    if r < n {
                        for l in 0..4 {
                            // SAFETY: this query's slice of this head is written by this task alone
                            unsafe { *op.at(r * d + h * hd + i0 + l) = t[qi][l] / sum[g * 4 + qi] };
                        }
                    }
                }
            }
        }
        let t3 = std::time::Instant::now();
        for (a, (x, y)) in [(0usize, (t0, t1)), (1, (t1, t2)), (2, (t2, t3))] {
            ATT_NS[a].fetch_add((y - x).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
    });
}

static ATT_NS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Runs f(offset, piece) over `out` cut into pieces of `unit` values (the last may be shorter), across the pool.
fn par_pieces(out: &mut [f32], unit: usize, f: &(dyn Fn(usize, &mut [f32]) + Sync)) {
    let len = out.len();
    let p = crate::pool::SyncPtr::new(out);
    crate::pool::par_for(len.div_ceil(unit), &|i| {
        let (a, b) = (i * unit, ((i + 1) * unit).min(len));
        // SAFETY: the pieces are disjoint
        f(a, unsafe { p.slice(a, b - a) });
    });
}

// ---------------------------------------------------------------------------------------------------------------------
// images in

/// The image size to read at: each side a multiple of 32 (a patch of 16, merged 2 x 2), in the proportions of the
/// picture, with at most `max_rows` merged rows. Returns the sides in units of 32.
pub fn grid_for(w: usize, h: usize, max_rows: usize) -> (usize, usize) {
    let max_rows = max_rows.clamp(1, MAX_ROWS);
    let (mut gw, mut gh) = (((w as f32 / 32.0).round() as usize).max(1), ((h as f32 / 32.0).round() as usize).max(1));
    if gw * gh > max_rows {
        let s = (max_rows as f32 / (gw * gh) as f32).sqrt();
        gw = ((gw as f32 * s).floor() as usize).max(1);
        gh = ((gh as f32 * s).floor() as usize).max(1);
    }
    (gw, gh)
}

/// Decodes a PNG or JPEG, scales it to the grid of `grid_for`, and cuts it into 16 x 16 patches (normalised with mean
/// and standard deviation 0.5) in merge-window order. Returns the patches, and the grid in patches (rows, columns).
pub fn patches_of(bytes: &[u8], max_rows: usize, patch: usize) -> Result<(Vec<f32>, usize, usize)> {
    let img = image::load_from_memory(bytes).map_err(|e| format::Error(format!("image: {e}")))?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    if w == 0 || h == 0 {
        return err("image: empty");
    }
    // transparency is shown over white
    let mut rgb = image::RgbImage::new(w as u32, h as u32);
    for (o, p) in rgb.pixels_mut().zip(rgba.pixels()) {
        let a = p[3] as u32;
        for c in 0..3 {
            o[c] = ((p[c] as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
        }
    }
    let (gw, gh) = grid_for(w, h, max_rows);
    let (tw, th) = (gw * 32, gh * 32);
    let rgb = if (tw, th) == (w, h) {
        rgb
    } else {
        image::imageops::resize(&rgb, tw as u32, th as u32, image::imageops::FilterType::CatmullRom)
    };
    let (pr, pc) = (th / patch, tw / patch);
    let (rows, cols) = merge_order(pr, pc, 2);
    let k = 3 * patch * patch;
    let mut out = vec![0.0f32; pr * pc * k];
    for (p, (&r, &c)) in rows.iter().zip(&cols).enumerate() {
        let o = &mut out[p * k..(p + 1) * k];
        for y in 0..patch {
            for x in 0..patch {
                let px = rgb.get_pixel((c * patch + x) as u32, (r * patch + y) as u32);
                for ch in 0..3 {
                    o[x + patch * y + patch * patch * ch] = (px[ch] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }
    }
    Ok((out, pr, pc))
}

/// The rows for the image bytes (PNG or JPEG), registered for the prompt: the ids to put in it. The same bytes are
/// encoded once; the encoded rows are kept for later requests.
pub fn embed(mp: &Mmproj, bytes: &[u8], max_rows: usize) -> Result<Vec<u32>> {
    let hash = *blake3::hash(bytes).as_bytes();
    {
        let reg = REG.lock().unwrap_or_else(|p| p.into_inner());
        if reg.iter().any(|e| e.hash == hash) {
            let img = reg.iter().find(|e| e.hash == hash).unwrap().img.clone();
            drop(reg);
            return Ok(register(hash, img));
        }
    }
    let t = std::time::Instant::now();
    let (patches, gh, gw) = patches_of(bytes, max_rows, mp.patch)?;
    let img = Arc::new(mp.encode(&patches, gh, gw)?);
    eprintln!("torc-server: image {} x {} rows read in {:.2} s", img.mh, img.mw, t.elapsed().as_secs_f64());
    Ok(register(hash, img))
}

/// The most merged rows an image may take (TORC_VISION_TOKENS, default 576; the tower's cost grows with the square).
pub fn max_rows() -> usize {
    crate::switches::num::<usize>("TORC_VISION_TOKENS").filter(|&n| n > 0).unwrap_or(576).min(MAX_ROWS)
}

// ---------------------------------------------------------------------------------------------------------------------
// the requests' images

static ON: AtomicBool = AtomicBool::new(false);

/// Whether a vision tower is loaded: only then do the prompt builders keep an image's place in the text.
pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

pub fn set_enabled(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// What stands in the prompt text where an image is: the chat template's own image markers.
pub const MARKER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// Whether a content part is an image: OpenAI's `image_url`, the Responses API's `input_image`, Anthropic's `image`.
pub fn is_image_part(p: &serde_json::Value) -> bool {
    matches!(p["type"].as_str(), Some("image_url" | "input_image" | "image"))
}

/// The image parts of the messages' contents, as bytes, in the order they appear (the order content_text puts its
/// markers in): a data URI with base64 data, or an Anthropic base64 source. A part that is not readable is an error,
/// so the prompt never has a marker without an image.
pub fn images_of(msgs: &[serde_json::Value]) -> Result<Vec<Vec<u8>>> {
    let bad = || format::Error("image: bad base64".into());
    let mut out = Vec::new();
    for m in msgs {
        let serde_json::Value::Array(parts) = &m["content"] else { continue };
        for p in parts.iter().filter(|p| is_image_part(p)) {
            if p["type"].as_str() == Some("image") {
                out.push(base64(p["source"]["data"].as_str().unwrap_or("")).ok_or_else(bad)?);
                continue;
            }
            // image_url is an object with url (chat completions) or the URL itself (Responses)
            let u = p["image_url"]["url"].as_str().or(p["image_url"].as_str()).unwrap_or("");
            match u.strip_prefix("data:").and_then(|d| d.split_once(";base64,")) {
                Some((_, b)) => out.push(base64(b).ok_or_else(bad)?),
                None => return err("image: only data: URIs with base64 data are read"),
            }
        }
    }
    Ok(out)
}

/// Standard base64 (the padding and any whitespace are allowed).
fn base64(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => return None,
        };
        acc = acc << 6 | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// Replaces each image marker's single pad token in `ids` by the next image's row ids.
pub fn expand(ids: &[u32], pad: u32, images: &[Vec<u32>]) -> Result<Vec<u32>> {
    let marks = ids.iter().filter(|&&t| t == pad).count();
    if marks != images.len() {
        return err(format!("the prompt has {marks} image marker(s) and the request {} image(s)", images.len()));
    }
    let mut it = images.iter();
    let mut out = Vec::with_capacity(ids.len());
    for &t in ids {
        if t == pad {
            out.extend_from_slice(it.next().unwrap());
        } else {
            out.push(t);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_reads_standard_text() {
        assert_eq!(base64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64("aGVs\nbG8gd29ybGQ=").unwrap(), b"hello world");
        assert!(base64("a$b").is_none());
    }

    #[test]
    fn grid_keeps_proportions_and_the_budget() {
        assert_eq!(grid_for(800, 300, 576), (25, 9));
        let (w, h) = grid_for(3840, 2160, 576);
        assert!(w * h <= 576 && w > h);
        assert_eq!(grid_for(10, 10, 576), (1, 1));
    }

    #[test]
    fn merge_order_makes_windows_consecutive() {
        let (r, c) = merge_order(4, 4, 2);
        assert_eq!(&r[..8], &[0, 0, 1, 1, 0, 0, 1, 1]);
        assert_eq!(&c[..8], &[0, 1, 0, 1, 2, 3, 2, 3]);
        assert_eq!((r[8], c[8]), (2, 0));
    }

    fn img(mh: usize, mw: usize) -> Arc<ImageRows> {
        Arc::new(ImageRows { mh, mw, dim: 4, rows: (0..mh * mw * 4).map(|i| i as f32).collect() })
    }

    /// Positions: text is its slot, an image's rows take the grid from the position after the marker, and the text
    /// after it continues from the larger side. Rows copy back, and ids are the same for the same bytes.
    #[test]
    fn positions_follow_the_image_grid() {
        let a = register([1; 32], img(2, 3));
        let again = register([1; 32], img(2, 3));
        assert_eq!(a, again);
        let b = register([2; 32], img(1, 2));
        assert_ne!(a[0] >> 12, b[0] >> 12);
        // slots: 0 1 2 = text, 3..9 = image a (2 x 3), 9 = text, 10 11 = image b (1 x 2), 12 = text
        let mut ids = vec![5, 6, 7];
        ids.extend(&a);
        ids.push(8);
        ids.extend(&b);
        ids.push(9);
        set_positions(&ids);
        assert_eq!(position(2), [2, 2, 2]);
        assert_eq!(position(3), [3, 3, 3]);
        assert_eq!(position(4), [3, 3, 4]);
        assert_eq!(position(6), [3, 4, 3]);
        assert_eq!(position(8), [3, 4, 5]);
        // text after the first image: 3 + max(2, 3) = 6 for slot 9
        assert_eq!(position(9), [6, 6, 6]);
        // the second image starts at slot 10 -> position 7; it is 1 x 2
        assert_eq!(position(10), [7, 7, 7]);
        assert_eq!(position(11), [7, 7, 8]);
        // after it: 7 + 2 = 9 for slot 12, and generated slots go on from there
        assert_eq!(position(12), [9, 9, 9]);
        assert_eq!(position(13), [10, 10, 10]);
        let mut row = [0.0f32; 4];
        assert!(copy_row(a[5], &mut row));
        assert_eq!(row, [20.0, 21.0, 22.0, 23.0]);
        set_positions(&[1, 2, 3]);
        assert_eq!(position(7), [7, 7, 7]);
    }

    #[test]
    fn expand_checks_the_count() {
        let ids = [1, 99, 2, 99, 3];
        assert_eq!(expand(&ids, 99, &[vec![ROW_BIT], vec![ROW_BIT | 1, ROW_BIT | 2]]).unwrap().len(), 6);
        assert!(expand(&ids, 99, &[vec![ROW_BIT]]).is_err());
    }
}

/// A timing and regression probe on the real tower: TORC_MMPROJ names the mmproj, TORC_TEST_IMAGE an image; with
/// TORC_VISION_OUT the rows are written there as f32 (a later run compares against them). Run with
/// `cargo test --release -p torc-model --lib vision_probe -- --ignored --nocapture`.
#[cfg(test)]
#[test]
#[ignore]
fn vision_probe() {
    let (Some(mp), Some(img)) = (crate::switches::get("TORC_MMPROJ"), crate::switches::get("TORC_TEST_IMAGE")) else {
        return;
    };
    crate::pool::init(0);
    let mp = Mmproj::open(&mp).unwrap();
    let bytes = std::fs::read(img).unwrap();
    let (p, gh, gw) = patches_of(&bytes, max_rows(), mp.patch).unwrap();
    for run in 0..2 {
        let t = std::time::Instant::now();
        let r = mp.encode(&p, gh, gw).unwrap();
        eprintln!("run {run}: {} x {} rows in {:.2} s", r.mh, r.mw, t.elapsed().as_secs_f64());
        if run == 1 {
            let bin: Vec<u8> = r.rows.iter().flat_map(|v| v.to_le_bytes()).collect();
            if let Some(out) = crate::switches::get("TORC_VISION_OUT") {
                if let Ok(old) = std::fs::read(&out) {
                    let old: Vec<f32> =
                        old.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for (a, b) in old.iter().zip(&r.rows) {
                        num += ((a - b) as f64).powi(2);
                        den += (*a as f64).powi(2);
                    }
                    eprintln!("relative difference from the saved rows: {:.3e}", (num / den).sqrt());
                } else {
                    std::fs::write(&out, bin).unwrap();
                    eprintln!("saved the rows");
                }
            }
        }
    }
}
