//! torc-kernels - the quantised kernels.
//!
//! Every path for a quant type (one vector or many, any rows a pass, SMMLA, VNNI) gives the same bits. The integer
//! parts are exact in any order, and the float parts keep a fixed order of operations, with fused multiply-adds only
//! where that order names them. The parity tests (tests/parity.rs) hold the paths to the same bits, and each product
//! to within float rounding of the type's definition.

/// The first len entries of an uninitialised stack buffer, zeroed. This lets a kernel keep its accumulators on the
/// stack without zeroing the whole buffer on every call - zeroing SMMLA's 16 KB array for each 16-row task cost more
/// than reading the weights the task needed at 2-3 vectors. All-zero bits are 0.0 in every lane of the vector types.
///
/// # Safety
/// len <= N, and T must be a plain vector type for which all-zero bits are a valid value.
#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn zeroed_prefix<T, const N: usize>(s: &mut core::mem::MaybeUninit<[T; N]>, len: usize) -> &mut [T] {
    assert!(len <= N);
    let p = s.as_mut_ptr().cast::<T>();
    unsafe {
        core::ptr::write_bytes(p, 0, len);
        core::slice::from_raw_parts_mut(p, len)
    }
}

pub mod ops;
pub mod platform;
pub mod quant;
pub mod switches;

// short paths for the model crate (kernels::neon, kernels::x86mm, kernels::q8k, ...)
pub use platform::cpu::{fvec, mix};
#[cfg(target_arch = "aarch64")]
pub use platform::cpu::{neon, neon_mm};
#[cfg(target_arch = "x86_64")]
pub use platform::cpu::{x86, x86_gemm};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
pub use platform::cpu::{x86dot, x86mm};
pub use quant::f16::*;
pub use quant::iq2_s::*;
pub use quant::iq3_s::*;
pub use quant::iq3_xxs::*;
pub use quant::iq4::*;
pub use quant::mxfp4::*;
pub use quant::q2_0::*;
pub use quant::q3_k::*;
pub use quant::q4_0::*;
pub use quant::q4_k::*;
pub use quant::q8_k::*;
pub use quant::{dequant, iq2s_grid, iq3s_grid, iq3xxs_grid, lay, q8k};

/// A Q4_0 dot product, using the fastest kernel this CPU has.
pub fn q4_0_dot(w: &[u8], x: &[Groups32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        return unsafe { neon::q4_0_dot(w, x) };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        return unsafe { x86::q4_0_dot(w, x) };
    }
    q4_0_dot_scalar(w, x)
}

/// Q4_0 matrix times one vector: y[r] is the dot product of weight row r (row_bytes bytes from w[r * row_bytes]) with
/// x.
pub fn q4_0_dot_rows(w: &[u8], row_bytes: usize, x: &[Groups32], y: &mut [f32]) {
    for (r, out) in y.iter_mut().enumerate() {
        *out = q4_0_dot(&w[r * row_bytes..(r + 1) * row_bytes], x);
    }
}

#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
pub use platform::npu::ggml;
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
pub use platform::npu::split as npu;

/// One step of the gated DeltaNet recurrence over rows = v.len() rows of an f32 state (qwen4exp's), in the NEON
/// kernel's lane order. For each row, 16 elements are handled per step in four 4-lane accumulators (lane l holds
/// elements i + 4l to i + 4l + 3, and an 8-element tail uses l = 0 and 1): r = r * alpha, then a[l] = fma(r, k, a[l]).
/// The lanes are summed as hsum4x2 does, (0 + 2) + (1 + 3), to give sk; then d = (v - sk) * beta, r = fma(d, k, r) and
/// a[l] = fma(r, q, a[l]), and the output is the lane sum of a[l] times q_scale. The state is read from `src` when one
/// is given (a revertible block's base) and written to `state`; otherwise it is updated in place. This is plain Rust
/// that matches the NEON kernel lane for lane.
#[allow(clippy::too_many_arguments)]
pub fn delta_rule_f32_scalar(
    src: Option<&[f32]>,
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    beta: f32,
    q_scale: f32,
    o: &mut [f32],
    dim: usize,
) {
    assert!(dim.is_multiple_of(8));
    // hsum4x2(a, b): s = a + b lane by lane, then (s0 + s2) + (s1 + s3)
    let hsum = |a: &[[f32; 4]; 4]| {
        let x = [a[0][0] + a[2][0], a[0][1] + a[2][1], a[0][2] + a[2][2], a[0][3] + a[2][3]];
        let y = [a[1][0] + a[3][0], a[1][1] + a[3][1], a[1][2] + a[3][2], a[1][3] + a[3][3]];
        let s = [x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]];
        (s[0] + s[2]) + (s[1] + s[3])
    };
    let steps = |dim: usize| {
        let full = dim / 16 * 16;
        let mut v: Vec<(usize, usize)> = (0..full).step_by(16).flat_map(|i| (0..4).map(move |l| (i, l))).collect();
        for i in (full..dim).step_by(8) {
            v.push((i, 0));
            v.push((i, 1));
        }
        v
    };
    let st = steps(dim);
    for j in 0..v.len() {
        let r = &mut state[j * dim..(j + 1) * dim];
        if let Some(s) = src {
            r.copy_from_slice(&s[j * dim..(j + 1) * dim]);
        }
        let mut a = [[0.0f32; 4]; 4];
        for &(i, l) in &st {
            for e in 0..4 {
                let p = i + 4 * l + e;
                r[p] *= alpha;
                a[l][e] = r[p].mul_add(k[p], a[l][e]);
            }
        }
        let sk = hsum(&a);
        let d = (v[j] - sk) * beta;
        let mut a = [[0.0f32; 4]; 4];
        for &(i, l) in &st {
            for e in 0..4 {
                let p = i + 4 * l + e;
                r[p] = d.mul_add(k[p], r[p]);
                a[l][e] = r[p].mul_add(q[p], a[l][e]);
            }
        }
        o[j] = hsum(&a) * q_scale;
    }
}

/// delta_rule_f32_scalar using the fastest kernel this CPU has: NEON on ARM64 (four rows a pass) and AVX2 on x86-64,
/// both giving the same bits.
#[allow(clippy::too_many_arguments, unused_mut)]
pub fn delta_rule_f32(
    src: Option<&[f32]>,
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    beta: f32,
    q_scale: f32,
    o: &mut [f32],
    dim: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if dim % 16 == 0 {
        neon::delta_rule_f32(src, state, q, k, v, alpha, beta, q_scale, o, dim);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dim.is_multiple_of(16)
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
    {
        // SAFETY: the if above checked the features
        unsafe { x86::delta_rule_f32(src, state, q, k, v, alpha, beta, q_scale, o, dim) };
        return;
    }
    delta_rule_f32_scalar(src, state, q, k, v, alpha, beta, q_scale, o, dim)
}

/// delta_rule_h in plain Rust, matching the AVX2 kernel (x86::delta_rule_h) lane for lane. For each row, 16 numbers are
/// handled per step in two 8-lane accumulators: the row's numbers are widened from fp16, decayed and dotted with k;
/// then d = (v - sum) * beta, and each number becomes fma(d, k, r), is stored back as fp16 (round to nearest even) and
/// is dotted with q in f32. dim must be a multiple of 16. This covers the shapes no native kernel takes.
#[allow(clippy::too_many_arguments)]
pub fn delta_rule_h_scalar(
    state: &mut [u16],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    beta: f32,
    q_scale: f32,
    o: &mut [f32],
    dim: usize,
) {
    assert!(dim.is_multiple_of(16) && state.len() >= v.len() * dim && o.len() >= v.len());
    // hsum8(a0 + a1): lanes l and l + 4 of the 8-lane sum, then (0 + 2) + (1 + 3)
    let hsum = |a: &[[f32; 8]; 2]| {
        let s: [f32; 8] = std::array::from_fn(|l| a[0][l] + a[1][l]);
        let t: [f32; 4] = std::array::from_fn(|l| s[l] + s[l + 4]);
        (t[0] + t[2]) + (t[1] + t[3])
    };
    let mut r = vec![0.0f32; dim];
    for j in 0..v.len() {
        let row = &mut state[j * dim..(j + 1) * dim];
        let mut a = [[0.0f32; 8]; 2];
        for i in 0..dim {
            r[i] = f16_to_f32(row[i]) * alpha;
            let (h, l) = ((i / 8) % 2, i % 8);
            a[h][l] = r[i].mul_add(k[i], a[h][l]);
        }
        let d = (v[j] - hsum(&a)) * beta;
        let mut a = [[0.0f32; 8]; 2];
        for i in 0..dim {
            let x = d.mul_add(k[i], r[i]);
            row[i] = q8k::f32_to_f16(x);
            let (h, l) = ((i / 8) % 2, i % 8);
            a[h][l] = x.mul_add(q[i], a[h][l]);
        }
        o[j] = hsum(&a) * q_scale;
    }
}

/// One step of the gated DeltaNet recurrence over rows = v.len() rows of its fp16 state, using a native kernel: NEON or
/// AVX2, four rows at a time, both giving the same bits. Returns false when no native kernel fits (no AVX2/FMA/F16C,
/// dim not a multiple of 16 or over 256, or rows not a multiple of 4), and the caller then falls back to the portable
/// version.
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn delta_rule_h(
    state: &mut [u16],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    beta: f32,
    q_scale: f32,
    o: &mut [f32],
    dim: usize,
) -> bool {
    if !dim.is_multiple_of(16) || dim > 256 || !v.len().is_multiple_of(4) {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::delta_rule_h(state, q, k, v, alpha, beta, q_scale, o, dim);
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        // SAFETY: the if above checked the features
        unsafe { x86::delta_rule_h(state, q, k, v, alpha, beta, q_scale, o, dim) };
        return true;
    }
    #[allow(unreachable_code)]
    false
}
