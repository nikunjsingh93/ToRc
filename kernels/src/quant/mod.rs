//! quant - the block formats (ggml's, MIT) and the code that reads them independently of any platform: the scalar
//! reference dequantisation, the Q8_K activations and IQ3_S's grid.

pub mod dequant;
pub mod f16;
pub mod iq2_s;
pub mod iq2s_grid;
pub mod iq3_s;
pub mod iq3_xxs;
pub mod iq3s_grid;
pub mod iq3xxs_grid;
pub mod iq4;
pub mod mxfp4;
pub mod q2_0;
pub mod q3_k;
pub mod q4_0;
pub mod q4_k;
pub mod q8_k;
pub mod q8k;

pub use f16::f16_to_f32;
pub use q8_k::{BlockQ8K, Blocks, Groups32, Q8, Q8m, Q8s, QK};
pub mod lay;
