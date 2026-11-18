//! qwen - the Qwen family and the relatives that share its forward pass (ERNIE 4.5, GLM-4.7-Flash, Moonlight, LFM2):
//! the hyperparameters (hparams.rs), the forward pass (forward.rs), Flash Next's hyper-connections (hc.rs), the MTP
//! blocks that draft for them (mtp.rs; mtp_hc.rs for Flash Next), and the chat side: the server half (serve.rs), which
//! serves the relatives too, each in its own chat format (models/ernie, glm, moonlight, lfm2), and the Qwen models' own
//! format (chat.rs). sample.rs picks each token, greedy or sampled.

pub mod chat;
pub mod forward;
pub mod hc;
pub mod hparams;
pub mod mtp;
pub mod mtp_hc;
pub mod sample;
pub mod serve;
pub mod vision;
