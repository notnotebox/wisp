//! Native Rust/Candle runtime for Wisp v0.1.2 4B.
//!
//! The library loads the merged Qwen3.5-4B + Wisp weights and scores the offered
//! options from next-token logits. It never generates text.

#[cfg(feature = "inference")]
pub mod inference;
