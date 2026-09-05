//! Core types and traits for the vq library.
//!
//! This module contains:
//! - [`Quantizer`](quantizer::Quantizer) - The unified trait implemented by all quantizers
//! - [`Distance`](distance::Distance) - Distance metrics for vector comparisons
//! - [`VqError`](error::VqError) and [`VqResult`](error::VqResult) - Error handling types
//! - [`Vector`](vector::Vector) - Generic vector type used internally
//!
//! Every quantizer also provides `to_bytes`, `from_bytes`, `save`, and `load` methods
//! for persisting trained models.

pub mod distance;
pub mod error;
#[cfg(feature = "simd")]
pub mod hsdlib_ffi;
pub(crate) mod persist;
pub mod quantizer;
pub mod vector;
