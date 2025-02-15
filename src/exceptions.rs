//! # Custom Errors Types
//!
//! This module defines custom error types for the `Vq` library. Use these errors to signal
//! issues like dimension mismatches, empty inputs, invalid parameters, or invalid metric parameters.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VqError {
    /// Indicates that two vectors do not have matching dimensions.
    #[error("Dimension mismatch: expected {expected}, got {found}")]
    DimensionMismatch { expected: usize, found: usize },

    /// Indicates that an operation was attempted on an empty input.
    #[error("Empty input: at least one vector is required.")]
    EmptyInput,

    /// Indicates that an invalid parameter was provided.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Indicates that a metric-specific parameter is invalid.
    #[error("Invalid metric parameter for {metric}: {details}")]
    InvalidMetricParameter { metric: String, details: String },
}

/// A convenience result type for operations in the `Vq` library.
pub type VqResult<T> = std::result::Result<T, VqError>;
