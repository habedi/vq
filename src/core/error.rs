use thiserror::Error;

/// Crate-specific error type for Vq operations.
#[derive(Debug, Error)]
pub enum VqError {
    /// Vectors have different dimensions where they were expected to match.
    #[error("Dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },

    /// Input data is empty when at least one element is required.
    #[error("Empty input: at least one vector is required")]
    EmptyInput,

    /// A configuration parameter is invalid (e.g., k=0, levels > 256).
    #[error("Invalid parameter '{parameter}': {reason}")]
    InvalidParameter {
        parameter: &'static str,
        reason: String,
    },

    /// Input data contains invalid values (NaN, Infinity, etc.).
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// FFI operation failed.
    #[error("FFI error: {0}")]
    FfiError(String),

    /// A quantizer could not be encoded to or decoded from bytes.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Reading or writing a model file failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A specialized Result type for Vq operations.
pub type VqResult<T> = std::result::Result<T, VqError>;
