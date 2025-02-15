//! # Binary Quantizer Implementation
//!
//! This module provides a simple binary quantizer that maps floating-point values
//! to one of two discrete levels. It assigns each value in an input vector to either the
//! "high" level or the "low" level based on a specified threshold. For large input vectors,
//! parallel processing (via Rayon) is used to speed up quantization.
//!
//! The quantizer also includes basic parameter checking using custom errors from the
//! exceptions module.
//!
//! # Examples
//! ```
//! use vq::vector::Vector;
//! use vq::bq::BinaryQuantizer;
//!
//! let quantizer = BinaryQuantizer::fit(0.5, 0, 1);
//! let input = Vector::new(vec![0.3, 0.5, 0.8]);
//! let quantized = quantizer.quantize(&input);
//! // quantized now contains [0, 1, 1]
//! ```

use crate::exceptions::VqError;
use crate::vector::{Vector, PARALLEL_THRESHOLD};
use rayon::prelude::*;

/// A simple binary quantizer that maps floating-point values to one of two discrete values (levels).
pub struct BinaryQuantizer {
    /// The threshold value used to determine whether an element is quantized to `high` or `low`.
    pub threshold: f32,
    /// The quantized value assigned to inputs that are below the threshold.
    pub low: u8,
    /// The quantized value assigned to inputs that are at or above the threshold.
    pub high: u8,
}

impl BinaryQuantizer {
    /// Creates a new `BinaryQuantizer` with the specified threshold and quantization levels.
    ///
    /// # Parameters
    /// - `threshold`: The threshold value used for quantization.
    /// - `low`: The quantized value to assign for input values below the threshold.
    /// - `high`: The quantized value to assign for input values at or above the threshold.
    ///
    /// # Panics
    /// Panics with a custom error if `low` is not less than `high`.
    pub fn fit(threshold: f32, low: u8, high: u8) -> Self {
        if low >= high {
            panic!(
                "{}",
                VqError::InvalidParameter(
                    "Low quantization level must be less than high quantization level".to_string()
                )
            );
        }
        Self {
            threshold,
            low,
            high,
        }
    }

    /// Quantizes an input vector by mapping each element to either the low or high value based on the threshold.
    ///
    /// For each element in the input vector:
    /// - If the value is greater than or equal to `self.threshold`, it is mapped to `self.high`.
    /// - Otherwise, it is mapped to `self.low`.
    ///
    /// If the input vector's length exceeds `PARALLEL_THRESHOLD`, the mapping is performed in parallel.
    ///
    /// # Parameters
    /// - `vector`: A reference to the input vector (`Vector<f32>`) to be quantized.
    ///
    /// # Returns
    /// A new vector (`Vector<u8>`) containing the quantized values.
    pub fn quantize(&self, vector: &Vector<f32>) -> Vector<u8> {
        let quantized_vector = if vector.data.len() > PARALLEL_THRESHOLD {
            // Use parallel iteration when the vector is large.
            vector
                .data
                .par_iter()
                .map(|&x| {
                    if x >= self.threshold {
                        self.high
                    } else {
                        self.low
                    }
                })
                .collect()
        } else {
            // Otherwise, use sequential iteration.
            vector
                .data
                .iter()
                .map(|&x| {
                    if x >= self.threshold {
                        self.high
                    } else {
                        self.low
                    }
                })
                .collect()
        };
        Vector::new(quantized_vector)
    }
}
