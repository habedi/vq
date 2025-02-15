//! # Scalar Quantizer Implementation
//!
//! This module provides a scalar quantizer that maps floating-point values to a set of discrete values (or levels).
//! The quantizer is configured with a minimum and maximum value and a specified number of levels.
//! Each input value is first clamped to the `[min, max]` range and then uniformly quantized into one of the levels.
//! The quantized result is represented as a `u8`. For large input vectors, parallel processing is used
//! to improve performance.
//!
//! Custom error handling is integrated to validate parameters. For example, the `fit` method will panic
//! with a custom error if the parameters are invalid (e.g. `max` is not greater than `min`, or if the number of levels
//! is not between 2 and 256).
//!
//! # Example
//! ```
//! use vq::vector::Vector;
//! use vq::sq::ScalarQuantizer;
//!
//! let quantizer = ScalarQuantizer::fit(0.0, 1.0, 256);
//! let input = Vector::new(vec![0.0, 0.5, 1.0]);
//! let output = quantizer.quantize(&input);
//! // output is a Vector<u8> with quantized values.
//! ```

use crate::exceptions::VqError;
use crate::vector::{Vector, PARALLEL_THRESHOLD};
use rayon::prelude::*;

/// A scalar quantizer that maps floating-point values to a set of discrete levels (levels).
pub struct ScalarQuantizer {
    /// The minimum value in the quantizer range.
    pub min: f32,
    /// The maximum value in the quantizer range.
    pub max: f32,
    /// The number of quantization levels (must be at least 2 and no more than 256).
    pub levels: usize,
    /// The step size computed as `(max - min) / (levels - 1)`.
    pub step: f32,
}

impl ScalarQuantizer {
    /// Creates a new `ScalarQuantizer`.
    ///
    /// # Parameters
    /// - `min`: The minimum value in the quantizer's range.
    /// - `max`: The maximum value in the quantizer's range. Must be greater than `min`.
    /// - `levels`: The number of quantization levels. Must be between 2 and 256.
    ///
    /// # Panics
    /// Panics with a custom error if `max` is not greater than `min`, or if `levels` is not within the valid range.
    pub fn fit(min: f32, max: f32, levels: usize) -> Self {
        if max <= min {
            panic!(
                "{}",
                VqError::InvalidParameter("max must be greater than min".to_string())
            );
        }
        if levels < 2 {
            panic!(
                "{}",
                VqError::InvalidParameter("levels must be at least 2".to_string())
            );
        }
        if levels > 256 {
            panic!(
                "{}",
                VqError::InvalidParameter("levels must be no more than 256".to_string())
            );
        }
        let step = (max - min) / (levels - 1) as f32;
        Self {
            min,
            max,
            levels,
            step,
        }
    }

    /// Quantizes an input vector by mapping each element to one of the discrete levels.
    ///
    /// Each element is clamped to the `[min, max]` range and then mapped to the nearest
    /// quantization level using uniform quantization. If the input vector's length exceeds
    /// `PARALLEL_THRESHOLD`, parallel iteration is used to improve performance.
    ///
    /// # Parameters
    /// - `vector`: A reference to the input vector (`Vector<f32>`) to quantize.
    ///
    /// # Returns
    /// A new vector (`Vector<u8>`) containing the quantized values.
    pub fn quantize(&self, vector: &Vector<f32>) -> Vector<u8> {
        let quantized_vector: Vec<u8> = if vector.data.len() > PARALLEL_THRESHOLD {
            // Use parallel iteration for large vectors.
            vector
                .data
                .par_iter()
                .map(|&x| self.quantize_scalar(x) as u8)
                .collect()
        } else {
            // Otherwise, process sequentially.
            vector
                .data
                .iter()
                .map(|&x| self.quantize_scalar(x) as u8)
                .collect()
        };
        Vector::new(quantized_vector)
    }

    /// Quantizes a single scalar value.
    ///
    /// The value is clamped to the `[min, max]` range and then uniformly quantized using the step size.
    ///
    /// # Parameters
    /// - `x`: The scalar value to quantize.
    ///
    /// # Returns
    /// The index (as `usize`) corresponding to the quantized level.
    fn quantize_scalar(&self, x: f32) -> usize {
        let clamped = if x < self.min {
            self.min
        } else if x > self.max {
            self.max
        } else {
            x
        };
        let index = ((clamped - self.min) / self.step).round() as usize;
        index.min(self.levels - 1)
    }
}
