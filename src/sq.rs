//! Scalar quantization (SQ) implementation.
//!
//! Scalar quantization uniformly divides a value range into discrete levels,
//! mapping each input value to its nearest quantization level.

use crate::core::error::{VqError, VqResult};
use crate::core::persist::impl_persist;
use crate::core::quantizer::Quantizer;
use serde::{Deserialize, Serialize};

/// Scalar quantizer that uniformly quantizes values in a range to discrete levels.
///
/// # Example
///
/// ```
/// use vq::ScalarQuantizer;
/// use vq::Quantizer;
///
/// let sq = ScalarQuantizer::new(0.0, 1.0, 11).unwrap(); // 0.0, 0.1, ..., 1.0
/// let quantized = sq.quantize(&[0.0, 0.5, 1.0]).unwrap();
/// assert_eq!(quantized, vec![0, 5, 10]);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    min: f32,
    max: f32,
    levels: usize,
    step: f32,
}

impl ScalarQuantizer {
    /// Creates a new scalar quantizer.
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum value in the quantization range
    /// * `max` - Maximum value in the quantization range
    /// * `levels` - Number of quantization levels (2-256)
    ///
    /// # Example
    ///
    /// ```
    /// use vq::ScalarQuantizer;
    /// use vq::Quantizer;
    ///
    /// // Create a quantizer for the range [-1, 1] with 256 levels
    /// let sq = ScalarQuantizer::new(-1.0, 1.0, 256).unwrap();
    ///
    /// // Quantize and reconstruct
    /// let input = vec![0.0, 0.5, -0.5];
    /// let quantized = sq.quantize(&input).unwrap();
    /// let reconstructed = sq.dequantize(&quantized).unwrap();
    ///
    /// // Reconstruction error is bounded
    /// for (orig, recon) in input.iter().zip(reconstructed.iter()) {
    ///     assert!((orig - recon).abs() < 0.01);
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `min` or `max` is NaN or Infinity
    /// - `max <= min`
    /// - `levels < 2` or `levels > 256`
    pub fn new(min: f32, max: f32, levels: usize) -> VqResult<Self> {
        if !min.is_finite() {
            return Err(VqError::InvalidParameter {
                parameter: "min",
                reason: "must be finite (not NaN or infinite)".to_string(),
            });
        }
        if !max.is_finite() {
            return Err(VqError::InvalidParameter {
                parameter: "max",
                reason: "must be finite (not NaN or infinite)".to_string(),
            });
        }
        if max <= min {
            return Err(VqError::InvalidParameter {
                parameter: "max",
                reason: "must be greater than min".to_string(),
            });
        }
        if levels < 2 {
            return Err(VqError::InvalidParameter {
                parameter: "levels",
                reason: "must be at least 2".to_string(),
            });
        }
        if levels > 256 {
            return Err(VqError::InvalidParameter {
                parameter: "levels",
                reason: "must be no more than 256 to fit in u8".to_string(),
            });
        }
        let step = (max - min) / (levels - 1) as f32;
        Ok(Self {
            min,
            max,
            levels,
            step,
        })
    }

    fn validate(self) -> VqResult<Self> {
        // Rebuild from the parameters so the step is recomputed and range checks rerun
        Self::new(self.min, self.max, self.levels)
    }

    /// Returns the minimum value in the quantization range.
    pub fn min(&self) -> f32 {
        self.min
    }

    /// Returns the maximum value in the quantization range.
    pub fn max(&self) -> f32 {
        self.max
    }

    /// Returns the number of quantization levels.
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Returns the step size between quantization levels.
    pub fn step(&self) -> f32 {
        self.step
    }

    fn quantize_scalar(&self, x: f32) -> usize {
        let clamped = x.clamp(self.min, self.max);
        let index = ((clamped - self.min) / self.step).round() as usize;
        index.min(self.levels - 1)
    }
}

impl_persist!(ScalarQuantizer);

impl Quantizer for ScalarQuantizer {
    type QuantizedOutput = Vec<u8>;

    fn quantize(&self, vector: &[f32]) -> VqResult<Self::QuantizedOutput> {
        // Safety assertion: levels was validated to be <= 256 in constructor
        debug_assert!(self.levels <= 256, "levels must be <= 256 to fit in u8");
        Ok(vector
            .iter()
            .map(|&x| {
                let idx = self.quantize_scalar(x);
                debug_assert!(idx < 256, "quantize_scalar returned index >= 256");
                idx as u8
            })
            .collect())
    }

    fn dequantize(&self, quantized: &Self::QuantizedOutput) -> VqResult<Vec<f32>> {
        Ok(quantized
            .iter()
            .map(|&idx| self.min + idx as f32 * self.step)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_on_scalars() {
        let sq = ScalarQuantizer::new(-1.0, 1.0, 5).unwrap();
        let test_values = vec![-1.2, -1.0, -0.8, -0.3, 0.0, 0.3, 0.6, 1.0, 1.2];
        for x in test_values {
            let indices = sq.quantize(&[x]).unwrap();
            assert_eq!(indices.len(), 1);
            let reconstructed = sq.min() + indices[0] as f32 * sq.step();
            let clamped = x.clamp(sq.min(), sq.max());
            let error = (reconstructed - clamped).abs();
            assert!(error <= sq.step() / 2.0 + 1e-6);
        }
    }

    #[test]
    fn test_large_vectors() {
        let sq = ScalarQuantizer::new(-1000.0, 1000.0, 256).unwrap();
        let input: Vec<f32> = (0..1024).map(|i| (i as f32) - 512.0).collect();
        let result = sq.quantize(&input).unwrap();
        assert_eq!(result.len(), 1024);
    }

    #[test]
    fn test_getters() {
        let sq = ScalarQuantizer::new(-1.0, 1.0, 5).unwrap();
        assert_eq!(sq.min(), -1.0);
        assert_eq!(sq.max(), 1.0);
        assert_eq!(sq.levels(), 5);
        assert_eq!(sq.step(), 0.5);
    }

    #[test]
    fn test_dequantize_matches_level_grid() {
        let sq = ScalarQuantizer::new(-1.0, 1.0, 5).unwrap();
        let recon = sq.dequantize(&vec![0, 1, 2, 3, 4]).unwrap();
        assert_eq!(recon, vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_invalid_range() {
        let result = ScalarQuantizer::new(1.0, -1.0, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_too_few_levels() {
        let result = ScalarQuantizer::new(-1.0, 1.0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_nan_min_rejected() {
        let result = ScalarQuantizer::new(f32::NAN, 1.0, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_nan_max_rejected() {
        let result = ScalarQuantizer::new(-1.0, f32::NAN, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_infinity_rejected() {
        let result = ScalarQuantizer::new(f32::NEG_INFINITY, 1.0, 256);
        assert!(result.is_err());

        let result = ScalarQuantizer::new(-1.0, f32::INFINITY, 256);
        assert!(result.is_err());
    }
}
