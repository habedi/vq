//! Binary quantization (BQ) implementation.
//!
//! Binary quantization is the simplest form of quantization, mapping each value
//! to one of two discrete levels based on a threshold comparison.

use crate::core::error::{VqError, VqResult};
use crate::core::persist::impl_persist;
use crate::core::quantizer::Quantizer;
use serde::{Deserialize, Serialize};

/// Binary quantizer that maps values above/below a threshold to two discrete levels.
///
/// # Example
///
/// ```
/// use vq::BinaryQuantizer;
/// use vq::Quantizer;
///
/// let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
/// let quantized = bq.quantize(&[-1.0, 0.5, 1.0]).unwrap();
/// assert_eq!(quantized, vec![0, 1, 1]);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryQuantizer {
    threshold: f32,
    low: u8,
    high: u8,
}

impl BinaryQuantizer {
    /// Creates a new binary quantizer.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Values >= threshold map to `high`, values < threshold map to `low`
    /// * `low` - The output value for inputs below the threshold
    /// * `high` - The output value for inputs at or above the threshold
    ///
    /// # Example
    ///
    /// ```
    /// use vq::BinaryQuantizer;
    /// use vq::Quantizer;
    ///
    /// // Create a binary quantizer with threshold 0.5
    /// let bq = BinaryQuantizer::new(0.5, 0, 1).unwrap();
    ///
    /// // Values >= 0.5 map to 1, values < 0.5 map to 0
    /// let result = bq.quantize(&[0.3, 0.5, 0.7]).unwrap();
    /// assert_eq!(result, vec![0, 1, 1]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `low >= high`
    /// - `threshold` is NaN or infinite
    pub fn new(threshold: f32, low: u8, high: u8) -> VqResult<Self> {
        if !threshold.is_finite() {
            return Err(VqError::InvalidParameter {
                parameter: "threshold",
                reason: "must be finite (not NaN or infinite)".to_string(),
            });
        }
        if low >= high {
            return Err(VqError::InvalidParameter {
                parameter: "low/high",
                reason: "low must be less than high".to_string(),
            });
        }
        Ok(Self {
            threshold,
            low,
            high,
        })
    }

    fn validate(self) -> VqResult<Self> {
        Self::new(self.threshold, self.low, self.high)
    }

    /// Returns the threshold value.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Returns the low quantization level.
    pub fn low(&self) -> u8 {
        self.low
    }

    /// Returns the high quantization level.
    pub fn high(&self) -> u8 {
        self.high
    }
}

impl_persist!(BinaryQuantizer);

impl Quantizer for BinaryQuantizer {
    type QuantizedOutput = Vec<u8>;

    fn quantize(&self, vector: &[f32]) -> VqResult<Self::QuantizedOutput> {
        Ok(vector
            .iter()
            .map(|&x| {
                if x >= self.threshold {
                    self.high
                } else {
                    self.low
                }
            })
            .collect())
    }

    fn dequantize(&self, quantized: &Self::QuantizedOutput) -> VqResult<Vec<f32>> {
        Ok(quantized
            .iter()
            .map(|&x| {
                if x >= self.high {
                    self.high as f32
                } else {
                    self.low as f32
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
        let input = vec![-1.0, 0.0, 1.0, -0.5, 0.5];
        let result = bq.quantize(&input).unwrap();
        assert_eq!(result, vec![0, 1, 1, 0, 1]);
    }

    #[test]
    fn test_large_vector() {
        let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
        let input: Vec<f32> = (0..1024).map(|i| (i as f32) - 512.0).collect();
        let result = bq.quantize(&input).unwrap();
        assert_eq!(result.len(), 1024);

        for (i, &val) in result.iter().enumerate() {
            let expected = if input[i] >= 0.0 { 1 } else { 0 };
            assert_eq!(val, expected);
        }
    }

    #[test]
    fn test_invalid_levels() {
        let result = BinaryQuantizer::new(0.0, 1, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_getters() {
        let bq = BinaryQuantizer::new(0.5, 10, 20).unwrap();
        assert_eq!(bq.threshold(), 0.5);
        assert_eq!(bq.low(), 10);
        assert_eq!(bq.high(), 20);
    }

    #[test]
    fn test_invalid_parameters() {
        // low == high
        let result = BinaryQuantizer::new(0.0, 5, 5);
        assert!(result.is_err());

        // low > high
        let result = BinaryQuantizer::new(0.0, 6, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input() {
        let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
        let input: Vec<f32> = vec![];
        let result = bq.quantize(&input).unwrap();
        assert!(result.is_empty());

        // Dequantize empty
        let empty_codes: Vec<u8> = vec![];
        let reconstructed = bq.dequantize(&empty_codes).unwrap();
        assert!(reconstructed.is_empty());
    }

    #[test]
    fn test_nan_threshold_rejected() {
        let result = BinaryQuantizer::new(f32::NAN, 0, 1);
        assert!(result.is_err());
    }
}
