//! Common traits for vector quantization algorithms.

use crate::core::error::VqResult;

/// A trait representing a vector quantizer.
///
/// All quantization algorithms implement this trait, providing a uniform
/// interface for encoding vectors into compact representations and
/// reconstructing approximate vectors from those representations.
///
/// # Type Parameters
///
/// * `QuantizedOutput` - The type of the quantized representation (e.g., `Vec<u8>`, `Vec<f16>`)
///
/// # Example
///
/// ```rust
/// use vq::{Quantizer, VqResult};
/// use vq::sq::ScalarQuantizer;
///
/// fn quantize_and_reconstruct<Q: Quantizer>(
///     quantizer: &Q,
///     vector: &[f32],
/// ) -> VqResult<Vec<f32>> {
///     let quantized = quantizer.quantize(vector)?;
///     quantizer.dequantize(&quantized)
/// }
/// ```
pub trait Quantizer {
    /// The output type of the quantization process.
    type QuantizedOutput;

    /// Quantizes a vector into a compact representation.
    ///
    /// # Arguments
    ///
    /// * `vector` - The input vector to quantize
    ///
    /// # Returns
    ///
    /// The quantized representation of the input vector
    ///
    /// # Errors
    ///
    /// Returns an error if the input vector has an invalid dimension or
    /// other algorithm-specific validation fails.
    fn quantize(&self, vector: &[f32]) -> VqResult<Self::QuantizedOutput>;

    /// Reconstructs an approximate vector from its quantized representation.
    ///
    /// # Arguments
    ///
    /// * `quantized` - The quantized representation to decode
    ///
    /// # Returns
    ///
    /// An approximate reconstruction of the original vector
    ///
    /// # Errors
    ///
    /// Returns an error if the quantized representation is invalid.
    fn dequantize(&self, quantized: &Self::QuantizedOutput) -> VqResult<Vec<f32>>;

    /// Quantizes several vectors at once.
    ///
    /// The default implementation calls [`quantize`](Self::quantize) on each vector.
    /// With the `parallel` feature, the vectors are processed concurrently.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by [`quantize`](Self::quantize).
    ///
    /// # Example
    ///
    /// ```rust
    /// use vq::{Quantizer, ScalarQuantizer};
    ///
    /// let sq = ScalarQuantizer::new(0.0, 1.0, 11).unwrap();
    /// let batch: Vec<&[f32]> = vec![&[0.0, 0.5], &[1.0, 0.2]];
    /// let codes = sq.quantize_batch(&batch).unwrap();
    /// assert_eq!(codes, vec![vec![0, 5], vec![10, 2]]);
    /// ```
    fn quantize_batch(&self, vectors: &[&[f32]]) -> VqResult<Vec<Self::QuantizedOutput>>
    where
        Self: Sync,
        Self::QuantizedOutput: Send,
    {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            vectors.par_iter().map(|v| self.quantize(v)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            vectors.iter().map(|v| self.quantize(v)).collect()
        }
    }

    /// Reconstructs several vectors at once.
    ///
    /// The default implementation calls [`dequantize`](Self::dequantize) on each item.
    /// With the `parallel` feature, the items are processed concurrently.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by [`dequantize`](Self::dequantize).
    fn dequantize_batch(&self, quantized: &[Self::QuantizedOutput]) -> VqResult<Vec<Vec<f32>>>
    where
        Self: Sync,
        Self::QuantizedOutput: Sync,
    {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            quantized.par_iter().map(|q| self.dequantize(q)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            quantized.iter().map(|q| self.dequantize(q)).collect()
        }
    }
}
