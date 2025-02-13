use crate::vector::Vector;
use rayon::prelude::*;

/// A scalar quantizer that maps floating-point values to a set of discrete values (levels).
///
/// The quantizer is configured with a minimum and maximum value and a specified number of levels.
/// Each input value is clamped between `min` and `max`, then quantized uniformly into one of the levels.
/// The resulting quantized value is represented as a `u8`. For large input vectors, parallel processing
/// is used to speed up quantization.
///
/// # Example
/// ```
/// # use vq::vector::Vector;
/// # use vq::sq::ScalarQuantizer;
/// let quantizer = ScalarQuantizer::new(0.0, 1.0, 256);
/// let input = Vector::new(vec![0.1, 0.5, 0.9]);
/// let output = quantizer.quantize(&input);
/// // output now contains quantized values for each input element.
/// ```
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
    /// Panics if `max` is not greater than `min`, or if `levels` is not within the valid range.
    pub fn new(min: f32, max: f32, levels: usize) -> Self {
        assert!(max > min, "max must be greater than min");
        assert!(levels >= 2, "levels must be at least 2");
        assert!(levels <= 256, "levels must be no more than 256");
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
    /// Each element is first clamped to the `[min, max]` range and then mapped to the nearest
    /// quantization level using uniform quantization. If the input vector's length exceeds
    /// `PARALLEL_THRESHOLD`, parallel iteration is used to improve performance.
    ///
    /// # Parameters
    /// - `vector`: A reference to the input vector (`Vector<f32>`) to quantize.
    ///
    /// # Returns
    /// A new vector (`Vector<u8>`) containing the quantized values.
    ///
    /// # Example
    /// ```
    /// # use vq::vector::Vector;
    /// # use vq::sq::ScalarQuantizer;
    /// let quantizer = ScalarQuantizer::new(0.0, 1.0, 256);
    /// let input = Vector::new(vec![0.0, 0.5, 1.0]);
    /// let output = quantizer.quantize(&input);
    /// // output is a Vector<u8> with quantized values.
    /// ```
    pub fn quantize(&self, vector: &Vector<f32>) -> Vector<u8> {
        // Define a threshold to decide when to use parallel processing.
        const PARALLEL_THRESHOLD: usize = 1024;

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
