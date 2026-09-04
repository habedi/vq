//! Product quantization (PQ) implementation.
//!
//! Product quantization splits vectors into subspaces and quantizes each
//! independently using learned codebooks, enabling efficient compression
//! and approximate nearest neighbor search.

use crate::core::distance::Distance;
use crate::core::error::{VqError, VqResult};
use crate::core::persist::impl_persist;
use crate::core::quantizer::Quantizer;
use crate::core::vector::{Vector, lbg_quantize};
use half::f16;
use serde::{Deserialize, Serialize};

/// Product quantizer that divides vectors into subspaces and quantizes each separately.
///
/// Product quantization (PQ) is a technique that splits high-dimensional vectors
/// into smaller subspaces and quantizes each subspace independently using learned
/// codebooks. This allows for efficient storage and fast approximate nearest
/// neighbor search.
///
/// # Example
///
/// ```
/// use vq::ProductQuantizer;
/// use vq::{Quantizer, Distance};
///
/// // Training data: 100 vectors of dimension 8
/// let training: Vec<Vec<f32>> = (0..100)
///     .map(|i| (0..8).map(|j| ((i + j) % 50) as f32).collect())
///     .collect();
/// let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
///
/// // Create PQ with 2 subspaces, 4 centroids each
/// let pq = ProductQuantizer::new(&training_refs, 2, 4, 10, Distance::Euclidean, 42).unwrap();
///
/// // Quantize a vector
/// let quantized = pq.quantize(&training[0]).unwrap();
/// assert_eq!(quantized.len(), 8);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductQuantizer {
    codebooks: Vec<Vec<Vector<f32>>>,
    sub_dim: usize,
    m: usize,
    dim: usize,
    distance: Distance,
}

impl ProductQuantizer {
    /// Creates a new product quantizer.
    ///
    /// # Arguments
    ///
    /// * `training_data` - Training vectors for learning codebooks
    /// * `m` - Number of subspaces to divide vectors into
    /// * `k` - Number of centroids per subspace
    /// * `max_iters` - Maximum iterations for codebook training
    /// * `distance` - Distance metric to use
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Example
    ///
    /// ```
    /// use vq::{ProductQuantizer, Distance, Quantizer};
    ///
    /// let training: Vec<Vec<f32>> = (0..50)
    ///     .map(|i| (0..12).map(|j| (i * j) as f32).collect())
    ///     .collect();
    /// let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
    ///
    /// // 3 subspaces (dim 12 / 3 = 4 each), 8 centroids per subspace
    /// let pq = ProductQuantizer::new(&refs, 3, 8, 20, Distance::Euclidean, 0).unwrap();
    ///
    /// assert_eq!(pq.num_subspaces(), 3);
    /// assert_eq!(pq.sub_dim(), 4);
    /// assert_eq!(pq.dim(), 12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Training data is empty
    /// - Data dimension is less than `m`
    /// - Data dimension is not divisible by `m`
    pub fn new(
        training_data: &[&[f32]],
        m: usize,
        k: usize,
        max_iters: usize,
        distance: Distance,
        seed: u64,
    ) -> VqResult<Self> {
        if training_data.is_empty() {
            return Err(VqError::EmptyInput);
        }
        let dim = training_data[0].len();

        // Validate all training vectors have the same dimension
        for vec in training_data.iter() {
            if vec.len() != dim {
                return Err(VqError::DimensionMismatch {
                    expected: dim,
                    found: vec.len(),
                });
            }
        }

        if m == 0 {
            return Err(VqError::InvalidParameter {
                parameter: "m",
                reason: "must be greater than 0".to_string(),
            });
        }
        if dim < m {
            return Err(VqError::InvalidParameter {
                parameter: "m",
                reason: format!("must be at most the data dimension ({})", dim),
            });
        }
        if dim % m != 0 {
            return Err(VqError::InvalidParameter {
                parameter: "m",
                reason: format!("dimension ({}) must be divisible by m", dim),
            });
        }
        let sub_dim = dim / m;

        let mut codebooks = Vec::with_capacity(m);
        for i in 0..m {
            let sub_training: Vec<Vector<f32>> = training_data
                .iter()
                .map(|v| {
                    let start = i * sub_dim;
                    let end = start + sub_dim;
                    Vector::new(v[start..end].to_vec())
                })
                .collect();
            let codebook = lbg_quantize(&sub_training, k, max_iters, seed.wrapping_add(i as u64))?;
            codebooks.push(codebook);
        }

        Ok(Self {
            codebooks,
            sub_dim,
            m,
            dim,
            distance,
        })
    }

    fn validate(self) -> VqResult<Self> {
        let invalid = |reason: &str| VqError::Serialization(reason.to_string());
        if self.m == 0 || self.sub_dim == 0 || self.m * self.sub_dim != self.dim {
            return Err(invalid("subspace layout does not match the dimension"));
        }
        if self.codebooks.len() != self.m {
            return Err(invalid(
                "number of codebooks does not match the subspace count",
            ));
        }
        for codebook in &self.codebooks {
            if codebook.is_empty() {
                return Err(invalid("codebook has no centroids"));
            }
            if codebook.iter().any(|c| c.len() != self.sub_dim) {
                return Err(invalid(
                    "centroid dimension does not match the subspace dimension",
                ));
            }
        }
        Ok(self)
    }

    /// Trains a product quantizer and quantizes the training data in one call.
    ///
    /// Returns the quantizer together with one code per training vector, in the
    /// same order. This is equivalent to [`new`](Self::new) followed by
    /// [`quantize_batch`](Quantizer::quantize_batch).
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`new`](Self::new).
    pub fn fit_transform(
        training_data: &[&[f32]],
        m: usize,
        k: usize,
        max_iters: usize,
        distance: Distance,
        seed: u64,
    ) -> VqResult<(Self, Vec<Vec<f16>>)> {
        let pq = Self::new(training_data, m, k, max_iters, distance, seed)?;
        let codes = pq.quantize_batch(training_data)?;
        Ok((pq, codes))
    }

    /// Returns the number of subspaces.
    pub fn num_subspaces(&self) -> usize {
        self.m
    }

    /// Returns the dimension of each subspace.
    pub fn sub_dim(&self) -> usize {
        self.sub_dim
    }

    /// Returns the expected input vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the name of the distance metric used.
    pub fn distance_metric(&self) -> &'static str {
        self.distance.name()
    }
}

impl_persist!(ProductQuantizer);

impl Quantizer for ProductQuantizer {
    type QuantizedOutput = Vec<f16>;

    fn quantize(&self, vector: &[f32]) -> VqResult<Self::QuantizedOutput> {
        let n = vector.len();
        if n != self.dim {
            return Err(VqError::DimensionMismatch {
                expected: self.dim,
                found: n,
            });
        }

        let mut result = Vec::with_capacity(n);
        for i in 0..self.m {
            let start = i * self.sub_dim;
            let end = start + self.sub_dim;
            let sub_vector = &vector[start..end];
            let codebook = &self.codebooks[i];

            let mut best_idx = 0;
            let mut best_dist = self.distance.compute(sub_vector, &codebook[0].data)?;
            for (j, centroid) in codebook.iter().enumerate().skip(1) {
                let dist = self.distance.compute(sub_vector, &centroid.data)?;
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j;
                }
            }

            for &val in &codebook[best_idx].data {
                result.push(f16::from_f32(val));
            }
        }

        Ok(result)
    }

    fn dequantize(&self, quantized: &Self::QuantizedOutput) -> VqResult<Vec<f32>> {
        if quantized.len() != self.dim {
            return Err(VqError::DimensionMismatch {
                expected: self.dim,
                found: quantized.len(),
            });
        }
        Ok(quantized.iter().map(|&x| f16::to_f32(x)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..dim).map(|j| ((i + j) % 100) as f32).collect())
            .collect()
    }

    #[test]
    fn test_basic() {
        let data: Vec<Vec<f32>> = generate_test_data(100, 10);
        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

        let pq = ProductQuantizer::new(&data_refs, 2, 4, 10, Distance::Euclidean, 42).unwrap();

        let quantized = pq.quantize(&data[0]).unwrap();
        assert_eq!(quantized.len(), 10);
    }

    #[test]
    fn test_empty_training() {
        let data: Vec<&[f32]> = vec![];
        let result = ProductQuantizer::new(&data, 2, 2, 10, Distance::Euclidean, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_inconsistent_training_dimensions() {
        let data = [vec![1.0, 2.0], vec![1.0, 2.0, 3.0]];
        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let result = ProductQuantizer::new(&data_refs, 1, 1, 10, Distance::Euclidean, 42);
        assert!(matches!(
            result,
            Err(VqError::DimensionMismatch {
                expected: 2,
                found: 3
            })
        ));
    }

    #[test]
    fn test_too_many_centroids_for_training_size() {
        let data = generate_test_data(3, 4);
        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let result = ProductQuantizer::new(&data_refs, 2, 4, 10, Distance::Euclidean, 42);
        assert!(matches!(
            result,
            Err(VqError::InvalidParameter { parameter: "k", .. })
        ));
    }

    #[test]
    fn test_dequantize_dimension_mismatch() {
        let data = generate_test_data(20, 4);
        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let pq = ProductQuantizer::new(&data_refs, 2, 2, 10, Distance::Euclidean, 42).unwrap();
        let codes = vec![f16::from_f32(1.0); 3];
        assert!(matches!(
            pq.dequantize(&codes),
            Err(VqError::DimensionMismatch {
                expected: 4,
                found: 3
            })
        ));
    }

    #[test]
    fn test_getters_and_roundtrip() {
        let data = generate_test_data(20, 6);
        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let pq = ProductQuantizer::new(&data_refs, 3, 2, 10, Distance::Manhattan, 42).unwrap();
        assert_eq!(pq.num_subspaces(), 3);
        assert_eq!(pq.sub_dim(), 2);
        assert_eq!(pq.dim(), 6);
        assert_eq!(pq.distance_metric(), "manhattan");
        let codes = pq.quantize(&data[0]).unwrap();
        let recon = pq.dequantize(&codes).unwrap();
        assert_eq!(recon.len(), 6);
        assert!(recon.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_more_subspaces_than_dimensions() {
        let data = generate_test_data(10, 2);
        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let result = ProductQuantizer::new(&data_refs, 4, 2, 10, Distance::Euclidean, 42);
        assert!(matches!(
            result,
            Err(VqError::InvalidParameter { parameter: "m", .. })
        ));
    }

    #[test]
    fn test_fit_transform_matches_new_and_quantize() {
        let data = generate_test_data(30, 6);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let (pq, codes) =
            ProductQuantizer::fit_transform(&refs, 3, 4, 10, Distance::Euclidean, 9).unwrap();
        let direct = ProductQuantizer::new(&refs, 3, 4, 10, Distance::Euclidean, 9).unwrap();
        assert_eq!(codes.len(), data.len());
        for (v, c) in data.iter().zip(&codes) {
            assert_eq!(c, &direct.quantize(v).unwrap());
            assert_eq!(c, &pq.quantize(v).unwrap());
        }
        assert!(ProductQuantizer::fit_transform(&refs, 4, 4, 10, Distance::Euclidean, 9).is_err());
    }

    #[test]
    fn test_dimension_not_divisible() {
        let data = [vec![1.0, 2.0, 3.0]];
        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let result = ProductQuantizer::new(&data_refs, 2, 2, 10, Distance::Euclidean, 42);
        assert!(result.is_err());
    }
}
