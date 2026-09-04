//! Tree-structured vector quantization (TSVQ) implementation.
//!
//! TSVQ builds a hierarchical binary tree of cluster centroids, allowing
//! efficient O(log k) quantization by traversing the tree to find the
//! nearest leaf node.

use crate::core::distance::Distance;
use crate::core::error::{VqError, VqResult};
use crate::core::quantizer::Quantizer;
use crate::core::vector::{Vector, mean_vector};
use half::f16;

struct TSVQNode {
    centroid: Vector<f32>,
    left: Option<Box<TSVQNode>>,
    right: Option<Box<TSVQNode>>,
}

impl TSVQNode {
    /// Build from vector references (avoids cloning)
    fn build_from_refs(training_data: &[&Vector<f32>], max_depth: usize) -> VqResult<Self> {
        if training_data.is_empty() {
            return Err(VqError::EmptyInput);
        }

        // Clone vectors for this call (only when actually needed)
        let owned_data: Vec<Vector<f32>> = training_data.iter().map(|&v| v.clone()).collect();
        Self::build(&owned_data, max_depth)
    }

    fn build(training_data: &[Vector<f32>], max_depth: usize) -> VqResult<Self> {
        if training_data.is_empty() {
            return Err(VqError::EmptyInput);
        }

        let centroid = mean_vector(training_data)?;

        if max_depth == 0 || training_data.len() <= 1 {
            return Ok(TSVQNode {
                centroid,
                left: None,
                right: None,
            });
        }

        let dim = centroid.len();
        let variances: Vec<f32> = (0..dim)
            .map(|i| {
                training_data
                    .iter()
                    .map(|v| {
                        let diff = v.data[i] - centroid.data[i];
                        diff * diff
                    })
                    .sum()
            })
            .collect();

        let split_dim = variances
            .iter()
            .enumerate()
            // Filter out NaN values, then find max
            .filter(|&(_, v)| !v.is_nan())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut values: Vec<f32> = training_data
            .iter()
            .map(|v| v.data[split_dim])
            .filter(|&x| !x.is_nan()) // Filter out NaN values before sorting
            .collect();

        // Every value on the split dimension was NaN, so there is nothing to split on
        if values.is_empty() {
            return Ok(TSVQNode {
                centroid,
                left: None,
                right: None,
            });
        }

        // Use total_cmp for stable sorting even with infinities
        values.sort_by(|a, b| a.total_cmp(b));

        let median = if values.len() % 2 == 0 {
            (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
        } else {
            values[values.len() / 2]
        };

        // Partition using indices to avoid cloning vectors
        let (left_indices, right_indices): (Vec<_>, Vec<_>) =
            (0..training_data.len()).partition(|&i| training_data[i].data[split_dim] <= median);

        // Build child nodes using vector references
        let left = if !left_indices.is_empty() && left_indices.len() < training_data.len() {
            let left_data: Vec<&Vector<f32>> =
                left_indices.iter().map(|&i| &training_data[i]).collect();
            Some(Box::new(TSVQNode::build_from_refs(
                &left_data,
                max_depth - 1,
            )?))
        } else {
            None
        };

        let right = if !right_indices.is_empty() && right_indices.len() < training_data.len() {
            let right_data: Vec<&Vector<f32>> =
                right_indices.iter().map(|&i| &training_data[i]).collect();
            Some(Box::new(TSVQNode::build_from_refs(
                &right_data,
                max_depth - 1,
            )?))
        } else {
            None
        };

        Ok(TSVQNode {
            centroid,
            left,
            right,
        })
    }

    fn find_leaf<'a>(&'a self, vector: &[f32], distance: &Distance) -> VqResult<&'a TSVQNode> {
        match (&self.left, &self.right) {
            (Some(left), Some(right)) => {
                let dist_left = distance.compute(vector, &left.centroid.data)?;
                let dist_right = distance.compute(vector, &right.centroid.data)?;
                if dist_left <= dist_right {
                    left.find_leaf(vector, distance)
                } else {
                    right.find_leaf(vector, distance)
                }
            }
            (Some(left), None) => left.find_leaf(vector, distance),
            (None, Some(right)) => right.find_leaf(vector, distance),
            (None, None) => Ok(self),
        }
    }
}

/// Tree-structured vector quantizer using hierarchical clustering.
///
/// TSVQ builds a binary tree where each node represents a cluster centroid.
/// Vectors are quantized by traversing the tree to find the nearest leaf node.
///
/// # Example
///
/// ```
/// use vq::{TSVQ, Distance, Quantizer};
///
/// // Training data: 50 vectors of dimension 6
/// let training: Vec<Vec<f32>> = (0..50)
///     .map(|i| (0..6).map(|j| ((i + j) % 30) as f32).collect())
///     .collect();
/// let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
///
/// // Create TSVQ with max depth 4
/// let tsvq = TSVQ::new(&refs, 4, Distance::Euclidean).unwrap();
///
/// // Quantize a vector
/// let quantized = tsvq.quantize(&training[0]).unwrap();
/// assert_eq!(quantized.len(), 6);
/// ```
pub struct TSVQ {
    root: TSVQNode,
    dim: usize,
    distance: Distance,
}

impl TSVQ {
    /// Creates a new tree-structured vector quantizer.
    ///
    /// # Arguments
    ///
    /// * `training_data` - Training vectors for building the tree
    /// * `max_depth` - Maximum depth of the tree
    /// * `distance` - Distance metric to use
    ///
    /// # Example
    ///
    /// ```
    /// use vq::{TSVQ, Distance, Quantizer};
    ///
    /// let data: Vec<Vec<f32>> = (0..100)
    ///     .map(|i| (0..8).map(|j| ((i * j) % 100) as f32).collect())
    ///     .collect();
    /// let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
    ///
    /// let tsvq = TSVQ::new(&refs, 5, Distance::SquaredEuclidean).unwrap();
    /// assert_eq!(tsvq.dim(), 8);
    ///
    /// // Quantize and dequantize
    /// let q = tsvq.quantize(&data[0]).unwrap();
    /// let reconstructed = tsvq.dequantize(&q).unwrap();
    /// assert_eq!(reconstructed.len(), 8);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if training data is empty.
    pub fn new(training_data: &[&[f32]], max_depth: usize, distance: Distance) -> VqResult<Self> {
        if training_data.is_empty() {
            return Err(VqError::EmptyInput);
        }

        let dim = training_data[0].len();

        // Validate all training vectors have the same dimension
        for vec_slice in training_data.iter() {
            if vec_slice.len() != dim {
                return Err(VqError::DimensionMismatch {
                    expected: dim,
                    found: vec_slice.len(),
                });
            }
        }

        let vectors: Vec<Vector<f32>> = training_data
            .iter()
            .map(|&slice| Vector::new(slice.to_vec()))
            .collect();

        let root = TSVQNode::build(&vectors, max_depth)?;
        Ok(TSVQ {
            root,
            dim,
            distance,
        })
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

impl Quantizer for TSVQ {
    type QuantizedOutput = Vec<f16>;

    fn quantize(&self, vector: &[f32]) -> VqResult<Self::QuantizedOutput> {
        if vector.len() != self.dim {
            return Err(VqError::DimensionMismatch {
                expected: self.dim,
                found: vector.len(),
            });
        }

        let leaf = self.root.find_leaf(vector, &self.distance)?;
        let result = leaf
            .centroid
            .data
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
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

    #[test]
    fn test_identical_vectors() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data: Vec<&[f32]> = (0..10).map(|_| vec.as_slice()).collect();

        let tsvq = TSVQ::new(&data, 3, Distance::SquaredEuclidean).unwrap();
        let quantized = tsvq.quantize(&vec).unwrap();

        assert_eq!(quantized.len(), vec.len());
        for (orig, q) in vec.iter().zip(quantized.iter()) {
            assert!((orig - f16::to_f32(*q)).abs() < 1e-2);
        }
    }

    #[test]
    fn test_random_vectors() {
        let data: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..10).map(|j| ((i + j) % 50) as f32).collect())
            .collect();
        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

        let tsvq = TSVQ::new(&data_refs, 3, Distance::SquaredEuclidean).unwrap();

        for vec in &data {
            let quantized = tsvq.quantize(vec).unwrap();
            assert_eq!(quantized.len(), vec.len());
        }
    }

    #[test]
    fn test_empty_training() {
        let data: Vec<&[f32]> = vec![];
        let result = TSVQ::new(&data, 3, Distance::Euclidean);
        assert!(result.is_err());
    }
}
