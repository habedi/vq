use crate::core::error::{VqError, VqResult};

#[cfg(feature = "simd")]
use crate::core::hsdlib_ffi;

/// Supported distance metrics for vector comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    /// Squared Euclidean distance (L2²). Efficient for comparisons as it avoids square roots.
    SquaredEuclidean,
    /// Euclidean distance (L2).
    Euclidean,
    /// Manhattan distance (L1). Sum of absolute differences.
    Manhattan,
    /// Cosine distance, defined as `1.0 - cosine_similarity`.
    CosineDistance,
}

impl Distance {
    /// Returns the name of this distance metric.
    pub const fn name(&self) -> &'static str {
        match self {
            Distance::SquaredEuclidean => "squared_euclidean",
            Distance::Euclidean => "euclidean",
            Distance::Manhattan => "manhattan",
            Distance::CosineDistance => "cosine",
        }
    }

    /// Computes the distance between two vectors using the specified metric.
    ///
    /// If the `simd` feature is enabled, this method will use SIMD-accelerated
    /// implementations when available (AVX/AVX2/FMA for x86, NEON for ARM).
    ///
    /// # Arguments
    ///
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    ///
    /// The computed distance as an `f32`.
    ///
    /// # Errors
    ///
    /// Returns `VqError::DimensionMismatch` if the vectors have different lengths.
    #[inline]
    pub fn compute(&self, a: &[f32], b: &[f32]) -> VqResult<f32> {
        if a.len() != b.len() {
            return Err(VqError::DimensionMismatch {
                expected: a.len(),
                found: b.len(),
            });
        }

        let result = match self {
            Distance::SquaredEuclidean => compute_squared_euclidean(a, b),
            Distance::Euclidean => compute_squared_euclidean(a, b).sqrt(),
            Distance::Manhattan => compute_manhattan(a, b),
            Distance::CosineDistance => compute_cosine_distance(a, b),
        };

        Ok(result)
    }
}

#[inline]
fn compute_squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        if let Some(result) = hsdlib_ffi::sqeuclidean_f32(a, b) {
            return result;
        }
    }
    // Fallback to scalar implementation
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

#[inline]
fn compute_manhattan(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        if let Some(result) = hsdlib_ffi::manhattan_f32(a, b) {
            return result;
        }
    }
    // Fallback to scalar implementation
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
}

#[inline]
fn compute_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        // hsdlib returns cosine similarity, we need cosine distance (1 - similarity)
        if let Some(similarity) = hsdlib_ffi::cosine_f32(a, b) {
            return 1.0 - similarity;
        }
    }
    // Fallback to scalar implementation
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

    // Use epsilon to handle near-zero norms (avoids division by very small numbers)
    const EPSILON: f32 = 1e-10;
    if norm_a < EPSILON || norm_b < EPSILON {
        // Zero or near-zero vectors are considered maximally distant
        1.0
    } else {
        // Cosine distance ranges over [0, 2]; clamp to absorb floating-point errors
        (1.0 - (dot / (norm_a * norm_b))).clamp(0.0, 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_squared_euclidean() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 6.0, 8.0];
        let result = Distance::SquaredEuclidean.compute(&a, &b).unwrap();
        assert!(approx_eq(result, 50.0, 1e-6));
    }

    #[test]
    fn test_euclidean() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 6.0, 8.0];
        let expected = 50.0f32.sqrt();
        let result = Distance::Euclidean.compute(&a, &b).unwrap();
        assert!(approx_eq(result, expected, 1e-6));
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let result = Distance::CosineDistance.compute(&a, &b).unwrap();
        assert!(approx_eq(result, 1.0, 1e-6));

        let a = vec![1.0f32, 1.0];
        let b = vec![1.0f32, 1.0];
        let result = Distance::CosineDistance.compute(&a, &b).unwrap();
        assert!(approx_eq(result, 0.0, 1e-6));
    }

    #[test]
    fn test_manhattan() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 6.0, 8.0];
        let result = Distance::Manhattan.compute(&a, &b).unwrap();
        assert!(approx_eq(result, 12.0, 1e-6));
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32];
        let result = Distance::Euclidean.compute(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_consistency() {
        use crate::core::hsdlib_ffi;

        let mut rng = rand::rng();
        use rand::RngExt;

        let len = 100;
        let a: Vec<f32> = (0..len).map(|_| rng.random::<f32>()).collect();
        let b: Vec<f32> = (0..len).map(|_| rng.random::<f32>()).collect();

        // Check L2 Squared
        let scalar_l2sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        let simd_l2sq = hsdlib_ffi::sqeuclidean_f32(&a, &b).unwrap();
        assert!(
            (scalar_l2sq - simd_l2sq).abs() < 1e-4,
            "L2 Squared mismatch: scalar={}, simd={}",
            scalar_l2sq,
            simd_l2sq
        );

        // Check Manhattan
        let scalar_l1: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        let simd_l1 = hsdlib_ffi::manhattan_f32(&a, &b).unwrap();
        assert!(
            (scalar_l1 - simd_l1).abs() < 1e-4,
            "Manhattan mismatch: scalar={}, simd={}",
            scalar_l1,
            simd_l1
        );

        // Check Cosine Similarity checking
        // Note: Distance::CosineDistance computes 1.0 - similarity
        // hsdlib returns similarity directly
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scalar_cos_sim = dot / (norm_a * norm_b);

        let simd_cos_sim = hsdlib_ffi::cosine_f32(&a, &b).unwrap();
        assert!(
            (scalar_cos_sim - simd_cos_sim).abs() < 1e-4,
            "Cosine mismatch: scalar={}, simd={}",
            scalar_cos_sim,
            simd_cos_sim
        );
    }
}
