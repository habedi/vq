//! # Distance Metrics
//!
//! This module defines the `Distance` enum for comparing vectors using different metrics.
//! Depending on input size, computations use Rayon for parallelism.
//!
//! # Panics
//! The `compute` method panics with a custom error if the input slices have different lengths
//! or if a metric-specific parameter is invalid.

use crate::exceptions::VqError;
use crate::vector::{Real, PARALLEL_THRESHOLD};
use rayon::prelude::*;

/// Sums mapped values over two slices using either parallel or sequential iterators.
#[inline]
fn zip_map_sum<T, F>(a: &[T], b: &[T], f: F) -> T
where
    T: Real + Send + Sync,
    F: Fn(T, T) -> T + Sync,
{
    if a.len() > PARALLEL_THRESHOLD {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| f(x, y))
            .reduce(|| T::zero(), |acc, val| acc + val)
    } else {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| f(x, y))
            .fold(T::zero(), |acc, val| acc + val)
    }
}

/// Reduces mapped values over two slices using the max operator.
#[inline]
fn zip_map_max<T, F>(a: &[T], b: &[T], f: F) -> T
where
    T: Real + Send + Sync,
    F: Fn(T, T) -> T + Sync,
{
    if a.len() > PARALLEL_THRESHOLD {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| f(x, y))
            .reduce(|| T::zero(), |acc, val| if val > acc { val } else { acc })
    } else {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| f(x, y))
            .fold(T::zero(), |acc, val| if val > acc { val } else { acc })
    }
}

/// Sums mapped values over a single slice.
#[inline]
fn map_sum<T, F>(a: &[T], f: F) -> T
where
    T: Real + Send + Sync,
    F: Fn(T) -> T + Sync,
{
    if a.len() > PARALLEL_THRESHOLD {
        a.par_iter()
            .map(|&x| f(x))
            .reduce(|| T::zero(), |acc, val| acc + val)
    } else {
        a.iter()
            .map(|&x| f(x))
            .fold(T::zero(), |acc, val| acc + val)
    }
}

/// Enum listing the available distance metrics.
pub enum Distance {
    /// Squared Euclidean distance (sum of squared differences).
    SquaredEuclidean,
    /// Euclidean distance (square root of the sum of squared differences).
    Euclidean,
    /// Cosine distance (1 minus the cosine similarity). If a vector has zero norm, returns 1.
    CosineDistance,
    /// Manhattan distance (sum of absolute differences).
    Manhattan,
    /// Chebyshev distance (maximum absolute difference).
    Chebyshev,
    /// Minkowski distance (generalized distance; `p` sets the norm order).
    Minkowski(f64),
    /// Hamming distance (count of positions where elements differ).
    Hamming,
}

impl Distance {
    /// Compute the distance between two slices `a` and `b` using the selected metric.
    ///
    /// # Type Parameters
    /// - `T`: A numeric type implementing `Real`, `Send`, and `Sync`.
    ///
    /// # Parameters
    /// - `a`: A slice representing the first vector.
    /// - `b`: A slice representing the second vector.
    ///
    /// # Returns
    /// The computed distance as a value of type `T`.
    ///
    /// # Panics
    /// Panics with a custom error if the lengths of `a` and `b` differ or if a metric-specific
    /// parameter is invalid.
    ///
    /// # Example
    /// ```
    /// use vq::distances::Distance;
    /// let a = vec![1.0, 2.0, 3.0];
    /// let b = vec![4.0, 5.0, 6.0];
    /// let d = Distance::Euclidean.compute(&a, &b);
    /// println!("Euclidean distance: {}", d);
    /// ```
    pub fn compute<T>(&self, a: &[T], b: &[T]) -> T
    where
        T: Real + Send + Sync,
    {
        if a.len() != b.len() {
            panic!(
                "{}",
                VqError::DimensionMismatch {
                    expected: a.len(),
                    found: b.len()
                }
            );
        }

        match self {
            Distance::SquaredEuclidean => zip_map_sum(a, b, |x, y| {
                let diff = x - y;
                diff * diff
            }),
            Distance::Euclidean => {
                let sum = zip_map_sum(a, b, |x, y| {
                    let diff = x - y;
                    diff * diff
                });
                sum.sqrt()
            }
            Distance::CosineDistance => {
                let dot = zip_map_sum(a, b, |x, y| x * y);
                let norm_a = map_sum(a, |x| x * x).sqrt();
                let norm_b = map_sum(b, |x| x * x).sqrt();

                if norm_a == T::zero() || norm_b == T::zero() {
                    T::one()
                } else {
                    T::one() - dot / (norm_a * norm_b)
                }
            }
            Distance::Manhattan => zip_map_sum(a, b, |x, y| (x - y).abs()),
            Distance::Chebyshev => zip_map_max(a, b, |x, y| (x - y).abs()),
            Distance::Minkowski(p) => {
                if *p <= 0.0 {
                    panic!(
                        "{}",
                        VqError::InvalidMetricParameter {
                            metric: "Minkowski".to_string(),
                            details: "p must be positive".to_string()
                        }
                    );
                }
                let p_val = T::from_f64(*p);
                let sum = zip_map_sum(a, b, |x, y| (x - y).abs().powf(p_val));
                sum.powf(T::one() / p_val)
            }
            Distance::Hamming => {
                zip_map_sum(a, b, |x, y| if x == y { T::zero() } else { T::one() })
            }
        }
    }
}
