use crate::vector::{Real, PARALLEL_THRESHOLD};
use rayon::prelude::*;

/// Represents various distance metrics for comparing vectors.
///
/// Each variant corresponds to a specific metric. When computing distances, if the length of the input
/// slices exceeds `PARALLEL_THRESHOLD`, parallel iterators (via Rayon) are used for improved performance.
pub enum Distance {
    /// Squared Euclidean distance: the sum of the squared differences.
    SquaredEuclidean,
    /// Euclidean distance: the square root of the sum of the squared differences.
    Euclidean,
    /// Cosine distance: defined as 1 minus the cosine similarity.
    /// If one of the vectors has zero norm, the distance is defined as 1.
    CosineDistance,
    /// Manhattan distance: the sum of absolute differences.
    Manhattan,
    /// Chebyshev distance: the maximum absolute difference.
    Chebyshev,
    /// Minkowski distance: a generalization of Euclidean and Manhattan distances.
    /// The parameter `p` defines the order of the norm.
    Minkowski(f64),
    /// Hamming distance: counts the number of positions where corresponding elements differ.
    Hamming,
}

impl Distance {
    /// Computes the distance between two slices `a` and `b` using the selected metric.
    ///
    /// If the length of the slices exceeds `PARALLEL_THRESHOLD`, the computation is performed in parallel
    /// using Rayon. Otherwise, sequential iteration is used.
    ///
    /// # Type Parameters
    /// - `T`: The numeric type of the elements in the slices, which must implement the `Real`, `Send`,
    ///   and `Sync` traits.
    ///
    /// # Parameters
    /// - `a`: A slice representing the first vector.
    /// - `b`: A slice representing the second vector.
    ///
    /// # Returns
    /// The computed distance as a value of type `T`.
    ///
    /// # Panics
    /// Panics if the lengths of `a` and `b` differ.
    pub fn compute<T>(&self, a: &[T], b: &[T]) -> T
    where
        T: Real + Send + Sync,
    {
        assert_eq!(a.len(), b.len(), "Input slices must have the same length");

        match self {
            Distance::SquaredEuclidean => {
                if a.len() > PARALLEL_THRESHOLD {
                    a.par_iter()
                        .zip(b.par_iter())
                        .map(|(&x, &y)| {
                            let diff = x - y;
                            diff * diff
                        })
                        .reduce(|| T::zero(), |acc, val| acc + val)
                } else {
                    a.iter()
                        .zip(b.iter())
                        .map(|(&x, &y)| {
                            let diff = x - y;
                            diff * diff
                        })
                        .fold(T::zero(), |acc, val| acc + val)
                }
            }
            Distance::Euclidean => {
                let sum = if a.len() > PARALLEL_THRESHOLD {
                    a.par_iter()
                        .zip(b.par_iter())
                        .map(|(&x, &y)| {
                            let diff = x - y;
                            diff * diff
                        })
                        .reduce(|| T::zero(), |acc, val| acc + val)
                } else {
                    a.iter()
                        .zip(b.iter())
                        .map(|(&x, &y)| {
                            let diff = x - y;
                            diff * diff
                        })
                        .fold(T::zero(), |acc, val| acc + val)
                };
                sum.sqrt()
            }
            Distance::CosineDistance => {
                let dot = if a.len() > PARALLEL_THRESHOLD {
                    a.par_iter()
                        .zip(b.par_iter())
                        .map(|(&x, &y)| x * y)
                        .reduce(|| T::zero(), |acc, val| acc + val)
                } else {
                    a.iter()
                        .zip(b.iter())
                        .map(|(&x, &y)| x * y)
                        .fold(T::zero(), |acc, val| acc + val)
                };

                let norm_a = if a.len() > PARALLEL_THRESHOLD {
                    a.par_iter()
                        .map(|&x| x * x)
                        .reduce(|| T::zero(), |acc, v| acc + v)
                        .sqrt()
                } else {
                    a.iter()
                        .map(|&x| x * x)
                        .fold(T::zero(), |acc, v| acc + v)
                        .sqrt()
                };

                let norm_b = if b.len() > PARALLEL_THRESHOLD {
                    b.par_iter()
                        .map(|&x| x * x)
                        .reduce(|| T::zero(), |acc, v| acc + v)
                        .sqrt()
                } else {
                    b.iter()
                        .map(|&x| x * x)
                        .fold(T::zero(), |acc, v| acc + v)
                        .sqrt()
                };

                if norm_a == T::zero() || norm_b == T::zero() {
                    // If either vector is zero, define cosine distance as 1.
                    T::one()
                } else {
                    T::one() - dot / (norm_a * norm_b)
                }
            }
            Distance::Manhattan => {
                if a.len() > PARALLEL_THRESHOLD {
                    a.par_iter()
                        .zip(b.par_iter())
                        .map(|(&x, &y)| (x - y).abs())
                        .reduce(|| T::zero(), |acc, val| acc + val)
                } else {
                    a.iter()
                        .zip(b.iter())
                        .map(|(&x, &y)| (x - y).abs())
                        .fold(T::zero(), |acc, val| acc + val)
                }
            }
            Distance::Chebyshev => {
                if a.len() > PARALLEL_THRESHOLD {
                    a.par_iter()
                        .zip(b.par_iter())
                        .map(|(&x, &y)| (x - y).abs())
                        .reduce(|| T::zero(), |acc, val| if val > acc { val } else { acc })
                } else {
                    a.iter()
                        .zip(b.iter())
                        .map(|(&x, &y)| (x - y).abs())
                        .fold(T::zero(), |acc, val| if val > acc { val } else { acc })
                }
            }
            Distance::Minkowski(p) => {
                let p_val = T::from_f64(*p);
                let sum = if a.len() > PARALLEL_THRESHOLD {
                    a.par_iter()
                        .zip(b.par_iter())
                        .map(|(&x, &y)| (x - y).abs().powf(p_val))
                        .reduce(|| T::zero(), |acc, val| acc + val)
                } else {
                    a.iter()
                        .zip(b.iter())
                        .map(|(&x, &y)| (x - y).abs().powf(p_val))
                        .fold(T::zero(), |acc, val| acc + val)
                };
                sum.powf(T::one() / p_val)
            }
            Distance::Hamming => {
                if a.len() > PARALLEL_THRESHOLD {
                    a.par_iter()
                        .zip(b.par_iter())
                        .map(|(&x, &y)| if x == y { T::zero() } else { T::one() })
                        .reduce(|| T::zero(), |acc, val| acc + val)
                } else {
                    a.iter()
                        .zip(b.iter())
                        .map(|(&x, &y)| if x == y { T::zero() } else { T::one() })
                        .fold(T::zero(), |acc, val| acc + val)
                }
            }
        }
    }
}
