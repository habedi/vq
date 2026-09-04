use half::f16;
use rand::prelude::{IndexedRandom, SeedableRng, StdRng};
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

use crate::core::error::{VqError, VqResult};

/// A trait for numeric types representing vector components.
///
/// This trait is implemented for `f32` (using standard `f32` operations)
/// and `f16` (using the `half` crate).
pub trait Real:
    Copy
    + Clone
    + Default
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + fmt::Display
    + Send
    + Sync
{
    /// Returns the additive identity.
    fn zero() -> Self;
    /// Returns the multiplicative identity.
    fn one() -> Self;
    /// Converts a `usize` to this type.
    fn from_usize(n: usize) -> Self;
    /// Computes the square root.
    fn sqrt(self) -> Self;
    /// Computes the absolute value.
    fn abs(self) -> Self;
}

impl Real for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_usize(n: usize) -> Self {
        n as f32
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Real for f16 {
    fn zero() -> Self {
        f16::from_f32(0.0)
    }
    fn one() -> Self {
        f16::from_f32(1.0)
    }
    fn from_usize(n: usize) -> Self {
        f16::from_f32(n as f32)
    }
    fn sqrt(self) -> Self {
        f16::from_f32(f16::to_f32(self).sqrt())
    }
    fn abs(self) -> Self {
        f16::from_f32(f16::to_f32(self).abs())
    }
}

/// A generic vector struct holding components of type `T`.
///
/// Wraps a standard `Vec<T>` and provides vector arithmetic operations
/// like addition, subtraction, dot product, and norm.
#[derive(Clone, Debug, PartialEq)]
pub struct Vector<T: Real> {
    /// The underlying data storage.
    pub data: Vec<T>,
}

impl<T: Real> Vector<T> {
    /// Creates a new `Vector` taking ownership of the data.
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a slice containing the vector data.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Computes the dot product with another vector.
    ///
    /// # Panics
    ///
    /// Panics if vectors have different dimensions.
    #[inline]
    pub fn dot(&self, other: &Self) -> T {
        assert_eq!(
            self.len(),
            other.len(),
            "Cannot compute dot product of vectors with different dimensions: {} vs {}",
            self.len(),
            other.len()
        );
        self.data
            .iter()
            .zip(other.data.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }

    /// Computes the Euclidean norm (length) of the vector.
    #[inline]
    pub fn norm(&self) -> T {
        self.dot(self).sqrt()
    }

    /// Computes the squared Euclidean distance to another vector.
    ///
    /// This is often more efficient than computing full Euclidean distance
    /// for comparisons (e.g. finding nearest neighbor).
    #[inline]
    pub fn distance2(&self, other: &Self) -> T {
        self.data
            .iter()
            .zip(other.data.iter())
            .fold(T::zero(), |acc, (&a, &b)| {
                let diff = a - b;
                acc + diff * diff
            })
    }
}

// Fallible arithmetic operations
impl<T: Real> Vector<T> {
    /// Adds two vectors element-wise, returning an error on dimension mismatch.
    ///
    /// # Errors
    ///
    /// Returns `VqError::DimensionMismatch` if vectors have different dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// use vq::core::vector::Vector;
    ///
    /// let a = Vector::new(vec![1.0, 2.0, 3.0]);
    /// let b = Vector::new(vec![4.0, 5.0, 6.0]);
    /// let c = a.try_add(&b).unwrap();
    /// assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
    /// ```
    pub fn try_add(&self, other: &Self) -> VqResult<Vector<T>> {
        if self.len() != other.len() {
            return Err(VqError::DimensionMismatch {
                expected: self.len(),
                found: other.len(),
            });
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Ok(Vector::new(data))
    }

    /// Subtracts two vectors element-wise, returning an error on dimension mismatch.
    ///
    /// # Errors
    ///
    /// Returns `VqError::DimensionMismatch` if vectors have different dimensions.
    pub fn try_sub(&self, other: &Self) -> VqResult<Vector<T>> {
        if self.len() != other.len() {
            return Err(VqError::DimensionMismatch {
                expected: self.len(),
                found: other.len(),
            });
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Ok(Vector::new(data))
    }

    /// Divides each element by a scalar, returning an error if scalar is zero.
    ///
    /// # Errors
    ///
    /// Returns `VqError::InvalidParameter` if scalar is zero.
    pub fn try_div(&self, scalar: T) -> VqResult<Vector<T>> {
        if scalar == T::zero() {
            return Err(VqError::InvalidParameter {
                parameter: "scalar",
                reason: "Cannot divide by zero".to_string(),
            });
        }
        let data = self.data.iter().map(|&a| a / scalar).collect();
        Ok(Vector::new(data))
    }
}

impl Vector<f32> {
    /// Checks if two vectors are approximately equal within an epsilon tolerance.
    ///
    /// This is useful for convergence checks in iterative algorithms where exact
    /// floating-point equality is too strict.
    ///
    /// # Arguments
    ///
    /// * `other` - The vector to compare with
    /// * `epsilon` - The maximum allowed difference per component (default: 1e-6)
    ///
    /// # Returns
    ///
    /// `true` if all components differ by less than epsilon, `false` otherwise
    pub fn approx_eq(&self, other: &Self, epsilon: f32) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(&a, &b)| (a - b).abs() < epsilon)
    }
}

impl<T: Real> Add for &Vector<T> {
    type Output = Vector<T>;

    /// Adds two vectors element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different dimensions.
    fn add(self, other: Self) -> Vector<T> {
        assert_eq!(
            self.len(),
            other.len(),
            "Cannot add vectors with different dimensions: {} vs {}",
            self.len(),
            other.len()
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Vector::new(data)
    }
}

impl<T: Real> Sub for &Vector<T> {
    type Output = Vector<T>;

    /// Subtracts two vectors element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different dimensions.
    fn sub(self, other: Self) -> Vector<T> {
        assert_eq!(
            self.len(),
            other.len(),
            "Cannot subtract vectors with different dimensions: {} vs {}",
            self.len(),
            other.len()
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Vector::new(data)
    }
}

impl<T: Real> Mul<T> for &Vector<T> {
    type Output = Vector<T>;

    fn mul(self, scalar: T) -> Vector<T> {
        let data = self.data.iter().map(|&a| a * scalar).collect();
        Vector::new(data)
    }
}

impl<T: Real> Div<T> for &Vector<T> {
    type Output = Vector<T>;

    /// Divides each element by a scalar.
    ///
    /// # Panics
    ///
    /// Panics if scalar is zero. This is consistent with Rust's default division behavior.
    /// If zero is possible, check before dividing or handle NaN/Infinity in results.
    fn div(self, scalar: T) -> Vector<T> {
        assert!(scalar != T::zero(), "Cannot divide vector by zero");
        let data = self.data.iter().map(|&a| a / scalar).collect();
        Vector::new(data)
    }
}

impl<T: Real> fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elements: Vec<String> = self.data.iter().map(|x| format!("{}", x)).collect();
        write!(f, "Vector [{}]", elements.join(", "))
    }
}

/// Computes the mean vector of a sequence of vectors.
///
/// # Errors
///
/// Returns `VqError::EmptyInput` if the input slice is empty.
pub fn mean_vector<T: Real>(vectors: &[Vector<T>]) -> VqResult<Vector<T>> {
    if vectors.is_empty() {
        return Err(VqError::EmptyInput);
    }
    let dim = vectors[0].len();
    let n = T::from_usize(vectors.len());
    let mut sum = vec![T::zero(); dim];

    for v in vectors {
        for (i, &val) in v.data.iter().enumerate() {
            sum[i] = sum[i] + val;
        }
    }

    let data = sum.into_iter().map(|s| s / n).collect();
    Ok(Vector::new(data))
}

/// Finds the index of the nearest centroid to a vector.
#[inline]
fn find_nearest_centroid(v: &Vector<f32>, centroids: &[Vector<f32>]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = v.distance2(&centroids[0]);
    for (j, c) in centroids.iter().enumerate().skip(1) {
        let dist = v.distance2(c);
        if dist < best_dist {
            best_dist = dist;
            best_idx = j;
        }
    }
    best_idx
}

/// Computes the mean vector from a slice of data using only the specified indices.
/// This avoids cloning vectors into temporary storage.
#[inline]
fn mean_vector_by_indices(data: &[Vector<f32>], indices: &[usize]) -> VqResult<Vector<f32>> {
    if indices.is_empty() {
        return Err(VqError::EmptyInput);
    }
    let dim = data[indices[0]].len();
    let n = indices.len() as f32;
    let mut sum = vec![0.0f32; dim];

    for &idx in indices {
        for (i, &val) in data[idx].data.iter().enumerate() {
            sum[i] += val;
        }
    }

    let result = sum.into_iter().map(|s| s / n).collect();
    Ok(Vector::new(result))
}

/// LBG/k-means quantization algorithm.
///
/// When compiled with the `parallel` feature, the assignment step is parallelized
/// using Rayon for improved performance on large datasets.
pub fn lbg_quantize(
    data: &[Vector<f32>],
    k: usize,
    max_iters: usize,
    seed: u64,
) -> VqResult<Vec<Vector<f32>>> {
    if data.is_empty() {
        return Err(VqError::EmptyInput);
    }
    if k == 0 {
        return Err(VqError::InvalidParameter {
            parameter: "k",
            reason: "must be greater than 0".to_string(),
        });
    }
    if data.len() < k {
        return Err(VqError::InvalidParameter {
            parameter: "k",
            reason: format!("not enough data points ({}) for {} clusters", data.len(), k),
        });
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids: Vec<Vector<f32>> = data.sample(&mut rng, k).cloned().collect();

    for _ in 0..max_iters {
        // Compute assignments (parallel when feature enabled)
        #[cfg(feature = "parallel")]
        let assignments: Vec<usize> = {
            use rayon::prelude::*;
            data.par_iter()
                .map(|v| find_nearest_centroid(v, &centroids))
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let assignments: Vec<usize> = data
            .iter()
            .map(|v| find_nearest_centroid(v, &centroids))
            .collect();

        // Build cluster indices (no cloning - just track which data points belong to each cluster)
        let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &cluster_idx) in assignments.iter().enumerate() {
            cluster_indices[cluster_idx].push(i);
        }

        // Update centroids
        let mut changed = false;
        const EPSILON: f32 = 1e-6;
        for j in 0..k {
            if !cluster_indices[j].is_empty() {
                let new_centroid = mean_vector_by_indices(data, &cluster_indices[j])?;
                // Use epsilon-based comparison instead of exact equality
                if !new_centroid.approx_eq(&centroids[j], EPSILON) {
                    changed = true;
                }
                centroids[j] = new_centroid;
            } else {
                #[allow(clippy::expect_used)]
                let random_point = data.choose(&mut rng).expect("data should not be empty");
                centroids[j] = random_point.clone();
                // The reseeded centroid has not been refined yet, so keep iterating
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    Ok(centroids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn get_data() -> Vec<Vector<f32>> {
        vec![
            Vector::new(vec![1.0, 2.0]),
            Vector::new(vec![2.0, 3.0]),
            Vector::new(vec![3.0, 4.0]),
            Vector::new(vec![4.0, 5.0]),
        ]
    }

    #[test]
    fn test_addition() {
        let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let b = Vector::new(vec![4.0f32, 5.0, 6.0]);
        let result = &a + &b;
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_subtraction() {
        let a = Vector::new(vec![4.0f32, 5.0, 6.0]);
        let b = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let result = &a - &b;
        assert_eq!(result.data, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_scalar_multiplication() {
        let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let result = &a * 2.0f32;
        assert_eq!(result.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_dot_product() {
        let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let b = Vector::new(vec![4.0f32, 5.0, 6.0]);
        let dot = a.dot(&b);
        assert!(approx_eq(dot, 32.0, 1e-6));
    }

    #[test]
    fn test_norm() {
        let a = Vector::new(vec![3.0f32, 4.0]);
        let norm = a.norm();
        assert!(approx_eq(norm, 5.0, 1e-6));
    }

    #[test]
    fn test_distance2() {
        let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let b = Vector::new(vec![4.0f32, 5.0, 6.0]);
        let dist2 = a.distance2(&b);
        assert!(approx_eq(dist2, 27.0, 1e-6));
    }

    #[test]
    fn test_mean_vector() {
        let vectors = vec![
            Vector::new(vec![1.0f32, 2.0, 3.0]),
            Vector::new(vec![4.0f32, 5.0, 6.0]),
            Vector::new(vec![7.0f32, 8.0, 9.0]),
        ];
        let mean = mean_vector(&vectors).unwrap();
        assert!(approx_eq(mean.data[0], 4.0, 1e-6));
        assert!(approx_eq(mean.data[1], 5.0, 1e-6));
        assert!(approx_eq(mean.data[2], 6.0, 1e-6));
    }

    #[test]
    fn test_mean_vector_empty() {
        let vectors: Vec<Vector<f32>> = vec![];
        let result = mean_vector(&vectors);
        assert!(result.is_err());
    }

    #[test]
    fn test_display() {
        let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let s = format!("{}", a);
        assert!(s.starts_with("Vector ["));
        assert!(s.ends_with("]"));
    }

    #[test]
    fn test_real_f16_constants_and_conversions() {
        assert_eq!(<f16 as Real>::zero().to_f32(), 0.0);
        assert_eq!(<f16 as Real>::one().to_f32(), 1.0);
        assert_eq!(<f16 as Real>::from_usize(7).to_f32(), 7.0);
        assert!(approx_eq(
            <f16 as Real>::sqrt(f16::from_f32(16.0)).to_f32(),
            4.0,
            1e-3
        ));
        assert_eq!(<f16 as Real>::abs(f16::from_f32(-2.5)).to_f32(), 2.5);
        assert_eq!(<f32 as Real>::from_usize(3), 3.0);
        assert_eq!(<f32 as Real>::abs(-1.5), 1.5);
    }

    #[test]
    fn test_accessors() {
        let v = Vector::new(vec![1.0f32, 2.0]);
        assert_eq!(v.len(), 2);
        assert!(!v.is_empty());
        assert_eq!(v.data(), &[1.0, 2.0]);
        assert!(Vector::<f32>::new(vec![]).is_empty());
    }

    #[test]
    fn test_try_ops_success() {
        let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let b = Vector::new(vec![4.0f32, 5.0, 6.0]);
        assert_eq!(a.try_add(&b).unwrap().data, vec![5.0, 7.0, 9.0]);
        assert_eq!(a.try_sub(&b).unwrap().data, vec![-3.0, -3.0, -3.0]);
        assert_eq!(b.try_div(2.0).unwrap().data, vec![2.0, 2.5, 3.0]);
    }

    #[test]
    fn test_try_div_by_zero() {
        let a = Vector::new(vec![1.0f32, 2.0]);
        assert!(matches!(
            a.try_div(0.0),
            Err(VqError::InvalidParameter {
                parameter: "scalar",
                ..
            })
        ));
    }

    #[test]
    fn test_scalar_division_operator() {
        let a = Vector::new(vec![2.0f32, 4.0]);
        assert_eq!((&a / 2.0).data, vec![1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "Cannot divide vector by zero")]
    fn test_scalar_division_operator_by_zero_panics() {
        let a = Vector::new(vec![2.0f32, 4.0]);
        let _ = &a / 0.0;
    }

    #[test]
    #[should_panic(expected = "Cannot add vectors with different dimensions")]
    fn test_add_operator_dimension_mismatch_panics() {
        let a = Vector::new(vec![1.0f32, 2.0]);
        let b = Vector::new(vec![1.0f32]);
        let _ = &a + &b;
    }

    #[test]
    #[should_panic(expected = "Cannot subtract vectors with different dimensions")]
    fn test_sub_operator_dimension_mismatch_panics() {
        let a = Vector::new(vec![1.0f32, 2.0]);
        let b = Vector::new(vec![1.0f32]);
        let _ = &a - &b;
    }

    #[test]
    fn test_approx_eq_length_mismatch() {
        let a = Vector::new(vec![1.0f32, 2.0]);
        let b = Vector::new(vec![1.0f32]);
        assert!(!a.approx_eq(&b, 1e-6));
    }

    #[test]
    fn test_mean_vector_f16() {
        let vs = vec![
            Vector::new(vec![f16::from_f32(1.0), f16::from_f32(3.0)]),
            Vector::new(vec![f16::from_f32(3.0), f16::from_f32(5.0)]),
        ];
        let m = mean_vector(&vs).unwrap();
        assert_eq!(m.data[0].to_f32(), 2.0);
        assert_eq!(m.data[1].to_f32(), 4.0);
    }

    #[test]
    fn test_lbg_quantize_invalid_parameters() {
        let data = vec![Vector::new(vec![0.0f32]), Vector::new(vec![1.0f32])];
        assert!(matches!(
            lbg_quantize(&[], 1, 10, 0),
            Err(VqError::EmptyInput)
        ));
        assert!(matches!(
            lbg_quantize(&data, 0, 10, 0),
            Err(VqError::InvalidParameter { parameter: "k", .. })
        ));
        assert!(matches!(
            lbg_quantize(&data, 3, 10, 0),
            Err(VqError::InvalidParameter { parameter: "k", .. })
        ));
    }

    #[test]
    fn test_lbg_quantize_zero_iterations_returns_sampled_points() {
        let data: Vec<Vector<f32>> = (0..5).map(|i| Vector::new(vec![i as f32])).collect();
        let centroids = lbg_quantize(&data, 2, 0, 3).unwrap();
        assert_eq!(centroids.len(), 2);
        assert!(centroids.iter().all(|c| data.contains(c)));
    }

    #[test]
    fn test_f16_operations() {
        let a = Vector::new(vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
        ]);
        let b = Vector::new(vec![
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ]);
        let dot = a.dot(&b);
        let dot_f32 = f32::from(dot);
        assert!((dot_f32 - 32.0).abs() < 1e-1);
    }

    #[test]
    fn lbg_quantize_basic() {
        let data = get_data();
        let centroids = lbg_quantize(&data, 2, 10, 42).unwrap();
        assert_eq!(centroids.len(), 2);
    }

    #[test]
    fn lbg_quantize_k_zero() {
        let data = vec![Vector::new(vec![1.0, 2.0])];
        let result = lbg_quantize(&data, 0, 10, 42);
        assert!(result.is_err());
    }

    #[test]
    fn lbg_quantize_not_enough_data() {
        let data = vec![Vector::new(vec![1.0, 2.0])];
        let result = lbg_quantize(&data, 2, 10, 42);
        assert!(result.is_err());
    }
}
