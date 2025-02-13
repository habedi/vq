use half::{bf16, f16};
use rayon::prelude::*;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

// Size threshold for enabling parallel computation.
pub const PARALLEL_THRESHOLD: usize = 1024;

/// Abstraction for real numbers.
pub trait Real:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn from_f64(x: f64) -> Self;
}

impl Real for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn abs(self) -> Self {
        f32::abs(self)
    }
    fn powf(self, n: Self) -> Self {
        f32::powf(self, n)
    }
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

impl Real for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn powf(self, n: Self) -> Self {
        f64::powf(self, n)
    }
    fn from_f64(x: f64) -> Self {
        x
    }
}

impl Real for f16 {
    fn zero() -> Self {
        f16::from_f32(0.0)
    }
    fn one() -> Self {
        f16::from_f32(1.0)
    }
    fn sqrt(self) -> Self {
        f16::from_f32(f32::from(self).sqrt())
    }
    fn abs(self) -> Self {
        if self < f16::from_f32(0.0) {
            -self
        } else {
            self
        }
    }
    fn powf(self, n: Self) -> Self {
        f16::from_f32(f32::from(self).powf(f32::from(n)))
    }
    fn from_f64(x: f64) -> Self {
        f16::from_f32(x as f32)
    }
}

impl Real for bf16 {
    fn zero() -> Self {
        bf16::from_f32(0.0)
    }
    fn one() -> Self {
        bf16::from_f32(1.0)
    }
    fn sqrt(self) -> Self {
        bf16::from_f32(f32::from(self).sqrt())
    }
    fn abs(self) -> Self {
        if self < bf16::from_f32(0.0) {
            -self
        } else {
            self
        }
    }
    fn powf(self, n: Self) -> Self {
        bf16::from_f32(f32::from(self).powf(f32::from(n)))
    }
    fn from_f64(x: f64) -> Self {
        bf16::from_f32(x as f32)
    }
}

impl Real for u8 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn sqrt(self) -> Self {
        (self as f32).sqrt() as u8
    }
    fn abs(self) -> Self {
        self
    }
    fn powf(self, n: Self) -> Self {
        f32::from(self).powf(f32::from(n)) as u8
    }
    fn from_f64(x: f64) -> Self {
        x as u8
    }
}

/// A vector of real numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T: Real> {
    pub data: Vec<T>,
}

impl<T: Real> Vector<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns a slice of the data.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Compute the dot product between two vectors.
    ///
    /// If the length is larger than `THRESHOLD`, the dot product is computed in parallel.
    /// Otherwise, it falls back to a sequential implementation.
    ///
    /// (This method requires that `T` implements `Send + Sync` so that parallel iteration is safe.)
    pub fn dot(&self, other: &Vector<T>) -> T
    where
        T: Send + Sync,
    {
        assert_eq!(self.len(), other.len(), "Vectors must be same length");
        if self.len() > PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .zip(other.data.par_iter())
                .map(|(&a, &b)| a * b)
                .reduce(|| T::zero(), |x, y| x + y)
        } else {
            self.data
                .iter()
                .zip(other.data.iter())
                .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
        }
    }

    /// Compute the Euclidean norm.
    pub fn norm(&self) -> T
    where
        T: Send + Sync,
    {
        self.dot(self).sqrt()
    }

    /// Compute the squared distance between two vectors.
    pub fn distance2(&self, other: &Vector<T>) -> T
    where
        T: Send + Sync,
    {
        let diff = self - other;
        diff.dot(&diff)
    }
}

/// Vector addition.
impl<'b, T: Real> Add<&'b Vector<T>> for &Vector<T> {
    type Output = Vector<T>;
    fn add(self, rhs: &'b Vector<T>) -> Vector<T> {
        assert_eq!(self.len(), rhs.len(), "Vectors must be same length");
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Vector::new(data)
    }
}

/// Vector subtraction.
impl<'b, T: Real> Sub<&'b Vector<T>> for &Vector<T> {
    type Output = Vector<T>;
    fn sub(self, rhs: &'b Vector<T>) -> Vector<T> {
        assert_eq!(self.len(), rhs.len(), "Vectors must be same length");
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Vector::new(data)
    }
}

/// Scalar multiplication.
impl<T: Real> Mul<T> for &Vector<T> {
    type Output = Vector<T>;
    fn mul(self, scalar: T) -> Vector<T> {
        let data = self.data.iter().map(|&a| a * scalar).collect();
        Vector::new(data)
    }
}

/// Compute the mean vector from a slice of vectors.
/// If there are more than `THRESHOLD` vectors, the summation is performed in parallel.
/// Assumes that all vectors have the same dimension.
pub fn mean_vector<T: Real + Send + Sync>(vectors: &[Vector<T>]) -> Vector<T> {
    assert!(!vectors.is_empty(), "Cannot compute mean of empty slice");
    let dim = vectors[0].len();
    for v in vectors {
        assert_eq!(v.len(), dim, "All vectors must have the same dimension");
    }
    let sum: Vec<T> = if vectors.len() > PARALLEL_THRESHOLD {
        // Parallel reduction over the vectors.
        // We first reduce the slice of vectors into one "sum" vector.
        let summed = vectors
            .par_iter()
            .cloned()
            .reduce(|| Vector::new(vec![T::zero(); dim]), |a, b| &a + &b);
        summed.data
    } else {
        let mut sum = vec![T::zero(); dim];
        for v in vectors {
            for i in 0..dim {
                sum[i] = sum[i] + v.data[i];
            }
        }
        sum
    };
    let n = T::from_f64(vectors.len() as f64);
    let mean_data = sum.into_iter().map(|s| s / n).collect();
    Vector::new(mean_data)
}

/// Custom Display implementation for Vector<T>.
impl<T: Real + fmt::Display> fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector [")?;
        let mut first = true;
        for elem in &self.data {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            write!(f, "{}", elem)?;
        }
        write!(f, "]")
    }
}
