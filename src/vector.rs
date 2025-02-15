//! # Vector Representation and Operations
//!
//! This module defines a `Vector` type and operations for real numbers. It includes basic
//! arithmetic (addition, subtraction, scalar multiplication), dot product, norm, and a function
//! to compute the mean vector from a slice of vectors. When the input size exceeds a threshold,
//! Rayon is used to perform operations in parallel for better performance.

use half::{bf16, f16};
use rayon::prelude::*;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

use crate::exceptions::VqError;

/// Size threshold for enabling parallel computation.
pub const PARALLEL_THRESHOLD: usize = 1024;

/// Trait for basic operations on real numbers.
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
    /// Create a new vector from a `Vec<T>`.
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a slice of the data.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Compute the dot product with another vector.
    ///
    /// If the vector length exceeds `PARALLEL_THRESHOLD`, this is computed in parallel.
    pub fn dot(&self, other: &Vector<T>) -> T
    where
        T: Send + Sync,
    {
        if self.len() != other.len() {
            panic!(
                "{}",
                VqError::DimensionMismatch {
                    expected: self.len(),
                    found: other.len()
                }
            );
        }
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
        if self.len() != rhs.len() {
            panic!(
                "{}",
                VqError::DimensionMismatch {
                    expected: self.len(),
                    found: rhs.len()
                }
            );
        }
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
        if self.len() != rhs.len() {
            panic!(
                "{}",
                VqError::DimensionMismatch {
                    expected: self.len(),
                    found: rhs.len()
                }
            );
        }
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
///
/// All vectors must have the same dimension. For many vectors (more than `PARALLEL_THRESHOLD`),
/// the summation is done in parallel.
pub fn mean_vector<T: Real + Send + Sync>(vectors: &[Vector<T>]) -> Vector<T> {
    if vectors.is_empty() {
        panic!("{}", VqError::EmptyInput);
    }
    let dim = vectors[0].len();
    for v in vectors {
        if v.len() != dim {
            panic!(
                "{}",
                VqError::DimensionMismatch {
                    expected: dim,
                    found: v.len()
                }
            );
        }
    }
    let sum: Vec<T> = if vectors.len() > PARALLEL_THRESHOLD {
        // Parallel reduction: sum all vectors into one.
        let summed = vectors
            .par_iter()
            .cloned()
            .reduce(|| Vector::new(vec![T::zero(); dim]), |a, b| &a + &b);
        summed.data
    } else {
        let mut sum = vec![T::zero(); dim];
        for v in vectors {
            // Replace explicit index loop with zip iterator.
            for (s, &value) in sum.iter_mut().zip(v.data.iter()) {
                *s = *s + value;
            }
        }
        sum
    };
    let n = T::from_f64(vectors.len() as f64);
    let mean_data = sum.into_iter().map(|s| s / n).collect();
    Vector::new(mean_data)
}

/// Custom display for vectors.
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
