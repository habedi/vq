use half::{bf16, f16};
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

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

#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T: Real> {
    pub data: Vec<T>,
}

impl<T: Real> Vector<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn dot(&self, other: &Vector<T>) -> T {
        assert_eq!(self.len(), other.len(), "Vectors must be same length");
        self.data
            .iter()
            .zip(other.data.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }

    pub fn norm(&self) -> T {
        self.dot(self).sqrt()
    }

    pub fn distance2(&self, other: &Vector<T>) -> T {
        let diff = self - other;
        diff.dot(&diff)
    }
}

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

impl<T: Real> Mul<T> for &Vector<T> {
    type Output = Vector<T>;
    fn mul(self, scalar: T) -> Vector<T> {
        let data = self.data.iter().map(|&a| a * scalar).collect();
        Vector::new(data)
    }
}

pub fn mean_vector<T: Real>(vectors: &[Vector<T>]) -> Vector<T> {
    assert!(!vectors.is_empty(), "Cannot compute mean of empty slice");
    let dim = vectors[0].len();
    let mut sum = vec![T::zero(); dim];
    for v in vectors {
        for i in 0..dim {
            sum[i] = sum[i] + v.data[i];
        }
    }
    let n = T::from_f64(vectors.len() as f64);
    let mean_data = sum.into_iter().map(|s| s / n).collect();
    Vector::new(mean_data)
}

// Custom Display implementation for Vector<T>
// This requires T to also implement fmt::Display.
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
