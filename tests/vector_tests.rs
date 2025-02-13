#[path = "utils.rs"]
mod utils;

use half::{bf16, f16};
use vq::vector::{mean_vector, Vector, PARALLEL_THRESHOLD};

// A small helper to compare floating point numbers with an epsilon.
fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
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
fn test_dot_product_sequential() {
    // Use a small vector to force sequential dot product.
    let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
    let b = Vector::new(vec![4.0f32, 5.0, 6.0]);
    // 1*4 + 2*5 + 3*6 = 32
    let dot = a.dot(&b);
    assert!(approx_eq(dot, 32.0, 1e-6));
}

#[test]
fn test_dot_product_parallel() {
    // Create vectors longer than PARALLEL_THRESHOLD so that parallel code is used.
    let len = PARALLEL_THRESHOLD + 1;
    let a = Vector::new((0..len).map(|i| i as f32).collect());
    let b = Vector::new((0..len).map(|i| (i as f32) * 2.0).collect());
    // dot = 2 * sum(i^2) for i in 0..len
    let expected: f32 = 2.0 * (0..len).map(|i| (i as f32).powi(2)).sum::<f32>();
    let dot = a.dot(&b);
    println!("Expected: {}, Actual: {}", expected, dot);
    assert!(approx_eq(dot, expected, 1e3)); // Using a larger epsilon due to error accumulation.
}

#[test]
fn test_norm() {
    // For a vector [3,4], norm should be 5.
    let a = Vector::new(vec![3.0f32, 4.0]);
    let norm = a.norm();
    assert!(approx_eq(norm, 5.0, 1e-6));
}

#[test]
fn test_distance2() {
    // Distance squared between [1,2,3] and [4,5,6]:
    // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
    let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
    let b = Vector::new(vec![4.0f32, 5.0, 6.0]);
    let dist2 = a.distance2(&b);
    assert!(approx_eq(dist2, 27.0, 1e-6));
}

#[test]
fn test_mean_vector_sequential() {
    let vectors = vec![
        Vector::new(vec![1.0f32, 2.0, 3.0]),
        Vector::new(vec![4.0f32, 5.0, 6.0]),
        Vector::new(vec![7.0f32, 8.0, 9.0]),
    ];
    let mean = mean_vector(&vectors);
    // Expected mean: ([1+4+7, 2+5+8, 3+6+9] / 3) = [4, 5, 6]
    assert!(approx_eq(mean.data[0], 4.0, 1e-6));
    assert!(approx_eq(mean.data[1], 5.0, 1e-6));
    assert!(approx_eq(mean.data[2], 6.0, 1e-6));
}

#[test]
fn test_mean_vector_parallel() {
    // Create more than PARALLEL_THRESHOLD vectors of identical content.
    let n = PARALLEL_THRESHOLD + 10;
    let dim = 5;
    let vectors: Vec<_> = (0..n)
        .map(|_| Vector::new((0..dim).map(|i| i as f32).collect()))
        .collect();
    let mean = mean_vector(&vectors);
    // Each vector is the same so the mean should be identical.
    let expected: Vec<f32> = (0..dim).map(|i| i as f32).collect();
    for (m, e) in mean.data.iter().zip(expected.iter()) {
        assert!(approx_eq(*m, *e, 1e-6));
    }
}

#[test]
#[should_panic(expected = "Vectors must be same length")]
fn test_addition_mismatched_dimensions() {
    let a = Vector::new(vec![1.0f32, 2.0]);
    let b = Vector::new(vec![1.0f32, 2.0, 3.0]);
    let _ = &a + &b;
}

#[test]
#[should_panic(expected = "Vectors must be same length")]
fn test_dot_product_mismatched_dimensions() {
    let a = Vector::new(vec![1.0f32, 2.0]);
    let b = Vector::new(vec![1.0f32]);
    let _ = a.dot(&b);
}

#[test]
#[should_panic(expected = "Cannot compute mean of empty slice")]
fn test_mean_vector_empty() {
    let vectors: Vec<Vector<f32>> = vec![];
    let _ = mean_vector(&vectors);
}

#[test]
fn test_display() {
    let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
    let s = format!("{}", a);
    // Check that the string starts with "Vector [" and ends with "]"
    assert!(s.starts_with("Vector ["));
    assert!(s.ends_with("]"));
}

// --- Tests using other Real types ---

#[test]
fn test_f64_operations() {
    let a = Vector::new(vec![1.0f64, 2.0, 3.0]);
    let b = Vector::new(vec![4.0f64, 5.0, 6.0]);
    let dot = a.dot(&b);
    assert!((dot - 32.0).abs() < 1e-6);
    let norm = a.norm();
    let expected_norm = (1.0f64 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0).sqrt();
    assert!((norm - expected_norm).abs() < 1e-6);
}

#[test]
fn test_f16_operations() {
    // Test basic operations using half-precision (f16).
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
    // Convert dot to f32 for comparison.
    let dot_f32 = f32::from(dot);
    assert!((dot_f32 - 32.0).abs() < 1e-1);
}

#[test]
fn test_bf16_operations() {
    // Test basic operations using bf16.
    let a = Vector::new(vec![
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
    ]);
    let b = Vector::new(vec![
        bf16::from_f32(4.0),
        bf16::from_f32(5.0),
        bf16::from_f32(6.0),
    ]);
    let dot = a.dot(&b);
    let dot_f32 = f32::from(dot);
    assert!((dot_f32 - 32.0).abs() < 1e-1);
}
