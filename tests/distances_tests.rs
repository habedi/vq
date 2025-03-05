#[path = "utils.rs"]
mod utils;

use vq::distances::Distance;
use vq::vector::PARALLEL_THRESHOLD;

// A helper function to compare two floating point numbers within a given tolerance.
fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

// ----------------------------
// Squared Euclidean Distance
// ----------------------------
#[test]
fn test_squared_euclidean_sequential() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 6.0, 8.0];
    // (1-4)² + (2-6)² + (3-8)² = 9 + 16 + 25 = 50
    let d = Distance::SquaredEuclidean;
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, 50.0, 1e-6));
}

#[test]
fn test_squared_euclidean_parallel() {
    let len = PARALLEL_THRESHOLD + 10;
    // Each difference is (i - (i+1)) = -1 so square is 1.
    let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..len).map(|i| (i as f32) + 1.0).collect();
    let d = Distance::SquaredEuclidean;
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, len as f32, 1e-6));
}

// ----------------------------
// Euclidean Distance
// ----------------------------
#[test]
fn test_euclidean_sequential() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 6.0, 8.0];
    // Squared distance is 50, so Euclidean distance = sqrt(50)
    let expected = 50.0f32.sqrt();
    let d = Distance::Euclidean;
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, expected, 1e-6));
}

#[test]
fn test_euclidean_parallel() {
    let len = PARALLEL_THRESHOLD + 10;
    let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..len).map(|i| (i as f32) + 1.0).collect();
    // Each pair differs by 1 so squared differences add to len.
    let expected = (len as f32).sqrt();
    let d = Distance::Euclidean;
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, expected, 1e-6));
}

// ----------------------------
// Cosine Distance
// ----------------------------
#[test]
fn test_cosine_distance_sequential() {
    // Orthogonal vectors: cosine similarity = 0, so distance = 1.
    let a = vec![1.0f32, 0.0];
    let b = vec![0.0f32, 1.0];
    let d = Distance::CosineDistance;
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, 1.0, 1e-6));

    // Identical vectors: cosine similarity = 1, so distance = 0.
    let a = vec![1.0f32, 1.0];
    let b = vec![1.0f32, 1.0];
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, 0.0, 1e-6));
}

#[test]
fn test_cosine_distance_parallel() {
    let len = PARALLEL_THRESHOLD + 10;
    // Use identical vectors so that cosine similarity is 1 and distance is 0.
    let a = vec![1.0f32; len];
    let b = vec![1.0f32; len];
    let d = Distance::CosineDistance;
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, 0.0, 1e-6));
}

// ----------------------------
// Manhattan Distance
// ----------------------------
#[test]
fn test_manhattan_sequential() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 6.0, 8.0];
    // |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
    let d = Distance::Manhattan;
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, 12.0, 1e-6));
}

#[test]
fn test_manhattan_parallel() {
    let len = PARALLEL_THRESHOLD + 10;
    let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..len).map(|i| (i as f32) + 2.0).collect();
    // Each difference is 2, so sum = 2 * len.
    let expected = 2.0 * (len as f32);
    let d = Distance::Manhattan;
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, expected, 1e-6));
}

// ----------------------------
// Chebyshev Distance
// ----------------------------
#[test]
fn test_chebyshev_sequential() {
    let a = vec![1.0f32, 5.0, 3.0];
    let b = vec![4.0f32, 2.0, 9.0];
    // Differences: |1-4|=3, |5-2|=3, |3-9|=6, so maximum is 6.
    let d = Distance::Chebyshev;
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, 6.0, 1e-6));
}

#[test]
fn test_chebyshev_parallel() {
    let len = PARALLEL_THRESHOLD + 10;
    // Create two vectors with nearly identical values except one coordinate.
    let mut a: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let mut b: Vec<f32> = (0..len).map(|i| i as f32).collect();
    // Introduce a large difference at the last element.
    a[len - 1] = 1000.0;
    b[len - 1] = 0.0;
    let d = Distance::Chebyshev;
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, 1000.0, 1e-6));
}

// ----------------------------
// Minkowski Distance (p = 3)
// ----------------------------
#[test]
fn test_minkowski_sequential() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 6.0, 8.0];
    // For p = 3:
    // |1-4|^3 = 27, |2-6|^3 = 64, |3-8|^3 = 125, sum = 216, cube root = 6.
    let d = Distance::Minkowski(3.0);
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, 6.0, 1e-6));
}

#[test]
fn test_minkowski_parallel() {
    let p = 3.0;
    let d = Distance::Minkowski(p);
    let len = PARALLEL_THRESHOLD + 10;
    let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..len).map(|i| (i as f32) + 1.0).collect();
    // Each difference is 1: |1|^3 = 1. Sum = len, then result = len^(1/3)
    let expected = (len as f32).powf(1.0 / 3.0);
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, expected, 1e-6));
}

// ----------------------------
// Hamming Distance
// ----------------------------
#[test]
fn test_hamming_sequential() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![1.0f32, 0.0, 3.0, 0.0];
    // Differences occur at index 1 and 3, so count = 2.
    let d = Distance::Hamming;
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, 2.0, 1e-6));
}

#[test]
fn test_hamming_parallel() {
    let len = PARALLEL_THRESHOLD + 10;
    let a: Vec<f32> = vec![1.0f32; len];
    // Make b differ on every odd index.
    let b: Vec<f32> = (0..len)
        .map(|i| if i % 2 == 0 { 1.0f32 } else { 0.0f32 })
        .collect();
    // Expected differences: about half the indices.
    let expected = if len % 2 == 0 {
        (len / 2) as f32
    } else {
        ((len / 2) + 1) as f32
    };
    let d = Distance::Hamming;
    let result = d.compute(&a, &b);
    assert!(approx_eq(result, expected, 1e-6));
}

// ----------------------------
// Mismatched Lengths
// ----------------------------
#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_compute_mismatched_lengths() {
    let a = vec![1.0f32, 2.0];
    let b = vec![1.0f32];
    let d = Distance::Euclidean;
    let _ = d.compute(&a, &b);
}
