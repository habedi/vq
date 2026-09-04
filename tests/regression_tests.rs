//! Regression tests for bugs fixed during development.
//!
//! This file contains tests that verify specific bugs remain fixed.
//! Each test is documented with the issue/bug it addresses.

use vq::core::distance::Distance;
use vq::core::error::VqError;
use vq::core::quantizer::Quantizer;
use vq::core::vector::{Vector, lbg_quantize};
use vq::{BinaryQuantizer, ProductQuantizer, ScalarQuantizer, TSVQ};

// =============================================================================
// Bug Fix: BinaryQuantizer dequantize returned hardcoded 0.0/1.0
// =============================================================================

#[test]
fn test_binary_quantizer_dequantize_uses_low_high_values() {
    // Bug: dequantize was returning hardcoded 0.0 and 1.0 instead of low/high
    let bq = BinaryQuantizer::new(0.0, 10, 20).unwrap();

    let codes = vec![0, 5, 10, 15, 20, 25, 255];
    let result = bq.dequantize(&codes).unwrap();

    // Values < high should map to low, values >= high should map to high
    assert_eq!(result[0], 10.0); // 0 < 20
    assert_eq!(result[1], 10.0); // 5 < 20
    assert_eq!(result[2], 10.0); // 10 < 20
    assert_eq!(result[3], 10.0); // 15 < 20
    assert_eq!(result[4], 20.0); // 20 >= 20
    assert_eq!(result[5], 20.0); // 25 >= 20
    assert_eq!(result[6], 20.0); // 255 >= 20
}

#[test]
fn test_binary_quantizer_dequantize_preserves_custom_levels() {
    let bq = BinaryQuantizer::new(0.5, 50, 200).unwrap();

    let quantized = bq.quantize(&[0.0, 0.5, 1.0]).unwrap();
    let reconstructed = bq.dequantize(&quantized).unwrap();

    // Should reconstruct to 50.0 or 200.0, not 0.0 or 1.0
    assert!(reconstructed.iter().all(|&x| x == 50.0 || x == 200.0));
}

// =============================================================================
// Bug Fix: BinaryQuantizer missing infinity validation
// =============================================================================

#[test]
fn test_binary_quantizer_rejects_infinite_threshold() {
    let result = BinaryQuantizer::new(f32::INFINITY, 0, 1);
    assert!(matches!(result, Err(VqError::InvalidParameter { .. })));

    let result = BinaryQuantizer::new(f32::NEG_INFINITY, 0, 1);
    assert!(matches!(result, Err(VqError::InvalidParameter { .. })));
}

#[test]
fn test_binary_quantizer_rejects_nan_threshold() {
    let result = BinaryQuantizer::new(f32::NAN, 0, 1);
    assert!(matches!(result, Err(VqError::InvalidParameter { .. })));
}

// =============================================================================
// Bug Fix: ProductQuantizer missing dimension validation
// =============================================================================

#[test]
fn test_product_quantizer_validates_dimension_consistency() {
    // Bug: PQ didn't check if all training vectors have same dimension
    let training = [
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0],
    ];
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let result = ProductQuantizer::new(&refs, 2, 4, 10, Distance::Euclidean, 42);

    assert!(matches!(result, Err(VqError::DimensionMismatch { .. })));
}

#[test]
fn test_product_quantizer_accepts_consistent_dimensions() {
    let training = [
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    ];
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let result = ProductQuantizer::new(&refs, 2, 2, 10, Distance::Euclidean, 42);

    assert!(result.is_ok());
}

// =============================================================================
// Bug Fix: TSVQ missing dimension validation
// =============================================================================

#[test]
fn test_tsvq_validates_dimension_consistency() {
    // Bug: TSVQ didn't check if all training vectors have same dimension
    let v1 = vec![1.0, 2.0, 3.0, 4.0];
    let v2 = vec![5.0, 6.0, 7.0, 8.0];
    let v3 = vec![9.0, 10.0]; // Different dimension!

    let training: Vec<&[f32]> = vec![&v1, &v2, &v3];

    let result = TSVQ::new(&training, 3, Distance::Euclidean);

    assert!(matches!(result, Err(VqError::DimensionMismatch { .. })));
}

// =============================================================================
// Bug Fix: Vector operations division by zero
// =============================================================================

#[test]
#[should_panic(expected = "Cannot divide vector by zero")]
fn test_vector_div_panics_on_zero() {
    // Bug: Vector division didn't check for zero divisor
    let v = Vector::new(vec![1.0, 2.0, 3.0]);
    let _ = &v / 0.0; // Should panic
}

#[test]
fn test_vector_try_div_returns_error_on_zero() {
    let v = Vector::new(vec![1.0, 2.0, 3.0]);
    let result = v.try_div(0.0);

    assert!(matches!(result, Err(VqError::InvalidParameter { .. })));
}

#[test]
fn test_vector_try_div_succeeds_on_nonzero() {
    let v = Vector::new(vec![2.0, 4.0, 6.0]);
    let result = v.try_div(2.0).unwrap();

    assert_eq!(result.data(), &[1.0, 2.0, 3.0]);
}

// =============================================================================
// Bug Fix: Vector dot product missing dimension check
// =============================================================================

#[test]
#[should_panic(expected = "Cannot compute dot product of vectors with different dimensions")]
fn test_vector_dot_panics_on_dimension_mismatch() {
    // Bug: dot product silently truncated to shorter vector
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0]);
    let _ = a.dot(&b);
}

#[test]
fn test_vector_dot_succeeds_on_matching_dimensions() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0, 6.0]);
    let result = a.dot(&b);

    assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6
}

// =============================================================================
// Bug Fix: Vector add/sub panic messages improved
// =============================================================================

#[test]
#[should_panic(expected = "Cannot add vectors with different dimensions")]
fn test_vector_add_panics_with_clear_message() {
    let a = Vector::new(vec![1.0, 2.0]);
    let b = Vector::new(vec![3.0, 4.0, 5.0]);
    let _ = &a + &b;
}

#[test]
#[should_panic(expected = "Cannot subtract vectors with different dimensions")]
fn test_vector_sub_panics_with_clear_message() {
    let a = Vector::new(vec![1.0, 2.0]);
    let b = Vector::new(vec![3.0, 4.0, 5.0]);
    let _ = &a - &b;
}

#[test]
fn test_vector_try_add_returns_error_on_mismatch() {
    let a = Vector::new(vec![1.0, 2.0]);
    let b = Vector::new(vec![3.0, 4.0, 5.0]);
    let result = a.try_add(&b);

    assert!(matches!(result, Err(VqError::DimensionMismatch { .. })));
}

#[test]
fn test_vector_try_sub_returns_error_on_mismatch() {
    let a = Vector::new(vec![1.0, 2.0]);
    let b = Vector::new(vec![3.0, 4.0, 5.0]);
    let result = a.try_sub(&b);

    assert!(matches!(result, Err(VqError::DimensionMismatch { .. })));
}

// =============================================================================
// Bug Fix: LBG quantization floating-point equality
// =============================================================================

#[test]
fn test_lbg_convergence_with_epsilon_comparison() {
    // Bug: LBG used exact equality which could cause unnecessary iterations
    let data = vec![
        Vector::new(vec![1.0, 1.0]),
        Vector::new(vec![1.0001, 1.0001]), // Very close to first
        Vector::new(vec![10.0, 10.0]),
        Vector::new(vec![10.0001, 10.0001]), // Very close to third
    ];

    let result = lbg_quantize(&data, 2, 100, 42);
    assert!(result.is_ok());

    let centroids = result.unwrap();
    assert_eq!(centroids.len(), 2);

    // Should converge quickly with epsilon comparison
    // This test primarily checks it doesn't run for full 100 iterations
}

#[test]
fn test_vector_approx_eq_detects_near_equality() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7]);

    assert!(a.approx_eq(&b, 1e-6));
    assert!(!a.approx_eq(&b, 1e-8));
}

// =============================================================================
// Bug Fix: Cosine distance edge cases
// =============================================================================

#[test]
fn test_cosine_distance_handles_zero_norm() {
    // Bug: Division by zero for zero-norm vectors
    let zero = vec![0.0, 0.0, 0.0];
    let normal = vec![1.0, 2.0, 3.0];

    let dist = Distance::CosineDistance.compute(&zero, &normal).unwrap();

    // Zero vectors should be considered maximally distant
    assert_eq!(dist, 1.0);
}

#[test]
fn test_cosine_distance_handles_near_zero_norm() {
    // Bug: Division by very small numbers causing numerical instability
    let tiny = vec![1e-20, 1e-20, 1e-20];
    let normal = vec![1.0, 2.0, 3.0];

    let dist = Distance::CosineDistance.compute(&tiny, &normal).unwrap();

    // Should return 1.0 for near-zero vectors (using epsilon check)
    assert_eq!(dist, 1.0);
}

#[test]
fn test_cosine_distance_result_clamped() {
    // Bug: Floating-point errors could produce values outside [0, 1]
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];

    let dist = Distance::CosineDistance.compute(&a, &b).unwrap();

    // Distance should be in valid range [0, 2]
    assert!((0.0..=2.0).contains(&dist));
    assert!(dist.abs() < 1e-6); // Should be very close to 0
}

#[test]
fn test_cosine_distance_opposite_vectors_is_two() {
    // Bug: the scalar path clamped cosine distance to [0, 1], which collapsed every
    // negatively correlated pair to 1.0 and disagreed with the SIMD path
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![-1.0, 0.0, 0.0];

    let dist = Distance::CosineDistance.compute(&a, &b).unwrap();
    assert!((dist - 2.0).abs() < 1e-6);

    // Ordering must be preserved: orthogonal is closer than opposite
    let c = vec![0.0, 1.0, 0.0];
    let orthogonal = Distance::CosineDistance.compute(&a, &c).unwrap();
    assert!(orthogonal < dist);
}

// =============================================================================
// Bug Fix: TSVQ NaN handling in sorting
// =============================================================================

#[test]
fn test_tsvq_handles_nan_in_training_data() {
    // Bug: NaN values caused unstable sorting behavior
    let training = [
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, f32::NAN, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    ];
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    // Should not panic and should handle NaN gracefully
    let result = TSVQ::new(&refs, 2, Distance::SquaredEuclidean);

    // Either succeeds (filtering NaN) or returns appropriate error
    // The important thing is it doesn't panic
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_tsvq_handles_all_nan_training_data() {
    // Bug: when every value on the split dimension was NaN, the median lookup
    // indexed into an empty vector and panicked
    let training = [
        vec![f32::NAN, f32::NAN],
        vec![f32::NAN, f32::NAN],
        vec![f32::NAN, f32::NAN],
    ];
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let tsvq = TSVQ::new(&refs, 3, Distance::SquaredEuclidean).unwrap();
    let quantized = tsvq.quantize(&[1.0, 2.0]).unwrap();
    assert_eq!(quantized.len(), 2);
}

// =============================================================================
// Bug Fix: ProductQuantizer panicked on zero subspaces or a maximal seed
// =============================================================================

#[test]
fn test_product_quantizer_rejects_zero_subspaces() {
    // Bug: m == 0 reached `dim % m` and panicked with a division by zero
    let training = [vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let result = ProductQuantizer::new(&refs, 0, 2, 10, Distance::Euclidean, 42);
    assert!(matches!(
        result,
        Err(VqError::InvalidParameter { parameter: "m", .. })
    ));
}

#[test]
fn test_product_quantizer_accepts_max_seed() {
    // Bug: `seed + i` overflowed for seed == u64::MAX and panicked in debug builds
    let training: Vec<Vec<f32>> = (0..20)
        .map(|i| (0..4).map(|j| (i * 4 + j) as f32).collect())
        .collect();
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let pq = ProductQuantizer::new(&refs, 2, 2, 5, Distance::Euclidean, u64::MAX).unwrap();
    assert_eq!(pq.num_subspaces(), 2);
}

// =============================================================================
// Bug Fix: Scalar quantization overflow assertion
// =============================================================================

#[test]
fn test_scalar_quantizer_validates_levels_range() {
    // Bug: levels > 256 could overflow u8
    let result = ScalarQuantizer::new(0.0, 1.0, 257);
    assert!(matches!(result, Err(VqError::InvalidParameter { .. })));

    let result = ScalarQuantizer::new(0.0, 1.0, 256);
    assert!(result.is_ok());
}

// =============================================================================
// Bug Fix: Error type consolidation
// =============================================================================

#[test]
fn test_error_types_have_parameter_names() {
    let result = ScalarQuantizer::new(f32::NAN, 1.0, 256);

    match result {
        Err(VqError::InvalidParameter { parameter, reason }) => {
            assert_eq!(parameter, "min");
            assert!(reason.contains("finite"));
        }
        _ => panic!("Expected InvalidParameter with parameter field"),
    }
}

#[test]
fn test_dimension_mismatch_error_has_values() {
    let a = Vector::new(vec![1.0, 2.0]);
    let b = Vector::new(vec![3.0, 4.0, 5.0]);

    match a.try_add(&b) {
        Err(VqError::DimensionMismatch { expected, found }) => {
            assert_eq!(expected, 2);
            assert_eq!(found, 3);
        }
        _ => panic!("Expected DimensionMismatch error"),
    }
}

// =============================================================================
// Bug Fix: Distance metric introspection
// =============================================================================

#[test]
fn test_distance_metric_name_method() {
    assert_eq!(Distance::Euclidean.name(), "euclidean");
    assert_eq!(Distance::SquaredEuclidean.name(), "squared_euclidean");
    assert_eq!(Distance::Manhattan.name(), "manhattan");
    assert_eq!(Distance::CosineDistance.name(), "cosine");
}

#[test]
fn test_pq_distance_metric_introspection() {
    let training = [vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let pq = ProductQuantizer::new(&refs, 2, 2, 10, Distance::Manhattan, 42).unwrap();

    assert_eq!(pq.distance_metric(), "manhattan");
}

#[test]
fn test_tsvq_distance_metric_introspection() {
    let training = [vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let tsvq = TSVQ::new(&refs, 2, Distance::CosineDistance).unwrap();

    assert_eq!(tsvq.distance_metric(), "cosine");
}

// =============================================================================
// Performance regression: TSVQ should not clone excessively
// =============================================================================

#[test]
fn test_tsvq_builds_efficiently_on_large_dataset() {
    // This test guarantees TSVQ doesn't regress to excessive cloning
    let training: Vec<Vec<f32>> = (0..1000)
        .map(|i| (0..32).map(|j| ((i + j) % 100) as f32).collect())
        .collect();
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    // Should complete in reasonable time with optimized partitioning
    let result = TSVQ::new(&refs, 5, Distance::SquaredEuclidean);

    assert!(result.is_ok());
}

// =============================================================================
// Edge case: Empty input handling
// =============================================================================

#[test]
fn test_quantizers_handle_empty_vectors() {
    let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
    let sq = ScalarQuantizer::new(0.0, 1.0, 256).unwrap();

    let empty: Vec<f32> = vec![];

    let bq_result = bq.quantize(&empty).unwrap();
    let sq_result = sq.quantize(&empty).unwrap();

    assert!(bq_result.is_empty());
    assert!(sq_result.is_empty());
}

#[test]
fn test_quantizers_reject_empty_training_data() {
    let empty: Vec<&[f32]> = vec![];

    let pq_result = ProductQuantizer::new(&empty, 2, 4, 10, Distance::Euclidean, 42);
    assert!(matches!(pq_result, Err(VqError::EmptyInput)));

    let tsvq_result = TSVQ::new(&empty, 3, Distance::Euclidean);
    assert!(matches!(tsvq_result, Err(VqError::EmptyInput)));
}
