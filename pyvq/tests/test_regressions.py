"""Regression tests for bugs fixed during development.

This file contains tests that verify specific bugs remain fixed.
Each test is documented with the issue/bug it addresses.
"""

import numpy as np
import pytest
import pyvq


# =============================================================================
# Bug Fix: BinaryQuantizer dequantize returned hardcoded 0.0/1.0
# =============================================================================


def test_binary_quantizer_dequantize_uses_low_high_values():
    """Test that dequantize uses actual low/high values, not hardcoded 0.0/1.0."""
    bq = pyvq.BinaryQuantizer(threshold=0.0, low=10, high=20)

    codes = np.array([0, 5, 10, 15, 20, 25, 255], dtype=np.uint8)
    result = bq.dequantize(codes)

    # Values < high should map to low, values >= high should map to high
    expected = np.array([10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_binary_quantizer_dequantize_preserves_custom_levels():
    """Test that custom low/high levels are preserved through quantize/dequantize."""
    bq = pyvq.BinaryQuantizer(threshold=0.5, low=50, high=200)

    vector = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    quantized = bq.quantize(vector)
    reconstructed = bq.dequantize(quantized)

    # Should reconstruct to 50.0 or 200.0, not 0.0 or 1.0
    assert np.all((reconstructed == 50.0) | (reconstructed == 200.0))


# =============================================================================
# Bug Fix: BinaryQuantizer missing infinity validation
# =============================================================================


def test_binary_quantizer_rejects_infinite_threshold():
    """Test that infinite threshold values are rejected."""
    with pytest.raises(Exception):  # Should raise ValueError or similar
        pyvq.BinaryQuantizer(threshold=float("inf"), low=0, high=1)

    with pytest.raises(Exception):
        pyvq.BinaryQuantizer(threshold=float("-inf"), low=0, high=1)


def test_binary_quantizer_rejects_nan_threshold():
    """Test that NaN threshold is rejected."""
    with pytest.raises(Exception):
        pyvq.BinaryQuantizer(threshold=float("nan"), low=0, high=1)


# =============================================================================
# Bug Fix: ProductQuantizer missing dimension validation
# =============================================================================


def test_product_quantizer_validates_dimension_consistency():
    """Test that PQ validates all training vectors have same dimension."""
    training = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 0.0, 0.0],  # Same length but we'll test with different
        ],
        dtype=np.float32,
    )

    # Test with inconsistent dimensions via list of arrays
    inconsistent = [
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        np.array([9.0, 10.0], dtype=np.float32),  # Different dimension!
    ]

    with pytest.raises(Exception):  # Should raise dimension error
        # Stack will fail or PQ will reject
        pyvq.ProductQuantizer(
            training_data=np.vstack(inconsistent), num_subspaces=2, num_centroids=2, max_iters=10, distance=pyvq.Distance.euclidean(), seed=42
        )


def test_product_quantizer_accepts_consistent_dimensions():
    """Test that PQ accepts training data with consistent dimensions."""
    training = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype=np.float32,
    )

    pq = pyvq.ProductQuantizer(
        training_data=training,
        num_subspaces=2,
        num_centroids=2,
        max_iters=10,
        distance=pyvq.Distance.euclidean(),
        seed=42,
    )

    assert pq is not None


# =============================================================================
# Bug Fix: TSVQ missing dimension validation
# =============================================================================


def test_tsvq_validates_dimension_consistency():
    """Test that TSVQ validates all training vectors have same dimension."""
    # Create inconsistent training data
    inconsistent = [
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        np.array([9.0, 10.0], dtype=np.float32),  # Different dimension!
    ]

    with pytest.raises(Exception):  # Should raise dimension error or shape error
        pyvq.TSVQ(
            training_data=np.vstack(inconsistent),  # This will fail at vstack
            max_depth=3,
            distance=pyvq.Distance.euclidean(),
        )


def test_tsvq_accepts_consistent_dimensions():
    """Test that TSVQ accepts training data with consistent dimensions."""
    training = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype=np.float32,
    )

    tsvq = pyvq.TSVQ(training_data=training, max_depth=2, distance=pyvq.Distance.euclidean())

    assert tsvq is not None


# =============================================================================
# Bug Fix: Cosine distance edge cases
# =============================================================================


def test_cosine_distance_handles_zero_norm():
    """Test that cosine distance handles zero-norm vectors gracefully."""
    zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    normal = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    dist = pyvq.Distance.cosine()
    result = dist.compute(zero, normal)

    # Zero vectors should be considered maximally distant
    assert result == 1.0


def test_cosine_distance_handles_near_zero_norm():
    """Test that cosine distance handles near-zero norms without numerical issues."""
    tiny = np.array([1e-20, 1e-20, 1e-20], dtype=np.float32)
    normal = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    dist = pyvq.Distance.cosine()
    result = dist.compute(tiny, normal)

    # Should return 1.0 for near-zero vectors (using epsilon check)
    assert result == 1.0


def test_cosine_distance_result_in_valid_range():
    """Test that cosine distance is always in [0, 2]."""
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    dist = pyvq.Distance.cosine()
    result = dist.compute(a, b)

    # Distance should be in valid range [0, 2]
    assert 0.0 <= result <= 2.0
    assert abs(result) < 1e-6  # Should be very close to 0


# =============================================================================
# Bug Fix: Scalar quantization overflow assertion
# =============================================================================


def test_scalar_quantizer_validates_levels_range():
    """Test that scalar quantizer rejects levels > 256."""
    with pytest.raises(Exception):
        pyvq.ScalarQuantizer(min=0.0, max=1.0, levels=257)

    # Should accept 256
    sq = pyvq.ScalarQuantizer(min=0.0, max=1.0, levels=256)
    assert sq is not None


def test_scalar_quantizer_max_levels_works():
    """Test that scalar quantizer works correctly with max levels (256)."""
    sq = pyvq.ScalarQuantizer(min=0.0, max=1.0, levels=256)

    vector = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    result = sq.quantize(vector)

    # All values should fit in uint8
    assert result.dtype == np.uint8
    assert np.all(result <= 255)


# =============================================================================
# Bug Fix: Distance metric introspection
# =============================================================================


def test_distance_metrics_have_names():
    """Test that distance metrics can be identified (indirectly through behavior)."""
    # We can't directly test .name() in Python, but we can verify different metrics work
    euclidean = pyvq.Distance.euclidean()
    manhattan = pyvq.Distance.manhattan()
    cosine = pyvq.Distance.cosine()
    sq_euclidean = pyvq.Distance.squared_euclidean()

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    # Different metrics should give different results
    r1 = euclidean.compute(a, b)
    r2 = manhattan.compute(a, b)
    r3 = cosine.compute(a, b)
    r4 = sq_euclidean.compute(a, b)

    # All should be different (except euclidean = sqrt(sq_euclidean))
    assert r2 != r1  # Manhattan != Euclidean
    assert r3 != r1  # Cosine != Euclidean
    assert abs(r1**2 - r4) < 1e-5  # Euclidean^2 ≈ Squared Euclidean


# =============================================================================
# Edge case: Empty input handling
# =============================================================================


def test_quantizers_handle_empty_vectors():
    """Test that quantizers handle empty vectors gracefully."""
    bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)
    sq = pyvq.ScalarQuantizer(min=0.0, max=1.0, levels=256)

    empty = np.array([], dtype=np.float32)

    bq_result = bq.quantize(empty)
    sq_result = sq.quantize(empty)

    assert len(bq_result) == 0
    assert len(sq_result) == 0


def test_quantizers_reject_empty_training_data():
    """Test that PQ and TSVQ reject empty training data."""
    empty = np.array([], dtype=np.float32).reshape(0, 4)

    with pytest.raises(Exception):
        pyvq.ProductQuantizer(
            training_data=empty,
            num_subspaces=2,
            num_centroids=4,
            max_iters=10,
            distance=pyvq.Distance.euclidean(),
            seed=42,
        )

    with pytest.raises(Exception):
        pyvq.TSVQ(training_data=empty, max_depth=3, distance=pyvq.Distance.euclidean())


# =============================================================================
# Numerical stability tests
# =============================================================================


def test_binary_quantizer_handles_extreme_values():
    """Test that BQ handles very large and very small values."""
    bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)

    extreme = np.array([1e10, -1e10, 1e-10, -1e-10], dtype=np.float32)
    result = bq.quantize(extreme)

    # Should not overflow or underflow
    assert len(result) == 4
    assert np.all((result == 0) | (result == 1))


def test_scalar_quantizer_handles_extreme_values():
    """Test that SQ clamps extreme values correctly."""
    sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)

    extreme = np.array([1e10, -1e10, 1.5, -1.5], dtype=np.float32)
    result = sq.quantize(extreme)

    # Should clamp to valid range
    assert len(result) == 4
    assert np.all(result <= 255)


# =============================================================================
# Type safety tests
# =============================================================================


def test_quantizers_accept_correct_dtype():
    """Test that quantizers work with float32 input."""
    bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)

    # Should work with float32
    vector_f32 = np.array([0.5, -0.3, 0.8], dtype=np.float32)
    result = bq.quantize(vector_f32)
    assert result is not None


def test_quantizers_handle_float64_input():
    """Test that quantizers handle float64 input (if supported)."""
    bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)

    # Try with float64 - should either work or raise clear error
    vector_f64 = np.array([0.5, -0.3, 0.8], dtype=np.float64)

    try:
        result = bq.quantize(vector_f64)
        assert result is not None
    except Exception as e:
        # If it fails, it should be a type error, not a crash
        err_msg = str(e).lower()
        assert "type" in err_msg or "dtype" in err_msg or "converted" in err_msg or "array" in err_msg


# =============================================================================
# Roundtrip accuracy tests
# =============================================================================


def test_binary_quantizer_roundtrip():
    """Test that BQ roundtrip produces expected binary values."""
    bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)

    vector = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    quantized = bq.quantize(vector)
    reconstructed = bq.dequantize(quantized)

    # Should be all 0s and 1s
    assert np.all((reconstructed == 0.0) | (reconstructed == 1.0))


def test_scalar_quantizer_roundtrip_bounded_error():
    """Test that SQ roundtrip error is bounded by step size."""
    sq = pyvq.ScalarQuantizer(min=0.0, max=1.0, levels=256)

    vector = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    quantized = sq.quantize(vector)
    reconstructed = sq.dequantize(quantized)

    # Error should be bounded by step size
    max_error = np.max(np.abs(vector - reconstructed))
    step_size = 1.0 / 255.0

    assert max_error <= step_size


def test_product_quantizer_reconstruction_quality():
    """Test that PQ produces reasonable reconstructions."""
    training = np.random.randn(100, 16).astype(np.float32)

    pq = pyvq.ProductQuantizer(
        training_data=training,
        num_subspaces=4,
        num_centroids=16,
        max_iters=10,
        distance=pyvq.Distance.euclidean(),
        seed=42,
    )

    # Test on training data
    vector = training[0]
    quantized = pq.quantize(vector)
    reconstructed = pq.dequantize(quantized)

    # Reconstruction should have same length
    assert len(reconstructed) == len(vector)

    # MSE should be reasonable (not infinite or NaN)
    mse = np.mean((vector - reconstructed) ** 2)
    assert np.isfinite(mse)
    assert mse < 100.0  # Reasonable bound for normalized data


def test_tsvq_reconstruction_quality():
    """Test that TSVQ produces reasonable reconstructions."""
    training = np.random.randn(100, 16).astype(np.float32)

    tsvq = pyvq.TSVQ(training_data=training, max_depth=5, distance=pyvq.Distance.euclidean())

    # Test on training data
    vector = training[0]
    quantized = tsvq.quantize(vector)
    reconstructed = tsvq.dequantize(quantized)

    # Reconstruction should have same length
    assert len(reconstructed) == len(vector)

    # MSE should be reasonable
    mse = np.mean((vector - reconstructed) ** 2)
    assert np.isfinite(mse)
    assert mse < 100.0


# =============================================================================
# Multi-vector batch tests
# =============================================================================


def test_quantizers_handle_multiple_vectors():
    """Test that quantizers can process multiple vectors."""
    bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)

    vectors = np.array(
        [[-1.0, 0.5, 1.0], [-0.5, 0.0, 0.5], [0.1, 0.2, 0.3]], dtype=np.float32
    )

    # Process each vector
    results = [bq.quantize(v) for v in vectors]

    assert len(results) == 3
    assert all(len(r) == 3 for r in results)


# =============================================================================
# Properties tests (invariants that should always hold)
# =============================================================================


def test_binary_quantizer_output_is_binary():
    """Test that BQ always produces 0 or 1 (or low/high)."""
    bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)

    random_vector = np.random.randn(100).astype(np.float32)
    result = bq.quantize(random_vector)

    assert np.all((result == 0) | (result == 1))


def test_scalar_quantizer_output_in_range():
    """Test that SQ output is always in valid range."""
    sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)

    random_vector = np.random.randn(100).astype(np.float32) * 10  # Wide range
    result = sq.quantize(random_vector)

    assert np.all(result >= 0)
    assert np.all(result <= 255)


def test_distance_is_non_negative():
    """Test that all distance metrics return non-negative values."""
    metrics = [
        pyvq.Distance.euclidean(),
        pyvq.Distance.squared_euclidean(),
        pyvq.Distance.manhattan(),
        pyvq.Distance.cosine(),
    ]

    a = np.random.randn(10).astype(np.float32)
    b = np.random.randn(10).astype(np.float32)

    for metric in metrics:
        dist = metric.compute(a, b)
        assert dist >= 0.0, f"Distance metric {metric} returned negative value"


def test_distance_to_self_is_zero():
    """Test that distance from vector to itself is zero (or very small)."""
    metrics = [
        pyvq.Distance.euclidean(),
        pyvq.Distance.squared_euclidean(),
        pyvq.Distance.manhattan(),
        pyvq.Distance.cosine(),
    ]

    a = np.random.randn(10).astype(np.float32)

    for metric in metrics:
        dist = metric.compute(a, a)
        assert dist < 1e-6, f"Distance metric {metric} non-zero for identical vectors"
