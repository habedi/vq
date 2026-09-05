"""Differential tests for pyvq.

Each test compares pyvq against an independent NumPy reference computed in float64.
"""

import numpy as np
import pyvq
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# =============================================================================
# Strategies
# =============================================================================


def f32_array(size, min_value=-100.0, max_value=100.0):
    return arrays(
        dtype=np.float32,
        shape=size,
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
    )


def f32_pair(max_len=67, min_value=-100.0, max_value=100.0):
    return st.integers(min_value=1, max_value=max_len).flatmap(
        lambda n: st.tuples(f32_array(n, min_value, max_value), f32_array(n, min_value, max_value))
    )


def f32_matrix(min_rows, max_rows, dim, min_value, max_value):
    return st.integers(min_value=min_rows, max_value=max_rows).flatmap(
        lambda n: f32_array((n, dim), min_value, max_value)
    )


METRICS = st.sampled_from(["squared_euclidean", "euclidean", "manhattan", "cosine"])


# =============================================================================
# Reference implementations
# =============================================================================


def ref_distance(metric, a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if metric == "squared_euclidean":
        return float(np.sum((a - b) ** 2))
    if metric == "euclidean":
        return float(np.sqrt(np.sum((a - b) ** 2)))
    if metric == "manhattan":
        return float(np.sum(np.abs(a - b)))
    if metric == "cosine":
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.clip(1.0 - sim, 0.0, 2.0))
    raise ValueError(metric)


def ref_nearest(metric, query, candidates):
    dists = [ref_distance(metric, query, c) for c in candidates]
    return int(np.argmin(dists))


def close(actual, expected, rel=1e-4, abs_tol=1e-4):
    return abs(actual - expected) <= abs_tol + rel * abs(expected)


# =============================================================================
# Distances
# =============================================================================


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(metric=METRICS, pair=f32_pair())
def test_distance_matches_numpy_reference(metric, pair):
    a, b = pair
    assume(np.linalg.norm(a) > 1e-3 and np.linalg.norm(b) > 1e-3)
    actual = pyvq.Distance(metric).compute(a, b)
    expected = ref_distance(metric, a, b)
    assert close(actual, expected), f"{metric}: {actual} vs {expected}"


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    metric=METRICS,
    query=f32_array(8, 0.5, 50.0),
    candidates=f32_matrix(2, 12, 8, 0.5, 50.0),
)
def test_distance_preserves_nearest_neighbor(metric, query, candidates):
    dist = pyvq.Distance(metric)
    actual = int(np.argmin([dist.compute(query, c) for c in candidates]))
    expected = ref_nearest(metric, query, candidates)
    # A different winner is acceptable only on a tie within float32 noise
    assert close(
        ref_distance(metric, query, candidates[actual]),
        ref_distance(metric, query, candidates[expected]),
        rel=1e-5,
        abs_tol=1e-5,
    )


# =============================================================================
# Binary and scalar quantizers
# =============================================================================


@settings(max_examples=200)
@given(
    values=f32_array(st.integers(0, 64), -50.0, 50.0),
    threshold=st.floats(-50.0, 50.0, width=32),
    low=st.integers(0, 254),
)
def test_bq_matches_numpy_reference(values, threshold, low):
    high = low + 1
    bq = pyvq.BinaryQuantizer(threshold, low, high)
    actual = bq.quantize(values)
    expected = np.where(values.astype(np.float64) >= threshold, high, low).astype(np.uint8)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(bq.dequantize(actual), expected.astype(np.float32))


@settings(max_examples=200)
@given(
    values=f32_array(st.integers(0, 64), -200.0, 200.0),
    min_value=st.floats(-100.0, 0.0, width=32),
    max_value=st.floats(1.0, 100.0, width=32),
    levels=st.integers(2, 256),
)
def test_sq_matches_numpy_reference(values, min_value, max_value, levels):
    sq = pyvq.ScalarQuantizer(min_value, max_value, levels)
    codes = sq.quantize(values)

    step = (float(max_value) - float(min_value)) / (levels - 1)
    position = (np.clip(values.astype(np.float64), min_value, max_value) - min_value) / step
    expected = np.minimum(np.round(position), levels - 1).astype(np.int64)

    mismatch = codes.astype(np.int64) != expected
    # The library rounds in float32, so only half-way points may differ, by one level
    half_way = np.abs(np.abs(position - np.floor(position)) - 0.5) < 1e-3
    assert np.all(~mismatch | (half_way & (np.abs(codes.astype(np.int64) - expected) == 1)))

    recon = sq.dequantize(codes)
    expected_recon = float(min_value) + codes.astype(np.float64) * step
    assert np.allclose(recon, expected_recon, rtol=1e-5, atol=1e-4)


# =============================================================================
# Product quantizer
# =============================================================================


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    metric=METRICS,
    training=f32_matrix(16, 40, 8, 0.5, 20.0),
    queries=f32_matrix(1, 5, 8, 0.5, 20.0),
    m=st.sampled_from([1, 2, 4]),
    k=st.integers(2, 4),
)
def test_pq_selects_nearest_observed_centroid(metric, training, queries, m, k):
    pq = pyvq.ProductQuantizer(training, m, k, max_iters=20, distance=pyvq.Distance(metric), seed=7)
    sub_dim = 8 // m

    observed = [[] for _ in range(m)]
    for v in training:
        recon = pq.dequantize(pq.quantize(v))
        for i in range(m):
            chunk = recon[i * sub_dim : (i + 1) * sub_dim]
            if not any(np.array_equal(chunk, c) for c in observed[i]):
                observed[i].append(chunk)

    for q in queries:
        recon = pq.dequantize(pq.quantize(q))
        for i in range(m):
            sub_q = q[i * sub_dim : (i + 1) * sub_dim]
            chosen = ref_distance(metric, sub_q, recon[i * sub_dim : (i + 1) * sub_dim])
            best = min(ref_distance(metric, sub_q, c) for c in observed[i])
            assert chosen <= best * 1.02 + 1e-2, (
                f"{metric}: subspace {i} chose {chosen}, best {best}"
            )


@settings(max_examples=100)
@given(codes=f32_array(8, -100.0, 100.0))
def test_pq_dequantize_is_float16_widening(codes):
    training = np.arange(128, dtype=np.float32).reshape(16, 8)
    pq = pyvq.ProductQuantizer(training, 2, 2, max_iters=5, seed=1)
    half = codes.astype(np.float16)
    np.testing.assert_array_equal(pq.dequantize(half), half.astype(np.float32))


# =============================================================================
# Tree-structured quantizer
# =============================================================================


@settings(max_examples=60, deadline=None)
@given(training=f32_matrix(1, 30, 6, -50.0, 50.0), query=f32_array(6, -50.0, 50.0))
def test_tsvq_depth_zero_is_mean(training, query):
    tsvq = pyvq.TSVQ(training, 0)
    actual = tsvq.dequantize(tsvq.quantize(query))
    expected = training.astype(np.float64).mean(axis=0).astype(np.float16).astype(np.float32)
    assert np.allclose(actual, expected, rtol=2e-3, atol=1e-2)


@settings(max_examples=60, deadline=None)
@given(
    training=f32_matrix(1, 40, 4, -50.0, 50.0),
    queries=f32_matrix(1, 40, 4, -50.0, 50.0),
    depth=st.integers(0, 5),
)
def test_tsvq_leaf_count_is_bounded(training, queries, depth):
    tsvq = pyvq.TSVQ(training, depth, pyvq.Distance.squared_euclidean())
    leaves = {tsvq.quantize(v).tobytes() for v in np.concatenate([training, queries])}
    assert len(leaves) <= min(2**depth, len(training))
