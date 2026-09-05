"""Tests for the batch quantize and dequantize methods."""

import numpy as np
import pytest

import pyvq


@pytest.fixture
def training():
    rng = np.random.default_rng(42)
    return rng.uniform(-1000.0, 1000.0, size=(40, 8)).astype(np.float32)


@pytest.fixture(
    params=["bq", "sq", "pq", "tsvq"],
)
def quantizer(request, training):
    if request.param == "bq":
        return pyvq.BinaryQuantizer(0.0, 0, 1)
    if request.param == "sq":
        return pyvq.ScalarQuantizer(-1000.0, 1000.0, 256)
    if request.param == "pq":
        return pyvq.ProductQuantizer(training, 2, 4, max_iters=10, seed=42)
    return pyvq.TSVQ(training, 3)


def test_batch_matches_single(quantizer, training):
    codes = quantizer.quantize_batch(training)
    assert codes.shape == training.shape
    for row, code in zip(training, codes):
        np.testing.assert_array_equal(code, quantizer.quantize(row))

    recon = quantizer.dequantize_batch(codes)
    assert recon.shape == training.shape
    assert recon.dtype == np.float32
    for code, r in zip(codes, recon):
        np.testing.assert_array_equal(r, quantizer.dequantize(code))


def test_batch_output_dtype(training):
    assert pyvq.BinaryQuantizer(0.0).quantize_batch(training).dtype == np.uint8
    assert pyvq.ScalarQuantizer(-1.0, 1.0).quantize_batch(training).dtype == np.uint8
    pq = pyvq.ProductQuantizer(training, 2, 4)
    assert pq.quantize_batch(training).dtype == np.float16
    tsvq = pyvq.TSVQ(training, 2)
    assert tsvq.quantize_batch(training).dtype == np.float16


def test_empty_batch(quantizer):
    empty = np.empty((0, 8), dtype=np.float32)
    codes = quantizer.quantize_batch(empty)
    assert codes.shape == (0, 8)
    recon = quantizer.dequantize_batch(codes)
    assert recon.shape == (0, 8)


def test_batch_dimension_mismatch(training):
    pq = pyvq.ProductQuantizer(training, 2, 4)
    with pytest.raises(ValueError):
        pq.quantize_batch(np.zeros((3, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        pq.dequantize_batch(np.zeros((3, 7), dtype=np.float16))


def test_batch_rejects_zero_columns(training):
    sq = pyvq.ScalarQuantizer(-1.0, 1.0)
    with pytest.raises(ValueError):
        sq.quantize_batch(np.zeros((3, 0), dtype=np.float32))


def test_batch_rejects_wrong_dtype(training):
    sq = pyvq.ScalarQuantizer(-1.0, 1.0)
    with pytest.raises(TypeError):
        sq.quantize_batch(training.astype(np.float64))


def test_batch_rejects_non_contiguous_view(training):
    # A strided view is not C-contiguous; a contiguous copy must be passed instead
    sq = pyvq.ScalarQuantizer(-1000.0, 1000.0)
    view = training.T[:, ::2].T
    with pytest.raises((TypeError, ValueError)):
        sq.quantize_batch(view)
    codes = sq.quantize_batch(np.ascontiguousarray(view))
    assert codes.shape == view.shape


def test_large_batch(training):
    rng = np.random.default_rng(1)
    data = rng.uniform(-1000.0, 1000.0, size=(5000, 16)).astype(np.float32)
    tsvq = pyvq.TSVQ(data, 4)
    a = tsvq.quantize_batch(data)
    b = tsvq.quantize_batch(data)
    np.testing.assert_array_equal(a, b)
    recon = tsvq.dequantize_batch(a)
    assert np.isfinite(recon).all()
