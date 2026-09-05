"""Tests for the fit_transform training shortcuts."""

import numpy as np
import pytest

import pyvq


@pytest.fixture
def training():
    rng = np.random.default_rng(3)
    return rng.uniform(-10.0, 10.0, size=(50, 8)).astype(np.float32)


def test_pq_fit_transform_matches_constructor(training):
    pq, codes = pyvq.ProductQuantizer.fit_transform(training, 2, 4, max_iters=10, seed=5)
    direct = pyvq.ProductQuantizer(training, 2, 4, max_iters=10, seed=5)
    assert isinstance(pq, pyvq.ProductQuantizer)
    assert codes.shape == training.shape
    assert codes.dtype == np.float16
    np.testing.assert_array_equal(codes, direct.quantize_batch(training))
    np.testing.assert_array_equal(codes, pq.quantize_batch(training))


def test_tsvq_fit_transform_matches_constructor(training):
    tsvq, codes = pyvq.TSVQ.fit_transform(training, 3, pyvq.Distance.manhattan())
    direct = pyvq.TSVQ(training, 3, pyvq.Distance.manhattan())
    assert isinstance(tsvq, pyvq.TSVQ)
    assert codes.shape == training.shape
    np.testing.assert_array_equal(codes, direct.quantize_batch(training))


def test_fit_transform_propagates_errors(training):
    with pytest.raises(ValueError):
        pyvq.ProductQuantizer.fit_transform(training, 3, 4)
    with pytest.raises(ValueError):
        pyvq.TSVQ.fit_transform(np.empty((0, 8), dtype=np.float32), 3)
