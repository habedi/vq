"""Tests for saving and loading quantizers."""

import numpy as np
import pytest

import pyvq


@pytest.fixture
def training():
    rng = np.random.default_rng(7)
    return rng.uniform(-100.0, 100.0, size=(40, 8)).astype(np.float32)


@pytest.fixture(params=["bq", "sq", "pq", "tsvq"])
def quantizer(request, training):
    if request.param == "bq":
        return pyvq.BinaryQuantizer(0.5, 2, 7)
    if request.param == "sq":
        return pyvq.ScalarQuantizer(-100.0, 100.0, 64)
    if request.param == "pq":
        return pyvq.ProductQuantizer(training, 2, 4, distance=pyvq.Distance.cosine(), seed=3)
    return pyvq.TSVQ(training, 3, pyvq.Distance.manhattan())


def test_bytes_roundtrip_preserves_outputs(quantizer, training):
    blob = quantizer.to_bytes()
    assert isinstance(blob, bytes) and len(blob) > 0
    restored = type(quantizer).from_bytes(blob)
    assert repr(restored) == repr(quantizer)
    np.testing.assert_array_equal(
        restored.quantize_batch(training), quantizer.quantize_batch(training)
    )


def test_save_and_load(quantizer, training, tmp_path):
    path = tmp_path / "model.vq"
    quantizer.save(path)
    assert path.stat().st_size > 0
    loaded = type(quantizer).load(str(path))
    np.testing.assert_array_equal(
        loaded.quantize_batch(training), quantizer.quantize_batch(training)
    )


def test_from_bytes_rejects_garbage():
    with pytest.raises(ValueError):
        pyvq.ProductQuantizer.from_bytes(b"")
    with pytest.raises(ValueError):
        pyvq.TSVQ.from_bytes(b"\xff" * 16)


def test_load_missing_file(tmp_path):
    with pytest.raises(OSError):
        pyvq.ScalarQuantizer.load(tmp_path / "missing.vq")


def test_bytes_are_deterministic(training):
    a = pyvq.TSVQ(training, 3).to_bytes()
    b = pyvq.TSVQ(training, 3).to_bytes()
    assert a == b
