mod batch;
mod bq;
mod distance;
mod pq;
mod sq;
mod tsvq;

use pyo3::prelude::*;

/// Get the name of the currently active SIMD backend.
///
/// Returns a string describing which SIMD implementation is being used
/// for distance computations, such as "AVX2 (Auto)" or "NEON (Auto)".
///
/// Example:
///     >>> import pyvq
///     >>> backend = pyvq.get_simd_backend()
///     >>> print(backend)  # e.g., "AVX2 (Auto)"
#[pyfunction]
fn get_simd_backend() -> String {
    vq::get_simd_backend()
}

/// Python bindings for the Vq vector quantization library.
///
/// This module provides efficient implementations of various vector quantization
/// algorithms, including Binary Quantization (BQ), Scalar Quantization (SQ),
/// Product Quantization (PQ), and Tree-Structured Vector Quantization (TSVQ).
///
/// Example:
///     >>> import pyvq
///     >>>
///     >>> # Binary Quantization
///     >>> bq = pyvq.BinaryQuantizer(threshold=0.5)
///     >>> codes = bq.quantize([0.3, 0.7, 0.5])
///     >>> print(codes)  # [0, 1, 1]
///     >>>
///     >>> # Distance computation
///     >>> dist = pyvq.Distance.euclidean()
///     >>> result = dist.compute([1.0, 2.0], [3.0, 4.0])
#[pymodule]
fn pyvq(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_simd_backend, m)?)?;
    m.add_class::<distance::Distance>()?;
    m.add_class::<bq::BinaryQuantizer>()?;
    m.add_class::<sq::ScalarQuantizer>()?;
    m.add_class::<pq::ProductQuantizer>()?;
    m.add_class::<tsvq::TSVQ>()?;
    Ok(())
}
