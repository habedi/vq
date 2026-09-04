use crate::batch;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use vq::Quantizer;
use vq::sq::ScalarQuantizer as VqScalarQuantizer;

/// Scalar quantizer that uniformly quantizes values to discrete levels.
///
/// Example:
///     >>> import numpy as np
///     >>> sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)
///     >>> data = np.array([0.0, 0.5, -0.5], dtype=np.float32)
///     >>> codes = sq.quantize(data)  # Returns np.array([128, 191, 64], dtype=np.uint8)
///     >>> reconstructed = sq.dequantize(codes)  # Returns approximate original values
#[pyclass]
pub struct ScalarQuantizer {
    quantizer: VqScalarQuantizer,
}

#[pymethods]
impl ScalarQuantizer {
    /// Create a new ScalarQuantizer.
    ///
    /// Args:
    ///     min: Minimum value in the quantization range.
    ///     max: Maximum value in the quantization range.
    ///     levels: Number of quantization levels (2-256).
    ///
    /// Raises:
    ///     ValueError: If max <= min, levels < 2 or > 256, or values are NaN/Infinity.
    #[new]
    #[pyo3(signature = (min, max, levels=256))]
    fn new(min: f32, max: f32, levels: usize) -> PyResult<Self> {
        VqScalarQuantizer::new(min, max, levels)
            .map(|q| ScalarQuantizer { quantizer: q })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Quantize a numpy array of floats to discrete levels.
    ///
    /// Args:
    ///     values: numpy array of floating-point values (float32).
    ///
    /// Returns:
    ///     numpy array of quantized level indices (uint8).
    fn quantize<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let input = values.as_slice()?;
        let result = self
            .quantizer
            .quantize(input)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Reconstruct approximate float values from quantized levels.
    ///
    /// Args:
    ///     codes: numpy array of quantized level indices (uint8).
    ///
    /// Returns:
    ///     numpy array of reconstructed float values (float32).
    fn dequantize<'py>(
        &self,
        py: Python<'py>,
        codes: PyReadonlyArray1<u8>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input = codes.as_slice()?.to_vec();
        let result = self
            .quantizer
            .dequantize(&input)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// The minimum value in the quantization range.
    /// Quantize every row of a 2-D array at once.
    fn quantize_batch<'py>(
        &self,
        py: Python<'py>,
        vectors: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let rows = batch::rows(&vectors)?;
        let result = py
            .detach(|| self.quantizer.quantize_batch(&rows))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        batch::to_array2(py, result, vectors.shape()[1])
    }

    /// Reconstruct every row of a 2-D array of codes at once.
    fn dequantize_batch<'py>(
        &self,
        py: Python<'py>,
        codes: PyReadonlyArray2<u8>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let rows: Vec<Vec<u8>> = batch::rows(&codes)?
            .into_iter()
            .map(|r| r.to_vec())
            .collect();
        let result = py
            .detach(|| self.quantizer.dequantize_batch(&rows))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        batch::to_array2(py, result, codes.shape()[1])
    }

    /// Encode the quantizer into bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self
            .quantizer
            .to_bytes()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Restore a quantizer from bytes produced by `to_bytes`.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        VqScalarQuantizer::from_bytes(data)
            .map(|q| Self { quantizer: q })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Write the quantizer to a file.
    fn save(&self, path: std::path::PathBuf) -> PyResult<()> {
        self.quantizer
            .save(path)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Read a quantizer from a file written by `save`.
    #[staticmethod]
    fn load(path: std::path::PathBuf) -> PyResult<Self> {
        VqScalarQuantizer::load(path)
            .map(|q| Self { quantizer: q })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    #[getter]
    fn min(&self) -> f32 {
        self.quantizer.min()
    }

    /// The maximum value in the quantization range.
    #[getter]
    fn max(&self) -> f32 {
        self.quantizer.max()
    }

    /// The number of quantization levels.
    #[getter]
    fn levels(&self) -> usize {
        self.quantizer.levels()
    }

    /// The step size between quantization levels.
    #[getter]
    fn step(&self) -> f32 {
        self.quantizer.step()
    }

    fn __repr__(&self) -> String {
        format!(
            "ScalarQuantizer(min={}, max={}, levels={})",
            self.quantizer.min(),
            self.quantizer.max(),
            self.quantizer.levels()
        )
    }
}
