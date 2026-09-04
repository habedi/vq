use crate::batch;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use vq::Quantizer;
use vq::bq::BinaryQuantizer as VqBinaryQuantizer;

/// Binary quantizer that maps values to 0 or 1 based on a threshold.
///
/// Example:
///     >>> import numpy as np
///     >>> bq = pyvq.BinaryQuantizer(threshold=0.5, low=0, high=1)
///     >>> data = np.array([0.3, 0.7, 0.5], dtype=np.float32)
///     >>> codes = bq.quantize(data)  # Returns np.array([0, 1, 1], dtype=np.uint8)
///     >>> reconstructed = bq.dequantize(codes)  # Returns np.array([0.0, 1.0, 1.0], dtype=np.float32)
#[pyclass]
pub struct BinaryQuantizer {
    quantizer: VqBinaryQuantizer,
}

#[pymethods]
impl BinaryQuantizer {
    /// Create a new BinaryQuantizer.
    ///
    /// Args:
    ///     threshold: Values >= threshold map to high, values < threshold map to low.
    ///     low: The output value for inputs below the threshold (0-255).
    ///     high: The output value for inputs at or above the threshold (0-255).
    ///
    /// Raises:
    ///     ValueError: If low >= high or threshold is NaN.
    #[new]
    #[pyo3(signature = (threshold, low=0, high=1))]
    fn new(threshold: f32, low: u8, high: u8) -> PyResult<Self> {
        VqBinaryQuantizer::new(threshold, low, high)
            .map(|q| BinaryQuantizer { quantizer: q })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Quantize a numpy array of floats to binary values.
    ///
    /// Args:
    ///     values: numpy array of floating-point values (float32).
    ///
    /// Returns:
    ///     numpy array of quantized values (uint8).
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

    /// Reconstruct approximate float values from quantized data.
    ///
    /// Args:
    ///     codes: numpy array of quantized values (uint8).
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

    /// The threshold value.
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
        VqBinaryQuantizer::from_bytes(data)
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
        VqBinaryQuantizer::load(path)
            .map(|q| Self { quantizer: q })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    #[getter]
    fn threshold(&self) -> f32 {
        self.quantizer.threshold()
    }

    /// The low quantization level.
    #[getter]
    fn low(&self) -> u8 {
        self.quantizer.low()
    }

    /// The high quantization level.
    #[getter]
    fn high(&self) -> u8 {
        self.quantizer.high()
    }

    fn __repr__(&self) -> String {
        format!(
            "BinaryQuantizer(threshold={}, low={}, high={})",
            self.quantizer.threshold(),
            self.quantizer.low(),
            self.quantizer.high()
        )
    }
}
