use crate::batch;
use half::f16;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use vq::tsvq::TSVQ as VqTSVQ;
use vq::{Distance as VqDistance, Quantizer};

use crate::distance::Distance;

/// Tree-structured vector quantizer using hierarchical clustering.
///
/// TSVQ builds a binary tree where each node represents a cluster centroid.
/// Vectors are quantized by traversing the tree to find the nearest leaf node.
///
/// Example:
///     >>> import numpy as np
///     >>> training = np.random.rand(100, 8).astype(np.float32)
///     >>> tsvq = pyvq.TSVQ(
///     ...     training_data=training,
///     ...     max_depth=5,
///     ...     distance=pyvq.Distance.euclidean()
///     ... )
///     >>> codes = tsvq.quantize(training[0])  # Returns float16 array
///     >>> reconstructed = tsvq.dequantize(codes)
#[pyclass]
pub struct TSVQ {
    quantizer: VqTSVQ,
}

#[pymethods]
impl TSVQ {
    /// Create a new Tree-Structured Vector Quantizer.
    ///
    /// Args:
    ///     training_data: 2D numpy array of training vectors (float32), shape (n_samples, dim).
    ///     max_depth: Maximum depth of the tree.
    ///     distance: Distance metric to use.
    ///
    /// Raises:
    ///     ValueError: If training data is empty.
    #[new]
    #[pyo3(signature = (training_data, max_depth, distance=None))]
    fn new(
        training_data: PyReadonlyArray2<f32>,
        max_depth: usize,
        distance: Option<Distance>,
    ) -> PyResult<Self> {
        let shape = training_data.shape();
        if shape[0] == 0 {
            return Err(PyValueError::new_err("Training data cannot be empty"));
        }

        // Convert 2D numpy array to Vec<Vec<f32>>
        let training_vec: Vec<Vec<f32>> = (0..shape[0])
            .map(|i| {
                (0..shape[1])
                    .map(|j| *training_data.get([i, j]).unwrap())
                    .collect()
            })
            .collect();

        let training_refs: Vec<&[f32]> = training_vec.iter().map(|v| v.as_slice()).collect();
        let dist = distance.map(|d| d.metric).unwrap_or(VqDistance::Euclidean);

        VqTSVQ::new(&training_refs, max_depth, dist)
            .map(|q| TSVQ { quantizer: q })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Quantize a vector.
    ///
    /// Args:
    ///     vector: Input vector as numpy array (float32).
    ///
    /// Returns:
    ///     Quantized representation (leaf centroid) as numpy array (float16).
    /// Build a tree and return it together with the codes of the training data.
    #[staticmethod]
    #[pyo3(signature = (training_data, max_depth, distance=None))]
    fn fit_transform<'py>(
        py: Python<'py>,
        training_data: PyReadonlyArray2<f32>,
        max_depth: usize,
        distance: Option<Distance>,
    ) -> PyResult<(Self, Bound<'py, PyArray2<f16>>)> {
        let quantizer = Self::new(training_data.clone(), max_depth, distance)?;
        let codes = quantizer.quantize_batch(py, training_data)?;
        Ok((quantizer, codes))
    }

    fn quantize<'py>(
        &self,
        py: Python<'py>,
        vector: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f16>>> {
        let input = vector.as_slice()?;
        let result = self
            .quantizer
            .quantize(input)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Reconstruct a vector from its quantized representation.
    ///
    /// Args:
    ///     codes: Quantized representation as numpy array (float16).
    ///
    /// Returns:
    ///     Reconstructed vector as numpy array (float32).
    fn dequantize<'py>(
        &self,
        py: Python<'py>,
        codes: PyReadonlyArray1<f16>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input = codes.as_slice()?.to_vec();
        let result = self
            .quantizer
            .dequantize(&input)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// The expected input vector dimension.
    /// Quantize every row of a 2-D array at once.
    fn quantize_batch<'py>(
        &self,
        py: Python<'py>,
        vectors: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f16>>> {
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
        codes: PyReadonlyArray2<f16>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let rows: Vec<Vec<f16>> = batch::rows(&codes)?
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
        VqTSVQ::from_bytes(data)
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
        VqTSVQ::load(path)
            .map(|q| Self { quantizer: q })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    #[getter]
    fn dim(&self) -> usize {
        self.quantizer.dim()
    }

    fn __repr__(&self) -> String {
        format!("TSVQ(dim={})", self.quantizer.dim())
    }
}
