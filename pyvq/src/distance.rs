use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vq::core::distance::Distance as VqDistance;

/// A Python class for computing vector distances.
///
/// Create using static methods like `Distance.euclidean()`.
///
/// Example:
///     >>> import numpy as np
///     >>> dist = pyvq.Distance.euclidean()
///     >>> a = np.array([1.0, 2.0], dtype=np.float32)
///     >>> b = np.array([3.0, 4.0], dtype=np.float32)
///     >>> dist.compute(a, b)
///     2.8284...
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Distance {
    pub(crate) metric: VqDistance,
}

#[pymethods]
impl Distance {
    /// Create a Distance with a specified metric name.
    ///
    /// Args:
    ///     metric: One of "euclidean", "squared_euclidean", "cosine", "manhattan"
    ///
    /// Raises:
    ///     ValueError: If the metric name is invalid.
    #[new]
    fn new(metric: &str) -> PyResult<Self> {
        let m = match metric.to_lowercase().as_str() {
            "euclidean" => VqDistance::Euclidean,
            "squaredeuclidean" | "squared_euclidean" => VqDistance::SquaredEuclidean,
            "cosine" | "cosine_distance" => VqDistance::CosineDistance,
            "manhattan" => VqDistance::Manhattan,
            _ => {
                return Err(PyValueError::new_err(
                    "Invalid distance metric. Choose from: euclidean, squared_euclidean, cosine, manhattan",
                ))
            }
        };
        Ok(Distance { metric: m })
    }

    /// Create a Euclidean distance metric.
    #[staticmethod]
    fn euclidean() -> Self {
        Distance {
            metric: VqDistance::Euclidean,
        }
    }

    /// Create a Squared Euclidean distance metric.
    #[staticmethod]
    fn squared_euclidean() -> Self {
        Distance {
            metric: VqDistance::SquaredEuclidean,
        }
    }

    /// Create a Manhattan (L1) distance metric.
    #[staticmethod]
    fn manhattan() -> Self {
        Distance {
            metric: VqDistance::Manhattan,
        }
    }

    /// Create a Cosine distance metric (1 - cosine similarity).
    #[staticmethod]
    fn cosine() -> Self {
        Distance {
            metric: VqDistance::CosineDistance,
        }
    }

    /// Compute the distance between two vectors.
    ///
    /// Args:
    ///     a: First vector as numpy array (float32).
    ///     b: Second vector as numpy array (float32).
    ///
    /// Returns:
    ///     The computed distance as a float.
    ///
    /// Raises:
    ///     ValueError: If vectors have different lengths.
    fn compute(&self, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
        let a_slice = a.as_slice()?;
        let b_slice = b.as_slice()?;
        self.metric
            .compute(a_slice, b_slice)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        let name = match self.metric {
            VqDistance::Euclidean => "euclidean",
            VqDistance::SquaredEuclidean => "squared_euclidean",
            VqDistance::Manhattan => "manhattan",
            VqDistance::CosineDistance => "cosine",
        };
        format!("Distance('{}')", name)
    }
}
