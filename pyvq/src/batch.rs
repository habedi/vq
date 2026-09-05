//! Helpers for converting between 2-D NumPy arrays and rows of vectors.

use numpy::ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Splits a C-contiguous 2-D array into one slice per row.
pub fn rows<'a, T: Element>(array: &'a PyReadonlyArray2<'_, T>) -> PyResult<Vec<&'a [T]>> {
    let shape = array.shape();
    let (n, dim) = (shape[0], shape[1]);
    if n == 0 {
        return Ok(Vec::new());
    }
    if dim == 0 {
        return Err(PyValueError::new_err(
            "batch input must have at least one column",
        ));
    }
    let flat = array.as_slice()?;
    Ok(flat.chunks(dim).take(n).collect())
}

/// Stacks equally sized rows into a 2-D array. `cols` is used when there are no rows.
pub fn to_array2<'py, T: Element>(
    py: Python<'py>,
    rows: Vec<Vec<T>>,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    let n = rows.len();
    let cols = rows.first().map_or(cols, |r| r.len());
    let flat: Vec<T> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n, cols), flat)
        .map(|a| a.into_pyarray(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
