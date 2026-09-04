//! FFI bindings for hsdlib SIMD distance functions.
//!
//! This module is only compiled when the `simd` feature is enabled.

use std::os::raw::c_int;

/// Status codes returned by hsdlib functions.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HsdStatus {
    Success = 0,
    ErrNullPtr = -1,
    ErrInvalidInput = -3,
    ErrCpuNotSupported = -4,
    Failure = -99,
}

impl HsdStatus {
    #[inline]
    pub fn is_success(self) -> bool {
        self == HsdStatus::Success
    }
}

impl From<c_int> for HsdStatus {
    fn from(value: c_int) -> Self {
        match value {
            0 => HsdStatus::Success,
            -1 => HsdStatus::ErrNullPtr,
            -3 => HsdStatus::ErrInvalidInput,
            -4 => HsdStatus::ErrCpuNotSupported,
            _ => HsdStatus::Failure,
        }
    }
}

unsafe extern "C" {
    /// Compute squared Euclidean distance between two float vectors.
    pub fn hsd_dist_sqeuclidean_f32(
        a: *const f32,
        b: *const f32,
        n: usize,
        result: *mut f32,
    ) -> c_int;

    /// Compute Manhattan distance (L1) between two float vectors.
    pub fn hsd_dist_manhattan_f32(
        a: *const f32,
        b: *const f32,
        n: usize,
        result: *mut f32,
    ) -> c_int;

    /// Compute cosine similarity for float vectors.
    pub fn hsd_sim_cosine_f32(a: *const f32, b: *const f32, n: usize, result: *mut f32) -> c_int;

    /// Compute dot product for float vectors.
    pub fn hsd_sim_dot_f32(a: *const f32, b: *const f32, n: usize, result: *mut f32) -> c_int;

    /// Get a human-readable description of the selected backend.
    pub fn hsd_get_backend() -> *const std::os::raw::c_char;
}

/// Compute squared Euclidean distance using SIMD acceleration.
///
/// Returns `None` if the C function fails.
#[inline]
pub fn sqeuclidean_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut result: f32 = 0.0;
    // SAFETY: Both slices are guaranteed to have the same length (checked above).
    // The C function only reads from the pointers and writes to result.
    // Slices guarantee valid memory regions.
    let status: HsdStatus =
        unsafe { hsd_dist_sqeuclidean_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) }.into();
    if status.is_success() {
        Some(result)
    } else {
        None
    }
}

/// Compute Manhattan distance using SIMD acceleration.
///
/// Returns `None` if the C function fails.
#[inline]
pub fn manhattan_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut result: f32 = 0.0;
    // SAFETY: Both slices are guaranteed to have the same length (checked above).
    // The C function only reads from the pointers and writes to result.
    // Slices guarantee valid memory regions.
    let status: HsdStatus =
        unsafe { hsd_dist_manhattan_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) }.into();
    if status.is_success() {
        Some(result)
    } else {
        None
    }
}

/// Compute cosine similarity using SIMD acceleration.
///
/// Returns `None` if the C function fails.
#[inline]
pub fn cosine_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut result: f32 = 0.0;
    // SAFETY: Both slices are guaranteed to have the same length (checked above).
    // The C function only reads from the pointers and writes to result.
    // Slices guarantee valid memory regions.
    let status: HsdStatus =
        unsafe { hsd_sim_cosine_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) }.into();
    if status.is_success() {
        Some(result)
    } else {
        None
    }
}

/// Returns the name of the currently active SIMD backend.
///
/// This function queries hsdlib to determine which SIMD implementation is being used.
/// Possible return values include:
/// - `"Scalar (Auto)"` - Fallback scalar implementation
/// - `"AVX (Auto)"` / `"AVX2 (Auto)"` / `"AVX512F (Auto)"` - x86 SIMD
/// - `"NEON (Auto)"` / `"SVE (Auto)"` - ARM SIMD
///
/// # Example
///
/// ```ignore
/// use vq::get_simd_backend;
///
/// if let Some(backend) = get_simd_backend() {
///     println!("Using SIMD backend: {}", backend);
/// }
/// ```
pub fn get_simd_backend() -> String {
    // SAFETY: hsd_get_backend() returns a pointer to a static string literal
    // managed by the C library. The pointer is valid for the lifetime of the program.
    // We check for null pointer before dereferencing.
    unsafe {
        let ptr = hsd_get_backend();
        if ptr.is_null() {
            return "Unknown".to_string();
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_conversion() {
        assert_eq!(HsdStatus::from(0), HsdStatus::Success);
        assert_eq!(HsdStatus::from(-1), HsdStatus::ErrNullPtr);
        assert_eq!(HsdStatus::from(-3), HsdStatus::ErrInvalidInput);
        assert_eq!(HsdStatus::from(-4), HsdStatus::ErrCpuNotSupported);
        assert_eq!(HsdStatus::from(-99), HsdStatus::Failure);
        assert_eq!(HsdStatus::from(12345), HsdStatus::Failure);
        assert!(HsdStatus::Success.is_success());
        assert!(!HsdStatus::Failure.is_success());
    }

    #[test]
    fn test_kernels_reject_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(manhattan_f32(&a, &b).is_none());
        assert!(cosine_f32(&a, &b).is_none());
    }

    #[test]
    fn test_kernels_reject_non_finite_input() {
        let a = vec![1.0, f32::INFINITY];
        let b = vec![1.0, 2.0];
        assert!(sqeuclidean_f32(&a, &b).is_none());
        assert!(manhattan_f32(&a, &b).is_none());
        assert!(cosine_f32(&a, &b).is_none());
    }

    #[test]
    fn test_kernels_accept_empty_input() {
        let empty: Vec<f32> = vec![];
        assert_eq!(sqeuclidean_f32(&empty, &empty), Some(0.0));
        assert_eq!(manhattan_f32(&empty, &empty), Some(0.0));
    }

    #[test]
    fn test_backend_detection() {
        let backend = get_simd_backend();
        assert!(!backend.is_empty());
        assert!(!backend.contains("Unknown"));
    }

    #[test]
    fn test_sqeuclidean_f32_valid() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // Dist: (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        let result = sqeuclidean_f32(&a, &b).expect("SIMD sqeuclidean failed");
        assert!((result - 27.0).abs() < 1e-5);
    }

    #[test]
    fn test_sqeuclidean_f32_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        let result = sqeuclidean_f32(&a, &b);
        assert!(result.is_none());
    }

    #[test]
    fn test_manhattan_f32_valid() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 1.0];
        // Manhattan: |1-3| + |2-1| = 2 + 1 = 3
        let result = manhattan_f32(&a, &b).expect("SIMD manhattan failed");
        assert!((result - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_f32_valid() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        // Cosine similarity of orthogonal vectors is 0
        let result = cosine_f32(&a, &b).expect("SIMD cosine failed");
        assert!(result.abs() < 1e-5);

        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        // Cosine similarity of identical vectors is 1
        let result = cosine_f32(&a, &b).expect("SIMD cosine failed");
        assert!((result - 1.0).abs() < 1e-5);
    }
}
