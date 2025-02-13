use crate::distances::Distance;
use crate::utils::lbg_quantize;
use crate::vector::Vector;
use half::f16;
use nalgebra::DMatrix;
use rayon::prelude::*;

/// An Optimized Product Quantizer (OPQ) that applies an optimal rotation
/// to the input data before performing product quantization.
///
/// The OPQ algorithm aims to reduce quantization error by first learning an optimal
/// rotation of the data, then partitioning the rotated data into `m` subspaces.
/// For each subspace, a codebook is learned (via the LBG algorithm) to quantize that subspace.
/// During quantization, the input vector is rotated and each subvector is quantized
/// by selecting the nearest codeword (using the stored distance metric).
/// The final quantized representation is the concatenation of the quantized subvectors,
/// converted to half-precision (`f16`).
pub struct OptimizedProductQuantizer {
    /// The learned rotation matrix (of size `dim x dim`).
    rotation: DMatrix<f32>,
    /// A vector of codebooks (one for each subspace). Each codebook is a vector of centroids.
    codebooks: Vec<Vec<Vector<f32>>>,
    /// The dimensionality of each subspace (i.e. `dim / m`).
    sub_dim: usize,
    /// The number of subspaces into which the rotated vector is partitioned.
    m: usize,
    /// The overall dimensionality of the input vectors.
    dim: usize,
    /// The distance metric used for selecting codewords during quantization.
    distance: Distance,
}

impl OptimizedProductQuantizer {
    /// Constructs a new `OptimizedProductQuantizer` from training data.
    ///
    /// # Parameters
    /// - `training_data`: A slice of training vectors (`Vector<f32>`) used for learning the quantizer.
    /// - `m`: The number of subspaces into which the rotated data will be partitioned.
    /// - `k`: The number of centroids (codewords) per subspace.
    /// - `max_iters`: The maximum number of iterations for the LBG quantization algorithm.
    /// - `opq_iters`: The number of OPQ iterations (i.e. the number of times the algorithm alternates
    ///    between codebook learning, reconstruction, rotation update, and re-rotation).
    /// - `distance`: The distance metric to use for comparing subvectors during codeword selection.
    /// - `seed`: A random seed for initializing LBG quantization (each subspace uses `seed + i`).
    ///
    /// # Panics
    /// Panics if:
    /// - `training_data` is empty.
    /// - The dimension of the training vectors is less than `m` or not divisible by `m`.
    pub fn new(
        training_data: &[Vector<f32>],
        m: usize,
        k: usize,
        max_iters: usize,
        opq_iters: usize,
        distance: Distance,
        seed: u64,
    ) -> Self {
        assert!(!training_data.is_empty());
        let dim = training_data[0].len();
        assert!(dim >= m, "Dimension must be at least m");
        assert_eq!(dim % m, 0, "Dimension must be divisible by m");
        let sub_dim = dim / m;
        let n = training_data.len();

        // Start with an identity rotation.
        let mut rotation = DMatrix::<f32>::identity(dim, dim);
        // Initially, no rotation is applied.
        let mut rotated_data: Vec<Vector<f32>> = training_data.to_vec();
        let mut codebooks = Vec::with_capacity(m);

        for _ in 0..opq_iters {
            // --- Codebook Learning ---
            // Learn a codebook for each subspace in parallel.
            codebooks = (0..m)
                .into_par_iter()
                .map(|i| {
                    let sub_training: Vec<Vector<f32>> = rotated_data
                        .iter()
                        .map(|v| {
                            let start = i * sub_dim;
                            let end = start + sub_dim;
                            Vector::new(v.data[start..end].to_vec())
                        })
                        .collect();
                    lbg_quantize(&sub_training, k, max_iters, seed + i as u64)
                })
                .collect();

            // --- Reconstruction ---
            // For each rotated vector, compute its reconstruction using the current codebooks.
            let reconstructions: Vec<Vector<f32>> = rotated_data
                .par_iter()
                .map(|v| {
                    let mut rec = Vec::with_capacity(dim);
                    for i in 0..m {
                        let start = i * sub_dim;
                        let end = start + sub_dim;
                        let sub_vector = &v.data[start..end];
                        let codebook = &codebooks[i];
                        let mut best_index = 0;
                        let mut best_dist = distance.compute(sub_vector, &codebook[0].data);
                        for (j, centroid) in codebook.iter().enumerate().skip(1) {
                            let dist = distance.compute(sub_vector, &centroid.data);
                            if dist < best_dist {
                                best_dist = dist;
                                best_index = j;
                            }
                        }
                        rec.extend_from_slice(&codebooks[i][best_index].data);
                    }
                    Vector::new(rec)
                })
                .collect();

            // --- Rotation Update ---
            // Prepare data matrices: x_mat for rotated_data, y_mat for reconstructions.
            let mut x_data: Vec<f32> = Vec::with_capacity(dim * n);
            let mut y_data: Vec<f32> = Vec::with_capacity(dim * n);
            // Flatten rotated_data and reconstructions.
            rotated_data.iter().for_each(|v| x_data.extend(&v.data));
            reconstructions.iter().for_each(|v| y_data.extend(&v.data));
            let x_mat = DMatrix::from_column_slice(dim, n, &x_data);
            let y_mat = DMatrix::from_column_slice(dim, n, &y_data);
            let a: DMatrix<f32> = &y_mat * x_mat.transpose();
            let svd = a.svd(true, true);
            let u = svd.u.expect("SVD failed to produce U");
            let v_t = svd.v_t.expect("SVD failed to produce Vᵀ");
            rotation = v_t.transpose() * u.transpose();

            // --- Re-rotate the Original Data ---
            rotated_data = training_data
                .par_iter()
                .map(|v| {
                    let x = DMatrix::from_column_slice(dim, 1, &v.data);
                    let y = &rotation * x;
                    let y_vec: Vec<f32> = y.column(0).iter().cloned().collect();
                    Vector::new(y_vec)
                })
                .collect();
        }

        Self {
            rotation,
            codebooks,
            sub_dim,
            m,
            dim,
            distance,
        }
    }

    /// Quantizes an input vector using the learned rotation and codebooks.
    ///
    /// The input vector is first rotated using the learned rotation matrix. It is then partitioned into `m`
    /// sub-vectors, each of dimension `sub_dim`. For each subspace, the nearest codeword is selected using the
    /// stored distance metric. The selected codewords (one from each subspace) are concatenated and converted
    /// to half-precision (`f16`), resulting in the final quantized representation.
    ///
    /// # Parameters
    /// - `vector`: The input vector (`Vector<f32>`) to be quantized.
    ///
    /// # Returns
    /// A quantized vector (`Vector<f16>`) representing the input vector.
    ///
    /// # Panics
    /// Panics if the input vector's dimension does not match the expected dimension.
    pub fn quantize(&self, vector: &Vector<f32>) -> Vector<f16> {
        assert_eq!(vector.len(), self.dim, "Input vector has wrong dimension");
        let x = DMatrix::from_column_slice(self.dim, 1, &vector.data);
        let y = &self.rotation * x;
        let y_vec: Vec<f32> = y.column(0).iter().cloned().collect();
        assert_eq!(y_vec.len(), self.sub_dim * self.m);
        let mut quantized_data = Vec::with_capacity(y_vec.len());
        for i in 0..self.m {
            let start = i * self.sub_dim;
            let end = start + self.sub_dim;
            let sub_vector = &y_vec[start..end];
            let codebook = &self.codebooks[i];
            let mut best_index = 0;
            let mut best_dist = self.distance.compute(sub_vector, &codebook[0].data);
            for (j, centroid) in codebook.iter().enumerate().skip(1) {
                let dist = self.distance.compute(sub_vector, &centroid.data);
                if dist < best_dist {
                    best_dist = dist;
                    best_index = j;
                }
            }
            for &val in &codebook[best_index].data {
                quantized_data.push(f16::from_f32(val));
            }
        }
        Vector::new(quantized_data)
    }
}
