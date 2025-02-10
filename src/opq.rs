use crate::distances::Distance;
use crate::utils::lbg_quantize;
use crate::vector::Vector;
use half::f16;
use nalgebra::DMatrix;

pub struct OptimizedProductQuantizer {
    rotation: DMatrix<f32>,
    codebooks: Vec<Vec<Vector<f32>>>,
    sub_dim: usize,
    m: usize,
    dim: usize,
}

impl OptimizedProductQuantizer {
    pub fn new(
        training_data: &[Vector<f32>],
        m: usize,
        k: usize,
        max_iters: usize,
        opq_iters: usize,
        seed: u64,
        distance: Distance,
    ) -> Self {
        assert!(!training_data.is_empty());
        let dim = training_data[0].len();
        assert!(dim >= m, "Dimension must be at least m");
        assert_eq!(dim % m, 0, "Dimension must be divisible by m");
        let sub_dim = dim / m;
        let n = training_data.len();

        let mut rotation = DMatrix::<f32>::identity(dim, dim);
        let mut rotated_data: Vec<Vector<f32>> = training_data.to_vec();
        let mut codebooks = Vec::with_capacity(m);

        for _ in 0..opq_iters {
            codebooks.clear();
            for i in 0..m {
                let sub_training: Vec<Vector<f32>> = rotated_data
                    .iter()
                    .map(|v| {
                        let start = i * sub_dim;
                        let end = start + sub_dim;
                        Vector::new(v.data[start..end].to_vec())
                    })
                    .collect();
                let sub_codebook = lbg_quantize(&sub_training, k, max_iters, seed + i as u64);
                codebooks.push(sub_codebook);
            }

            let mut reconstructions = Vec::with_capacity(n);
            for v in &rotated_data {
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
                reconstructions.push(Vector::new(rec));
            }

            let mut x_data: Vec<f32> = Vec::with_capacity(dim * n);
            let mut y_data: Vec<f32> = Vec::with_capacity(dim * n);
            for v in &rotated_data {
                x_data.extend(&v.data);
            }
            for v in &reconstructions {
                y_data.extend(&v.data);
            }
            let x_mat = DMatrix::from_column_slice(dim, n, &x_data);
            let y_mat = DMatrix::from_column_slice(dim, n, &y_data);
            let a: DMatrix<f32> = &y_mat * x_mat.transpose();
            let svd = a.svd(true, true);
            let u = svd.u.expect("SVD failed to produce U");
            let v_t = svd.v_t.expect("SVD failed to produce Vᵀ");
            rotation = v_t.transpose() * u.transpose();

            rotated_data = training_data
                .iter()
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
        }
    }

    pub fn quantize(&self, vector: &Vector<f32>, distance: Distance) -> Vector<f16> {
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
            let mut best_dist = distance.compute(sub_vector, &codebook[0].data);
            for (j, centroid) in codebook.iter().enumerate().skip(1) {
                let dist = distance.compute(sub_vector, &centroid.data);
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
