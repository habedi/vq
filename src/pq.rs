use crate::distances::Distance;
use crate::utils::lbg_quantize;
use crate::vector::Vector;
use half::f16;

pub struct ProductQuantizer {
    codebooks: Vec<Vec<Vector<f32>>>,
    sub_dim: usize,
    m: usize,
}

impl ProductQuantizer {
    pub fn new(
        training_data: &[Vector<f32>],
        m: usize,
        k: usize,
        max_iters: usize,
        seed: u64,
    ) -> Self {
        assert!(!training_data.is_empty());
        let n = training_data[0].len();
        assert!(n >= m);
        assert_eq!(n % m, 0);
        let sub_dim = n / m;
        let mut codebooks = Vec::with_capacity(m);
        for i in 0..m {
            let sub_training: Vec<Vector<f32>> = training_data
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
        Self {
            codebooks,
            sub_dim,
            m,
        }
    }

    pub fn quantize(&self, vector: &Vector<f32>, distance: Distance) -> Vector<f16> {
        let n = vector.len();
        assert_eq!(n, self.sub_dim * self.m);
        let mut quantized_data = Vec::with_capacity(n);
        for i in 0..self.m {
            let start = i * self.sub_dim;
            let end = start + self.sub_dim;
            let sub_vector = &vector.data[start..end];
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
