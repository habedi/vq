use crate::distances::Distance;
use crate::utils::lbg_quantize;
use crate::vector::Vector;
use half::f16;

pub struct ResidualQuantizer {
    stages: usize,
    codebooks: Vec<Vec<Vector<f32>>>,
    dim: usize,
}

impl ResidualQuantizer {
    pub fn new(
        training_data: &[Vector<f32>],
        stages: usize,
        k: usize,
        max_iters: usize,
        seed: u64,
        distance: Distance,
    ) -> Self {
        assert!(!training_data.is_empty());
        let dim = training_data[0].len();
        let mut codebooks = Vec::with_capacity(stages);
        let mut residuals = training_data.to_vec();
        for stage in 0..stages {
            let codebook = lbg_quantize(&residuals, k, max_iters, seed + stage as u64);
            codebooks.push(codebook.clone());
            for res in residuals.iter_mut() {
                let mut best_index = 0;
                let mut best_dist = distance.compute(&res.data, &codebooks[stage][0].data);
                for (j, centroid) in codebooks[stage].iter().enumerate().skip(1) {
                    let dist = distance.compute(&res.data, &centroid.data);
                    if dist < best_dist {
                        best_dist = dist;
                        best_index = j;
                    }
                }
                *res = &*res - &codebooks[stage][best_index];
            }
        }
        Self {
            stages,
            codebooks,
            dim,
        }
    }

    pub fn quantize(&self, vector: &Vector<f32>, distance: Distance) -> Vector<f16> {
        assert_eq!(vector.len(), self.dim, "Input vector has wrong dimension");
        let mut residual = vector.clone();
        let mut quantized_sum = Vector::new(vec![0.0; self.dim]);
        for stage in 0..self.stages {
            let codebook = &self.codebooks[stage];
            let mut best_index = 0;
            let mut best_dist = distance.compute(&residual.data, &codebook[0].data);
            for (j, centroid) in codebook.iter().enumerate().skip(1) {
                let dist = distance.compute(&residual.data, &centroid.data);
                if dist < best_dist {
                    best_dist = dist;
                    best_index = j;
                }
            }
            let chosen = &codebook[best_index];
            quantized_sum = &quantized_sum + chosen;
            residual = &residual - chosen;
        }
        let quantized_f16: Vec<f16> = quantized_sum
            .data
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        Vector::new(quantized_f16)
    }
}
