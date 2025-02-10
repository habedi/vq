use crate::vector::Vector;

pub struct ScalarQuantizer {
    pub min: f32,
    pub max: f32,
    pub levels: usize,
    pub step: f32,
}

impl ScalarQuantizer {
    pub fn new(min: f32, max: f32, levels: usize) -> Self {
        assert!(max > min);
        assert!(levels >= 2);
        assert!(levels <= 256);
        let step = (max - min) / (levels - 1) as f32;
        Self {
            min,
            max,
            levels,
            step,
        }
    }

    pub fn quantize(&self, vector: &Vector<f32>) -> Vector<u8> {
        let quantized_vector = vector
            .data
            .iter()
            .map(|&x| self.quantize_scalar(x) as u8)
            .collect();
        Vector::new(quantized_vector)
    }

    fn quantize_scalar(&self, x: f32) -> usize {
        let clamped = if x < self.min {
            self.min
        } else if x > self.max {
            self.max
        } else {
            x
        };
        let index = ((clamped - self.min) / self.step).round() as usize;
        index.min(self.levels - 1)
    }
}
