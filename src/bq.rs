use crate::vector::Vector;

pub struct BinaryQuantizer {
    pub threshold: f32,
    pub low: u8,
    pub high: u8,
}

impl BinaryQuantizer {
    pub fn new(threshold: f32, low: u8, high: u8) -> Self {
        Self {
            threshold,
            low,
            high,
        }
    }

    pub fn quantize(&self, vector: &Vector<f32>) -> Vector<u8> {
        let quantized_vector = vector
            .data
            .iter()
            .map(|&x| {
                if x >= self.threshold {
                    self.high
                } else {
                    self.low
                }
            })
            .collect();
        Vector::new(quantized_vector)
    }
}
