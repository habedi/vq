use crate::distances::Distance;
use crate::vector::{mean_vector, Vector};
use half::f16;

pub struct TSVQNode {
    pub centroid: Vector<f32>,
    pub left: Option<Box<TSVQNode>>,
    pub right: Option<Box<TSVQNode>>,
}

impl TSVQNode {
    pub fn new(training_data: &[Vector<f32>], max_depth: usize) -> Self {
        let centroid = mean_vector(training_data);
        if max_depth == 0 || training_data.len() <= 1 {
            return TSVQNode {
                centroid,
                left: None,
                right: None,
            };
        }
        let dim = centroid.len();
        let mut variances = vec![0.0; dim];
        for v in training_data {
            for (i, &val) in v.data.iter().enumerate() {
                let diff = val - centroid.data[i];
                variances[i] += diff * diff;
            }
        }
        let (split_dim, _) = variances
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let mut values: Vec<f32> = training_data.iter().map(|v| v.data[split_dim]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = values[values.len() / 2];
        let (left_data, right_data): (Vec<Vector<f32>>, Vec<Vector<f32>>) = training_data
            .iter()
            .cloned()
            .partition(|v| v.data[split_dim] <= median);
        let left = if !left_data.is_empty() && left_data.len() < training_data.len() {
            Some(Box::new(TSVQNode::new(&left_data, max_depth - 1)))
        } else {
            None
        };
        let right = if !right_data.is_empty() && right_data.len() < training_data.len() {
            Some(Box::new(TSVQNode::new(&right_data, max_depth - 1)))
        } else {
            None
        };
        TSVQNode {
            centroid,
            left,
            right,
        }
    }

    pub fn quantize_with_distance(&self, vector: &Vector<f32>, distance: &Distance) -> &TSVQNode {
        match (&self.left, &self.right) {
            (Some(left), Some(right)) => {
                let dist_left = distance.compute(&vector.data, &left.centroid.data);
                let dist_right = distance.compute(&vector.data, &right.centroid.data);
                if dist_left <= dist_right {
                    left.quantize_with_distance(vector, distance)
                } else {
                    right.quantize_with_distance(vector, distance)
                }
            }
            (Some(left), None) => left.quantize_with_distance(vector, distance),
            (None, Some(right)) => right.quantize_with_distance(vector, distance),
            (None, None) => self,
        }
    }
}

pub struct TSVQ {
    root: TSVQNode,
}

impl TSVQ {
    pub fn new(training_data: &[Vector<f32>], max_depth: usize) -> Self {
        let root = TSVQNode::new(training_data, max_depth);
        TSVQ { root }
    }

    pub fn quantize(&self, vector: &Vector<f32>, distance: Distance) -> Vector<f16> {
        let leaf = self.root.quantize_with_distance(vector, &distance);
        let centroid_f16: Vec<f16> = leaf
            .centroid
            .data
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        Vector::new(centroid_f16)
    }
}
