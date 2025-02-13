use crate::distances::Distance;
use crate::vector::{mean_vector, Vector};
use half::f16;
use rayon::prelude::*;

/// A node in the Tree-Structured Vector Quantizer (TSVQ) tree.
///
/// Each node holds a centroid (the mean of the training data at that node)
/// and optionally left/right child nodes representing further splits.
pub struct TSVQNode {
    /// The centroid of the training data at this node.
    pub centroid: Vector<f32>,
    /// Left subtree (if any).
    pub left: Option<Box<TSVQNode>>,
    /// Right subtree (if any).
    pub right: Option<Box<TSVQNode>>,
}

impl TSVQNode {
    /// Recursively builds a TSVQ node from the given training data.
    ///
    /// # Parameters
    /// - `training_data`: A slice of training vectors used to build this node.
    /// - `max_depth`: The maximum depth of recursion. When 0 or if there is only one
    ///   training vector, the node is a leaf.
    ///
    /// # Returns
    /// A `TSVQNode` containing the centroid and (optionally) left/right child nodes.
    pub fn new(training_data: &[Vector<f32>], max_depth: usize) -> Self {
        // Compute the centroid of the training data.
        let centroid = mean_vector(training_data);
        // If we've reached maximum depth or have one or fewer vectors, make a leaf.
        if max_depth == 0 || training_data.len() <= 1 {
            return TSVQNode {
                centroid,
                left: None,
                right: None,
            };
        }
        let dim = centroid.len();

        // Compute variances in parallel for each dimension.
        let variances: Vec<f32> = (0..dim)
            .into_par_iter()
            .map(|i| {
                training_data
                    .iter()
                    .map(|v| {
                        let diff = v.data[i] - centroid.data[i];
                        diff * diff
                    })
                    .sum()
            })
            .collect();

        // Select the dimension with maximum variance for splitting.
        let (split_dim, _) = variances
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        // Extract the values along the chosen dimension and sort them.
        let mut values: Vec<f32> = training_data.iter().map(|v| v.data[split_dim]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute the median: if even number of elements, use the average of the two middle values.
        let median = if values.len() % 2 == 0 {
            (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
        } else {
            values[values.len() / 2]
        };

        // Partition the training data based on the median along the split dimension.
        let (left_data, right_data): (Vec<Vector<f32>>, Vec<Vector<f32>>) = training_data
            .iter()
            .cloned()
            .partition(|v| v.data[split_dim] <= median);

        // Recursively build left and right children in parallel.
        let (left, right) = rayon::join(
            || {
                if !left_data.is_empty() && left_data.len() < training_data.len() {
                    Some(Box::new(TSVQNode::new(&left_data, max_depth - 1)))
                } else {
                    None
                }
            },
            || {
                if !right_data.is_empty() && right_data.len() < training_data.len() {
                    Some(Box::new(TSVQNode::new(&right_data, max_depth - 1)))
                } else {
                    None
                }
            },
        );

        TSVQNode {
            centroid,
            left,
            right,
        }
    }

    /// Recursively traverses the TSVQ tree to quantize an input vector.
    ///
    /// At each node, the distance between the input vector and the centroids of
    /// the child nodes is computed using the provided distance metric. The traversal
    /// proceeds into the child with the smaller distance until a leaf node is reached.
    ///
    /// # Parameters
    /// - `vector`: The input vector to quantize.
    /// - `distance`: A reference to the distance metric used for comparing vectors.
    ///
    /// # Returns
    /// A reference to the leaf `TSVQNode` whose centroid best approximates the input.
    pub fn quantize_with_distance<'a>(
        &'a self,
        vector: &Vector<f32>,
        distance: &Distance,
    ) -> &'a TSVQNode {
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

/// A Tree-Structured Vector Quantizer (TSVQ) that builds a binary tree for quantization.
///
/// The TSVQ is constructed from a set of training data by recursively partitioning
/// the data along the dimension of maximum variance. Each node stores the mean
/// (centroid) of its data, and leaf nodes provide the final quantized representations.
pub struct TSVQ {
    /// The root node of the TSVQ tree.
    pub root: TSVQNode,
    /// The distance metric used for traversing the tree.
    pub distance: Distance,
}

impl TSVQ {
    /// Constructs a new TSVQ from the given training data.
    ///
    /// # Parameters
    /// - `training_data`: A slice of training vectors used to build the tree.
    /// - `max_depth`: The maximum depth of the TSVQ tree. A larger value allows finer partitions.
    /// - `distance`: The distance metric to use for comparing vectors during tree traversal.
    ///
    /// # Returns
    /// A new `TSVQ` instance with the constructed tree and stored distance metric.
    pub fn new(training_data: &[Vector<f32>], max_depth: usize, distance: Distance) -> Self {
        let root = TSVQNode::new(training_data, max_depth);
        TSVQ { root, distance }
    }

    /// Quantizes an input vector by traversing the TSVQ tree.
    ///
    /// The traversal uses the stored distance metric to determine which branch to follow at each node.
    /// Once a leaf node is reached, its centroid is returned as the quantized representation.
    /// The resulting vector is converted to half-precision (`f16`).
    ///
    /// # Parameters
    /// - `vector`: The input vector to quantize.
    ///
    /// # Returns
    /// A quantized vector (`Vector<f16>`) corresponding to the centroid of the selected leaf node.
    ///
    /// # Panics
    /// Panics if the input vector's dimension does not match the expected dimension.
    pub fn quantize(&self, vector: &Vector<f32>) -> Vector<f16> {
        // Traverse the tree using the stored distance metric.
        let leaf = self.root.quantize_with_distance(vector, &self.distance);
        // Convert the leaf centroid from f32 to f16.
        let centroid_f16: Vec<f16> = leaf
            .centroid
            .data
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        Vector::new(centroid_f16)
    }
}
