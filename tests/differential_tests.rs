//! Differential tests for the vq crate.
//!
//! Each test compares the library against an independent reference implementation
//! written in plain f64 arithmetic inside this file. The library output must agree
//! with the reference for randomly generated inputs.
//!
//! The same tests run under every feature set, so running them with and without
//! `simd` and `parallel` (see `make test-diff`) checks the SIMD distance kernels and
//! the parallel training path against the scalar reference.

use half::f16;
use proptest::prelude::*;
use vq::core::vector::{Vector, lbg_quantize};
use vq::{BinaryQuantizer, Distance, ProductQuantizer, Quantizer, ScalarQuantizer, TSVQ};

// =============================================================================
// Reference implementations
// =============================================================================

mod reference {
    pub fn squared_euclidean(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = x as f64 - y as f64;
                d * d
            })
            .sum()
    }

    pub fn euclidean(a: &[f32], b: &[f32]) -> f64 {
        squared_euclidean(a, b).sqrt()
    }

    pub fn manhattan(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - y as f64).abs())
            .sum()
    }

    /// Cosine distance in [0, 2]. Callers must avoid zero-norm inputs because the
    /// library's scalar and SIMD paths define that case differently.
    pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
        let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        (1.0 - dot / (na * nb)).clamp(0.0, 2.0)
    }

    pub fn distance(metric: super::Distance, a: &[f32], b: &[f32]) -> f64 {
        match metric {
            super::Distance::SquaredEuclidean => squared_euclidean(a, b),
            super::Distance::Euclidean => euclidean(a, b),
            super::Distance::Manhattan => manhattan(a, b),
            super::Distance::CosineDistance => cosine(a, b),
        }
    }

    pub fn binary_quantize(input: &[f32], threshold: f32, low: u8, high: u8) -> Vec<u8> {
        input
            .iter()
            .map(|&x| {
                if x as f64 >= threshold as f64 {
                    high
                } else {
                    low
                }
            })
            .collect()
    }

    /// Returns the exact real-valued position of `x` on the level grid before rounding.
    pub fn scalar_position(x: f32, min: f32, max: f32, levels: usize) -> f64 {
        let step = (max as f64 - min as f64) / (levels as f64 - 1.0);
        let clamped = (x as f64).clamp(min as f64, max as f64);
        (clamped - min as f64) / step
    }

    pub fn scalar_dequantize(code: u8, min: f32, max: f32, levels: usize) -> f64 {
        let step = (max as f64 - min as f64) / (levels as f64 - 1.0);
        min as f64 + code as f64 * step
    }

    pub fn mean(vectors: &[Vec<f32>]) -> Vec<f64> {
        let dim = vectors[0].len();
        let n = vectors.len() as f64;
        (0..dim)
            .map(|i| vectors.iter().map(|v| v[i] as f64).sum::<f64>() / n)
            .collect()
    }

    /// Index of the candidate nearest to `query` under `metric`.
    pub fn nearest(metric: super::Distance, query: &[f32], candidates: &[Vec<f32>]) -> usize {
        let mut best = 0;
        let mut best_dist = f64::INFINITY;
        for (i, c) in candidates.iter().enumerate() {
            let d = distance(metric, query, c);
            if d < best_dist {
                best_dist = d;
                best = i;
            }
        }
        best
    }
}

// =============================================================================
// Strategies
// =============================================================================

fn vec_f32(
    len: impl Strategy<Value = usize>,
    min: f32,
    max: f32,
) -> impl Strategy<Value = Vec<f32>> {
    len.prop_flat_map(move |n| prop::collection::vec(min..max, n))
}

fn vec_pair(min: f32, max: f32) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    // Lengths up to 67 cover SIMD register widths plus a scalar tail
    (1usize..=67).prop_flat_map(move |n| {
        (
            prop::collection::vec(min..max, n),
            prop::collection::vec(min..max, n),
        )
    })
}

fn any_metric() -> impl Strategy<Value = Distance> {
    prop_oneof![
        Just(Distance::SquaredEuclidean),
        Just(Distance::Euclidean),
        Just(Distance::Manhattan),
        Just(Distance::CosineDistance),
    ]
}

fn training_data(
    n: impl Strategy<Value = usize>,
    dim: usize,
    min: f32,
    max: f32,
) -> impl Strategy<Value = Vec<Vec<f32>>> {
    n.prop_flat_map(move |n| prop::collection::vec(prop::collection::vec(min..max, dim), n))
}

fn to_refs(data: &[Vec<f32>]) -> Vec<&[f32]> {
    data.iter().map(|v| v.as_slice()).collect()
}

fn f16_round(v: &[f64]) -> Vec<f32> {
    v.iter()
        .map(|&x| f16::from_f32(x as f32).to_f32())
        .collect()
}

fn assert_close(
    actual: f32,
    expected: f64,
    rel: f64,
    abs: f64,
    context: &str,
) -> Result<(), TestCaseError> {
    let diff = (actual as f64 - expected).abs();
    prop_assert!(
        diff <= abs + rel * expected.abs(),
        "{context}: library={actual}, reference={expected}, diff={diff}"
    );
    Ok(())
}

// =============================================================================
// Distances
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Every metric agrees with the f64 reference on non-degenerate inputs.
    #[test]
    fn diff_distance_matches_reference(
        metric in any_metric(),
        (a, b) in vec_pair(-100.0, 100.0),
    ) {
        // Keep away from the zero-norm special case of the cosine metric
        let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        prop_assume!(na > 1e-3 && nb > 1e-3);

        let actual = metric.compute(&a, &b).unwrap();
        let expected = reference::distance(metric, &a, &b);
        assert_close(actual, expected, 1e-4, 1e-4, metric.name())?;
    }

    /// Distances agree with the reference for very large magnitudes, where the
    /// order of accumulation matters most.
    #[test]
    fn diff_distance_large_magnitude(
        metric in any_metric(),
        (a, b) in vec_pair(-1e6, 1e6),
    ) {
        let actual = metric.compute(&a, &b).unwrap();
        let expected = reference::distance(metric, &a, &b);
        assert_close(actual, expected, 1e-3, 1e-3, metric.name())?;
    }

    /// The library ranks candidates the same way as the reference.
    #[test]
    fn diff_distance_preserves_nearest_neighbor(
        metric in any_metric(),
        query in vec_f32(Just(8usize), 0.5, 50.0),
        candidates in training_data(2usize..=12, 8, 0.5, 50.0),
    ) {
        let expected = reference::nearest(metric, &query, &candidates);
        let expected_dist = reference::distance(metric, &query, &candidates[expected]);
        let actual = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, metric.compute(&query, c).unwrap()))
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .map(|(i, _)| i)
            .unwrap();
        let actual_dist = reference::distance(metric, &query, &candidates[actual]);
        // Allow a different winner only when the reference distances tie within f32 noise
        assert_close(actual_dist as f32, expected_dist, 1e-5, 1e-5, metric.name())?;
    }
}

// =============================================================================
// Binary and scalar quantizers
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn diff_bq_matches_reference(
        input in vec_f32(0usize..=64, -50.0, 50.0),
        threshold in -50.0f32..50.0,
        (low, high) in (0u8..255).prop_flat_map(|l| (Just(l), (l + 1)..=255)),
    ) {
        let bq = BinaryQuantizer::new(threshold, low, high).unwrap();
        let actual = bq.quantize(&input).unwrap();
        let expected = reference::binary_quantize(&input, threshold, low, high);
        prop_assert_eq!(actual, expected);
    }

    /// Integer-valued inputs and thresholds exercise the exact equality boundary.
    #[test]
    fn diff_bq_matches_reference_at_boundary(
        input in prop::collection::vec((-5i8..=5).prop_map(|x| x as f32), 0..=32),
        threshold in (-5i8..=5).prop_map(|x| x as f32),
    ) {
        let bq = BinaryQuantizer::new(threshold, 0, 1).unwrap();
        let actual = bq.quantize(&input).unwrap();
        let expected = reference::binary_quantize(&input, threshold, 0, 1);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn diff_sq_quantize_matches_reference(
        input in vec_f32(0usize..=64, -200.0, 200.0),
        (min, max) in (-100.0f32..0.0, 1.0f32..100.0),
        levels in 2usize..=256,
    ) {
        let sq = ScalarQuantizer::new(min, max, levels).unwrap();
        let actual = sq.quantize(&input).unwrap();
        for (&x, &code) in input.iter().zip(&actual) {
            let position = reference::scalar_position(x, min, max, levels);
            let expected = position.round().min((levels - 1) as f64) as u8;
            if code != expected {
                // The library rounds in f32, so it may pick the other side of a
                // half-way point. Any other disagreement is a bug.
                let frac = (position.fract() - 0.5).abs();
                prop_assert!(
                    frac < 1e-3 && (code as i32 - expected as i32).abs() == 1,
                    "x={x}, position={position}, library={code}, reference={expected}"
                );
            }
        }
    }

    #[test]
    fn diff_sq_dequantize_matches_reference(
        codes in prop::collection::vec(any::<u8>(), 0..=64),
        (min, max) in (-100.0f32..0.0, 1.0f32..100.0),
    ) {
        let levels = 256;
        let sq = ScalarQuantizer::new(min, max, levels).unwrap();
        let actual = sq.dequantize(&codes).unwrap();
        for (&code, &value) in codes.iter().zip(&actual) {
            let expected = reference::scalar_dequantize(code, min, max, levels);
            assert_close(value, expected, 1e-5, 1e-4, "sq dequantize")?;
        }
    }
}

// =============================================================================
// Product quantizer
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    /// The sub-centroid PQ picks for a query is the nearest one under the reference
    /// distance, among all sub-centroids that PQ ever emits.
    #[test]
    fn diff_pq_selects_nearest_observed_centroid(
        metric in any_metric(),
        training in training_data(16usize..=40, 8, 0.5, 20.0),
        queries in training_data(1usize..=5, 8, 0.5, 20.0),
        m in prop_oneof![Just(1usize), Just(2), Just(4)],
        k in 2usize..=4,
    ) {
        let refs = to_refs(&training);
        let pq = ProductQuantizer::new(&refs, m, k, 20, metric, 7).unwrap();
        let sub_dim = 8 / m;

        // Recover the codebooks the library uses from its own outputs
        let mut observed: Vec<Vec<Vec<f32>>> = vec![Vec::new(); m];
        for v in &training {
            let codes = pq.quantize(v).unwrap();
            let recon = pq.dequantize(&codes).unwrap();
            for (i, chunk) in recon.chunks(sub_dim).enumerate() {
                if !observed[i].contains(&chunk.to_vec()) {
                    observed[i].push(chunk.to_vec());
                }
            }
        }

        for q in &queries {
            let recon = pq.dequantize(&pq.quantize(q).unwrap()).unwrap();
            for (i, chunk) in recon.chunks(sub_dim).enumerate() {
                let sub_q = &q[i * sub_dim..(i + 1) * sub_dim];
                let chosen = reference::distance(metric, sub_q, chunk);
                let best = reference::nearest(metric, sub_q, &observed[i]);
                let best_dist = reference::distance(metric, sub_q, &observed[i][best]);
                // f16 storage perturbs centroids slightly, so allow a small margin
                prop_assert!(
                    chosen <= best_dist * 1.02 + 1e-2,
                    "{}: subspace {i} chose distance {chosen} but {best_dist} was available",
                    metric.name()
                );
            }
        }
    }

    /// Dequantization is exactly the f16 to f32 widening of each code.
    #[test]
    fn diff_pq_dequantize_is_f16_widening(
        codes in prop::collection::vec(-100.0f32..100.0, 8),
    ) {
        let training: Vec<Vec<f32>> = (0..16)
            .map(|i| (0..8).map(|j| (i * 8 + j) as f32).collect())
            .collect();
        let pq = ProductQuantizer::new(&to_refs(&training), 2, 2, 5, Distance::Euclidean, 1).unwrap();
        let f16_codes: Vec<f16> = codes.iter().map(|&x| f16::from_f32(x)).collect();
        let actual = pq.dequantize(&f16_codes).unwrap();
        let expected: Vec<f32> = f16_codes.iter().map(|&x| x.to_f32()).collect();
        prop_assert_eq!(actual, expected);
    }
}

// =============================================================================
// Tree-structured quantizer
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(60))]

    /// A depth-zero tree is a single leaf holding the mean of the training data.
    #[test]
    fn diff_tsvq_depth_zero_is_mean(
        training in training_data(1usize..=30, 6, -50.0, 50.0),
        query in vec_f32(Just(6usize), -50.0, 50.0),
    ) {
        let tsvq = TSVQ::new(&to_refs(&training), 0, Distance::Euclidean).unwrap();
        let actual = tsvq.dequantize(&tsvq.quantize(&query).unwrap()).unwrap();
        let expected = f16_round(&reference::mean(&training));
        for (a, e) in actual.iter().zip(&expected) {
            // f16 rounding of the f32 mean versus the f64 mean
            assert_close(*a, *e as f64, 2e-3, 1e-2, "tsvq mean")?;
        }
    }

    /// A tree of depth d has at most min(2^d, n) distinct leaves.
    #[test]
    fn diff_tsvq_leaf_count_is_bounded(
        training in training_data(1usize..=40, 4, -50.0, 50.0),
        queries in training_data(1usize..=40, 4, -50.0, 50.0),
        depth in 0usize..=5,
    ) {
        let tsvq = TSVQ::new(&to_refs(&training), depth, Distance::SquaredEuclidean).unwrap();
        let mut leaves: Vec<Vec<f16>> = Vec::new();
        for v in training.iter().chain(&queries) {
            let codes = tsvq.quantize(v).unwrap();
            if !leaves.contains(&codes) {
                leaves.push(codes);
            }
        }
        let bound = (1usize << depth).min(training.len());
        prop_assert!(leaves.len() <= bound, "{} leaves, bound {bound}", leaves.len());
    }

    #[test]
    fn diff_tsvq_dequantize_is_f16_widening(
        codes in prop::collection::vec(-100.0f32..100.0, 4),
    ) {
        let training: Vec<Vec<f32>> = (0..8)
            .map(|i| (0..4).map(|j| (i * 4 + j) as f32).collect())
            .collect();
        let tsvq = TSVQ::new(&to_refs(&training), 2, Distance::Euclidean).unwrap();
        let f16_codes: Vec<f16> = codes.iter().map(|&x| f16::from_f32(x)).collect();
        let actual = tsvq.dequantize(&f16_codes).unwrap();
        let expected: Vec<f32> = f16_codes.iter().map(|&x| x.to_f32()).collect();
        prop_assert_eq!(actual, expected);
    }
}

// =============================================================================
// Codebook training
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    /// Converged Lloyd iterations return centroids that are the means of their own
    /// clusters. The reference recomputes the assignment and the mean in f64.
    #[test]
    fn diff_lbg_centroids_are_cluster_means(
        data in training_data(4usize..=30, 3, -20.0, 20.0),
        k in 1usize..=3,
        seed in any::<u64>(),
    ) {
        let vectors: Vec<Vector<f32>> = data.iter().map(|v| Vector::new(v.clone())).collect();
        let centroids = lbg_quantize(&vectors, k, 500, seed).unwrap();
        prop_assert_eq!(centroids.len(), k);

        let plain: Vec<Vec<f32>> = centroids.iter().map(|c| c.data.clone()).collect();
        let mut members: Vec<Vec<&Vec<f32>>> = vec![Vec::new(); k];
        for v in &data {
            members[reference::nearest(Distance::SquaredEuclidean, v, &plain)].push(v);
        }
        for (c, cluster) in plain.iter().zip(&members) {
            prop_assert!(!cluster.is_empty(), "centroid {c:?} owns no points");
            let owned: Vec<Vec<f32>> = cluster.iter().map(|v| (*v).clone()).collect();
            let mean = reference::mean(&owned);
            for (x, m) in c.iter().zip(&mean) {
                assert_close(*x, *m, 1e-4, 1e-4, "lbg centroid")?;
            }
        }
    }
}

// =============================================================================
// Vector arithmetic in f16 versus f32
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn diff_f16_vector_ops_match_f32(
        (a, b) in vec_pair(-8.0, 8.0),
    ) {
        let a16 = Vector::new(a.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>());
        let b16 = Vector::new(b.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>());
        // Reference: the same operations on the f16-rounded values in f64
        let ar: Vec<f32> = a16.data.iter().map(|x| x.to_f32()).collect();
        let br: Vec<f32> = b16.data.iter().map(|x| x.to_f32()).collect();
        let dot: f64 = ar.iter().zip(&br).map(|(&x, &y)| x as f64 * y as f64).sum();
        let d2 = reference::squared_euclidean(&ar, &br);

        // f16 accumulates with about three significant digits
        let tol = 2e-2 * a.len() as f64;
        assert_close(a16.dot(&b16).to_f32(), dot, tol, tol, "f16 dot")?;
        assert_close(a16.distance2(&b16).to_f32(), d2, tol, tol, "f16 distance2")?;
        let sum = a16.try_add(&b16).unwrap();
        for ((s, &x), &y) in sum.data.iter().zip(&ar).zip(&br) {
            assert_close(s.to_f32(), x as f64 + y as f64, 2e-3, 2e-3, "f16 add")?;
        }
    }
}

// =============================================================================
// SIMD kernels against the reference (only meaningful with the simd feature)
// =============================================================================

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use vq::core::hsdlib_ffi;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn diff_simd_kernels_match_reference((a, b) in vec_pair(-100.0, 100.0)) {
            let l2 = hsdlib_ffi::sqeuclidean_f32(&a, &b).unwrap();
            assert_close(l2, reference::squared_euclidean(&a, &b), 1e-4, 1e-4, "simd sqeuclidean")?;
            let l1 = hsdlib_ffi::manhattan_f32(&a, &b).unwrap();
            assert_close(l1, reference::manhattan(&a, &b), 1e-4, 1e-4, "simd manhattan")?;
            let sim = hsdlib_ffi::cosine_f32(&a, &b).unwrap();
            assert_close(1.0 - sim, reference::cosine(&a, &b), 1e-4, 1e-4, "simd cosine")?;
        }

        /// Every length from 0 through 70 exercises the vector tails of each kernel.
        #[test]
        fn diff_simd_kernels_all_tail_lengths(seed in 0u32..1000) {
            for n in 0usize..=70 {
                let a: Vec<f32> = (0..n).map(|i| ((i as u32 * 7 + seed) % 13) as f32 - 6.0).collect();
                let b: Vec<f32> = (0..n).map(|i| ((i as u32 * 11 + seed) % 17) as f32 - 8.0).collect();
                let l2 = hsdlib_ffi::sqeuclidean_f32(&a, &b).unwrap();
                assert_close(l2, reference::squared_euclidean(&a, &b), 1e-5, 1e-5, "sqeuclidean tail")?;
                let l1 = hsdlib_ffi::manhattan_f32(&a, &b).unwrap();
                assert_close(l1, reference::manhattan(&a, &b), 1e-5, 1e-5, "manhattan tail")?;
            }
        }
    }

    #[test]
    fn diff_simd_rejects_non_finite_and_library_falls_back() {
        let a = vec![1.0, f32::NAN, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(hsdlib_ffi::cosine_f32(&a, &b).is_none());
        // The public API must still produce a result through the scalar fallback
        let d = Distance::CosineDistance.compute(&a, &b).unwrap();
        assert!(d.is_nan());
    }
}
