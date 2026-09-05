//! Tests for the batch quantization and dequantization methods of the `Quantizer` trait.

mod common;

use vq::{BinaryQuantizer, Distance, ProductQuantizer, Quantizer, ScalarQuantizer, TSVQ, VqError};

fn training() -> Vec<Vec<f32>> {
    common::generate_test_data(&mut common::seeded_rng(), 40, 8)
        .into_iter()
        .map(|v| v.data)
        .collect()
}

fn refs(data: &[Vec<f32>]) -> Vec<&[f32]> {
    data.iter().map(|v| v.as_slice()).collect()
}

/// Batch results must equal the per-vector results, and round trip through dequantize_batch.
fn assert_batch_matches_single<Q>(quantizer: &Q, data: &[Vec<f32>])
where
    Q: Quantizer + Sync,
    Q::QuantizedOutput: Send + Sync + PartialEq + std::fmt::Debug,
{
    let batch = quantizer.quantize_batch(&refs(data)).unwrap();
    assert_eq!(batch.len(), data.len());
    for (v, q) in data.iter().zip(&batch) {
        assert_eq!(q, &quantizer.quantize(v).unwrap());
    }

    let recon = quantizer.dequantize_batch(&batch).unwrap();
    assert_eq!(recon.len(), data.len());
    for (q, r) in batch.iter().zip(&recon) {
        assert_eq!(r, &quantizer.dequantize(q).unwrap());
    }
}

#[test]
fn test_bq_batch_matches_single() {
    let data = training();
    let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
    assert_batch_matches_single(&bq, &data);
}

#[test]
fn test_sq_batch_matches_single() {
    let data = training();
    let sq = ScalarQuantizer::new(-1000.0, 1000.0, 256).unwrap();
    assert_batch_matches_single(&sq, &data);
}

#[test]
fn test_pq_batch_matches_single() {
    let data = training();
    let pq = ProductQuantizer::new(&refs(&data), 2, 4, 10, Distance::Euclidean, 42).unwrap();
    assert_batch_matches_single(&pq, &data);
}

#[test]
fn test_tsvq_batch_matches_single() {
    let data = training();
    let tsvq = TSVQ::new(&refs(&data), 3, Distance::SquaredEuclidean).unwrap();
    assert_batch_matches_single(&tsvq, &data);
}

#[test]
fn test_empty_batch() {
    let data = training();
    let pq = ProductQuantizer::new(&refs(&data), 2, 4, 10, Distance::Euclidean, 42).unwrap();
    let empty: Vec<&[f32]> = vec![];
    assert!(pq.quantize_batch(&empty).unwrap().is_empty());
    assert!(pq.dequantize_batch(&[]).unwrap().is_empty());
}

#[test]
fn test_batch_reports_first_invalid_vector() {
    let data = training();
    let pq = ProductQuantizer::new(&refs(&data), 2, 4, 10, Distance::Euclidean, 42).unwrap();
    let short = vec![1.0f32; 3];
    let batch: Vec<&[f32]> = vec![data[0].as_slice(), short.as_slice(), data[1].as_slice()];
    assert!(matches!(
        pq.quantize_batch(&batch),
        Err(VqError::DimensionMismatch {
            expected: 8,
            found: 3
        })
    ));

    let codes = pq.quantize_batch(&refs(&data[..2])).unwrap();
    let mut bad = codes.clone();
    bad[1].pop();
    assert!(matches!(
        pq.dequantize_batch(&bad),
        Err(VqError::DimensionMismatch {
            expected: 8,
            found: 7
        })
    ));
}

#[test]
fn test_large_batch_is_deterministic() {
    let data = common::generate_test_data(&mut common::seeded_rng(), 2_000, 16);
    let plain: Vec<Vec<f32>> = data.into_iter().map(|v| v.data).collect();
    let tsvq = TSVQ::new(&refs(&plain), 4, Distance::Euclidean).unwrap();
    let a = tsvq.quantize_batch(&refs(&plain)).unwrap();
    let b = tsvq.quantize_batch(&refs(&plain)).unwrap();
    assert_eq!(a, b);
}
