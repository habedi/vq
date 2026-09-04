//! Tests for saving and loading quantizers.

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

fn temp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("vq-persist-{}-{}.bin", std::process::id(), name));
    p
}

#[test]
fn test_bq_roundtrip() {
    let bq = BinaryQuantizer::new(0.25, 3, 9).unwrap();
    let restored = BinaryQuantizer::from_bytes(&bq.to_bytes().unwrap()).unwrap();
    assert_eq!(restored, bq);
    assert_eq!(restored.threshold(), 0.25);
    assert_eq!((restored.low(), restored.high()), (3, 9));
}

#[test]
fn test_sq_roundtrip() {
    let sq = ScalarQuantizer::new(-2.0, 3.0, 17).unwrap();
    let restored = ScalarQuantizer::from_bytes(&sq.to_bytes().unwrap()).unwrap();
    assert_eq!(restored, sq);
    let input = vec![-3.0, -1.0, 0.0, 1.5, 4.0];
    assert_eq!(
        restored.quantize(&input).unwrap(),
        sq.quantize(&input).unwrap()
    );
}

#[test]
fn test_pq_roundtrip_preserves_outputs() {
    let data = training();
    let pq = ProductQuantizer::new(&refs(&data), 2, 4, 10, Distance::CosineDistance, 42).unwrap();
    let restored = ProductQuantizer::from_bytes(&pq.to_bytes().unwrap()).unwrap();
    assert_eq!(restored, pq);
    assert_eq!(restored.distance_metric(), "cosine");
    for v in &data {
        assert_eq!(restored.quantize(v).unwrap(), pq.quantize(v).unwrap());
    }
}

#[test]
fn test_tsvq_roundtrip_preserves_outputs() {
    let data = training();
    let tsvq = TSVQ::new(&refs(&data), 4, Distance::Manhattan).unwrap();
    let restored = TSVQ::from_bytes(&tsvq.to_bytes().unwrap()).unwrap();
    assert_eq!(restored, tsvq);
    assert_eq!(restored.distance_metric(), "manhattan");
    for v in &data {
        assert_eq!(restored.quantize(v).unwrap(), tsvq.quantize(v).unwrap());
    }
}

#[test]
fn test_save_and_load_file() {
    let data = training();
    let pq = ProductQuantizer::new(&refs(&data), 4, 3, 10, Distance::Euclidean, 1).unwrap();
    let path = temp_path("pq");
    pq.save(&path).unwrap();
    let loaded = ProductQuantizer::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(loaded, pq);
}

#[test]
fn test_load_missing_file_is_io_error() {
    let result = TSVQ::load(temp_path("does-not-exist"));
    assert!(matches!(result, Err(VqError::Io(_))));
}

#[test]
fn test_malformed_bytes_are_rejected() {
    assert!(matches!(
        BinaryQuantizer::from_bytes(&[]),
        Err(VqError::Serialization(_))
    ));
    assert!(matches!(
        ProductQuantizer::from_bytes(&[0xff; 7]),
        Err(VqError::Serialization(_))
    ));
    // Truncated payload
    let data = training();
    let tsvq = TSVQ::new(&refs(&data), 3, Distance::Euclidean).unwrap();
    let bytes = tsvq.to_bytes().unwrap();
    assert!(TSVQ::from_bytes(&bytes[..bytes.len() / 2]).is_err());
}

#[test]
fn test_bytes_encoding_invalid_parameters_are_rejected() {
    // Mirrors the field layout of ScalarQuantizer so postcard produces compatible bytes
    #[derive(serde::Serialize)]
    struct RawScalar {
        min: f32,
        max: f32,
        levels: usize,
        step: f32,
    }
    let bad = RawScalar {
        min: 0.0,
        max: 1.0,
        levels: 300,
        step: 1.0 / 299.0,
    };
    let bytes = postcard::to_stdvec(&bad).unwrap();
    assert!(matches!(
        ScalarQuantizer::from_bytes(&bytes),
        Err(VqError::InvalidParameter {
            parameter: "levels",
            ..
        })
    ));

    #[derive(serde::Serialize)]
    struct RawBinary {
        threshold: f32,
        low: u8,
        high: u8,
    }
    let bytes = postcard::to_stdvec(&RawBinary {
        threshold: 0.0,
        low: 5,
        high: 5,
    })
    .unwrap();
    assert!(matches!(
        BinaryQuantizer::from_bytes(&bytes),
        Err(VqError::InvalidParameter { .. })
    ));
}

#[test]
fn test_bytes_are_deterministic() {
    let data = training();
    let a = TSVQ::new(&refs(&data), 3, Distance::Euclidean).unwrap();
    let b = TSVQ::new(&refs(&data), 3, Distance::Euclidean).unwrap();
    assert_eq!(a.to_bytes().unwrap(), b.to_bytes().unwrap());
}
