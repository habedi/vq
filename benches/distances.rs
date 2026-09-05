//! Benchmarks for the distance metrics at typical embedding dimensions.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use vq::Distance;

const DIMS: [usize; 4] = [64, 128, 768, 1536];

fn vector(dim: usize, seed: u32) -> Vec<f32> {
    (0..dim)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 1000) as f32 / 500.0 - 1.0)
        .collect()
}

fn bench_distances(c: &mut Criterion) {
    let metrics = [
        Distance::SquaredEuclidean,
        Distance::Euclidean,
        Distance::Manhattan,
        Distance::CosineDistance,
    ];
    let mut group = c.benchmark_group("distance");
    for &dim in &DIMS {
        let a = vector(dim, 1);
        let b = vector(dim, 2);
        group.throughput(Throughput::Elements(dim as u64));
        for metric in metrics {
            group.bench_with_input(BenchmarkId::new(metric.name(), dim), &dim, |bencher, _| {
                bencher.iter(|| metric.compute(black_box(&a), black_box(&b)))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_distances);
criterion_main!(benches);
