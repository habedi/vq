use criterion::criterion_main;

mod bench_bq;
mod bench_opq;
mod bench_pq;
mod bench_rvq;
mod bench_sq;
mod bench_tsvq;

criterion_main!(
    bench_bq::benches,
    bench_sq::benches,
    bench_pq::benches,
    bench_opq::benches,
    bench_tsvq::benches,
    bench_rvq::benches
);
