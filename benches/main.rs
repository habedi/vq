use criterion::criterion_main;

mod bq_bench;
mod opq_bench;
mod pq_bench;
mod rvq_bench;
mod sq_bench;
mod tsvq_bench;

criterion_main!(
    bq_bench::benches,
    sq_bench::benches,
    pq_bench::benches,
    opq_bench::benches,
    tsvq_bench::benches,
    rvq_bench::benches
);
