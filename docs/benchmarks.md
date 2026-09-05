# Benchmarks

Results of the Criterion suite under `benches/`, run with `make bench`.
The suite is run twice: once with the default features (scalar distances, single-threaded training) and once with the `all` feature set (SIMD distances through Hsdlib and Rayon-parallel training and batch quantization).

Test machine:

| Item | Value |
|---|---|
| CPU | AMD Ryzen 5 7600X, 6 cores, 12 threads |
| SIMD backend | AVX-512 (reported as `Auto (AVX512VPOPCNTDQ Capable)`) |
| Rust | 1.85.0 |
| Vq | 0.3.0 |
| Criterion settings | 0.5 s warm-up, 1.5 s measurement per benchmark |

The times are Criterion's point estimates. Treat them as indicative; run `make bench` on your own hardware for numbers that matter to you.

## Distances

One call to `Distance::compute` on two vectors of the given dimension.

| Benchmark | Default features | `all` features | Speedup |
|---|---|---|---|
| squared euclidean, dim 64 | 24.533 ns | 5.6861 ns | 4.3x |
| euclidean, dim 64 | 27.666 ns | 7.4838 ns | 3.7x |
| manhattan, dim 64 | 26.887 ns | 5.7689 ns | 4.7x |
| cosine, dim 64 | 73.002 ns | 10.782 ns | 6.8x |
| squared euclidean, dim 128 | 58.425 ns | 8.5753 ns | 6.8x |
| euclidean, dim 128 | 59.325 ns | 10.972 ns | 5.4x |
| manhattan, dim 128 | 63.248 ns | 8.8861 ns | 7.1x |
| cosine, dim 128 | 156.94 ns | 13.21 ns | 11.9x |
| squared euclidean, dim 768 | 438.52 ns | 36.788 ns | 11.9x |
| euclidean, dim 768 | 491.67 ns | 37.152 ns | 13.2x |
| manhattan, dim 768 | 427.88 ns | 32.481 ns | 13.2x |
| cosine, dim 768 | 1.52 µs | 49.014 ns | 31.0x |
| squared euclidean, dim 1536 | 859.81 ns | 72.213 ns | 11.9x |
| euclidean, dim 1536 | 848.22 ns | 72.863 ns | 11.6x |
| manhattan, dim 1536 | 871.84 ns | 65.146 ns | 13.4x |
| cosine, dim 1536 | 2.7356 µs | 95.798 ns | 28.6x |

The SIMD kernels are 4x to 30x faster than the scalar fallbacks, and the gap widens with dimension.
Cosine distance gains the most because the scalar path makes three passes over the data.

## Training

Training on 2000 vectors of dimension 128.

| Benchmark | Default features | `all` features | Speedup |
|---|---|---|---|
| PQ, 8 subspaces, 256 centroids, 5 iterations | 98.652 ms | 27.048 ms | 3.6x |
| TSVQ, depth 8 | 2.0654 ms | 2.081 ms | 1.0x |

PQ training benefits from both parallel k-means assignment and SIMD distances. TSVQ training is dominated by variance computation and sorting, which neither feature accelerates.

## Quantization

Single-vector calls use vectors of dimension 128. Batch calls quantize 2000 such vectors with `quantize_batch`.

| Benchmark | Default features | `all` features | Speedup |
|---|---|---|---|
| BQ, one vector | 46.801 ns | 45.303 ns | 1.0x |
| SQ, one vector | 393.71 ns | 386.53 ns | 1.0x |
| PQ, one vector | 11.623 µs | 10.848 µs | 1.1x |
| TSVQ, one vector | 1.1805 µs | 345.66 ns | 3.4x |
| BQ, batch of 2000 | 120.82 µs | 133.1 µs | 0.9x |
| SQ, batch of 2000 | 817.97 µs | 315.04 µs | 2.6x |
| PQ, batch of 2000 | 23.342 ms | 4.2699 ms | 5.5x |
| TSVQ, batch of 2000 | 2.532 ms | 256.93 µs | 9.9x |
| PQ dequantize, batch of 2000 | 341.32 µs | 161.6 µs | 2.1x |

With the `all` features, `quantize_batch` runs rows in parallel. The gain is largest for PQ and TSVQ, where each vector costs microseconds. For BQ, a single vector takes under 50 ns, so the thread scheduling overhead cancels out the parallelism and the batch call is no faster than a loop.

## Persistence

Encoding and decoding trained quantizers with `to_bytes` and `from_bytes`. The PQ model here has 8 codebooks of 256 centroids with 16 dimensions each; the TSVQ model has depth 8 over dimension 128.

| Benchmark | Default features | `all` features | Speedup |
|---|---|---|---|
| PQ to_bytes | 25.073 µs | 37.193 µs | 0.7x |
| PQ from_bytes | 62.962 µs | 71.092 µs | 0.9x |
| TSVQ to_bytes | 38.591 µs | 34.994 µs | 1.1x |
| TSVQ from_bytes | 68.018 µs | 62.708 µs | 1.1x |

Persistence does not use SIMD or threads, so the two columns differ only by measurement noise.

## Reproducing

```bash
cargo bench                  # Default features
cargo bench --features all   # SIMD and parallel
```

Criterion writes detailed reports to `target/criterion/`. To pass Criterion options, target one bench binary directly, for example `cargo bench --bench distances -- --measurement-time 5`.
