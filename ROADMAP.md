## Feature Roadmap

This document includes the roadmap for the Vq project.
It outlines features to be implemented and their current status.

> [!IMPORTANT]
> This roadmap is a work in progress and is subject to change.

### 1. Quantization Algorithms

* [x] Binary quantizer (BQ)
* [x] Scalar quantizer (SQ)
* [x] Product quantizer (PQ)
* [x] Tree-structured vector quantizer (TSVQ)

### 2. Distances

* [x] (Squared) Euclidean (L2) distance
* [x] Manhattan (L1) distance
* [x] Cosine distance

### 3. Core Features

* [x] Unified `Quantizer` trait
* [x] Generic `Vector` struct
* [x] Codebook training (LBG/k-means)
* [x] `dequantize` support
* [x] Persistent serialization (save and load models)
* [ ] Streaming training support
* [x] Batch quantization (quantize multiple vectors at once)
* [x] `fit_transform` convenience method

### 4. Performance Optimizations

* [x] Parallel training using multithreading
* [x] Inline hints for hot paths
* [x] Zero-copy training (allocation reduction)
* [x] SIMD Acceleration for Intel and AMD CPUs (AVX/AVX2/AVX512)
* [x] SIMD Acceleration for ARM CPUs (NEON/SVE)
* [x] Runtime CPU feature detection
* [ ] SIMD for `f16` (half-precision)

### 5. Language Bindings

* [x] Python bindings (`pyvq`)
* [ ] C bindings
* [ ] Node.js bindings

### 6. Tools and Binaries

* [x] `eval` tool for algorithm comparison
* [ ] CLI for direct file quantization
* [x] Benchmark suite (internal `cargo bench`)

### 7. Documentation and Testing

* [x] Rust unit tests
* [x] Integration tests
* [x] Regression tests
* [x] Property-based tests (Rust and Python)
* [x] Differential tests against reference implementations (Rust and Python)
* [x] Documentation examples
* [x] Code coverage setup
* [x] Python API documentation
* [x] Performance benchmarks report
