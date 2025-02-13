# Vq

[<img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/habedi/vq/tests.yml?label=Tests&style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/habedi/vq/actions/workflows/tests.yml)
[<img alt="Code Coverage" src="https://img.shields.io/codecov/c/github/habedi/vq?style=for-the-badge&labelColor=555555&logo=codecov" height="20">](https://codecov.io/gh/habedi/vq)
[<img alt="CodeFactor" src="https://img.shields.io/codefactor/grade/github/habedi/vq?style=for-the-badge&labelColor=555555&logo=codefactor" height="20">](https://www.codefactor.io/repository/github/habedi/vq)
[<img alt="Crates.io" src="https://img.shields.io/crates/v/vq.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/vq)
[<img alt="Docs.rs" src="https://img.shields.io/badge/docs.rs-vq-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/vq)
[<img alt="Downloads" src="https://img.shields.io/crates/d/vq?style=for-the-badge&labelColor=555555&logo=rust" height="20">](https://crates.io/crates/vq)
[<img alt="Docs" src="https://img.shields.io/badge/docs-latest-3776ab?style=for-the-badge&labelColor=555555&logo=readthedocs" height="20">](docs)
[<img alt="License" src="https://img.shields.io/badge/license-MIT%2FApache--2.0-007ec6?style=for-the-badge&labelColor=555555&logo=open-source-initiative" height="20">](https://github.com/habedi/vq)

Vq (**v**[ector] **q**[uantiztion]) is a Rust library that implements several
popular [vector quantization](https://en.wikipedia.org/wiki/Vector_quantization) algorithms including binary, scalar,
and product quantization algorithms.
It provides a simple, efficient API for data compression that help reduce memory usage and computational overhead.

## Features

- Implemented Algorithms:
    - [**Binary Quantization (BQ)**](src/bq.rs)
    - [**Scalar Quantization (SQ)**](src/sq.rs)
    - [**Product Quantization (PQ)**](https://ieeexplore.ieee.org/document/5432202)
    - [**Optimized Product Quantization (OPQ)**](https://ieeexplore.ieee.org/document/6619223)
    - [**Tree-structured Vector Quantization (TSVQ)**](https://ieeexplore.ieee.org/document/515493)
    - [**Residual Vector Quantization (RVQ)**](https://pmc.ncbi.nlm.nih.gov/articles/PMC3231071/)

- Parallelized vector operations for large vectors using [Rayon](https://crates.io/crates/rayon).
- Flexible quantization algorithm implementations that support custom distance functions (e.g., Euclidean, Cosine,
  Chebyshev, etc.).
- Support for quantizing vectors of `f32` to `f16` (using [half](https://github.com/starkat99/half-rs/tree/main/src)) or `u8` data types.
- Simple and intuitive API for all quantization algorithms.

## Installation

```bash
cargo add vq
```

## Documentation

Find the latest documentation [here](docs) or on [docs.rs](https://docs.rs/vq).

Check out the [tests](tests/) directory for detailed examples of using Vq.

### Quick Example

Here's a simple example using the scalar quantization:

```rust
use vq::sq::ScalarQuantizer;
use vq::vector::Vector;

fn main() {
    // Create a scalar quantizer for values in the range [0.0, 1.0] with 256 levels.
    let quantizer = ScalarQuantizer::new(0.0, 1.0, 256);

    // Create an input vector.
    let input = Vector::new(vec![0.1, 0.5, -0.8, -0.3, 0.9]);

    // Quantize the input vector.
    let quantized_input = quantizer.quantize(&input);

    println!("Quantized input vector: {}", quantized_input);
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

## License

Vq is available under the terms of either of the following licenses:

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
