# Vq: A Vector Quantization Library for Rust

[<img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/habedi/vq/tests.yml?label=Tests&style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/habedi/vq/actions/workflows/tests.yml)
[<img alt="Code Coverage" src="https://img.shields.io/codecov/c/github/habedi/vq?style=for-the-badge&labelColor=555555&logo=codecov" height="20">](https://codecov.io/gh/habedi/vq)
[<img alt="CodeFactor" src="https://img.shields.io/codefactor/grade/github/habedi/vq?style=for-the-badge&labelColor=555555&logo=codefactor" height="20">](https://www.codefactor.io/repository/github/habedi/vq)
[<img alt="Crates.io" src="https://img.shields.io/crates/v/vq.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/vq)
[<img alt="Docs.rs" src="https://img.shields.io/badge/docs.rs-vq-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/vq)
[<img alt="Downloads" src="https://img.shields.io/crates/d/vq?style=for-the-badge&labelColor=555555&logo=rust" height="20">](https://crates.io/crates/vq)
[<img alt="Docs" src="https://img.shields.io/badge/docs-latest-3776ab?style=for-the-badge&labelColor=555555&logo=readthedocs" height="20">](docs)
[<img alt="License" src="https://img.shields.io/badge/license-MIT%2FApache--2.0-007ec6?style=for-the-badge&labelColor=555555&logo=open-source-initiative" height="20">](https://github.com/habedi/vq)

**Vq** is a Rust library that implements several popular vector quantization algorithms.
It's designed to be fast, simple, and easy to use.

## What It Offers

- [**Binary Quantization (BQ)**](src/bq.rs)
- [**Scalar Quantization (SQ)**](src/sq.rs)
- [**Product Quantization (PQ)**](https://ieeexplore.ieee.org/document/5432202)
- [**Optimized Product Quantization (OPQ)**](https://ieeexplore.ieee.org/document/6619223)
- [**Tree-structured Vector Quantization (TSVQ)**](https://ieeexplore.ieee.org/document/515493)
- [**Residual Vector Quantization (RVQ)**](https://pmc.ncbi.nlm.nih.gov/articles/PMC3231071/)

[//]: # (It uses SIMD for speed and has a clean API for working with vectors.)

## Installation

```bash
cargo add vq
```

## Documentation

The documentation for the latest release can be found [here](docs).

Additionally, check out the [tests](tests/) directory for detailed examples of how to use Vq.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

## License

Spart is available under the terms of either of the following licenses:

* MIT License ([LICENSE-MIT](LICENSE-MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
