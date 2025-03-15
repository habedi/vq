## Vq

[<img alt="tests" src="https://img.shields.io/github/actions/workflow/status/habedi/vq/tests.yml?label=tests&style=flat&labelColor=555555&logo=github" height="20">](https://github.com/habedi/vq/actions/workflows/tests.yml)
[<img alt="lints" src="https://img.shields.io/github/actions/workflow/status/habedi/vq/lints.yml?label=lints&style=flat&labelColor=555555&logo=github" height="20">](https://github.com/habedi/vq/actions/workflows/lints.yml)
[<img alt="code coverage" src="https://img.shields.io/codecov/c/github/habedi/vq?style=flat&labelColor=555555&logo=codecov" height="20">](https://codecov.io/gh/habedi/vq)
[<img alt="codefactor" src="https://img.shields.io/codefactor/grade/github/habedi/vq?style=flat&labelColor=555555&logo=codefactor" height="20">](https://www.codefactor.io/repository/github/habedi/vq)
[<img alt="crates.io" src="https://img.shields.io/crates/v/vq.svg?label=crates.io&style=flat&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/vq)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-vq-66c2a5?label=docs.rs&style=flat&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/vq)
[<img alt="downloads" src="https://img.shields.io/crates/d/vq?label=downloads&style=flat&labelColor=555555&logo=rust" height="20">](https://crates.io/crates/vq)
[<img alt="msrv" src="https://img.shields.io/badge/msrv-1.83.0-orange?label=msrv&style=flat&labelColor=555555&logo=rust" height="20">](https://github.com/rust-lang/rust/releases/tag/1.83.0)
[<img alt="dependencies" src="https://deps.rs/repo/github/habedi/vq/status.svg">](https://deps.rs/repo/github/habedi/vq)
[<img alt="license" src="https://img.shields.io/badge/license-mit%2Fapache--2.0-007ec6?label=license&style=flat&labelColor=555555&logo=open-source-initiative" height="20">](https://github.com/habedi/vq)

Vq (**v**[ector] **q**[uantizer]) is a Rust library that implements several
popular [vector quantization](https://en.wikipedia.org/wiki/Vector_quantization) algorithms including binary, scalar,
and product quantization.
It provides a simple API for data compression that helps reduce memory usage and computational overhead.

### Features

- Implemented Algorithms:
    - [Binary Quantization (BQ)](src/bq.rs)
    - [Scalar Quantization (SQ)](src/sq.rs)
    - [Product Quantization (PQ)](https://ieeexplore.ieee.org/document/5432202)
    - [Optimized Product Quantization (OPQ)](https://ieeexplore.ieee.org/document/6619223)
    - [Tree-structured Vector Quantization (TSVQ)](https://ieeexplore.ieee.org/document/515493)
    - [Residual Vector Quantization (RVQ)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3231071/)

- Parallelized vector operations for large vectors using [Rayon](https://crates.io/crates/rayon).
- Flexible quantization algorithm implementations that support using various distance metrics such as Euclidean, Cosine,
  and Manhattan distances.
- Support for quantizing vectors of `f32` to `f16` (using [half](https://crates.io/crates/half)) or `u8` data types.
- Simple, intuitive, and uniform API for all quantization algorithms.

### Installation

```bash
cargo add vq
```

*Vq requires Rust 1.83 or later.*

### Documentation

Find the latest API documentation on [docs.rs](https://docs.rs/vq).

Check out [vq_examples.rs](src/bin/vq_examples.rs) the [tests](tests) directory for detailed examples of using Vq.

#### Quick Example

Here's a simple example using the SQ algorithm to quantize a vector:

```rust
use vq::sq::ScalarQuantizer;
use vq::vector::Vector;

fn main() {
    // Create a scalar quantizer for values in the range [0.0, 1.0] with 256 levels.
    let quantizer = ScalarQuantizer::fit(0.0, 1.0, 256);

    // Create an input vector.
    let input = Vector::new(vec![0.1, 0.5, -0.8, -0.3, 0.9]);

    // Quantize the input vector.
    let quantized_input = quantizer.quantize(&input);

    println!("Quantized input vector: {}", quantized_input);
}
```

### Performance

Check out the [notebooks](notebooks/) directory for information on how to evaluate the performance of the implemented
algorithms.
Additionally, see the content of [src/bin](src/bin/) directory for the scripts used for the evaluation.

> On a ThinkPad T14 laptop with an Intel i7-1355U CPU and 32GB of RAM, the performance of the PQ algorithm for
> quantizing one million vectors of 128 dimensions (into 16 subspaces with 256 centroids per subspace) is as follows:
>   - Training Time: 232.5 seconds
>   - Quantization Time: 34.1 seconds
>   - Reconstruction Error (MSE): 0.02
>   - Recall@10: 0.19

### Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

### License

Vq is available under the terms of either of the following licenses:

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
