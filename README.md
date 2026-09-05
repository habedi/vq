## Vq

[![Tests](https://img.shields.io/github/actions/workflow/status/CogitatorTech/vq/tests.yml?label=tests&style=flat&labelColor=282c34&logo=github)](https://github.com/CogitatorTech/vq/actions/workflows/tests.yml)
[![Code Coverage](https://img.shields.io/codecov/c/github/CogitatorTech/vq?style=flat&labelColor=282c34&logo=codecov)](https://codecov.io/gh/CogitatorTech/vq)
[![Crates.io](https://img.shields.io/crates/v/vq?label=crates.io&style=flat&labelColor=282c34&logo=rust)](https://crates.io/crates/vq)
[![PyPI](https://img.shields.io/pypi/v/pyvq?label=pypi&style=flat&labelColor=282c34&logo=pypi&logoColor=white)](https://pypi.org/project/pyvq/)
[![Docs.rs](https://img.shields.io/docsrs/vq?label=docs.rs&style=flat&labelColor=282c34&logo=docs.rs)](https://docs.rs/vq)
[![Documentation](https://img.shields.io/badge/docs-read-507ec6?style=flat&labelColor=282c34&logo=readthedocs)](https://CogitatorTech.github.io/vq)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-007ec6?label=license&style=flat&labelColor=282c34&logo=open-source-initiative)](https://github.com/CogitatorTech/vq#license)

Vq (**v**[ector] **q**[uantizer]) is a vector quantization library for Rust.
It provides implementations of popular quantization algorithms, including binary quantization (BQ), scalar quantization (SQ),
product quantization (PQ), and tree-structured vector quantization (TSVQ).

Vector quantization is a technique to reduce the size of high-dimensional vectors by approximating them with a smaller set of representative vectors.
It can be used for various applications such as data compression and nearest neighbor search to reduce the memory footprint and speed up search.
For example, vector quantization can be used to reduce the size of data stored in a vector database or speed up the response time of a RAG-based
application.
For more information about vector quantization, check out
[this article](https://docs.weaviate.io/weaviate/concepts/vector-quantization) from Weaviate documentation.

### Features

- A simple and generic API for all quantizers
- Can reduce storage size of input vectors, at least 50% (2x)
- Good performance via SIMD acceleration (using [Hsdlib](https://github.com/habedi/hsdlib)), multi-threading, and zero-copying
- Support for multiple distances including Euclidean, cosine, and Manhattan distances
- Batch quantization and saving and loading of trained quantizers
- Python 🐍 bindings via [PyVq](https://pypi.org/project/pyvq/) package

See [ROADMAP.md](ROADMAP.md) for the list of implemented and planned features and the [benchmarks page](docs/benchmarks.md) for performance numbers.

> [!IMPORTANT]
> Vq is in early development, so bugs and breaking changes are expected.
> Please use the [issues page](https://github.com/CogitatorTech/vq/issues) to report bugs or request features.

### Supported Algorithms

| Algorithm           | Training Complexity | Quantization Complexity | Supported Distances | Input Type | Output Type | Storage Size Reduction |
|---------------------|---------------------|-------------------------|---------------------|------------|-------------|------------------------|
| [BQ](src/bq.rs)     | $O(1)$              | $O(nd)$                 | —                   | `&[f32]`   | `Vec<u8>`   | 75%                    |
| [SQ](src/sq.rs)     | $O(1)$              | $O(nd)$                 | —                   | `&[f32]`   | `Vec<u8>`   | 75%                    |
| [PQ](src/pq.rs)     | $O(nkd)$            | $O(nd)$                 | All                 | `&[f32]`   | `Vec<f16>`  | 50%                    |
| [TSVQ](src/tsvq.rs) | $O(n \log k)$       | $O(d \log k)$           | All                 | `&[f32]`   | `Vec<f16>`  | 50%                    |

- $n$: number of vectors
- $d$: dimensionality of vectors
- $k$: number of centroids or clusters

### Quantization Demo

Below is a visual comparison of different quantization algorithms applied to a 1024×1024 PNG image
(using [this Python script](pyvq/scripts/image_quantization_demo.py)):

<table>
<tr>
<td align="center"><b>Original (1.5 MB)</b><br><img src="docs/assets/images/nixon_visions_base_1024.png" width="200"></td>
<td align="center"><b>Scalar 8 (288 KB, -81.7%)</b><br><img src="docs/assets/images/nixon_visions_scalar8_1024.png" width="200"></td>
<td align="center"><b>Scalar 16 (421 KB, -73.3%)</b><br><img src="docs/assets/images/nixon_visions_scalar16_1024.png" width="200"></td>
</tr>
<tr>
<td align="center"><b>Binary (72 KB, -95.5%)</b><br><img src="docs/assets/images/nixon_visions_binary_1024.png" width="200"></td>
<td align="center"><b>PQ 8×16 (61 KB, -96.2%)</b><br><img src="docs/assets/images/nixon_visions_pq8x16_1024.png" width="200"></td>
<td align="center"><b>TSVQ depth=6 (111 KB, -92.9%)</b><br><img src="docs/assets/images/nixon_visions_tsvq6_1024.png" width="200"></td>
</tr>
</table>

> [!NOTE]
> The binary and scalar quantizers are applied per-channel (each pixel value independently),
> while PQ and TSVQ are applied per-row (each image row as a vector).
> PQ and TSVQ treat each image row as a high-dimensional vector, which causes the horizontal banding artifacts.
> Vq is primarily designed for embedding vector compression (like the ones stored in a vector database),
> where PQ and TSVQ are applied to vectors of typical dimensions 128–1536.

---

### Getting Started

#### Installing Vq

```bash
cargo add vq --features parallel,simd
```

> [!NOTE]
> The `parallel` and `simd` features enables multi-threading support and SIMD acceleration support for training phase of PQ and TSVQ algorithms.
> This can significantly speed up training time, especially for large datasets.
> Note that to enable the `simd` feature, a modern C compiler (like GCC or Clang) that supports C11 standard is needed.

*Vq requires Rust 1.85 or later.*

#### Installing PyVq

```bash
pip install pyvq
```

Python bindings for Vq are available via [PyVq](https://pypi.org/project/pyvq/) package.
For more information, check out the [pyvq](pyvq) directory.

---

### Documentation

The Vq documentation is available [here](https://CogitatorTech.github.io/vq)
and the Rust API reference is available on [docs.rs/vq](https://docs.rs/vq).

#### Quick Example

Here's a simple example using the BQ and SQ algorithms to quantize vectors:

```rust
use vq::{BinaryQuantizer, ScalarQuantizer, Quantizer, VqResult};

fn main() -> VqResult<()> {
    // Binary quantization
    let bq = BinaryQuantizer::new(0.0, 0, 1)?;
    let quantized = bq.quantize(&[0.5, -0.3, 0.8])?;

    // Scalar quantization
    let sq = ScalarQuantizer::new(0.0, 1.0, 256)?;
    let quantized = sq.quantize(&[0.1, 0.5, 0.9])?;

    Ok(())
}
```

#### Product Quantizer Example

```rust
use vq::{ProductQuantizer, Distance, Quantizer, VqResult};

fn main() -> VqResult<()> {
    // Training data (each inner slice is a vector)
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..10).map(|j| ((i + j) % 50) as f32).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    // Train the quantizer
    let pq = ProductQuantizer::new(
        &training_refs,
        2,  // m: number of subspaces
        4,  // k: centroids per subspace
        10, // max_iters
        Distance::Euclidean,
        42, // seed
    )?;

    // Quantize a vector
    let quantized = pq.quantize(&training[0])?;

    Ok(())
}
```

### Benchmarks

You can follow the instructions below to run the benchmarks locally on your machine.

```bash
git clone --recursive https://github.com/CogitatorTech/vq.git
cd vq
```

```bash
make eval-all
```

> [!NOTE]
> To run the benchmarks, you need to have GNU Make installed.
> The `make eval-all` command will run each quantizer on a set of high-dimensional synthetic data and report runtime (in millisecond)
> and reconstruction error (in mean squared error).
> See [src/bin/common.rs](src/bin/common.rs) for parameters used in the benchmarks like size of the training data, dimensions, etc.

---

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

### License

Vq is available under either of the following licenses:

* MIT License ([LICENSE-MIT](LICENSE-MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

### Acknowledgements

* This project uses [Hsdlib](https://github.com/habedi/hsdlib) C library for SIMD acceleration.
