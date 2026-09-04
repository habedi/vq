# API Reference

This page provides an overview of Vq's public API. For detailed documentation, see [docs.rs/vq](https://docs.rs/vq).

## Core Trait

### `Quantizer`

All quantization algorithms implement this trait:

```rust
pub trait Quantizer {
    type QuantizedOutput;

    fn quantize(&self, vector: &[f32]) -> VqResult<Self::QuantizedOutput>;
    fn dequantize(&self, quantized: &Self::QuantizedOutput) -> VqResult<Vec<f32>>;

    // Provided methods; parallel when the `parallel` feature is enabled
    fn quantize_batch(&self, vectors: &[&[f32]]) -> VqResult<Vec<Self::QuantizedOutput>>;
    fn dequantize_batch(&self, quantized: &[Self::QuantizedOutput]) -> VqResult<Vec<Vec<f32>>>;
}
```

`quantize_batch` and `dequantize_batch` process a slice of inputs and stop at the first error.

## Persistence

Every quantizer can be encoded to bytes or written to a file and restored later:

```rust
let bytes = pq.to_bytes()?;
let restored = ProductQuantizer::from_bytes(&bytes)?;

pq.save("model.vq")?;
let loaded = ProductQuantizer::load("model.vq")?;
```

The encoding is compact and versioned only by the crate: bytes written by one version of Vq are not guaranteed to load in another. Decoding validates the model and returns `VqError::Serialization` or `VqError::InvalidParameter` for malformed input, and `VqError::Io` when a file cannot be read or written.

## Quantizers

### `BinaryQuantizer`

Maps values above/below a threshold to two discrete levels.

```rust
// Constructor
BinaryQuantizer::new(threshold: f32, low: u8, high: u8) -> VqResult<Self>

// Getters
fn threshold(&self) -> f32
fn low(&self) -> u8
fn high(&self) -> u8

// Output type: Vec<u8>
```

### `ScalarQuantizer`

Uniformly quantizes values in a range to discrete levels (2-256).

```rust
// Constructor
ScalarQuantizer::new(min: f32, max: f32, levels: usize) -> VqResult<Self>

// Getters
fn min(&self) -> f32
fn max(&self) -> f32
fn levels(&self) -> usize
fn step(&self) -> f32

// Output type: Vec<u8>
```

### `ProductQuantizer`

Divides vectors into subspaces and quantizes each using learned codebooks.

```rust
// Constructor (requires training)
ProductQuantizer::new(
    training_data: &[&[f32]],
    m: usize,           // number of subspaces
    k: usize,           // centroids per subspace
    max_iters: usize,
    distance: Distance,
    seed: u64,
) -> VqResult<Self>

// Getters
fn num_subspaces(&self) -> usize
fn sub_dim(&self) -> usize
fn dim(&self) -> usize
fn distance_metric(&self) -> &'static str  // "euclidean", "cosine", etc.

// Output type: Vec<f16>
```

### `TSVQ`

Tree-structured vector quantizer using hierarchical clustering.

```rust
// Constructor (requires training)
TSVQ::new(
    training_data: &[&[f32]],
    max_depth: usize,
    distance: Distance,
) -> VqResult<Self>

// Getters
fn dim(&self) -> usize
fn distance_metric(&self) -> &'static str  // "euclidean", "cosine", etc.

// Output type: Vec<f16>
```

## Distance Metrics

### `Distance`

Enum for computing vector distances:

```rust
pub enum Distance {
    SquaredEuclidean,  // L2²
    Euclidean,         // L2
    Manhattan,         // L1
    CosineDistance,    // 1 - cosine_similarity
}

// Usage
Distance::Euclidean.compute(a: &[f32], b: &[f32]) -> VqResult<f32>

// Get metric name
Distance::Euclidean.name() -> &'static str  // "euclidean"
```

## Vector Operations (Advanced)

The `Vector<T>` type provides fallible arithmetic operations:

```rust
use vq::core::vector::Vector;

// Fallible operations (return Result)
vec_a.try_add(&vec_b) -> VqResult<Vector<T>>
vec_a.try_sub(&vec_b) -> VqResult<Vector<T>>
vec_a.try_div(scalar) -> VqResult<Vector<T>>

// Trait-based operations (panic on error)
&vec_a + &vec_b  // panics if dimensions mismatch
&vec_a - &vec_b
&vec_a * scalar
&vec_a / scalar

// Other methods
vec_a.dot(&vec_b) -> T     // panics if dimensions mismatch
vec_a.norm() -> T
vec_a.distance2(&vec_b) -> T
```

**Note:** Use `try_*` methods for safe error handling, or the trait operators for internal code where dimensions are guaranteed to match.

## Error Handling

### `VqError`

All operations return `VqResult<T>`, which is `Result<T, VqError>`:

```rust
pub enum VqError {
    /// Vectors have mismatched dimensions
    DimensionMismatch { expected: usize, found: usize },

    /// Input data is empty
    EmptyInput,

    /// A configuration parameter is invalid
    InvalidParameter { parameter: &'static str, reason: String },

    /// Input data contains invalid values (NaN, Infinity, etc.)
    InvalidData(String),

    /// FFI operation failed
    FfiError(String),
}
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `parallel` | Multi-threaded training for PQ and TSVQ |
| `simd` | SIMD acceleration (AVX/AVX2/AVX512/NEON/SVE) |

```bash
cargo add vq --features parallel,simd
```
