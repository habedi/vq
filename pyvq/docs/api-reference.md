# API Reference

Complete API documentation for PyVq.

## Distance

```python
class Distance:
    """Compute vector distances with SIMD acceleration."""
    
    @staticmethod
    def euclidean() -> Distance
    
    @staticmethod
    def squared_euclidean() -> Distance
    
    @staticmethod
    def manhattan() -> Distance
    
    @staticmethod
    def cosine() -> Distance
    
    def compute(self, a: np.ndarray, b: np.ndarray) -> float
        """Compute distance between two float32 arrays."""
```

**Example:**
```python
dist = pyvq.Distance.euclidean()
result = dist.compute(np.array([1.0, 2.0], dtype=np.float32),
                      np.array([3.0, 4.0], dtype=np.float32))
```

---

## BinaryQuantizer

```python
class BinaryQuantizer:
    """Maps values to 0 or 1 based on a threshold."""
    
    def __init__(self, threshold: float, low: int = 0, high: int = 1)
    
    def quantize(self, values: np.ndarray) -> np.ndarray
        """Input: float32, Output: uint8"""
    
    def dequantize(self, codes: np.ndarray) -> np.ndarray
    def quantize_batch(self, vectors: np.ndarray) -> np.ndarray
    def dequantize_batch(self, codes: np.ndarray) -> np.ndarray
        """Input: uint8, Output: float32"""
    
    # Properties
    threshold: float
    low: int
    high: int
```

**Example:**
```python
bq = pyvq.BinaryQuantizer(threshold=0.0)
codes = bq.quantize(np.array([-0.5, 0.5], dtype=np.float32))
# Returns: [0, 1]
```

---

## ScalarQuantizer

```python
class ScalarQuantizer:
    """Uniformly quantizes values to discrete levels."""
    
    def __init__(self, min: float, max: float, levels: int = 256)
    
    def quantize(self, values: np.ndarray) -> np.ndarray
        """Input: float32, Output: uint8"""
    
    def dequantize(self, codes: np.ndarray) -> np.ndarray
    def quantize_batch(self, vectors: np.ndarray) -> np.ndarray
    def dequantize_batch(self, codes: np.ndarray) -> np.ndarray
        """Input: uint8, Output: float32"""
    
    # Properties
    min: float
    max: float
    levels: int
    step: float
```

**Example:**
```python
sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)
codes = sq.quantize(np.array([0.0, 0.5], dtype=np.float32))
reconstructed = sq.dequantize(codes)
```

---

## ProductQuantizer

```python
class ProductQuantizer:
    """Divides vectors into subspaces and quantizes each separately."""
    
    def __init__(
        self,
        training_data: np.ndarray,  # 2D float32 array
        num_subspaces: int,
        num_centroids: int,
        max_iters: int = 10,
        distance: Distance = None,
        seed: int = 42
    )
    
    def quantize(self, vector: np.ndarray) -> np.ndarray
        """Input: float32, Output: float16"""
    
    def dequantize(self, codes: np.ndarray) -> np.ndarray
    def quantize_batch(self, vectors: np.ndarray) -> np.ndarray
    def dequantize_batch(self, codes: np.ndarray) -> np.ndarray
        """Input: float16, Output: float32"""
    
    # Properties
    num_subspaces: int
    sub_dim: int
    dim: int
```

**Example:**
```python
training = np.random.randn(100, 16).astype(np.float32)
pq = pyvq.ProductQuantizer(
    training_data=training,
    num_subspaces=4,
    num_centroids=8,
    distance=pyvq.Distance.euclidean()
)
codes = pq.quantize(training[0])
```

---

## TSVQ

```python
class TSVQ:
    """Tree-structured vector quantizer using hierarchical clustering."""
    
    def __init__(
        self,
        training_data: np.ndarray,  # 2D float32 array
        max_depth: int,
        distance: Distance = None
    )
    
    def quantize(self, vector: np.ndarray) -> np.ndarray
        """Input: float32, Output: float16"""
    
    def dequantize(self, codes: np.ndarray) -> np.ndarray
    def quantize_batch(self, vectors: np.ndarray) -> np.ndarray
    def dequantize_batch(self, codes: np.ndarray) -> np.ndarray
        """Input: float16, Output: float32"""
    
    # Properties
    dim: int
```

**Example:**
```python
training = np.random.randn(100, 32).astype(np.float32)
tsvq = pyvq.TSVQ(
    training_data=training,
    max_depth=5,
    distance=pyvq.Distance.squared_euclidean()
)
codes = tsvq.quantize(training[0])
```

---

## Batch Methods

Every quantizer also accepts a 2-D array of shape `(n, dim)`:

```python
codes = sq.quantize_batch(np.random.rand(1000, 128).astype(np.float32))
reconstructed = sq.dequantize_batch(codes)
```

`quantize_batch` and `dequantize_batch` release the GIL and, when the extension is built with the `parallel` feature, process rows in parallel. They raise `ValueError` on the first row with an invalid dimension. Inputs must be C-contiguous.

## Persistence

Trained quantizers can be saved and restored:

```python
pq.save("model.vq")
restored = pyvq.ProductQuantizer.load("model.vq")

blob = tsvq.to_bytes()
restored = pyvq.TSVQ.from_bytes(blob)
```

`from_bytes` raises `ValueError` for malformed data. `save` and `load` raise `OSError` on file errors. Files written by the Rust crate and by PyVq use the same format.

## Utility Functions

```python
def get_simd_backend() -> str:
    """Returns the active SIMD backend (e.g., 'AVX2 (Auto)')."""
```

## Type Summary

| Quantizer | Input | quantize() Output | dequantize() Output |
|-----------|-------|-------------------|---------------------|
| BinaryQuantizer | float32 | uint8 | float32 |
| ScalarQuantizer | float32 | uint8 | float32 |
| ProductQuantizer | float32 | float16 | float32 |
| TSVQ | float32 | float16 | float32 |
