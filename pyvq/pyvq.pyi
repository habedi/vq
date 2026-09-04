"""
Type stubs for PyVq - Python bindings for Vq vector quantization library.

This file provides type hints for IDEs and type checkers.
"""

import os
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def get_simd_backend() -> str:
    """
    Get the name of the currently active SIMD backend.

    Returns a string describing which SIMD implementation is being used
    for distance computations, such as "AVX2 (Auto)" or "NEON (Auto)".

    Returns:
        A string identifying the SIMD backend.
    """
    ...


class Distance:
    """
    A class for computing vector distances.

    Create using static methods like `Distance.euclidean()`.

    Example:
        >>> import numpy as np
        >>> dist = pyvq.Distance.euclidean()
        >>> a = np.array([1.0, 2.0], dtype=np.float32)
        >>> b = np.array([3.0, 4.0], dtype=np.float32)
        >>> dist.compute(a, b)
        2.8284...
    """

    def __init__(self, metric: str) -> None:
        """
        Create a Distance with a specified metric name.

        Args:
            metric: One of "euclidean", "squared_euclidean", "cosine", "manhattan"

        Raises:
            ValueError: If the metric name is invalid.
        """
        ...

    @staticmethod
    def euclidean() -> "Distance":
        """Create a Euclidean distance metric."""
        ...

    @staticmethod
    def squared_euclidean() -> "Distance":
        """Create a Squared Euclidean distance metric."""
        ...

    @staticmethod
    def manhattan() -> "Distance":
        """Create a Manhattan (L1) distance metric."""
        ...

    @staticmethod
    def cosine() -> "Distance":
        """Create a Cosine distance metric (1 - cosine similarity)."""
        ...

    def compute(self, a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        """
        Compute the distance between two vectors.

        Args:
            a: First vector as numpy array (float32).
            b: Second vector as numpy array (float32).

        Returns:
            The computed distance as a float.

        Raises:
            ValueError: If vectors have different lengths.
        """
        ...

    def __repr__(self) -> str: ...


class BinaryQuantizer:
    """
    Binary quantizer that maps values to 0 or 1 based on a threshold.

    Example:
        >>> import numpy as np
        >>> bq = pyvq.BinaryQuantizer(threshold=0.5, low=0, high=1)
        >>> data = np.array([0.3, 0.7, 0.5], dtype=np.float32)
        >>> codes = bq.quantize(data)  # Returns np.array([0, 1, 1], dtype=np.uint8)
    """

    def __init__(self, threshold: float, low: int = 0, high: int = 1) -> None:
        """
        Create a new BinaryQuantizer.

        Args:
            threshold: Values >= threshold map to high, values < threshold map to low.
            low: The output value for inputs below the threshold (0-255).
            high: The output value for inputs at or above the threshold (0-255).

        Raises:
            ValueError: If low >= high or threshold is NaN.
        """
        ...

    def quantize(self, values: NDArray[np.float32]) -> NDArray[np.uint8]:
        """
        Quantize a numpy array of floats to binary values.

        Args:
            values: numpy array of floating-point values (float32).

        Returns:
            numpy array of quantized values (uint8).
        """
        ...

    def dequantize(self, codes: NDArray[np.uint8]) -> NDArray[np.float32]:
        """
        Reconstruct approximate float values from quantized data.

        Args:
            codes: numpy array of quantized values (uint8).

        Returns:
            numpy array of reconstructed float values (float32).
        """
        ...

    def quantize_batch(self, vectors: NDArray[np.float32]) -> NDArray[np.uint8]:
        """
        Quantize every row of a 2-D array at once.

        Args:
            vectors: 2-D numpy array of shape (n, dim) with float32 values.

        Returns:
            2-D numpy array of shape (n, dim) with quantized values.

        Raises:
            ValueError: If a row has an invalid dimension.
        """
        ...

    def dequantize_batch(self, codes: NDArray[np.uint8]) -> NDArray[np.float32]:
        """
        Reconstruct every row of a 2-D array of codes at once.

        Args:
            codes: 2-D numpy array of shape (n, dim) with quantized values.

        Returns:
            2-D numpy array of shape (n, dim) with float32 values.

        Raises:
            ValueError: If a row has an invalid dimension.
        """
        ...

    def to_bytes(self) -> bytes:
        """Encode the quantizer into bytes that `from_bytes` can restore."""
        ...

    @staticmethod
    def from_bytes(data: bytes) -> "BinaryQuantizer":
        """
        Restore a quantizer from bytes produced by `to_bytes`.

        Raises:
            ValueError: If the bytes are malformed or describe an invalid quantizer.
        """
        ...

    def save(self, path: str | os.PathLike[str]) -> None:
        """
        Write the quantizer to a file.

        Raises:
            OSError: If the file cannot be written.
        """
        ...

    @staticmethod
    def load(path: str | os.PathLike[str]) -> "BinaryQuantizer":
        """
        Read a quantizer from a file written by `save`.

        Raises:
            OSError: If the file cannot be read or is malformed.
        """
        ...

    @property
    def threshold(self) -> float:
        """The threshold value."""
        ...

    @property
    def low(self) -> int:
        """The low quantization level."""
        ...

    @property
    def high(self) -> int:
        """The high quantization level."""
        ...

    def __repr__(self) -> str: ...


class ScalarQuantizer:
    """
    Scalar quantizer that uniformly quantizes values to discrete levels.

    Example:
        >>> import numpy as np
        >>> sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)
        >>> data = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        >>> codes = sq.quantize(data)
    """

    def __init__(self, min: float, max: float, levels: int = 256) -> None:
        """
        Create a new ScalarQuantizer.

        Args:
            min: Minimum value in the quantization range.
            max: Maximum value in the quantization range.
            levels: Number of quantization levels (2-256).

        Raises:
            ValueError: If max <= min, levels < 2 or > 256, or values are NaN/Infinity.
        """
        ...

    def quantize(self, values: NDArray[np.float32]) -> NDArray[np.uint8]:
        """
        Quantize a numpy array of floats to discrete levels.

        Args:
            values: numpy array of floating-point values (float32).

        Returns:
            numpy array of quantized level indices (uint8).
        """
        ...

    def dequantize(self, codes: NDArray[np.uint8]) -> NDArray[np.float32]:
        """
        Reconstruct approximate float values from quantized levels.

        Args:
            codes: numpy array of quantized level indices (uint8).

        Returns:
            numpy array of reconstructed float values (float32).
        """
        ...

    def quantize_batch(self, vectors: NDArray[np.float32]) -> NDArray[np.uint8]:
        """
        Quantize every row of a 2-D array at once.

        Args:
            vectors: 2-D numpy array of shape (n, dim) with float32 values.

        Returns:
            2-D numpy array of shape (n, dim) with quantized values.

        Raises:
            ValueError: If a row has an invalid dimension.
        """
        ...

    def dequantize_batch(self, codes: NDArray[np.uint8]) -> NDArray[np.float32]:
        """
        Reconstruct every row of a 2-D array of codes at once.

        Args:
            codes: 2-D numpy array of shape (n, dim) with quantized values.

        Returns:
            2-D numpy array of shape (n, dim) with float32 values.

        Raises:
            ValueError: If a row has an invalid dimension.
        """
        ...

    def to_bytes(self) -> bytes:
        """Encode the quantizer into bytes that `from_bytes` can restore."""
        ...

    @staticmethod
    def from_bytes(data: bytes) -> "ScalarQuantizer":
        """
        Restore a quantizer from bytes produced by `to_bytes`.

        Raises:
            ValueError: If the bytes are malformed or describe an invalid quantizer.
        """
        ...

    def save(self, path: str | os.PathLike[str]) -> None:
        """
        Write the quantizer to a file.

        Raises:
            OSError: If the file cannot be written.
        """
        ...

    @staticmethod
    def load(path: str | os.PathLike[str]) -> "ScalarQuantizer":
        """
        Read a quantizer from a file written by `save`.

        Raises:
            OSError: If the file cannot be read or is malformed.
        """
        ...

    @property
    def min(self) -> float:
        """The minimum value in the quantization range."""
        ...

    @property
    def max(self) -> float:
        """The maximum value in the quantization range."""
        ...

    @property
    def levels(self) -> int:
        """The number of quantization levels."""
        ...

    @property
    def step(self) -> float:
        """The step size between quantization levels."""
        ...

    def __repr__(self) -> str: ...


class ProductQuantizer:
    """
    Product quantizer that divides vectors into subspaces and quantizes each separately.

    Product quantization (PQ) splits high-dimensional vectors into smaller subspaces
    and quantizes each subspace independently using learned codebooks.

    Example:
        >>> import numpy as np
        >>> training = np.random.rand(100, 16).astype(np.float32)
        >>> pq = pyvq.ProductQuantizer(
        ...     training_data=training,
        ...     num_subspaces=4,
        ...     num_centroids=8,
        ...     max_iters=20,
        ...     distance=pyvq.Distance.euclidean(),
        ...     seed=42
        ... )
    """

    def __init__(
        self,
        training_data: NDArray[np.float32],
        num_subspaces: int,
        num_centroids: int,
        max_iters: int = 10,
        distance: Optional[Distance] = None,
        seed: int = 42,
    ) -> None:
        """
        Create a new ProductQuantizer.

        Args:
            training_data: 2D numpy array of training vectors (float32), shape (n_samples, dim).
            num_subspaces: Number of subspaces to divide vectors into (m).
            num_centroids: Number of centroids per subspace (k).
            max_iters: Maximum iterations for codebook training.
            distance: Distance metric to use.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If training data is empty, dimension < num_subspaces,
                        or dimension not divisible by num_subspaces.
        """
        ...

    def quantize(self, vector: NDArray[np.float32]) -> NDArray[np.float16]:
        """
        Quantize a vector.

        Args:
            vector: Input vector as numpy array (float32).

        Returns:
            Quantized representation as numpy array (float16).
        """
        ...

    def dequantize(self, codes: NDArray[np.float16]) -> NDArray[np.float32]:
        """
        Reconstruct a vector from its quantized representation.

        Args:
            codes: Quantized representation as numpy array (float16).

        Returns:
            Reconstructed vector as numpy array (float32).
        """
        ...

    def quantize_batch(self, vectors: NDArray[np.float32]) -> NDArray[np.float16]:
        """
        Quantize every row of a 2-D array at once.

        Args:
            vectors: 2-D numpy array of shape (n, dim) with float32 values.

        Returns:
            2-D numpy array of shape (n, dim) with quantized values.

        Raises:
            ValueError: If a row has an invalid dimension.
        """
        ...

    def dequantize_batch(self, codes: NDArray[np.float16]) -> NDArray[np.float32]:
        """
        Reconstruct every row of a 2-D array of codes at once.

        Args:
            codes: 2-D numpy array of shape (n, dim) with quantized values.

        Returns:
            2-D numpy array of shape (n, dim) with float32 values.

        Raises:
            ValueError: If a row has an invalid dimension.
        """
        ...

    def to_bytes(self) -> bytes:
        """Encode the quantizer into bytes that `from_bytes` can restore."""
        ...

    @staticmethod
    def from_bytes(data: bytes) -> "ProductQuantizer":
        """
        Restore a quantizer from bytes produced by `to_bytes`.

        Raises:
            ValueError: If the bytes are malformed or describe an invalid quantizer.
        """
        ...

    def save(self, path: str | os.PathLike[str]) -> None:
        """
        Write the quantizer to a file.

        Raises:
            OSError: If the file cannot be written.
        """
        ...

    @staticmethod
    def load(path: str | os.PathLike[str]) -> "ProductQuantizer":
        """
        Read a quantizer from a file written by `save`.

        Raises:
            OSError: If the file cannot be read or is malformed.
        """
        ...

    @property
    def num_subspaces(self) -> int:
        """The number of subspaces."""
        ...

    @property
    def sub_dim(self) -> int:
        """The dimension of each subspace."""
        ...

    @property
    def dim(self) -> int:
        """The expected input vector dimension."""
        ...

    def __repr__(self) -> str: ...


class TSVQ:
    """
    Tree-structured vector quantizer using hierarchical clustering.

    TSVQ builds a binary tree where each node represents a cluster centroid.
    Vectors are quantized by traversing the tree to find the nearest leaf node.

    Example:
        >>> import numpy as np
        >>> training = np.random.rand(100, 8).astype(np.float32)
        >>> tsvq = pyvq.TSVQ(
        ...     training_data=training,
        ...     max_depth=5,
        ...     distance=pyvq.Distance.euclidean()
        ... )
    """

    def __init__(
        self,
        training_data: NDArray[np.float32],
        max_depth: int,
        distance: Optional[Distance] = None,
    ) -> None:
        """
        Create a new Tree-Structured Vector Quantizer.

        Args:
            training_data: 2D numpy array of training vectors (float32), shape (n_samples, dim).
            max_depth: Maximum depth of the tree.
            distance: Distance metric to use.

        Raises:
            ValueError: If training data is empty.
        """
        ...

    def quantize(self, vector: NDArray[np.float32]) -> NDArray[np.float16]:
        """
        Quantize a vector.

        Args:
            vector: Input vector as numpy array (float32).

        Returns:
            Quantized representation (leaf centroid) as numpy array (float16).
        """
        ...

    def dequantize(self, codes: NDArray[np.float16]) -> NDArray[np.float32]:
        """
        Reconstruct a vector from its quantized representation.

        Args:
            codes: Quantized representation as numpy array (float16).

        Returns:
            Reconstructed vector as numpy array (float32).
        """
        ...

    def quantize_batch(self, vectors: NDArray[np.float32]) -> NDArray[np.float16]:
        """
        Quantize every row of a 2-D array at once.

        Args:
            vectors: 2-D numpy array of shape (n, dim) with float32 values.

        Returns:
            2-D numpy array of shape (n, dim) with quantized values.

        Raises:
            ValueError: If a row has an invalid dimension.
        """
        ...

    def dequantize_batch(self, codes: NDArray[np.float16]) -> NDArray[np.float32]:
        """
        Reconstruct every row of a 2-D array of codes at once.

        Args:
            codes: 2-D numpy array of shape (n, dim) with quantized values.

        Returns:
            2-D numpy array of shape (n, dim) with float32 values.

        Raises:
            ValueError: If a row has an invalid dimension.
        """
        ...

    def to_bytes(self) -> bytes:
        """Encode the quantizer into bytes that `from_bytes` can restore."""
        ...

    @staticmethod
    def from_bytes(data: bytes) -> "TSVQ":
        """
        Restore a quantizer from bytes produced by `to_bytes`.

        Raises:
            ValueError: If the bytes are malformed or describe an invalid quantizer.
        """
        ...

    def save(self, path: str | os.PathLike[str]) -> None:
        """
        Write the quantizer to a file.

        Raises:
            OSError: If the file cannot be written.
        """
        ...

    @staticmethod
    def load(path: str | os.PathLike[str]) -> "TSVQ":
        """
        Read a quantizer from a file written by `save`.

        Raises:
            OSError: If the file cannot be read or is malformed.
        """
        ...

    @property
    def dim(self) -> int:
        """The expected input vector dimension."""
        ...

    def __repr__(self) -> str: ...
