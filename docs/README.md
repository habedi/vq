# Vq Documentation

The building blocks of the Vq library are the **vectors** and **quantizers**.

Currently, Vq supports the following quantization algorithms:

- **Binary Quantization (BQ)**: This is the simplest form of quantization where each dimension of the input vector is
  quantized to a binary values.
- **Scalar Quantization (SQ)**: This algorithm quantizes the input vector to a set of scalar values called levels.
- **Product Quantization (PQ)**: This algorithm quantizes the input vector into multiple subvectors and then quantizes
  each
  subvector independently.
- **Optimized Product Quantization (OPQ)**: This is an optimized version of the PQ algorithm that learns a rotation
  matrix
  to minimize the quantization error.
- **Tree-structured Vector Quantization (TSVQ)**: This algorithm quantizes the input vector using a tree structure
  recursively.
- **Residual Vector Quantization (RVQ)**: This algorithm quantizes the input vector by encoding the residual between the
  input vector and the quantized vector.

