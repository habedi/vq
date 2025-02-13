# Performance Evaluation

This directory contains the Jupyter notebooks used to evaluate the performance of the implemented algorithms.

## Setup Python Environment

```bash
# Install poetry
pip install poetry # or uv tool install poetry
```

```bash
# Create Python environment and install dependencies
poetry install --no-root
```

```bash
# Start Jupyter Notebook
poetry run jupyter notebook
```

## Generate Experiment Data

```bash
# Run all experiments and store results in the `data` directory
make eval-all
```

Check the [../src/bin/utils.rs](../src/bin/utils.rs) for the parameters used in the experiments.

## Evaluation Metrics

The performance of each implementation is measured using a few metrics that capture both efficiency and quality.
These metrics include:

- **Training Time (ms):**  
  The time needed to train the quantization algorithm on the synthetic dataset.

- **Quantization Time (ms):**  
  The time taken to quantize all the vectors in the dataset.

- **Reconstruction Error:**  
  The [mean squared error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) between the original vectors and
  their quantized versions is used to measure the reconstruction quality.
  A lower error indicates better reconstruction quality.

- **Recall@10:**  
  The fraction of shared nearest neighbors between the original and quantized vectors, using the 10 nearest neighbors
  determined by Euclidean distance.
  A higher recall means better preservation of neighborhood relationships.

Additionally, the dataset is created using a uniform random distribution over the interval \([0.0, 1.0]\) with a fixed
seed for reproducibility.

### Example CSV Output

```csv
n_samples,n_dims,training_time_ms,quantization_time_ms,reconstruction_error,recall
1000,128,XX.XX,YY.YY,ZZ.ZZZZ,AA.AAAA
5000,128,XX.XX,YY.YY,ZZ.ZZZZ,AA.AAAA
...
```

### Acknowledgements

The evaluation framework is inspired by the code from
[here](https://github.com/oramasearch/vector_quantizer/blob/main/src/bin/quality_check.rs).
