# AGENTS.md

This file provides guidance to coding agents collaborating on this repository.

## Mission

Vq is a vector quantization library for Rust.
It provides implementations of binary quantization (BQ), scalar quantization (SQ), product quantization (PQ), and tree-structured vector quantization (TSVQ).
The project also ships Python bindings through the `pyvq` package.

Priorities, in order:

1. Correctness of quantization, dequantization, distance calculations, and reconstruction behavior.
2. A stable, simple Rust API centered on the `Quantizer` trait and public quantizer types.
3. Reliable Python bindings that match the Rust behavior where exposed.
4. Performance for training, quantization, and evaluation without sacrificing correctness.
5. Memory safety across Rust code, C SIMD FFI, and Python bindings.

## Core Rules

- Use English for code, comments, docs, and tests.
- Prefer small, focused changes over large refactoring.
- Add comments only when they clarify non-obvious behavior.
- Do not add features, error handling, dependencies, or abstractions beyond what is needed for the current task.
- Keep public behavior deterministic where seeds are accepted or required.
- Preserve compatibility with Rust 1.85 or later unless a task explicitly changes the MSRV.

## Writing Style

- Write in simple, plain English. Use short sentences and everyday words.
- Use Oxford commas in inline lists: "a, b, and c" not "a, b, c".
- Do not use em dashes. Restructure the sentence, or use a colon or semicolon instead.
- Avoid colorful adjectives and adverbs. Write "vector quantization library" not "powerful vector quantization library".
- Use noun phrases for checklist items, not imperative verbs. Write "distance metric coverage" not "cover distance metrics".
- Headings in Markdown files must be in title case: "Build from Source" not "Build from source". Minor words (a, an, the, and, but, or, for, in, on,
  at, to, by, of, is, are, was, were, be) stay lowercase unless they are the first word.

## Repository Layout

- `src/lib.rs`: Rust crate root. Re-exports public quantizers, distance types, error types, and core traits.
- `src/bq.rs`: Binary quantization implementation.
- `src/sq.rs`: Scalar quantization implementation.
- `src/pq.rs`: Product quantization implementation.
- `src/tsvq.rs`: Tree-structured vector quantization implementation.
- `src/core/`: Shared distance, error, quantizer trait, vector, and optional SIMD FFI modules.
- `src/bin/`: Evaluation binaries used by `make eval` and `make eval-all`.
- `tests/`: Rust integration tests and test data helpers.
- `external/hsdlib/`: C SIMD acceleration library used by the `simd` feature through `build.rs`.
- `pyvq/`: Python package and bindings built with Maturin.
- `docs/`: Main project documentation and assets.
- `.github/workflows/`: CI workflows for tests, linting, documentation, and packaging.
- `Cargo.toml`: Rust package metadata, feature flags, dependencies, and profile settings.
- `pyproject.toml`: Python development environment, test, coverage, mypy, and Ruff configuration.
- `Makefile`: Main automation entry point for formatting, tests, linting, builds, docs, evaluation, and Python workflows.

## Architecture

### Rust Crate

The Rust crate targets edition 2024 and Rust 1.85 or later. Public consumers primarily use the quantizer types (`BinaryQuantizer`, `ScalarQuantizer`,
`ProductQuantizer`, and `TSVQ`), the `Quantizer` trait, `Distance`, and the `VqResult` / `VqError` error types. Keep public API changes deliberate
because they affect both Rust users and Python bindings.

### Quantizer API

Quantizers expose training or construction, quantization, and dequantization paths through concrete types and the shared `Quantizer` trait. New or
changed quantizer behavior should include tests for:

- Valid inputs and expected output shapes.
- Invalid dimensions, invalid parameters, and empty data when applicable.
- Distance metric behavior where the quantizer supports distances.
- Deterministic output when a seed is part of the API.

### Feature Flags

- `default`: No optional features.
- `binaries`: Evaluation binaries.
- `parallel`: Enables Rayon-backed parallel training paths.
- `simd`: Enables C SIMD acceleration through `external/hsdlib` and `build.rs`.
- `all`: Enables `binaries`, `parallel`, and `simd`.

When changing feature-gated code, verify that the relevant feature combinations still compile. At minimum, test the feature set touched by the change
and the `all` feature set when practical.

### SIMD and FFI

The `simd` feature uses `external/hsdlib` through C FFI. Treat changes in `build.rs`, `external/hsdlib/`, and FFI modules as memory-safety-sensitive.
Validate pointer lifetimes, slice lengths, alignment assumptions, CPU feature detection, and fallback behavior. Run sanitizer or careful checks when
changing unsafe Rust or C integration paths.

### Python Bindings

The Python package lives under `pyvq/` and is built with Maturin. Python tests are configured through `pyproject.toml` and run with `make test-py`.
When Rust API changes affect exported Python behavior, update Python bindings, type stubs, docs, and tests together.

## Rust Conventions

- Rust edition: 2024.
- MSRV: Rust 1.85.
- Formatting: `cargo fmt` through `make format`.
- Linting: `cargo clippy` through `make lint`, with warnings denied and `unwrap` / `expect` disallowed.
- Naming: follow standard Rust naming conventions (`snake_case` functions and variables, `CamelCase` types, `SCREAMING_SNAKE_CASE` constants).
- Error handling: prefer `VqResult` and `VqError` for library errors. Do not introduce panics for recoverable invalid input.
- Randomness: expose or use deterministic seeds for tests and reproducible algorithms.
- Dependencies: avoid new dependencies unless they are necessary and justified.

## Python Conventions

- Python version: `>=3.10,<4.0`.
- Package tooling: `uv`, Maturin, Pytest, MyPy, and Ruff.
- Formatting and lint settings are in `pyproject.toml`.
- Use type annotations for new Python code.
- Keep Python behavior aligned with the Rust implementation.

## Required Validation

Run the relevant targets for any change:

| Target          | Command           | What It Runs                                      |
|-----------------|-------------------|---------------------------------------------------|
| Rust format     | `make format`     | `cargo fmt`                                       |
| Rust tests      | `make test`       | Format, doctests, and `cargo test --features all` |
| Rust doctests   | `make doctest`    | Rust documentation tests with all features        |
| Differential    | `make test-diff`  | Reference comparisons with and without features   |
| Rust benchmarks | `make bench`      | Criterion benchmarks with all features            |
| Rust lint       | `make lint`       | `cargo clippy` with warnings denied               |
| Rust build      | `make build`      | Release build                                     |
| Coverage        | `make coverage`   | Tarpaulin XML and HTML coverage reports           |
| Security audit  | `make audit`      | `cargo audit`                                     |
| Careful checks  | `make careful`    | `cargo careful run`                               |
| Rust docs       | `make docs`       | Rust documentation generation                     |
| Evaluation      | `make eval-all`   | Evaluation binaries for BQ, SQ, PQ, and TSVQ      |
| Python develop  | `make develop-py` | Maturin development install                       |
| Python tests    | `make test-py`    | Pytest for `pyvq`                                 |
| Python docs     | `make docs-py`    | PyVq MkDocs build                                 |
| Project docs    | `make docs-build` | Main MkDocs build                                 |
| Git hooks       | `make test-hooks` | Pre-commit hooks on all files                     |

For documentation-only changes, tests may be skipped after reviewing the changed Markdown. For public Rust API changes, run `make test`, `make lint`,
and relevant documentation checks. For Python binding changes, run `make test-py` and update type stubs when needed. For unsafe Rust, FFI, SIMD, or
memory-sensitive changes, run `make careful` or another appropriate sanitizer-style check.

## First Contribution Flow

1. Read the relevant module under `src/` and any matching tests under `tests/`.
2. Identify whether the change affects the Rust API, Python bindings, feature flags, or SIMD FFI.
3. Implement the smallest change that covers the requirement.
4. Add or update tests for new behavior, bug fixes, and edge cases.
5. Run `make format` and the relevant validation targets.
6. Update docs, examples, or type stubs when public behavior changes.

Good first tasks:

- A regression test for a quantizer edge case.
- A small documentation correction in `README.md`, `docs/`, or `pyvq/docs/`.
- A targeted fix for invalid input handling.
- A benchmark or evaluation cleanup that does not change public API behavior.

## Testing Expectations

- Prefer deterministic tests with fixed seeds for randomized training or sampling.
- Cover BQ, SQ, PQ, and TSVQ behavior when shared logic changes.
- Include shape and dimensionality checks for vector inputs and outputs.
- Include negative tests for invalid parameters and incompatible vector dimensions.
- Compare floating-point results with tolerances instead of exact equality when appropriate.
- Differential tests in `tests/differential_tests.rs` and `pyvq/tests/test_differential.py` compare the library against reference implementations; extend them when adding a quantizer or distance.
- Keep long-running benchmark or evaluation workloads out of regular unit tests.
- Test Python binding behavior when exposed Python APIs change.

## Change Design Checklist

Before coding:

1. Public API surface: Rust exports, Python bindings, docs, and type stubs.
2. Feature flags: default, `parallel`, `simd`, `binaries`, and `all`.
3. Algorithm behavior: output shape, reconstruction error expectations, and distance semantics.
4. Error behavior: invalid inputs, empty data, and dimension mismatches.
5. Memory safety: unsafe Rust, FFI pointers, C buffers, and ownership across Python bindings.
6. Performance impact: avoid unnecessary allocations and preserve existing parallel or SIMD paths.

Before submitting:

1. `make format` has been run or the change is Markdown-only.
2. Relevant Rust tests pass for the affected feature set.
3. `make lint` passes when Rust code changes.
4. Python tests pass when `pyvq/` changes.
5. Docs or examples are updated when public behavior changes.
6. Unsafe, FFI, or memory-sensitive changes have an additional safety validation.

## Commit and PR Hygiene

- Keep commits scoped to one logical change.
- Do not commit generated artifacts such as coverage reports, built wheels, `target/`, or `site/` unless explicitly requested.
- PR descriptions should include:
    1. Behavioral change summary.
    2. Tests added or updated.
    3. Validation commands run locally, such as `make test`, `make lint`, or `make test-py`.
