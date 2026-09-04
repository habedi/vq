# Variables
REPO             := github.com/CogitatorTech/vq
BINARY_NAME     := $(or $(PROJ_BINARY), $(notdir $(REPO)))
BINARY          := target/release/$(BINARY_NAME)
PATH            := /snap/bin:$(PATH)
DEBUG_VQ        := 1
RUST_LOG        := info
RUST_BACKTRACE  := off
WHEEL_DIR       := dist
PYVQ_DIR        := pyvq
PY_DEP_MNGR     := uv # Use `uv sync --all-extras` to make the environment
TEST_DATA_DIR   := tests/testdata
SHELL           := /bin/bash
MSRV            := 1.85

# Pinned versions for Rust development tools
TARPAULIN_VERSION=0.32.8
NEXTEST_VERSION=0.9.95
AUDIT_VERSION=0.21.2
CAREFUL_VERSION=0.4.8

# Find the latest built Python wheel file
WHEEL_FILE := $(shell ls $(PYVQ_DIR)/$(WHEEL_DIR)/pyvq-*.whl 2>/dev/null | head -n 1)

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show the help message for each target
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | \
	   awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

########################################################################################
## Rust targets
########################################################################################

.PHONY: format
format: ## Format Rust files
	@echo "Formatting Rust files..."
	@cargo fmt

.PHONY: test
test: format doctest ## Run the tests
	@echo "Running tests..."
	@DEBUG_VQ=$(DEBUG_VQ) RUST_LOG=debug RUST_BACKTRACE=$(RUST_BACKTRACE) cargo test --features all --all-targets \
	--workspace -- --nocapture

.PHONY: doctest
doctest: ## Run documentation tests (Rust code examples in doc comments)
	@echo "Running documentation tests..."
	@cargo test --doc --features all

.PHONY: bench
bench: ## Run the benchmarks (using Criterion) with all features
	@echo "Running benchmarks..."
	@cargo bench --features all

.PHONY: test-diff
test-diff: ## Run differential tests against reference implementations (with and without SIMD and parallel features)
	@echo "Running differential tests (default features)..."
	@cargo test --test differential_tests
	@echo "Running differential tests (all features)..."
	@cargo test --test differential_tests --features all

.PHONY: coverage
coverage: format doctest ## Generate test coverage report
	@echo "Generating test coverage report..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo tarpaulin --features all --out Xml --out Html

.PHONY: build
build: format ## Build the binary for the current platform
	@echo "Building the project..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo build --release

.PHONY: run
run: build ## Build and run the binary
	@echo "Running binary: $(BINARY)"
	@DEBUG_VQ=$(DEBUG_VQ) ./$(BINARY)

.PHONY: clean
clean: ## Remove generated and temporary files
	@echo "Cleaning up..."
	@cargo clean
	@rm -rf $(WHEEL_DIR) dist/ $(PYVQ_DIR)/$(WHEEL_DIR)
	@rm -f $(PYVQ_DIR)/*.so
	@rm -f benchmark_results.csv eval_*.csv

.PHONY: submodule-init
submodule-init: ## Initialize git submodules (required for simd feature)
	@echo "Initializing git submodules..."
	@git submodule update --init --recursive

.PHONY: install-snap
install-snap: ## Install dependencies using Snapcraft
	@echo "Installing snap dependencies..."
	@sudo apt-get update && sudo apt-get install -y snapd
	@sudo snap refresh
	@sudo snap install rustup --classic

.PHONY: install-deps
install-deps: install-snap ## Install development dependencies
	@echo "Installing development dependencies..."
	@rustup component add rustfmt clippy
	# Install each tool with a specific, pinned version
	@cargo install cargo-tarpaulin --locked --version ${TARPAULIN_VERSION}
	@cargo install cargo-nextest --locked --version ${NEXTEST_VERSION}
	@cargo install cargo-audit --locked --version ${AUDIT_VERSION}
	@cargo install cargo-careful --locked --version ${CAREFUL_VERSION}
	@sudo apt-get install python3-pip libfontconfig1-dev
	@pip install $(PY_DEP_MNGR)

.PHONY: lint
lint: format ## Run linters on Rust files
	@echo "Linting Rust files..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo clippy -- -D warnings -D clippy::unwrap_used -D clippy::expect_used

.PHONY: publish
publish: ## Publish the package to crates.io (requires CARGO_REGISTRY_TOKEN to be set)
	@echo "Publishing package to Cargo registry..."
	@cargo publish --token $(CARGO_REGISTRY_TOKEN)

.PHONY: audit
audit: ## Run security audit on Rust dependencies
	@echo "Running security audit..."
	@cargo audit

.PHONY: careful
careful: ## Run security checks on Rust code
	@echo "Running security checks..."
	@DEBUG_VQ=$(DEBUG_VQ) RUST_BACKTRACE=$(RUST_BACKTRACE) cargo careful run

.PHONY: docs
docs: format ## Generate the documentation
	@echo "Generating documentation..."
	@cargo doc --no-deps --document-private-items

.PHONY: fix-lint
fix-lint: ## Fix the linter warnings
	@echo "Fixing linter warnings..."
	@cargo clippy --fix --allow-dirty --all-targets --workspace --all-features -- -D warnings -D clippy::unwrap_used -D clippy::expect_used

.PHONY: nextest
nextest: ## Run tests using nextest
	@echo "Running tests using nextest..."
	@DEBUG_VQ=$(DEBUG_VQ) RUST_BACKTRACE=$(RUST_BACKTRACE) cargo nextest run --features all

.PHONY: eval
eval: ## Evaluate an implementation (ALG must be: bq, sq, pq, or tsvq)
	@echo && \
	if [ -z "$(ALG)" ]; then \
	  echo "Please provide the ALG argument"; exit 1; \
	fi
	@echo "Evaluating implementation: $(ALG)"
	@cargo run --release --features binaries --bin eval_$(ALG)

.PHONY: eval-all
eval-all: ## Evaluate all the implementations (bq, sq, pq, and tsvq)
	@echo "Evaluating all implementations..."
	@$(MAKE) eval ALG=bq
	@$(MAKE) eval ALG=sq
	@$(MAKE) eval ALG=pq
	@$(MAKE) eval ALG=tsvq

.PHONY: testdata
testdata: ## Download the datasets used in tests
	@echo "Downloading test data..."
	@$(SHELL) $(TEST_DATA_DIR)/download_datasets.sh $(TEST_DATA_DIR)

.PHONY: install-msrv
install-msrv: ## Install the minimum supported Rust version (MSRV) for development
	@echo "Installing the minimum supported Rust version..."
	@rustup toolchain install $(MSRV)
	@rustup default $(MSRV)

########################################################################################
## Python targets
########################################################################################

.PHONY: develop-py
develop-py: ## Build and install PyVq in the current Python environment
	@echo "Building and installing PyVq..."
	# Note: Maturin does not work when CONDA_PREFIX and VIRTUAL_ENV are both set
	@(cd $(PYVQ_DIR) && unset CONDA_PREFIX && maturin develop)

.PHONY: wheel
wheel: ## Build the wheel file for PyVq
	@echo "Building the PyVq wheel..."
	@(cd $(PYVQ_DIR) && maturin build --release --out $(WHEEL_DIR) --auditwheel check)

.PHONY: wheel-manylinux
wheel-manylinux: ## Build the wheel file for PyVq (use CI for true manylinux)
	@echo "Building the PyVq wheel (local build, use CI for manylinux compliance)..."
	@(cd $(PYVQ_DIR) && maturin build --release --out $(WHEEL_DIR))

.PHONY: test-py
test-py: develop-py ## Run Python tests
	@echo "Running Python tests..."
	@$(PY_DEP_MNGR) run pytest

.PHONY: docs-py
docs-py: develop-py ## Generate PyVq MkDocs documentation
	@echo "Generating MkDocs documentation..."
	@$(PY_DEP_MNGR) run mkdocs build --config-file pyvq/mkdocs.yml

.PHONY: docs-serve-py
docs-serve-py: develop-py ## Serve PyVq MkDocs documentation locally
	@echo "Serving MkDocs documentation locally..."
	@$(PY_DEP_MNGR) run mkdocs serve --config-file pyvq/mkdocs.yml

.PHONY: rundocs
rundocs: develop-py ## Test all code examples in PyVq documentation using rundoc
	@echo "Testing documentation code examples..."
	@failed=0; \
	for f in $(PYVQ_DIR)/docs/examples/*.md; do \
		echo "=== Testing $$(basename $$f) ==="; \
		if echo | rundoc run "$$f" 2>&1 | grep -q "Failed"; then \
			echo "FAILED: $$f"; \
			failed=$$((failed + 1)); \
		else \
			echo "PASSED: $$f"; \
		fi; \
	done; \
	if [ $$failed -gt 0 ]; then \
		echo "$$failed file(s) had failures"; \
		exit 1; \
	else \
		echo "All documentation examples passed!"; \
	fi

.PHONY: docs-serve
docs-serve: ## Serve Vq MkDocs locally
	@echo "Serving Vq MkDocs..."
	@uv run mkdocs serve

.PHONY: docs-build
docs-build: ## Generate Vq MkDocs documentation
	@echo "Building Vq MkDocs..."
	@uv run mkdocs build

.PHONY: publish-py
publish-py: wheel-manylinux ## Publish the PyVq wheel to PyPI (requires PYPI_TOKEN to be set)
	@echo "Publishing PyVq to PyPI..."
	@if [ -z "$(WHEEL_FILE)" ]; then \
	   echo "Error: No wheel file found. Please run 'make wheel' first."; \
	   exit 1; \
	fi
	@echo "Found wheel file: $(WHEEL_FILE)"
	@twine upload -u __token__ -p $(PYPI_TOKEN) $(WHEEL_FILE)

.PHONY: generate-ci
generate-ci: ## Generate CI configuration files (GitHub Actions workflow)
	@echo "Generating CI configuration files..."
	@(cd $(PYVQ_DIR) && maturin generate-ci --zig --pytest --platform all -o ../.github/workflows/ci.yml github)

########################################################################################
## Additional targets
########################################################################################

.PHONY: setup-hooks
setup-hooks: ## Install Git hooks (pre-commit and pre-push)
	@echo "Installing Git hooks..."
	@pre-commit install --hook-type pre-commit
	@pre-commit install --hook-type pre-push
	@pre-commit install-hooks

.PHONY: test-hooks
test-hooks: ## Test Git hooks on all files
	@echo "Testing Git hooks..."
	@pre-commit run --all-files
