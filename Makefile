# Variables
REPO_URL := github.com/habedi/vq
BINARY_NAME := $(or $(PROJ_BINARY), $(notdir $(REPO_URL)-examples))
BINARY := $(BINARY_NAME)
PATH := /snap/bin:$(PATH)
CARGO_TERM_COLOR := always
RUST_BACKTRACE := 0
RUST_LOG := info
DEBUG_VQ := 0

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; \
 	{printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: format
format: ## Format Rust files
	@echo "Formatting Rust files..."
	@cargo fmt

.PHONY: test
test: format ## Run the tests
	@echo "Running tests..."
	@DEBUG_VQ=$(DEBUG_VQ) RUST_BACKTRACE=$(RUST_BACKTRACE) cargo test -- --nocapture

.PHONY: coverage
coverage: format ## Generate test coverage report
	@echo "Generating test coverage report..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo tarpaulin --out Xml --out Html

.PHONY: build
build: format ## Build the binary for the current platform
	@echo "Building the project..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo build --release --features binaries

.PHONY: run
run: build ## Build and run the binary
	@echo "Running the $(BINARY) binary..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo run --release --features binaries --bin $(BINARY)

.PHONY: clean
clean: ## Remove generated and temporary files
	@echo "Cleaning up..."
	@cargo clean
	@rm -f benchmark_results.csv
	@rm -f eval_*.csv

.PHONY: install-snap
install-snap: ## Install a few dependencies using Snapcraft
	@echo "Installing the snap package..."
	@sudo apt-get update
	@sudo apt-get install -y snapd
	@sudo snap refresh
	@sudo snap install rustup --classic

.PHONY: install-deps
install-deps: install-snap ## Install development dependencies
	@echo "Installing dependencies..."
	@rustup component add rustfmt clippy
	@cargo install cargo-tarpaulin
	@cargo install cargo-audit
	@cargo install nextest

.PHONY: lint
lint: format ## Run the linters
	@echo "Linting Rust files..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo clippy -- -D warnings

.PHONY: publish
publish: ## Publish the package to crates.io (needs CARGO_REGISTRY_TOKEN to be set)
	@echo "Publishing the package to Cargo registry..."
	@cargo publish --token $(CARGO_REGISTRY_TOKEN)

.PHONY: bench
bench: ## Run the benchmarks
	@echo "Running benchmarks..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo bench

.PHONY: eval
eval: ## Evaluate an implementation (ALG must be: bq, sq, pq, opq, tsvq, or rvq)
	@echo && if [ -z "$(ALG)" ]; then echo "Please provide the ALG argument"; exit 1; fi
	@echo "Evaluating implementation with argument: $(ALG)"
	@cargo run --release --features binaries --bin eval -- --eval $(ALG)

.PHONY: eval-all
eval-all: ## Evaluate all the implementations (bq, sq, pq, opq, tsvq, rvq)
	@echo "Evaluating all implementations..."
	@make eval ALG=bq
	@make eval ALG=sq
	@make eval ALG=pq
	@make eval ALG=opq
	@make eval ALG=tsvq
	@make eval ALG=rvq

.PHONY: fix-lint
fix-lint: ## Fix the linter warnings
	@echo "Fixing linter warnings..."
	@cargo clippy --fix --allow-dirty --allow-staged --all-targets --workspace --all-features -- -D warnings

.PHONY: nextest
nextest: ## Run tests using nextest
	@echo "Running tests using nextest..."
	@DEBUG_VQ=$(DEBUG_VQ) RUST_BACKTRACE=$(RUST_BACKTRACE) cargo nextest run

.PHONY: doc
doc: format ## Generate the documentation
	@echo "Generating documentation..."
	@cargo doc --no-deps --document-private-items
