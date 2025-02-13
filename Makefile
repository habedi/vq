# Variables
PKG = github.com/habedi/vq
BINARY_NAME = $(or $(PROJ_BINARY), $(notdir $(PKG)-examples))
BINARY = target/release/$(BINARY_NAME)
PATH := /snap/bin:$(PATH)
CARGO_TERM_COLOR = always
RUST_BACKTRACE = 1
RUST_LOG = info
DEBUG_VQ = 0

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: format
format: ## Format Rust files
	@echo "Formatting Rust files..."
	@cargo fmt

.PHONY: test
test: format ## Run tests
	@echo "Running tests..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo test -- --nocapture

.PHONY: coverage
coverage: format ## Generate test coverage report
	@echo "Generating test coverage report..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo tarpaulin --out Xml --out Html

.PHONY: build
build: format ## Build the binary for the current platform
	@echo "Building the project..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo build --release

.PHONY: run
run: build ## Build and run the binary
	@echo "Running the $(BINARY) binary..."
	@DEBUG_VQ=$(DEBUG_VQ) ./$(BINARY)

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

.PHONY: lint
lint: format ## Run linters on Rust files
	@echo "Linting Rust files..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo clippy -- -D warnings

.PHONY: publish
publish: ## Publish the package to crates.io (requires CARGO_REGISTRY_TOKEN to be set)
	@echo "Publishing the package to Cargo registry..."
	@cargo publish --token $(CARGO_REGISTRY_TOKEN)

.PHONY: bench
bench: ## Run benchmarks
	@echo "Running benchmarks..."
	@DEBUG_VQ=$(DEBUG_VQ) cargo bench

.PHONY: eval
eval: ## Evaluate an implementation (the ALG should be the algorithm name, e.g., bq, sq, pq, opq, tsvq, rvq)
	@echo && if [ -z "$(ALG)" ]; then echo "Please provide the ALG argument"; exit 1; fi
	@echo "Evaluating implementation with argument: $(ALG)"
	@cargo run --release --bin eval -- --eval $(ALG)

.PHONY: eval-all
eval-all: ## Evaluate all the implementations (bq, sq, pq, opq, tsvq, rvq)
	@echo "Evaluating all implementations..."
	@cargo run --release --bin eval -- --eval bq
	@cargo run --release --bin eval -- --eval sq
	@cargo run --release --bin eval -- --eval pq
	@cargo run --release --bin eval -- --eval opq
	@cargo run --release --bin eval -- --eval tsvq
	@cargo run --release --bin eval -- --eval rvq
