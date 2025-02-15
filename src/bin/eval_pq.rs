use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use tracing::{info, span, Level};

use vq::distances::Distance;
use vq::pq::ProductQuantizer;
use vq::vector::Vector;

// Import the helper functions and BenchmarkResult from utils.
#[path = "utils.rs"]
mod utils;
use utils::{
    calculate_recall, calculate_reconstruction_error, generate_synthetic_data, BenchmarkResult,
    SEED,
};
use utils::{DIM, K, M, MAX_ITERS, NUM_SAMPLES};

const OUTPUT_FILENAME: &str = "notebooks/data/eval_pq_results.csv";

fn run_benchmark(
    n_samples: usize,
    n_dims: usize,
    m: usize,
    k: usize,
    max_iters: usize,
) -> Result<BenchmarkResult> {
    // Create a span for data generation.
    let data_gen_span = span!(Level::INFO, "Data Generation", n_samples);
    let _data_gen_enter = data_gen_span.enter();

    // 1. Generate synthetic data as a Vec<Vector<f32>>.
    let training_data = generate_synthetic_data(n_samples, n_dims, SEED);
    drop(_data_gen_enter);

    // 2. Train the product quantizer.
    let training_span = span!(Level::INFO, "Training Phase", n_samples, n_dims, m, k);
    let _training_enter = training_span.enter();
    let distance = Distance::Euclidean;
    let seed = 42;
    let training_start = Instant::now();
    let pq = ProductQuantizer::fit(&training_data, m, k, max_iters, distance, seed);
    let training_time_ms = training_start.elapsed().as_secs_f64() * 1000.0;
    drop(_training_enter);

    // 3. Quantize all vectors.
    let quantization_span = span!(Level::INFO, "Quantization Phase", n_samples);
    let _quantization_enter = quantization_span.enter();
    let quantization_start = Instant::now();
    let reconstructed_data: Vec<Vector<f32>> = training_data
        .iter()
        .map(|vec| {
            // Each quantized vector is a Vector<f16>; convert back to f32.
            let quantized = pq.quantize(vec);
            let data_f32: Vec<f32> = quantized.data.into_iter().map(|val| val.to_f32()).collect();
            Vector::new(data_f32)
        })
        .collect();
    let quantization_time_ms = quantization_start.elapsed().as_secs_f64() * 1000.0;
    drop(_quantization_enter);

    // 4. Evaluate quality metrics.
    let reconstruction_error = calculate_reconstruction_error(&training_data, &reconstructed_data);
    let recall = calculate_recall(&training_data, &reconstructed_data, 10)?;

    // Log the metrics.
    info!("Training time: {:.2}ms", training_time_ms);
    info!("Quantization time: {:.2}ms", quantization_time_ms);
    info!("Reconstruction error: {:.4}", reconstruction_error);
    info!("Recall@10: {:.4}", recall);

    Ok(BenchmarkResult {
        n_samples,
        n_dims,
        training_time_ms,
        quantization_time_ms,
        reconstruction_error,
        recall,
        memory_reduction_ratio: 0.0, // Not applicable
    })
}

pub fn main() -> Result<()> {
    // Initialize the tracing subscriber for hierarchical logging.
    tracing_subscriber::fmt::init();

    // Create an overall span for the benchmark run.
    let overall_span = span!(Level::INFO, "Benchmark Run");
    let _overall_enter = overall_span.enter();

    let mut results = Vec::new();
    for n_samples in NUM_SAMPLES {
        let bench_span = span!(Level::INFO, "Benchmark", n_samples);
        let _bench_enter = bench_span.enter();
        info!("Running benchmark with {} samples...", n_samples);
        let result = run_benchmark(n_samples, DIM, M, K, MAX_ITERS)?;
        results.push(result);
        drop(_bench_enter);
    }

    // Write results to a CSV file (excluding memory reduction).
    let mut file = File::create(OUTPUT_FILENAME)?;
    writeln!(
        file,
        "n_samples,n_dims,training_time_ms,quantization_time_ms,reconstruction_error,recall"
    )?;
    for result in &results {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            result.n_samples,
            result.n_dims,
            result.training_time_ms,
            result.quantization_time_ms,
            result.reconstruction_error,
            result.recall
        )?;
    }

    // Log the summarized results.
    for result in &results {
        info!("\nResults for {} samples:", result.n_samples);
        info!("Training time: {:.2}ms", result.training_time_ms);
        info!("Quantization time: {:.2}ms", result.quantization_time_ms);
        info!("Reconstruction Error: {:.4}", result.reconstruction_error);
        info!("Recall@10: {:.4}", result.recall);
    }

    Ok(())
}
