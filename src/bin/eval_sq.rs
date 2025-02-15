use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use tracing::{info, span, Level};

use vq::sq::ScalarQuantizer;
use vq::vector::Vector;

// Import helper functions and BenchmarkResult from utils.
#[path = "utils.rs"]
mod utils;
use utils::{
    calculate_recall, calculate_reconstruction_error, generate_synthetic_data, BenchmarkResult,
};
use utils::{DIM, NUM_SAMPLES, SEED};

const OUTPUT_FILENAME: &str = "notebooks/data/eval_sq_results.csv";

// Parameters for the ScalarQuantizer.
const MIN_VAL: f32 = 0.0;
const MAX_VAL: f32 = 1.0;
const LEVELS: usize = 256;

fn run_benchmark(
    n_samples: usize,
    n_dims: usize,
    min_val: f32,
    max_val: f32,
    levels: usize,
) -> Result<BenchmarkResult> {
    // 1. Generate synthetic data as a Vec<Vector<f32>>.
    let data_gen_span = span!(Level::INFO, "Data Generation", n_samples);
    let _data_gen_enter = data_gen_span.enter();
    let original_data = generate_synthetic_data(n_samples, n_dims, SEED);
    drop(_data_gen_enter);

    // 2. Initialize the ScalarQuantizer.
    let sq = ScalarQuantizer::fit(min_val, max_val, levels);

    // 3. Quantize all vectors and measure quantization time.
    let quantization_span = span!(Level::INFO, "Quantization Phase", n_samples);
    let _quantization_enter = quantization_span.enter();
    let quantization_start = Instant::now();
    let quantized_data: Vec<Vector<u8>> =
        original_data.iter().map(|vec| sq.quantize(vec)).collect();
    let quantization_time_ms = quantization_start.elapsed().as_secs_f64() * 1000.0;
    drop(_quantization_enter);

    // 4. Reconstruct the quantized data.
    // Each quantized index is converted back to f32 using:
    //   reconstructed_value = min_val + (index as f32) * step
    // where step = (max_val - min_val) / (levels - 1)
    let step = (max_val - min_val) / (levels - 1) as f32;
    let reconstructed_data: Vec<Vector<f32>> = quantized_data
        .iter()
        .map(|vec_u8| {
            let rec: Vec<f32> = vec_u8
                .data
                .iter()
                .map(|&index| min_val + (index as f32) * step)
                .collect();
            Vector::new(rec)
        })
        .collect();

    // 5. Evaluate quality metrics.
    let reconstruction_error = calculate_reconstruction_error(&original_data, &reconstructed_data);
    let recall = calculate_recall(&original_data, &reconstructed_data, 10)?;

    info!("Quantization time: {:.2}ms", quantization_time_ms);
    info!("Reconstruction error: {:.4}", reconstruction_error);
    info!("Recall@10: {:.4}", recall);

    // There is no training phase for ScalarQuantizer.
    Ok(BenchmarkResult {
        n_samples,
        n_dims,
        training_time_ms: 0.0,
        quantization_time_ms,
        reconstruction_error,
        recall,
        memory_reduction_ratio: 0.0, // Not applicable
    })
}

pub fn main() -> Result<()> {
    // Initialize tracing subscriber for logging.
    tracing_subscriber::fmt::init();

    let overall_span = span!(Level::INFO, "Benchmark Run");
    let _overall_enter = overall_span.enter();

    let mut results = Vec::new();
    for n_samples in NUM_SAMPLES {
        let bench_span = span!(Level::INFO, "Benchmark", n_samples);
        let _bench_enter = bench_span.enter();
        info!("Running benchmark with {} samples...", n_samples);
        let result = run_benchmark(n_samples, DIM, MIN_VAL, MAX_VAL, LEVELS)?;
        results.push(result);
        drop(_bench_enter);
    }

    // Write results to a CSV file.
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

    // Log summarized results.
    for result in &results {
        info!("\nResults for {} samples:", result.n_samples);
        info!("Quantization time: {:.2}ms", result.quantization_time_ms);
        info!("Reconstruction error: {:.4}", result.reconstruction_error);
        info!("Recall@10: {:.4}", result.recall);
    }

    Ok(())
}
