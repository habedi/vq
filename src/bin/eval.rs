use anyhow::Result;
use clap::Parser;

mod eval_bq;
mod eval_opq;
mod eval_pq;
mod eval_rvq;
mod eval_sq;
mod eval_tsvq;

/// Simple CLI to run different evaluations.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Which evaluation to run: opq, pq, tsvq, rvq, bq, sq
    #[arg(short, long)]
    eval: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.eval.as_str() {
        "opq" => eval_opq::main()?,
        "pq" => eval_pq::main()?,
        "tsvq" => eval_tsvq::main()?,
        "rvq" => eval_rvq::main()?,
        "bq" => eval_bq::main()?,
        "sq" => eval_sq::main()?,
        other => {
            eprintln!("Unknown evaluation: {}", other);
            std::process::exit(1);
        }
    }

    Ok(())
}
