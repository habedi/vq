//! # Internal Settings for Vq
//!
//! This module configures settings for the `Vq` library, including setting up logging.
//! It reads the `DEBUG_VQ` environment variable to decide whether to enable debug logging.
//! If enabled, it initializes the tracing subscriber with the DEBUG log level.

use ctor::ctor;
use std::env;
use tracing::Level;

#[ctor]
fn set_debug_level() {
    // Read the DEBUG_VQ environment variable and enable debug logging if appropriate.
    let enable_debug = env::var("DEBUG_VQ")
        .map(|v| {
            let v = v.trim().to_lowercase();
            // Treat these values as false: "0", "false", "no", "off", or empty.
            !(v == "0" || v == "false" || v == "no" || v == "off" || v.is_empty())
        })
        .unwrap_or(false);

    if enable_debug {
        // Initialize tracing subscriber with DEBUG level.
        if let Err(e) = tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .try_init()
        {
            eprintln!("Failed to initialize tracing subscriber: {}", e);
        }
    }
}
