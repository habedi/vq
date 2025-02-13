use ctor::ctor;
use std::env;
use tracing::Level;
use tracing_subscriber;

#[ctor]
fn set_debug_level() {
    // Read the DEBUG_VQ environment variable.
    // If the variable is missing, default to disabling debug logging.
    let enable_debug = env::var("DEBUG_VQ")
        .map(|v| {
            // Normalize the string for case-insensitive comparison.
            let v = v.trim().to_lowercase();
            // Consider these values as "false".
            !(v == "0" || v == "false" || v == "no" || v == "off" || v == "")
        })
        .unwrap_or(false);

    if enable_debug {
        // Initialize the subscriber with a maximum log level of DEBUG.
        // Use try_init() to avoid panics if a subscriber is already set.
        if let Err(e) = tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .try_init()
        {
            eprintln!("Failed to initialize tracing subscriber: {}", e);
        }
    }
}
