//! Head-to-head reference bench using huggingface/tokenizers Rust crate.
//!
//! Loads the same tokenizer.json fixture as the Zig bench, encodes the
//! same text repeatedly, reports ns/encode and encodes/sec on the same
//! scale as the Zig output.

use std::time::Instant;
use tokenizers::Tokenizer;

const FIXTURE: &str = "../../tests/fixtures/tokenizer.json";
const TEXT: &str = "The quick brown fox jumps over the lazy dog. The five boxing wizards jump quickly. Pack my box with five dozen liquor jugs.";
const ITERS: u64 = 1_000;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_file(FIXTURE)?;

    // Warmup
    {
        let _ = tokenizer.encode(TEXT, false)?;
    }

    let start = Instant::now();
    let mut total_tokens = 0u64;
    for _ in 0..ITERS {
        let enc = tokenizer.encode(TEXT, false)?;
        total_tokens += enc.get_ids().len() as u64;
    }
    let elapsed_ns = start.elapsed().as_nanos() as u64;

    let ns_per_encode = elapsed_ns / ITERS;
    let bytes_per_sec = (ITERS as u128 * TEXT.len() as u128 * 1_000_000_000) / (elapsed_ns as u128);
    let mb_per_sec = bytes_per_sec / 1_000_000;
    let tokens_per_encode = total_tokens / ITERS;

    println!("fixture        {}", FIXTURE);
    println!("text_bytes     {}", TEXT.len());
    println!("iters          {}", ITERS);
    println!("encode");
    println!("  ns_per_encode    {}", ns_per_encode);
    println!("  us_per_encode    {}", ns_per_encode / 1000);
    println!("  tokens_per_encode {}", tokens_per_encode);
    println!("  bytes_per_sec     {}", bytes_per_sec);
    println!("  MB_per_sec        {}", mb_per_sec);

    Ok(())
}
