//! Standalone normalizer dump: applies just the normalizer stage (no
//! pre-tokenization, no model encode) to isolate whether a
//! differential-correctness mismatch comes from normalization or from
//! the model's subword matching. Prints normalized text as JSON.
use std::env;
use std::fs;
use tokenizers::tokenizer::normalizer::NormalizedString;
use tokenizers::Normalizer;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: norm_dump <fixture.json> <corpus.jsonl>");
        std::process::exit(2);
    }
    let tokenizer = Tokenizer::from_file(&args[1])?;
    let corpus = fs::read_to_string(&args[2])?;

    for (i, raw_line) in corpus.lines().enumerate() {
        if raw_line.trim().is_empty() {
            continue;
        }
        let text: String = match serde_json::from_str(raw_line) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut normalized = NormalizedString::from(text.as_str());
        if let Some(n) = tokenizer.get_normalizer() {
            n.normalize(&mut normalized)?;
        }
        let out = serde_json::json!({"line": i, "normalized": normalized.get()});
        println!("{}", serde_json::to_string(&out)?);
    }
    Ok(())
}
