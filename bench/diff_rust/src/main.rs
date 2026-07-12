//! Differential dump: reads a JSONL corpus (one JSON-encoded string per
//! line) and a tokenizer.json fixture, encodes every line with
//! add_special_tokens=true (matching Zig's Tokenizer.encode, which always
//! applies TemplateProcessing), and prints one JSON object per line:
//!   {"line": N, "ok": true, "ids": [...]}
//!   {"line": N, "ok": false, "error": "..."}
//! Used to differential-test against the Zig-side dump (tools/diff_dump.zig)
//! — exact token-ID equality is the pass criterion, not timing.

use serde::Serialize;
use std::env;
use std::fs;
use tokenizers::Tokenizer;

#[derive(Serialize)]
struct LineResult {
    line: usize,
    ok: bool,
    ids: Option<Vec<u32>>,
    error: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: diff_dump <fixture.json> <corpus.jsonl>");
        std::process::exit(2);
    }
    let fixture_path = &args[1];
    let corpus_path = &args[2];

    let tokenizer = Tokenizer::from_file(fixture_path)
        .map_err(|e| format!("failed to load fixture {fixture_path}: {e}"))?;

    let corpus = fs::read_to_string(corpus_path)
        .map_err(|e| format!("failed to read corpus {corpus_path}: {e}"))?;

    for (i, raw_line) in corpus.lines().enumerate() {
        if raw_line.trim().is_empty() {
            continue;
        }
        let text: String = match serde_json::from_str(raw_line) {
            Ok(s) => s,
            Err(e) => {
                let r = LineResult {
                    line: i,
                    ok: false,
                    ids: None,
                    error: Some(format!("corpus line {i} is not a valid JSON string: {e}")),
                };
                println!("{}", serde_json::to_string(&r)?);
                continue;
            }
        };
        let result = match tokenizer.encode(text.as_str(), true) {
            Ok(enc) => LineResult {
                line: i,
                ok: true,
                ids: Some(enc.get_ids().to_vec()),
                error: None,
            },
            Err(e) => LineResult {
                line: i,
                ok: false,
                ids: None,
                error: Some(e.to_string()),
            },
        };
        println!("{}", serde_json::to_string(&result)?);
    }

    Ok(())
}
