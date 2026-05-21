//! Real TinyLlama tokenizer encode bench (Zig side of the Zig-vs-Rust
//! head-to-head). Loads tests/fixtures/tokenizer.json, encodes the
//! same text as bench/bench_rust/src/main.rs N times, reports
//! comparable ns/encode + bytes/sec numbers.

const std = @import("std");
const tokenizers = @import("tokenizers");

const TEXT =
    "The quick brown fox jumps over the lazy dog. The five boxing wizards jump quickly. Pack my box with five dozen liquor jugs.";

const FIXTURE = "tests/fixtures/tokenizer.json";
const ITERS = 1000;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    const json = std.Io.Dir.cwd().readFileAlloc(io, FIXTURE, alloc, std.Io.Limit.limited(64 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("fixture {s} not present.\n", .{FIXTURE});
            std.process.exit(2);
        },
        else => return err,
    };
    defer alloc.free(json);

    var tok = try tokenizers.fromJson(alloc, json);
    defer tok.deinit();

    // Warmup
    {
        const ids = try tok.encode(alloc, TEXT);
        alloc.free(ids);
    }

    var total_tokens: u64 = 0;
    const t0 = std.Io.Clock.awake.now(io);
    var i: usize = 0;
    while (i < ITERS) : (i += 1) {
        const ids = try tok.encode(alloc, TEXT);
        total_tokens += ids.len;
        alloc.free(ids);
    }
    const t1 = std.Io.Clock.awake.now(io);
    const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);

    const ns_per_encode = ns / ITERS;
    const bytes_per_sec = @as(u64, ITERS) * TEXT.len * std.time.ns_per_s / ns;
    const tokens_per_encode = total_tokens / ITERS;

    std.debug.print("fixture        {s}\n", .{FIXTURE});
    std.debug.print("text_bytes     {d}\n", .{TEXT.len});
    std.debug.print("iters          {d}\n", .{ITERS});
    std.debug.print("encode\n", .{});
    std.debug.print("  ns_per_encode    {d}\n", .{ns_per_encode});
    std.debug.print("  us_per_encode    {d}\n", .{ns_per_encode / 1000});
    std.debug.print("  tokens_per_encode {d}\n", .{tokens_per_encode});
    std.debug.print("  bytes_per_sec     {d}\n", .{bytes_per_sec});
    std.debug.print("  MB_per_sec        {d}\n", .{bytes_per_sec / 1_000_000});
}
