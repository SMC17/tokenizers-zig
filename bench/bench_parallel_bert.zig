//! Parallel encodeBatch scaling bench on bert-base-uncased.
//! Runs the same text array through encodeBatch (1 thread) and
//! encodeBatchParallel (N threads). Reports relative throughput.

const std = @import("std");
const tokenizers = @import("tokenizers");

const FIXTURE = "tests/fixtures/bert-base-uncased-tokenizer.json";
const BATCH = 200;
const TEXT =
    "The quick brown fox jumps over the lazy dog. The five boxing wizards jump quickly. Pack my box with five dozen liquor jugs.";

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

    var texts: [BATCH][]const u8 = undefined;
    for (&texts) |*t| t.* = TEXT;

    // Warm
    {
        const batch = try tok.encodeBatch(alloc, &texts);
        tokenizers.Tokenizer.freeBatch(alloc, batch);
    }

    std.debug.print("fixture        {s}\n", .{FIXTURE});
    std.debug.print("batch_size     {d}\n", .{BATCH});
    std.debug.print("text_bytes     {d}\n", .{TEXT.len});

    for ([_]usize{ 1, 2, 4, 8 }) |n| {
        const t0 = std.Io.Clock.awake.now(io);
        const batch = try tok.encodeBatchParallel(alloc, &texts, n);
        const t1 = std.Io.Clock.awake.now(io);
        tokenizers.Tokenizer.freeBatch(alloc, batch);

        const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);
        const ms = ns / 1_000_000;
        const per_text_ns = ns / BATCH;
        const throughput = (@as(u64, BATCH) * std.time.ns_per_s) / ns;
        std.debug.print("threads {d}  total_ms {d}  per_text_ns {d}  texts/sec {d}\n", .{ n, ms, per_text_ns, throughput });
    }
}
