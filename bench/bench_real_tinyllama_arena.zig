//! Same as bench_real_tinyllama.zig but encode uses a per-iter arena
//! allocator instead of GPA. Pre-reg: if Zig encode allocator-overhead
//! dominated by GPA, arena will close most of the 2.07x gap to Rust HF.

const std = @import("std");
const tokenizers = @import("tokenizers");

const TEXT =
    "The quick brown fox jumps over the lazy dog. The five boxing wizards jump quickly. Pack my box with five dozen liquor jugs.";

const FIXTURE = "tests/fixtures/tokenizer.json";
const ITERS = 1000;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    const json = std.Io.Dir.cwd().readFileAlloc(io, FIXTURE, gpa, std.Io.Limit.limited(64 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("fixture {s} not present.\n", .{FIXTURE});
            std.process.exit(2);
        },
        else => return err,
    };
    defer gpa.free(json);

    var tok = try tokenizers.fromJson(gpa, json);
    defer tok.deinit();

    // Per-iter arena: reset between encodes, single backing buffer reused.
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const aalloc = arena.allocator();

    // Warmup
    {
        const ids = try tok.encode(aalloc, TEXT);
        _ = ids;
        _ = arena.reset(.retain_capacity);
    }

    var total_tokens: u64 = 0;
    const t0 = std.Io.Clock.awake.now(io);
    var i: usize = 0;
    while (i < ITERS) : (i += 1) {
        const ids = try tok.encode(aalloc, TEXT);
        total_tokens += ids.len;
        _ = arena.reset(.retain_capacity);
    }
    const t1 = std.Io.Clock.awake.now(io);
    const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);

    const ns_per_encode = ns / ITERS;
    const bytes_per_sec = @as(u64, ITERS) * TEXT.len * std.time.ns_per_s / ns;
    const tokens_per_encode = total_tokens / ITERS;

    std.debug.print("fixture        {s}\n", .{FIXTURE});
    std.debug.print("text_bytes     {d}\n", .{TEXT.len});
    std.debug.print("iters          {d}\n", .{ITERS});
    std.debug.print("allocator      ArenaAllocator (per-iter reset, retain_capacity)\n", .{});
    std.debug.print("encode\n", .{});
    std.debug.print("  ns_per_encode    {d}\n", .{ns_per_encode});
    std.debug.print("  us_per_encode    {d}\n", .{ns_per_encode / 1000});
    std.debug.print("  tokens_per_encode {d}\n", .{tokens_per_encode});
    std.debug.print("  bytes_per_sec     {d}\n", .{bytes_per_sec});
    std.debug.print("  MB_per_sec        {d}\n", .{bytes_per_sec / 1_000_000});
}
