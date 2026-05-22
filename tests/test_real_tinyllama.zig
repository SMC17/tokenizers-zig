//! Real TinyLlama-1.1B tokenizer integration test.
//!
//! Loads tests/fixtures/tokenizer.json (1.8 MB, 32K vocab, 61249
//! merges, Llama / SentencePiece pipeline) and validates encode +
//! decode round-trip against known-good Llama tokenizations.
//!
//! Reference IDs verified against HF tokenizers Rust crate v0.21
//! on the same fixture (see bench/bench_rust for the head-to-head).

const std = @import("std");
const tokenizers = @import("tokenizers");

const fixture_path = "tests/fixtures/tokenizer.json";

fn loadTokenizer(alloc: std.mem.Allocator, io: std.Io) !tokenizers.Tokenizer {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, fixture_path, alloc, std.Io.Limit.limited(64 * 1024 * 1024));
    defer alloc.free(bytes);
    return tokenizers.fromJson(alloc, bytes);
}

test "real TinyLlama tokenizer.json parses (32000 vocab, 61249 merges)" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    var f = std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print(
                "\n[skip] {s} not present. Copy from " ++
                    "~/safetensors-zig/tests/fixtures/tinyllama-real/tokenizer.json\n",
                .{fixture_path},
            );
            return error.SkipZigTest;
        },
        else => return err,
    };
    f.close(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // 32000 vocab + 3 added tokens — total vocab keys in BPE model
    // should be 32000. The added_tokens are separate in the JSON.
    try std.testing.expect(tok.model == .bpe);
    try std.testing.expectEqual(@as(usize, 32000), tok.model.bpe.vocab.count());
    try std.testing.expectEqual(@as(usize, 61249), tok.model.bpe.merges.count());

    // The SP space marker should have been detected via the
    // normalizer Sequence(Prepend, Replace) pattern.
    try std.testing.expect(tok.sp_space_marker != null);
    try std.testing.expectEqualStrings("\xE2\x96\x81", tok.sp_space_marker.?);
}

test "real TinyLlama — encodeBase 'Hello, how are you today?' produces known base IDs" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    var f = std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    f.close(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // Bare BPE (no TemplateProcessing wrap):
    //   15043=Hello, 29892=",", 920=how, 526=are, 366=you, 9826=today, 29973="?"
    const ids = try tok.encodeBase(alloc, "Hello, how are you today?");
    defer alloc.free(ids);
    const expected = [_]u32{ 15043, 29892, 920, 526, 366, 9826, 29973 };
    try std.testing.expectEqualSlices(u32, &expected, ids);
}

test "real TinyLlama — encode prepends BOS via TemplateProcessing" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    var f = std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    f.close(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // TinyLlama TemplateProcessing prepends <s>=1 (BOS) before the sequence.
    const ids = try tok.encode(alloc, "Hello, how are you today?");
    defer alloc.free(ids);
    const expected = [_]u32{ 1, 15043, 29892, 920, 526, 366, 9826, 29973 };
    try std.testing.expectEqualSlices(u32, &expected, ids);
}

test "real TinyLlama — encodeWithOffsets gives byte ranges per token" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    var f = std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    f.close(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    const text = "Hello world";
    const r = try tok.encodeWithOffsets(alloc, text);
    defer r.deinit(alloc);

    // TinyLlama BPE+SP. encodeWithOffsets returns raw model output
    // (no TemplateProcessing wrap). First real token is "▁Hello"=15043.
    try std.testing.expect(r.ids.len >= 1);
    try std.testing.expectEqual(@as(u32, 15043), r.ids[0]);
    // First token covers "Hello" → starts at byte 0.
    try std.testing.expectEqual(@as(u32, 0), r.offsets[0].start);
    // All offsets stay within the text bounds + monotonic non-decreasing.
    var prev_end: u32 = 0;
    for (r.offsets) |off| {
        try std.testing.expect(off.start <= text.len);
        try std.testing.expect(off.end <= text.len);
        try std.testing.expect(off.start <= off.end);
        try std.testing.expect(off.start >= prev_end);
        prev_end = off.end;
    }
}

test "real TinyLlama — encodeBase + decode round-trip preserves text" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    var f = std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    f.close(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    const samples = [_][]const u8{
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Tokenization is a strange and beautiful art.",
    };
    for (samples) |s| {
        const ids = try tok.encodeBase(alloc, s);
        defer alloc.free(ids);
        const back = try tok.decode(alloc, ids);
        defer alloc.free(back);
        try std.testing.expectEqualStrings(s, back);
    }
}
