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
    try std.testing.expectEqual(@as(usize, 32000), tok.model.vocab.count());
    try std.testing.expectEqual(@as(usize, 61249), tok.model.merges.count());

    // The SP space marker should have been detected via the
    // normalizer Sequence(Prepend, Replace) pattern.
    try std.testing.expect(tok.sp_space_marker != null);
    try std.testing.expectEqualStrings("\xE2\x96\x81", tok.sp_space_marker.?);
}

test "real TinyLlama — encode 'Hello, how are you today?' produces known IDs" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    var f = std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    f.close(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    const ids = try tok.encode(alloc, "Hello, how are you today?");
    defer alloc.free(ids);

    // Reference: Llama-2 / TinyLlama tokenization of this exact text
    //   15043=Hello, 29892=",", 920=how, 526=are, 366=you, 9826=today, 29973="?"
    const expected = [_]u32{ 15043, 29892, 920, 526, 366, 9826, 29973 };
    try std.testing.expectEqualSlices(u32, &expected, ids);
}

test "real TinyLlama — encode + decode round-trip preserves text" {
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
        const ids = try tok.encode(alloc, s);
        defer alloc.free(ids);
        const back = try tok.decode(alloc, ids);
        defer alloc.free(back);
        try std.testing.expectEqualStrings(s, back);
    }
}
