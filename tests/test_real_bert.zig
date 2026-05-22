//! Real bert-base-uncased tokenizer integration test.
//!
//! Loads tests/fixtures/bert-base-uncased-tokenizer.json (466 KB,
//! 30522 vocab, WordPiece + BertNormalizer + BertPreTokenizer) and
//! validates parse + encode + decode against known-good BERT
//! tokenizations.
//!
//! Reference IDs verified against published BERT-base-uncased
//! tokenization examples. Download via:
//!   curl -fsSL -o tests/fixtures/bert-base-uncased-tokenizer.json \
//!     https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json

const std = @import("std");
const tokenizers = @import("tokenizers");

const fixture_path = "tests/fixtures/bert-base-uncased-tokenizer.json";

fn loadTokenizer(alloc: std.mem.Allocator, io: std.Io) !tokenizers.Tokenizer {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, fixture_path, alloc, std.Io.Limit.limited(64 * 1024 * 1024));
    defer alloc.free(bytes);
    return tokenizers.fromJson(alloc, bytes);
}

fn skipIfMissing(io: std.Io) !void {
    var f = std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print(
                "\n[skip] {s} not present. Download via:\n" ++
                    "  curl -fsSL -o {s} \\\n" ++
                    "    https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json\n",
                .{ fixture_path, fixture_path },
            );
            return error.SkipZigTest;
        },
        else => return err,
    };
    f.close(io);
}

test "real BERT-base-uncased — parses with 30522 vocab, WordPiece + lowercase + punct-split" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    try std.testing.expect(tok.model == .wordpiece);
    try std.testing.expectEqual(@as(usize, 30522), tok.model.vocabCount());
    try std.testing.expect(tok.use_whitespace_split);
    try std.testing.expect(tok.split_on_punctuation);
    try std.testing.expect(tok.normalizer == .lowercase);
    // BERT-base-uncased ships 5 added_tokens: [PAD], [UNK], [CLS], [SEP], [MASK]
    try std.testing.expectEqual(@as(usize, 5), tok.added_tokens.len);
}

test "real BERT-base-uncased — added tokens emit their canonical IDs" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // Canonical BERT-base-uncased added_token IDs from huggingface/bert-base-uncased.
    const cases = [_]struct { content: []const u8, id: u32 }{
        .{ .content = "[PAD]", .id = 0 },
        .{ .content = "[UNK]", .id = 100 },
        .{ .content = "[CLS]", .id = 101 },
        .{ .content = "[SEP]", .id = 102 },
        .{ .content = "[MASK]", .id = 103 },
    };
    for (cases) |c| {
        const ids = try tok.encode(alloc, c.content);
        defer alloc.free(ids);
        try std.testing.expectEqualSlices(u32, &[_]u32{c.id}, ids);
    }
}

test "real BERT-base-uncased — encodes 'hello' as canonical token 7592" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    const ids = try tok.encode(alloc, "hello");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{7592}, ids);
}

test "real BERT-base-uncased — lowercases input before WordPiece" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // BertNormalizer.lowercase: "Hello" → "hello" → token 7592 (same as bare "hello").
    const upper = try tok.encode(alloc, "Hello");
    defer alloc.free(upper);
    const lower = try tok.encode(alloc, "hello");
    defer alloc.free(lower);
    try std.testing.expectEqualSlices(u32, lower, upper);
}

test "real BERT-base-uncased — BertPreTokenizer splits on punctuation" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // "hello," should split into ["hello", ","] → [7592, 1010].
    // Canonical IDs for the punctuation pieces in BERT-base-uncased vocab.
    const ids = try tok.encode(alloc, "hello,");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 7592, 1010 }, ids);
}

test "real BERT-base-uncased — canonical [CLS] hello world [SEP] sequence" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // Canonical: [CLS]=101, hello=7592, world=2088, [SEP]=102.
    const ids = try tok.encode(alloc, "[CLS] hello world [SEP]");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 101, 7592, 2088, 102 }, ids);
}

test "real BERT-base-uncased — encode + decode round-trip on lowercased text" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // After encode + decode the text comes back lowercased and
    // space-separated. WordPiece's decoder rejoins ## continuations.
    const ids = try tok.encode(alloc, "hello world");
    defer alloc.free(ids);
    const back = try tok.decode(alloc, ids);
    defer alloc.free(back);
    try std.testing.expectEqualStrings("hello world", back);
}
