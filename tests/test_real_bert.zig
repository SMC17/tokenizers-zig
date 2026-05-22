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
    // BertNormalizer composes Sequence([clean_text, handle_chinese_chars,
    // NFD, strip_accents, lowercase]) on a fully-default uncased config.
    try std.testing.expect(tok.normalizer == .sequence);
    // BERT-base-uncased ships 5 added_tokens: [PAD], [UNK], [CLS], [SEP], [MASK]
    try std.testing.expectEqual(@as(usize, 5), tok.added_tokens.len);
}

test "real BERT-base-uncased — encodeBase emits added tokens at their canonical IDs" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // encodeBase skips TemplateProcessing so we see raw added-token IDs.
    const cases = [_]struct { content: []const u8, id: u32 }{
        .{ .content = "[PAD]", .id = 0 },
        .{ .content = "[UNK]", .id = 100 },
        .{ .content = "[CLS]", .id = 101 },
        .{ .content = "[SEP]", .id = 102 },
        .{ .content = "[MASK]", .id = 103 },
    };
    for (cases) |c| {
        const ids = try tok.encodeBase(alloc, c.content);
        defer alloc.free(ids);
        try std.testing.expectEqualSlices(u32, &[_]u32{c.id}, ids);
    }
}

test "real BERT-base-uncased — encode 'hello' produces canonical [CLS,7592,SEP]" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // TemplateProcessing wraps base [7592] with [CLS]=101 / [SEP]=102.
    const ids = try tok.encode(alloc, "hello");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 101, 7592, 102 }, ids);
}

test "real BERT-base-uncased — lowercases input before WordPiece" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

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

    // "hello," → [hello, ","] base, then TemplateProcessing wraps.
    const ids = try tok.encode(alloc, "hello,");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 101, 7592, 1010, 102 }, ids);
}

test "real BERT-base-uncased — encode 'hello world' produces [CLS, hello, world, SEP]" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    const ids = try tok.encode(alloc, "hello world");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 101, 7592, 2088, 102 }, ids);
}

test "real BERT-base-uncased — encodeBase skips TemplateProcessing wrap" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    const ids = try tok.encodeBase(alloc, "hello world");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 7592, 2088 }, ids);
}

test "real BERT-base-uncased — encodePair wraps with [CLS] A [SEP] B [SEP]" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // Canonical: [CLS] hello [SEP] world [SEP]
    const ids = try tok.encodePair(alloc, "hello", "world");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 101, 7592, 102, 2088, 102 }, ids);
}

test "real BERT-base-uncased — encodePairWithTypes produces canonical type_ids" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    const result = try tok.encodePairWithTypes(alloc, "hello", "world");
    defer result.deinit(alloc);

    try std.testing.expectEqualSlices(u32, &[_]u32{ 101, 7592, 102, 2088, 102 }, result.ids);
    // BERT segment ids: A=0 for first three (CLS, hello, SEP), B=1 for
    // last two (world, SEP). Matches HF tokenizer output exactly.
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 1, 1 }, result.type_ids);
}

test "real BERT-base-uncased — encode + decode round-trip emits wrapped text" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // Round-trip preserves the TemplateProcessing wrap because added_tokens
    // decode emits their content verbatim.
    const ids = try tok.encode(alloc, "hello world");
    defer alloc.free(ids);
    const back = try tok.decode(alloc, ids);
    defer alloc.free(back);
    try std.testing.expectEqualStrings("[CLS]hello world[SEP]", back);
}
