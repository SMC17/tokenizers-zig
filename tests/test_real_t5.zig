//! Real t5-base tokenizer integration test.
//!
//! Loads tests/fixtures/t5-base-tokenizer.json (1.4 MB, 32100 vocab,
//! Unigram + Sequence(WhitespaceSplit, Metaspace)) and validates parse
//! + encode + decode against known-good T5 tokenizations.
//!
//! Reference IDs verified against published T5-base tokenization +
//! HuggingFace `tokenizer.json` byte inspection. Download via:
//!   curl -fsSL -o tests/fixtures/t5-base-tokenizer.json \
//!     https://huggingface.co/t5-base/resolve/main/tokenizer.json

const std = @import("std");
const tokenizers = @import("tokenizers");

const fixture_path = "tests/fixtures/t5-base-tokenizer.json";

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
                    "    https://huggingface.co/t5-base/resolve/main/tokenizer.json\n",
                .{ fixture_path, fixture_path },
            );
            return error.SkipZigTest;
        },
        else => return err,
    };
    f.close(io);
}

test "real t5-base — parses with 32100 vocab, Unigram + Metaspace" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    try std.testing.expect(tok.model == .unigram);
    try std.testing.expectEqual(@as(usize, 32100), tok.model.vocabCount());
    try std.testing.expect(tok.metaspace_replacement != null);
    try std.testing.expectEqualStrings("\u{2581}", tok.metaspace_replacement.?);
    // t5-base ships 103 added_tokens: <pad>, </s>, <unk>, <extra_id_0..99>.
    try std.testing.expectEqual(@as(usize, 103), tok.added_tokens.len);
}

test "real t5-base — encodeBase emits added tokens at their canonical IDs" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    const cases = [_]struct { content: []const u8, id: u32 }{
        .{ .content = "<pad>", .id = 0 },
        .{ .content = "</s>", .id = 1 },
        .{ .content = "<unk>", .id = 2 },
        .{ .content = "<extra_id_0>", .id = 32099 },
        .{ .content = "<extra_id_99>", .id = 32000 },
    };
    for (cases) |c| {
        const ids = try tok.encodeBase(alloc, c.content);
        defer alloc.free(ids);
        try std.testing.expectEqualSlices(u32, &[_]u32{c.id}, ids);
    }
}

test "real t5-base — encode 'hello' produces [▁hello, </s>] via TemplateProcessing" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // T5's TemplateProcessing appends </s>=1 after the sequence.
    const ids = try tok.encode(alloc, "hello");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 21820, 1 }, ids);
}

test "real t5-base — encodeBase 'hello' produces bare [▁hello]" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    const ids = try tok.encodeBase(alloc, "hello");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{21820}, ids);
}

test "real t5-base — encode 'hello </s>' produces [▁hello, </s>, </s>]" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    // Base: [▁hello, </s>] (literal </s> matched as added_token).
    // TemplateProcessing appends </s>=1 after the sequence.
    const ids = try tok.encode(alloc, "hello </s>");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 21820, 1, 1 }, ids);
}

test "real t5-base — encode + decode round-trip emits text with appended </s>" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    try skipIfMissing(io);

    var tok = try loadTokenizer(alloc, io);
    defer tok.deinit();

    const ids = try tok.encode(alloc, "hello");
    defer alloc.free(ids);
    const back = try tok.decode(alloc, ids);
    defer alloc.free(back);
    try std.testing.expectEqualStrings("hello</s>", back);
}
