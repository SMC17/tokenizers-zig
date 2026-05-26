//! Property-based invariant fuzzer.
//!
//! For each real fixture, generate N random ASCII texts (whitespace +
//! punctuation + letters + digits), encode, decode, and verify
//! invariants that should hold regardless of input:
//!
//!   1. ids.len > 0 (non-empty input produces ≥1 token)
//!   2. all offsets stay within [0, text.len]
//!   3. offsets are monotonic non-decreasing
//!   4. attention_mask + type_ids + offsets are parallel to ids
//!   5. decode does not error
//!
//! These catch entire classes of bugs (off-by-one in offsets, mismatched
//! parallel arrays, decode panics on adversarial input) without
//! requiring a per-input known-good answer.

const std = @import("std");
const tokenizers = @import("tokenizers");

const N_ITERS_PER_FIXTURE = 200;

fn loadTokenizer(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !?tokenizers.Tokenizer {
    var f = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch return null;
    f.close(io);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, std.Io.Limit.limited(64 * 1024 * 1024));
    defer alloc.free(bytes);
    return try tokenizers.fromJson(alloc, bytes);
}

fn randomAsciiText(rng: *std.Random, buf: []u8) []const u8 {
    const len = rng.intRangeAtMost(usize, 1, buf.len);
    for (buf[0..len]) |*b| {
        const r = rng.intRangeAtMost(u8, 0, 100);
        b.* = switch (r) {
            0...60 => rng.intRangeAtMost(u8, 'a', 'z'),
            61...75 => rng.intRangeAtMost(u8, 'A', 'Z'),
            76...85 => rng.intRangeAtMost(u8, '0', '9'),
            86...92 => ' ',
            else => switch (rng.intRangeAtMost(u8, 0, 5)) {
                0 => '.',
                1 => ',',
                2 => '!',
                3 => '?',
                4 => '-',
                else => '\'',
            },
        };
    }
    return buf[0..len];
}

fn checkInvariants(tok: *tokenizers.Tokenizer, alloc: std.mem.Allocator, text: []const u8) !void {
    // encodeWithOffsets path
    const r = try tok.encodeWithOffsets(alloc, text);
    defer r.deinit(alloc);

    if (text.len > 0) try std.testing.expect(r.ids.len > 0);
    try std.testing.expectEqual(r.ids.len, r.offsets.len);

    var prev_end: u32 = 0;
    for (r.offsets) |off| {
        try std.testing.expect(off.start <= text.len);
        try std.testing.expect(off.end <= text.len);
        try std.testing.expect(off.start <= off.end);
        if (off.start != 0 or off.end != 0) {
            try std.testing.expect(off.start >= prev_end);
            prev_end = off.end;
        }
    }

    // encodeFull — all parallel arrays aligned
    const enc = try tok.encodeFull(alloc, text);
    defer enc.deinit(alloc);
    try std.testing.expectEqual(enc.ids.len, enc.type_ids.len);
    try std.testing.expectEqual(enc.ids.len, enc.attention_mask.len);
    try std.testing.expectEqual(enc.ids.len, enc.offsets.len);
    try std.testing.expectEqual(enc.ids.len, enc.tokens.len);
    try std.testing.expectEqual(enc.ids.len, enc.special_tokens_mask.len);

    // Cross-method id consistency: encode + encodeFull + encodeWithTypes
    // should all produce the same id sequence.
    const plain = try tok.encode(alloc, text);
    defer alloc.free(plain);
    try std.testing.expectEqualSlices(u32, enc.ids, plain);

    const types_r = try tok.encodeWithTypes(alloc, text);
    defer types_r.deinit(alloc);
    try std.testing.expectEqualSlices(u32, enc.ids, types_r.ids);
    try std.testing.expectEqualSlices(u8, enc.type_ids, types_r.type_ids);
    try std.testing.expectEqualSlices(u8, enc.attention_mask, types_r.attention_mask);

    // Decode must succeed.
    const decoded = try tok.decode(alloc, enc.ids);
    defer alloc.free(decoded);
}

fn checkPairInvariants(tok: *tokenizers.Tokenizer, alloc: std.mem.Allocator, text_a: []const u8, text_b: []const u8) !void {
    if (tok.template_pair == null) return; // skip if no pair template

    const enc = try tok.encodePairFull(alloc, text_a, text_b);
    defer enc.deinit(alloc);

    // All parallel arrays same length
    try std.testing.expectEqual(enc.ids.len, enc.type_ids.len);
    try std.testing.expectEqual(enc.ids.len, enc.attention_mask.len);
    try std.testing.expectEqual(enc.ids.len, enc.offsets.len);
    try std.testing.expectEqual(enc.ids.len, enc.tokens.len);
    try std.testing.expectEqual(enc.ids.len, enc.special_tokens_mask.len);
    try std.testing.expect(enc.sequence_ids != null);
    try std.testing.expectEqual(enc.ids.len, enc.sequence_ids.?.len);

    // sequence_ids contains only 0 or 1
    for (enc.sequence_ids.?) |s| try std.testing.expect(s <= 1);

    // Pair encode + encodePair should produce the same ids
    const pair_ids = try tok.encodePair(alloc, text_a, text_b);
    defer alloc.free(pair_ids);
    try std.testing.expectEqualSlices(u32, enc.ids, pair_ids);
}

fn checkBatchInvariants(tok: *tokenizers.Tokenizer, alloc: std.mem.Allocator, texts: []const []const u8) !void {
    // Sequential vs parallel batch encode produce identical output
    const seq = try tok.encodeBatch(alloc, texts);
    defer tokenizers.Tokenizer.freeBatch(alloc, seq);
    const par = try tok.encodeBatchParallel(alloc, texts, 4);
    defer tokenizers.Tokenizer.freeBatch(alloc, par);

    try std.testing.expectEqual(seq.len, par.len);
    for (seq, par) |s, p| try std.testing.expectEqualSlices(u32, s, p);
}

test "fuzz — BERT-base-uncased invariants over 200 random texts" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    const maybe_tok = loadTokenizer(alloc, io, "tests/fixtures/bert-base-uncased-tokenizer.json") catch return;
    var tok = maybe_tok orelse return error.SkipZigTest;
    defer tok.deinit();

    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    var rng = prng.random();
    var buf: [256]u8 = undefined;

    var i: usize = 0;
    while (i < N_ITERS_PER_FIXTURE) : (i += 1) {
        const text = randomAsciiText(&rng, &buf);
        checkInvariants(&tok, alloc, text) catch |err| {
            std.debug.print("\nfuzz failure on text: {s}\n", .{text});
            return err;
        };
    }
}

test "fuzz — T5-base invariants over 200 random texts" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    const maybe_tok = loadTokenizer(alloc, io, "tests/fixtures/t5-base-tokenizer.json") catch return;
    var tok = maybe_tok orelse return error.SkipZigTest;
    defer tok.deinit();

    var prng = std.Random.DefaultPrng.init(0xCAFEBABE);
    var rng = prng.random();
    var buf: [256]u8 = undefined;

    var i: usize = 0;
    while (i < N_ITERS_PER_FIXTURE) : (i += 1) {
        const text = randomAsciiText(&rng, &buf);
        checkInvariants(&tok, alloc, text) catch |err| {
            std.debug.print("\nfuzz failure on text: {s}\n", .{text});
            return err;
        };
    }
}

test "fuzz — TinyLlama invariants over 200 random texts" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    const maybe_tok = loadTokenizer(alloc, io, "tests/fixtures/tokenizer.json") catch return;
    var tok = maybe_tok orelse return error.SkipZigTest;
    defer tok.deinit();

    var prng = std.Random.DefaultPrng.init(0xFACEFEED);
    var rng = prng.random();
    var buf: [256]u8 = undefined;

    var i: usize = 0;
    while (i < N_ITERS_PER_FIXTURE) : (i += 1) {
        const text = randomAsciiText(&rng, &buf);
        checkInvariants(&tok, alloc, text) catch |err| {
            std.debug.print("\nfuzz failure on text: {s}\n", .{text});
            return err;
        };
    }
}

test "fuzz — BERT pair encodePairFull invariants over 100 pairs" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    const maybe_tok = loadTokenizer(alloc, io, "tests/fixtures/bert-base-uncased-tokenizer.json") catch return;
    var tok = maybe_tok orelse return error.SkipZigTest;
    defer tok.deinit();

    var prng = std.Random.DefaultPrng.init(0xBEEFCAFE);
    var rng = prng.random();
    var buf_a: [128]u8 = undefined;
    var buf_b: [128]u8 = undefined;

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const a = randomAsciiText(&rng, &buf_a);
        const b = randomAsciiText(&rng, &buf_b);
        checkPairInvariants(&tok, alloc, a, b) catch |err| {
            std.debug.print("\npair fuzz failure on a={s} b={s}\n", .{ a, b });
            return err;
        };
    }
}

test "fuzz — sequential vs parallel batch identical (BERT, 50 batches)" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    const maybe_tok = loadTokenizer(alloc, io, "tests/fixtures/bert-base-uncased-tokenizer.json") catch return;
    var tok = maybe_tok orelse return error.SkipZigTest;
    defer tok.deinit();

    var prng = std.Random.DefaultPrng.init(0x10010);
    var rng = prng.random();
    var buf: [128]u8 = undefined;

    // Generate 50 different batches of varying size
    var batch_i: usize = 0;
    while (batch_i < 50) : (batch_i += 1) {
        const batch_size = rng.intRangeAtMost(usize, 2, 10);
        const texts = try alloc.alloc([]u8, batch_size);
        defer {
            for (texts) |t| alloc.free(t);
            alloc.free(texts);
        }
        for (texts) |*t| {
            const tx = randomAsciiText(&rng, &buf);
            t.* = try alloc.dupe(u8, tx);
        }

        const texts_const = try alloc.alloc([]const u8, batch_size);
        defer alloc.free(texts_const);
        for (texts, 0..) |t, i| texts_const[i] = t;

        try checkBatchInvariants(&tok, alloc, texts_const);
    }
}
