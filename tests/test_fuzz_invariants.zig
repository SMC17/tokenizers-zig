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
        // Monotonic: each offset's start must be >= previous offset's end
        // (or both can be 0,0 for added tokens / synthetic).
        if (off.start != 0 or off.end != 0) {
            try std.testing.expect(off.start >= prev_end);
            prev_end = off.end;
        }
    }

    // encodeFull path — all parallel arrays aligned
    const enc = try tok.encodeFull(alloc, text);
    defer enc.deinit(alloc);
    try std.testing.expectEqual(enc.ids.len, enc.type_ids.len);
    try std.testing.expectEqual(enc.ids.len, enc.attention_mask.len);
    try std.testing.expectEqual(enc.ids.len, enc.offsets.len);
    try std.testing.expectEqual(enc.ids.len, enc.tokens.len);
    try std.testing.expectEqual(enc.ids.len, enc.special_tokens_mask.len);

    // Decode must succeed (no panic / no error).
    const decoded = try tok.decode(alloc, enc.ids);
    defer alloc.free(decoded);
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
