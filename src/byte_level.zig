//! ByteLevel pre-tokenizer — GPT-2 / Llama-3 / most modern BPE
//! tokenizer style. Maps each input byte (0..255) to a unique
//! printable Unicode codepoint so that the BPE algorithm operates on
//! "characters" (UTF-8 encoded codepoints) rather than raw bytes.
//!
//! The mapping is bidirectional and canonical: GPT-2 defined it,
//! HuggingFace tokenizers reproduces it. Space (0x20) maps to
//! U+0120 (Ġ), the canonical "space marker" at the start of tokens.
//!
//! Reference:
//!   https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
//!   bytes_to_unicode()

const std = @import("std");

/// Static table: byte → unicode codepoint. Encoded once at comptime.
pub const byte_to_codepoint: [256]u21 = blk: {
    @setEvalBranchQuota(20000);
    var out: [256]u21 = undefined;
    var n: u32 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        if ((b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE and b <= 0xFF)) {
            out[b] = @intCast(b);
        } else {
            out[b] = @intCast(256 + n);
            n += 1;
        }
    }
    break :blk out;
};

/// Inverse: unicode codepoint → byte. Returns 256 if codepoint is
/// not in the ByteLevel domain (shouldn't happen on valid input).
pub fn codepointToByte(cp: u21) u16 {
    var b: u16 = 0;
    while (b < 256) : (b += 1) {
        if (byte_to_codepoint[b] == cp) return b;
    }
    return 256;
}

/// Encode UTF-8 input bytes into the ByteLevel "string" representation.
/// Output is a UTF-8 string where each input byte has been mapped to
/// its canonical ByteLevel codepoint. Caller frees the returned slice.
pub fn encode(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    // Worst case: every byte maps to a 2-byte UTF-8 codepoint (since
    // codepoints 256-323 are 2 UTF-8 bytes each). Reserve accordingly.
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);

    var utf8_buf: [4]u8 = undefined;
    for (input) |b| {
        const cp = byte_to_codepoint[b];
        const n = try std.unicode.utf8Encode(cp, &utf8_buf);
        try out.appendSlice(alloc, utf8_buf[0..n]);
    }
    return out.toOwnedSlice(alloc);
}

/// Encode with a parallel `[]u32` per-output-byte index back to the
/// originating input byte. `result.encoded.len == result.input_index.len`;
/// `input_index[i]` is the input byte position that produced output[i].
/// Caller frees both slices.
pub fn encodeWithOffsets(alloc: std.mem.Allocator, input: []const u8) !struct { encoded: []u8, input_index: []u32 } {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var idx: std.ArrayList(u32) = .empty;
    errdefer idx.deinit(alloc);

    var utf8_buf: [4]u8 = undefined;
    for (input, 0..) |b, in_pos| {
        const cp = byte_to_codepoint[b];
        const n = try std.unicode.utf8Encode(cp, &utf8_buf);
        try out.appendSlice(alloc, utf8_buf[0..n]);
        for (0..n) |_| try idx.append(alloc, @intCast(in_pos));
    }
    return .{
        .encoded = try out.toOwnedSlice(alloc),
        .input_index = try idx.toOwnedSlice(alloc),
    };
}

/// Decode a ByteLevel "string" back to the original UTF-8 bytes.
/// Caller frees the returned slice.
pub fn decode(alloc: std.mem.Allocator, encoded: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);

    var i: usize = 0;
    while (i < encoded.len) {
        const cp_len = try std.unicode.utf8ByteSequenceLength(encoded[i]);
        if (i + cp_len > encoded.len) return error.InvalidUtf8;
        const cp: u21 = @intCast(try std.unicode.utf8Decode(encoded[i..][0..cp_len]));
        const b = codepointToByte(cp);
        if (b >= 256) return error.InvalidByteLevelCodepoint;
        try out.append(alloc, @intCast(b));
        i += cp_len;
    }
    return out.toOwnedSlice(alloc);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "ByteLevel — space is mapped to U+0120 (Ġ)" {
    try testing.expectEqual(@as(u21, 0x0120), byte_to_codepoint[' ']);
}

test "ByteLevel — printable ASCII is identity" {
    try testing.expectEqual(@as(u21, 'A'), byte_to_codepoint['A']);
    try testing.expectEqual(@as(u21, 'a'), byte_to_codepoint['a']);
    try testing.expectEqual(@as(u21, '0'), byte_to_codepoint['0']);
    try testing.expectEqual(@as(u21, '!'), byte_to_codepoint['!']);
    try testing.expectEqual(@as(u21, '~'), byte_to_codepoint['~']);
}

test "ByteLevel — table is a bijection over 256 bytes" {
    var seen = [_]bool{false} ** 324; // up to U+0144
    for (byte_to_codepoint) |cp| {
        const idx: usize = @intCast(cp);
        try testing.expect(!seen[idx]);
        seen[idx] = true;
    }
}

test "ByteLevel — encode + decode round-trip for ASCII" {
    const input = "Hello, world!";
    const encoded = try encode(testing.allocator, input);
    defer testing.allocator.free(encoded);
    const decoded = try decode(testing.allocator, encoded);
    defer testing.allocator.free(decoded);
    try testing.expectEqualStrings(input, decoded);
}

test "ByteLevel — space encoded as Ġ-prefix" {
    const input = " ";
    const encoded = try encode(testing.allocator, input);
    defer testing.allocator.free(encoded);
    // Ġ is U+0120, UTF-8 = C4 A0
    try testing.expectEqualSlices(u8, &[_]u8{ 0xC4, 0xA0 }, encoded);
}

test "ByteLevel — non-ASCII bytes round-trip" {
    const input = "café"; // c, a, f, é (é = 0xC3 0xA9 in UTF-8)
    const encoded = try encode(testing.allocator, input);
    defer testing.allocator.free(encoded);
    const decoded = try decode(testing.allocator, encoded);
    defer testing.allocator.free(decoded);
    try testing.expectEqualStrings(input, decoded);
}

test "ByteLevel — newline (control char) round-trips" {
    const input = "line1\nline2";
    const encoded = try encode(testing.allocator, input);
    defer testing.allocator.free(encoded);
    const decoded = try decode(testing.allocator, encoded);
    defer testing.allocator.free(decoded);
    try testing.expectEqualStrings(input, decoded);
}

test "ByteLevel — encodeWithOffsets tracks input→output byte mapping" {
    const input = "ab";
    const r = try encodeWithOffsets(testing.allocator, input);
    defer testing.allocator.free(r.encoded);
    defer testing.allocator.free(r.input_index);
    // 'a' is printable, maps to 'a' (1 byte). 'b' → 'b' (1 byte).
    try testing.expectEqualStrings("ab", r.encoded);
    try testing.expectEqual(@as(u32, 0), r.input_index[0]);
    try testing.expectEqual(@as(u32, 1), r.input_index[1]);
}

test "ByteLevel — encodeWithOffsets handles 2-byte codepoint expansion" {
    const input = " "; // space → U+0120 (Ġ), encodes to 2 UTF-8 bytes.
    const r = try encodeWithOffsets(testing.allocator, input);
    defer testing.allocator.free(r.encoded);
    defer testing.allocator.free(r.input_index);
    try testing.expectEqualSlices(u8, &[_]u8{ 0xC4, 0xA0 }, r.encoded);
    // Both output bytes point back to input byte 0.
    try testing.expectEqual(@as(u32, 0), r.input_index[0]);
    try testing.expectEqual(@as(u32, 0), r.input_index[1]);
}
