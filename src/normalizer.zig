//! Normalizer — pre-BPE text normalization stage.
//!
//! v0.1 scope: the SentencePiece-style normalizer used by Llama / Mistral
//! tokenizers, which is a Sequence of:
//!   1. Prepend (constant string prefix, typically U+2581 "▁")
//!   2. Replace (pattern → content, typically " " → U+2581)
//!
//! Result: spaces become ▁ and the entire text is prefixed with ▁.
//! BPE then operates on this ▁-marked stream. Decoding does the
//! inverse (▁ → " ", strip leading ▁).
//!
//! What's covered:
//!   - Prepend with arbitrary string content
//!   - Replace with String pattern (not Regex)
//!   - Sequence of normalizers applied left-to-right
//!
//! What's NOT covered:
//!   - NFC / NFD / NFKC / NFKD Unicode normalization
//!   - Lowercase / StripAccents
//!   - Regex Replace
//!   - BertNormalizer

const std = @import("std");
const unicode_norm = @import("unicode_norm.zig");

pub const Error = error{
    OutOfMemory,
    InvalidNormalizerJson,
    InvalidUtf8,
};

pub const NormalizerKind = union(enum) {
    none: void,
    prepend: []const u8,
    replace: struct { pattern: []const u8, content: []const u8 },
    sequence: []const NormalizerKind,
    /// ASCII lowercase (a-z → A-Z mapping). Bytes outside ASCII pass
    /// through unchanged. Full Unicode lowercase is v0.5 work.
    lowercase: void,
    /// Unicode normalization forms — partial coverage (Latin-1 + Latin
    /// Extended-A only). Codepoints outside the supported range pass
    /// through unchanged. See src/unicode_norm.zig for the coverage table.
    nfc: void,
    nfd: void,
    nfkc: void,
    nfkd: void,
    /// Strip combining-mark codepoints (U+0300–U+036F range; Combining
    /// Diacritical Marks block). Typically used after NFD to remove
    /// diacritics, matching BertNormalizer.strip_accents semantics.
    strip_accents: void,
    /// BertNormalizer "handle_chinese_chars": surround each CJK
    /// ideograph with spaces so the downstream whitespace pre-tokenize
    /// treats each ideograph as its own word.
    handle_chinese_chars: void,
    /// BertNormalizer "clean_text": drop NULL / U+FFFD / control chars
    /// (except tab/CR/LF) and replace whitespace variants with space.
    clean_text: void,
};

/// Sentinel value in an `input_index` table marking a synthetic output
/// byte (introduced by Prepend, Replace content, handle_chinese_chars
/// spaces, etc.) that has no source position in the input.
pub const SYNTHETIC: u32 = std.math.maxInt(u32);

/// Apply with per-output-byte input-index tracking. Used for offset
/// tracking through Prepend / Replace / Sequence / Lowercase chains
/// (the typical Llama / Mistral normalizer pipeline). For unsupported
/// normalizers (NFC/NFD/strip_accents/handle_chinese_chars/clean_text)
/// the input_index is approximate. Caller frees both slices.
pub fn applyWithOffsets(
    alloc: std.mem.Allocator,
    norm: NormalizerKind,
    input: []const u8,
) !struct { output: []u8, input_index: []u32 } {
    switch (norm) {
        .none => {
            const out = try alloc.dupe(u8, input);
            const idx = try alloc.alloc(u32, input.len);
            for (0..input.len) |i| idx[i] = @intCast(i);
            return .{ .output = out, .input_index = idx };
        },
        .prepend => |prefix| {
            const out = try alloc.alloc(u8, prefix.len + input.len);
            @memcpy(out[0..prefix.len], prefix);
            @memcpy(out[prefix.len..], input);
            const idx = try alloc.alloc(u32, out.len);
            for (0..prefix.len) |i| idx[i] = SYNTHETIC;
            for (0..input.len) |i| idx[prefix.len + i] = @intCast(i);
            return .{ .output = out, .input_index = idx };
        },
        .replace => |r| {
            if (r.pattern.len == 0) return applyWithOffsets(alloc, .none, input);
            var out: std.ArrayList(u8) = .empty;
            errdefer out.deinit(alloc);
            var idx: std.ArrayList(u32) = .empty;
            errdefer idx.deinit(alloc);
            var i: usize = 0;
            while (i < input.len) {
                if (i + r.pattern.len <= input.len and
                    std.mem.eql(u8, input[i .. i + r.pattern.len], r.pattern))
                {
                    // Replace: content bytes are synthetic (or could map
                    // to first matched input byte; HF marks them synthetic).
                    try out.appendSlice(alloc, r.content);
                    for (0..r.content.len) |_| try idx.append(alloc, SYNTHETIC);
                    i += r.pattern.len;
                } else {
                    try out.append(alloc, input[i]);
                    try idx.append(alloc, @intCast(i));
                    i += 1;
                }
            }
            return .{
                .output = try out.toOwnedSlice(alloc),
                .input_index = try idx.toOwnedSlice(alloc),
            };
        },
        .sequence => |seq| {
            // Apply step-by-step, composing offset tables.
            // step output's byte i → step input byte step_idx[i]
            // → input byte input_idx[step_idx[i]] (if not SYNTHETIC).
            var current_out: []u8 = try alloc.dupe(u8, input);
            errdefer alloc.free(current_out);
            var current_idx: []u32 = try alloc.alloc(u32, input.len);
            errdefer alloc.free(current_idx);
            for (0..input.len) |i| current_idx[i] = @intCast(i);

            for (seq) |step| {
                const r = try applyWithOffsets(alloc, step, current_out);
                // Compose: for each byte in r.output, r.input_index[k] gives
                // the byte in current_out (the step's input). Map through
                // current_idx to get the original input byte.
                const new_idx = try alloc.alloc(u32, r.output.len);
                for (r.input_index, 0..) |stepi, k| {
                    if (stepi == SYNTHETIC) {
                        new_idx[k] = SYNTHETIC;
                    } else if (stepi < current_idx.len) {
                        new_idx[k] = current_idx[stepi];
                    } else {
                        new_idx[k] = SYNTHETIC;
                    }
                }
                alloc.free(current_out);
                alloc.free(current_idx);
                alloc.free(r.input_index);
                current_out = r.output;
                current_idx = new_idx;
            }
            return .{ .output = current_out, .input_index = current_idx };
        },
        .lowercase => {
            const out = try alloc.alloc(u8, input.len);
            for (input, 0..) |b, i| out[i] = std.ascii.toLower(b);
            const idx = try alloc.alloc(u32, input.len);
            for (0..input.len) |i| idx[i] = @intCast(i);
            return .{ .output = out, .input_index = idx };
        },
        else => {
            // Unsupported normalizer for offset tracking: fall back to
            // applying without offsets and approximating identity.
            const out = try apply(alloc, norm, input);
            const idx = try alloc.alloc(u32, out.len);
            // Best-effort linear interpolation: out[i] ≈ input[i*input.len/out.len]
            if (out.len == 0) {
                return .{ .output = out, .input_index = idx };
            }
            if (input.len == 0) {
                for (0..out.len) |i| idx[i] = SYNTHETIC;
            } else {
                for (0..out.len) |i| {
                    idx[i] = @intCast((i * input.len) / out.len);
                }
            }
            return .{ .output = out, .input_index = idx };
        },
    }
}

/// Apply a normalizer to UTF-8 input. Caller frees output.
pub fn apply(alloc: std.mem.Allocator, norm: NormalizerKind, input: []const u8) ![]u8 {
    return switch (norm) {
        .none => alloc.dupe(u8, input),
        .prepend => |prefix| {
            const buf = try alloc.alloc(u8, prefix.len + input.len);
            @memcpy(buf[0..prefix.len], prefix);
            @memcpy(buf[prefix.len..], input);
            return buf;
        },
        .replace => |r| try replaceAll(alloc, input, r.pattern, r.content),
        .sequence => |seq| {
            // Apply each in order, threading the buffer through.
            var current: []u8 = try alloc.dupe(u8, input);
            for (seq) |n| {
                const next = try apply(alloc, n, current);
                alloc.free(current);
                current = next;
            }
            return current;
        },
        .lowercase => {
            const out = try alloc.alloc(u8, input.len);
            for (input, 0..) |b, i| out[i] = std.ascii.toLower(b);
            return out;
        },
        .nfc => return unicode_norm.nfc(alloc, input),
        .nfd => return unicode_norm.nfd(alloc, input),
        .nfkc => return unicode_norm.nfkc(alloc, input),
        .nfkd => return unicode_norm.nfkd(alloc, input),
        .strip_accents => return stripAccents(alloc, input),
        .handle_chinese_chars => return handleChineseChars(alloc, input),
        .clean_text => return cleanText(alloc, input),
    };
}

/// Surround each CJK ideograph codepoint with spaces. Used by
/// BertNormalizer.handle_chinese_chars before whitespace pre-tokenize
/// so each ideograph becomes its own WordPiece word.
fn handleChineseChars(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    while (i < input.len) {
        const n = std.unicode.utf8ByteSequenceLength(input[i]) catch {
            try out.append(alloc, input[i]);
            i += 1;
            continue;
        };
        if (i + n > input.len) return Error.InvalidUtf8;
        const cp = std.unicode.utf8Decode(input[i..][0..n]) catch {
            try out.appendSlice(alloc, input[i..][0..n]);
            i += n;
            continue;
        };
        if (isCjkIdeograph(cp)) {
            try out.append(alloc, ' ');
            try out.appendSlice(alloc, input[i..][0..n]);
            try out.append(alloc, ' ');
        } else {
            try out.appendSlice(alloc, input[i..][0..n]);
        }
        i += n;
    }
    return out.toOwnedSlice(alloc);
}

fn isCjkIdeograph(cp: u21) bool {
    return switch (cp) {
        0x4E00...0x9FFF => true, // CJK Unified Ideographs
        0x3400...0x4DBF => true, // CJK Extension A
        0xF900...0xFAFF => true, // CJK Compatibility Ideographs
        // Higher planes (Ext B, C, D, E, Compatibility-Supplement) — u21
        // is the upper limit; we'd need to widen everywhere to cover them.
        // For v0.6 these are documented as out-of-scope.
        else => false,
    };
}

/// BertNormalizer.clean_text: drop NULL / U+FFFD / control chars
/// (except tab/CR/LF) and replace any whitespace codepoint with a
/// regular ASCII space.
fn cleanText(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    while (i < input.len) {
        const n = std.unicode.utf8ByteSequenceLength(input[i]) catch {
            i += 1;
            continue;
        };
        if (i + n > input.len) return Error.InvalidUtf8;
        const cp = std.unicode.utf8Decode(input[i..][0..n]) catch {
            i += n;
            continue;
        };
        if (cp == 0 or cp == 0xFFFD) {
            // drop
        } else if (isControl(cp)) {
            // drop
        } else if (isWhitespaceCp(cp)) {
            try out.append(alloc, ' ');
        } else {
            try out.appendSlice(alloc, input[i..][0..n]);
        }
        i += n;
    }
    return out.toOwnedSlice(alloc);
}

fn isControl(cp: u21) bool {
    return switch (cp) {
        // Tab, LF, CR are whitespace per HF; explicitly exclude.
        0x00...0x08, 0x0B, 0x0C, 0x0E...0x1F => true,
        0x7F, 0x80...0x9F => true,
        else => false,
    };
}

fn isWhitespaceCp(cp: u21) bool {
    return switch (cp) {
        0x09, 0x0A, 0x0D, 0x20 => true,
        0x00A0 => true, // no-break space
        0x1680 => true, // ogham space mark
        0x2000...0x200A => true, // en-quad through hair space
        0x2028, 0x2029 => true,
        0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

/// Strip Combining Diacritical Marks (U+0300–U+036F). All other code-
/// points pass through unchanged. Typically applied after NFD so e.g.
/// "é" → "e" + ◌́ → "e".
fn stripAccents(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    while (i < input.len) {
        const n = std.unicode.utf8ByteSequenceLength(input[i]) catch {
            try out.append(alloc, input[i]);
            i += 1;
            continue;
        };
        if (i + n > input.len) return Error.InvalidUtf8;
        const cp = std.unicode.utf8Decode(input[i..][0..n]) catch {
            try out.appendSlice(alloc, input[i..][0..n]);
            i += n;
            continue;
        };
        if (cp >= 0x0300 and cp <= 0x036F) {
            // Skip combining mark
        } else {
            try out.appendSlice(alloc, input[i..][0..n]);
        }
        i += n;
    }
    return out.toOwnedSlice(alloc);
}

fn replaceAll(alloc: std.mem.Allocator, input: []const u8, pattern: []const u8, content: []const u8) ![]u8 {
    if (pattern.len == 0) return alloc.dupe(u8, input);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    while (i <= input.len) {
        if (i + pattern.len <= input.len and std.mem.eql(u8, input[i .. i + pattern.len], pattern)) {
            try out.appendSlice(alloc, content);
            i += pattern.len;
        } else if (i < input.len) {
            try out.append(alloc, input[i]);
            i += 1;
        } else {
            break;
        }
    }
    return out.toOwnedSlice(alloc);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "Prepend adds prefix" {
    const r = try apply(testing.allocator, .{ .prepend = ">>" }, "hi");
    defer testing.allocator.free(r);
    try testing.expectEqualStrings(">>hi", r);
}

test "Replace single substring" {
    const r = try apply(
        testing.allocator,
        .{ .replace = .{ .pattern = "abc", .content = "Z" } },
        "abcdefabc",
    );
    defer testing.allocator.free(r);
    try testing.expectEqualStrings("ZdefZ", r);
}

test "Replace space with U+2581 (SentencePiece convention)" {
    const r = try apply(
        testing.allocator,
        .{ .replace = .{ .pattern = " ", .content = "\u{2581}" } },
        "hello world",
    );
    defer testing.allocator.free(r);
    // "hello" + ▁ + "world" — ▁ is 3-byte UTF-8 e2 96 81
    try testing.expectEqualStrings("hello\xE2\x96\x81world", r);
}

test "Sequence: Llama-style Prepend + Replace" {
    const sub_norms = [_]NormalizerKind{
        .{ .prepend = "\u{2581}" },
        .{ .replace = .{ .pattern = " ", .content = "\u{2581}" } },
    };
    const r = try apply(testing.allocator, .{ .sequence = &sub_norms }, "hi there");
    defer testing.allocator.free(r);
    // Expected: ▁hi▁there (prepend ▁, then replace space with ▁)
    try testing.expectEqualStrings("\u{2581}hi\u{2581}there", r);
}

test "none is identity" {
    const r = try apply(testing.allocator, .none, "hello");
    defer testing.allocator.free(r);
    try testing.expectEqualStrings("hello", r);
}

test "empty pattern is identity (replace)" {
    const r = try apply(
        testing.allocator,
        .{ .replace = .{ .pattern = "", .content = "X" } },
        "abc",
    );
    defer testing.allocator.free(r);
    try testing.expectEqualStrings("abc", r);
}

test "empty input survives" {
    const r = try apply(testing.allocator, .{ .prepend = "X" }, "");
    defer testing.allocator.free(r);
    try testing.expectEqualStrings("X", r);
}

test "applyWithOffsets — Llama Sequence(Prepend, Replace) tracks input mapping" {
    const sub_norms = [_]NormalizerKind{
        .{ .prepend = "\u{2581}" },
        .{ .replace = .{ .pattern = " ", .content = "\u{2581}" } },
    };
    const r = try applyWithOffsets(testing.allocator, .{ .sequence = &sub_norms }, "hi there");
    defer testing.allocator.free(r.output);
    defer testing.allocator.free(r.input_index);
    // "hi there" → prepend ▁ → "▁hi there" → replace " " → "▁hi▁there"
    // 3 synthetic + h(0) + i(1) + 3 synthetic + t(3) + h(4) + e(5) + r(6) + e(7)
    try testing.expectEqualStrings("\u{2581}hi\u{2581}there", r.output);
    try testing.expectEqual(SYNTHETIC, r.input_index[0]); // ▁
    try testing.expectEqual(SYNTHETIC, r.input_index[1]);
    try testing.expectEqual(SYNTHETIC, r.input_index[2]);
    try testing.expectEqual(@as(u32, 0), r.input_index[3]); // h
    try testing.expectEqual(@as(u32, 1), r.input_index[4]); // i
    try testing.expectEqual(SYNTHETIC, r.input_index[5]); // ▁ (was space)
    try testing.expectEqual(SYNTHETIC, r.input_index[6]);
    try testing.expectEqual(SYNTHETIC, r.input_index[7]);
    try testing.expectEqual(@as(u32, 3), r.input_index[8]); // t
}

test "lowercase ASCII works" {
    const r = try apply(testing.allocator, .lowercase, "Hello, World!");
    defer testing.allocator.free(r);
    try testing.expectEqualStrings("hello, world!", r);
}

test "lowercase passes non-ASCII through unchanged" {
    const r = try apply(testing.allocator, .lowercase, "Café É");
    defer testing.allocator.free(r);
    // ASCII lowercase: 'C' → 'c', 'a', 'f', non-ASCII é stays, ' ', non-ASCII É stays
    try testing.expectEqualStrings("café É", r);
}
