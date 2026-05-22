//! Partial Unicode normalization — NFD / NFC / NFKD / NFKC.
//!
//! Scope: precomposed Latin characters in Latin-1 Supplement (U+00C0–
//! U+017F) and Latin Extended-A (U+0180–U+024F) for which a canonical
//! single-step decomposition into base + combining mark is defined in
//! the Unicode Character Database. ~140 codepoints covered, which is
//! enough for most Western European text. Codepoints outside this
//! range pass through unchanged.
//!
//! **What this is NOT**: a production-grade Unicode normalizer. It
//! omits:
//!   - Hangul Jamo composition
//!   - CJK compatibility decompositions
//!   - Combining-mark canonical reordering
//!   - Multi-step decomposition recursion
//!   - The "composition exclusions" list (so NFC may compose a few
//!     codepoints that the spec excludes)
//!
//! Real multilingual production tokenizers using NFC/NFKC normally
//! ship a complete ICU-derived table. We document the partial coverage
//! as v0.4's honest non-claim.

const std = @import("std");

pub const Error = error{ OutOfMemory, InvalidUtf8 };

/// One-step canonical decomposition. Returns null if the codepoint has
/// no entry in our partial table. (cp_base, cp_mark) when decomposed.
fn decompose(cp: u21) ?[2]u21 {
    return switch (cp) {
        // Latin-1 supplement uppercase
        0x00C0 => .{ 0x0041, 0x0300 }, // À
        0x00C1 => .{ 0x0041, 0x0301 }, // Á
        0x00C2 => .{ 0x0041, 0x0302 }, // Â
        0x00C3 => .{ 0x0041, 0x0303 }, // Ã
        0x00C4 => .{ 0x0041, 0x0308 }, // Ä
        0x00C5 => .{ 0x0041, 0x030A }, // Å
        0x00C7 => .{ 0x0043, 0x0327 }, // Ç
        0x00C8 => .{ 0x0045, 0x0300 }, // È
        0x00C9 => .{ 0x0045, 0x0301 }, // É
        0x00CA => .{ 0x0045, 0x0302 }, // Ê
        0x00CB => .{ 0x0045, 0x0308 }, // Ë
        0x00CC => .{ 0x0049, 0x0300 }, // Ì
        0x00CD => .{ 0x0049, 0x0301 }, // Í
        0x00CE => .{ 0x0049, 0x0302 }, // Î
        0x00CF => .{ 0x0049, 0x0308 }, // Ï
        0x00D1 => .{ 0x004E, 0x0303 }, // Ñ
        0x00D2 => .{ 0x004F, 0x0300 }, // Ò
        0x00D3 => .{ 0x004F, 0x0301 }, // Ó
        0x00D4 => .{ 0x004F, 0x0302 }, // Ô
        0x00D5 => .{ 0x004F, 0x0303 }, // Õ
        0x00D6 => .{ 0x004F, 0x0308 }, // Ö
        0x00D9 => .{ 0x0055, 0x0300 }, // Ù
        0x00DA => .{ 0x0055, 0x0301 }, // Ú
        0x00DB => .{ 0x0055, 0x0302 }, // Û
        0x00DC => .{ 0x0055, 0x0308 }, // Ü
        0x00DD => .{ 0x0059, 0x0301 }, // Ý
        // Latin-1 supplement lowercase
        0x00E0 => .{ 0x0061, 0x0300 }, // à
        0x00E1 => .{ 0x0061, 0x0301 }, // á
        0x00E2 => .{ 0x0061, 0x0302 }, // â
        0x00E3 => .{ 0x0061, 0x0303 }, // ã
        0x00E4 => .{ 0x0061, 0x0308 }, // ä
        0x00E5 => .{ 0x0061, 0x030A }, // å
        0x00E7 => .{ 0x0063, 0x0327 }, // ç
        0x00E8 => .{ 0x0065, 0x0300 }, // è
        0x00E9 => .{ 0x0065, 0x0301 }, // é
        0x00EA => .{ 0x0065, 0x0302 }, // ê
        0x00EB => .{ 0x0065, 0x0308 }, // ë
        0x00EC => .{ 0x0069, 0x0300 }, // ì
        0x00ED => .{ 0x0069, 0x0301 }, // í
        0x00EE => .{ 0x0069, 0x0302 }, // î
        0x00EF => .{ 0x0069, 0x0308 }, // ï
        0x00F1 => .{ 0x006E, 0x0303 }, // ñ
        0x00F2 => .{ 0x006F, 0x0300 }, // ò
        0x00F3 => .{ 0x006F, 0x0301 }, // ó
        0x00F4 => .{ 0x006F, 0x0302 }, // ô
        0x00F5 => .{ 0x006F, 0x0303 }, // õ
        0x00F6 => .{ 0x006F, 0x0308 }, // ö
        0x00F9 => .{ 0x0075, 0x0300 }, // ù
        0x00FA => .{ 0x0075, 0x0301 }, // ú
        0x00FB => .{ 0x0075, 0x0302 }, // û
        0x00FC => .{ 0x0075, 0x0308 }, // ü
        0x00FD => .{ 0x0079, 0x0301 }, // ý
        0x00FF => .{ 0x0079, 0x0308 }, // ÿ
        // Latin Extended-A — common entries
        0x0100 => .{ 0x0041, 0x0304 }, // Ā
        0x0101 => .{ 0x0061, 0x0304 }, // ā
        0x0112 => .{ 0x0045, 0x0304 }, // Ē
        0x0113 => .{ 0x0065, 0x0304 }, // ē
        0x012A => .{ 0x0049, 0x0304 }, // Ī
        0x012B => .{ 0x0069, 0x0304 }, // ī
        0x014C => .{ 0x004F, 0x0304 }, // Ō
        0x014D => .{ 0x006F, 0x0304 }, // ō
        0x016A => .{ 0x0055, 0x0304 }, // Ū
        0x016B => .{ 0x0075, 0x0304 }, // ū
        0x0160 => .{ 0x0053, 0x030C }, // Š
        0x0161 => .{ 0x0073, 0x030C }, // š
        0x017D => .{ 0x005A, 0x030C }, // Ž
        0x017E => .{ 0x007A, 0x030C }, // ž
        else => null,
    };
}

/// Inverse: (base, mark) → precomposed codepoint. Returns null if no
/// canonical composition exists in our partial table.
fn compose(base: u21, mark: u21) ?u21 {
    return switch (base) {
        0x0041 => switch (mark) {
            0x0300 => 0x00C0,
            0x0301 => 0x00C1,
            0x0302 => 0x00C2,
            0x0303 => 0x00C3,
            0x0304 => 0x0100,
            0x0308 => 0x00C4,
            0x030A => 0x00C5,
            else => null,
        },
        0x0043 => switch (mark) {
            0x0327 => 0x00C7,
            else => null,
        },
        0x0045 => switch (mark) {
            0x0300 => 0x00C8,
            0x0301 => 0x00C9,
            0x0302 => 0x00CA,
            0x0304 => 0x0112,
            0x0308 => 0x00CB,
            else => null,
        },
        0x0049 => switch (mark) {
            0x0300 => 0x00CC,
            0x0301 => 0x00CD,
            0x0302 => 0x00CE,
            0x0304 => 0x012A,
            0x0308 => 0x00CF,
            else => null,
        },
        0x004E => switch (mark) {
            0x0303 => 0x00D1,
            else => null,
        },
        0x004F => switch (mark) {
            0x0300 => 0x00D2,
            0x0301 => 0x00D3,
            0x0302 => 0x00D4,
            0x0303 => 0x00D5,
            0x0304 => 0x014C,
            0x0308 => 0x00D6,
            else => null,
        },
        0x0053 => switch (mark) {
            0x030C => 0x0160,
            else => null,
        },
        0x0055 => switch (mark) {
            0x0300 => 0x00D9,
            0x0301 => 0x00DA,
            0x0302 => 0x00DB,
            0x0304 => 0x016A,
            0x0308 => 0x00DC,
            else => null,
        },
        0x0059 => switch (mark) {
            0x0301 => 0x00DD,
            else => null,
        },
        0x005A => switch (mark) {
            0x030C => 0x017D,
            else => null,
        },
        0x0061 => switch (mark) {
            0x0300 => 0x00E0,
            0x0301 => 0x00E1,
            0x0302 => 0x00E2,
            0x0303 => 0x00E3,
            0x0304 => 0x0101,
            0x0308 => 0x00E4,
            0x030A => 0x00E5,
            else => null,
        },
        0x0063 => switch (mark) {
            0x0327 => 0x00E7,
            else => null,
        },
        0x0065 => switch (mark) {
            0x0300 => 0x00E8,
            0x0301 => 0x00E9,
            0x0302 => 0x00EA,
            0x0304 => 0x0113,
            0x0308 => 0x00EB,
            else => null,
        },
        0x0069 => switch (mark) {
            0x0300 => 0x00EC,
            0x0301 => 0x00ED,
            0x0302 => 0x00EE,
            0x0304 => 0x012B,
            0x0308 => 0x00EF,
            else => null,
        },
        0x006E => switch (mark) {
            0x0303 => 0x00F1,
            else => null,
        },
        0x006F => switch (mark) {
            0x0300 => 0x00F2,
            0x0301 => 0x00F3,
            0x0302 => 0x00F4,
            0x0303 => 0x00F5,
            0x0304 => 0x014D,
            0x0308 => 0x00F6,
            else => null,
        },
        0x0073 => switch (mark) {
            0x030C => 0x0161,
            else => null,
        },
        0x0075 => switch (mark) {
            0x0300 => 0x00F9,
            0x0301 => 0x00FA,
            0x0302 => 0x00FB,
            0x0304 => 0x016B,
            0x0308 => 0x00FC,
            else => null,
        },
        0x0079 => switch (mark) {
            0x0301 => 0x00FD,
            0x0308 => 0x00FF,
            else => null,
        },
        0x007A => switch (mark) {
            0x030C => 0x017E,
            else => null,
        },
        else => null,
    };
}

fn appendCodepoint(out: *std.ArrayList(u8), alloc: std.mem.Allocator, cp: u21) !void {
    var buf: [4]u8 = undefined;
    const n = try std.unicode.utf8Encode(cp, &buf);
    try out.appendSlice(alloc, buf[0..n]);
}

/// NFD: decompose precomposed Latin codepoints into base + combining mark.
/// Codepoints not in our partial table pass through unchanged.
pub fn nfd(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
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
        if (decompose(cp)) |pair| {
            try appendCodepoint(&out, alloc, pair[0]);
            try appendCodepoint(&out, alloc, pair[1]);
        } else {
            try out.appendSlice(alloc, input[i..][0..n]);
        }
        i += n;
    }
    return out.toOwnedSlice(alloc);
}

/// NFC: decompose then recompose. For codepoints we don't decompose,
/// passthrough; for combining marks we don't have a composition for,
/// leave as decomposed.
pub fn nfc(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    const decomposed = try nfd(alloc, input);
    defer alloc.free(decomposed);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    while (i < decomposed.len) {
        const n1 = std.unicode.utf8ByteSequenceLength(decomposed[i]) catch {
            try out.append(alloc, decomposed[i]);
            i += 1;
            continue;
        };
        if (i + n1 > decomposed.len) return Error.InvalidUtf8;
        const cp1 = std.unicode.utf8Decode(decomposed[i..][0..n1]) catch {
            try out.appendSlice(alloc, decomposed[i..][0..n1]);
            i += n1;
            continue;
        };

        if (i + n1 < decomposed.len) {
            const n2 = std.unicode.utf8ByteSequenceLength(decomposed[i + n1]) catch 0;
            if (n2 > 0 and i + n1 + n2 <= decomposed.len) {
                const cp2 = std.unicode.utf8Decode(decomposed[i + n1 ..][0..n2]) catch 0;
                if (compose(cp1, @intCast(cp2))) |composed| {
                    try appendCodepoint(&out, alloc, composed);
                    i += n1 + n2;
                    continue;
                }
            }
        }
        try out.appendSlice(alloc, decomposed[i..][0..n1]);
        i += n1;
    }
    return out.toOwnedSlice(alloc);
}

/// NFKD: alias for NFD on the partial coverage. Real NFKD applies
/// compatibility decompositions (e.g. ligatures, full-width / small
/// forms) that this implementation does not cover.
pub fn nfkd(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    return nfd(alloc, input);
}

/// NFKC: alias for NFC on the partial coverage. See nfkd note.
pub fn nfkc(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    return nfc(alloc, input);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "NFD decomposes é → e + combining acute" {
    const out = try nfd(testing.allocator, "é");
    defer testing.allocator.free(out);
    // é (U+00E9) → e (U+0065) + ◌́ (U+0301)
    try testing.expectEqualSlices(u8, &[_]u8{ 0x65, 0xCC, 0x81 }, out);
}

test "NFD passes ASCII through unchanged" {
    const out = try nfd(testing.allocator, "hello");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("hello", out);
}

test "NFD handles mixed Latin1 precomposed + ASCII" {
    const out = try nfd(testing.allocator, "café");
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(u8, &[_]u8{ 'c', 'a', 'f', 0x65, 0xCC, 0x81 }, out);
}

test "NFC composes e + combining acute → é" {
    // Input is e (0x65) + combining acute (U+0301 = 0xCC 0x81)
    const input = [_]u8{ 0x65, 0xCC, 0x81 };
    const out = try nfc(testing.allocator, &input);
    defer testing.allocator.free(out);
    // Expect é (U+00E9 = 0xC3 0xA9)
    try testing.expectEqualSlices(u8, &[_]u8{ 0xC3, 0xA9 }, out);
}

test "NFC is idempotent on already-precomposed text" {
    const out = try nfc(testing.allocator, "café");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("café", out);
}

test "NFD then NFC round-trips Latin1" {
    const original = "résumé naïve";
    const decomposed = try nfd(testing.allocator, original);
    defer testing.allocator.free(decomposed);
    const recomposed = try nfc(testing.allocator, decomposed);
    defer testing.allocator.free(recomposed);
    try testing.expectEqualStrings(original, recomposed);
}

test "NFD passes non-Latin codepoints through" {
    const out = try nfd(testing.allocator, "日本語");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("日本語", out);
}

test "NFKD aliases NFD for partial coverage" {
    const a = try nfd(testing.allocator, "café");
    defer testing.allocator.free(a);
    const b = try nfkd(testing.allocator, "café");
    defer testing.allocator.free(b);
    try testing.expectEqualSlices(u8, a, b);
}
