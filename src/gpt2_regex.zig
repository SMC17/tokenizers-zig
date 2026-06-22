//! GPT-2 word-boundary pre-tokenizer.
//!
//! Approximates the canonical GPT-2 regex:
//!   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//!
//! Used by GPT-2 / GPT-3 / RoBERTa-family BPE tokenizers (HF
//! `ByteLevel` pre-tokenizer with `use_regex: true`). The output is a
//! list of substrings of the input which the downstream BPE encoder
//! processes one at a time.
//!
//! Scope (v0.4):
//!   - Contractions: 's 't 're 've 'm 'll 'd (case-sensitive)
//!   - ASCII letter runs (A–Z, a–z) with optional leading space
//!   - Latin-1 letter runs (selected ranges) with optional leading space
//!   - ASCII digit runs with optional leading space
//!   - Other (non-letter, non-digit, non-whitespace) runs with optional
//!     leading space
//!   - Whitespace runs
//!
//! Unicode letter coverage is partial — full \p{L} would require a
//! Unicode property table. CJK / Cyrillic / Arabic / Indic scripts are
//! NOT classified as letters and will fall into the "other" bucket.

const std = @import("std");

pub const Error = error{ OutOfMemory, InvalidUtf8 };

const contractions = [_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };

fn isAsciiLetter(b: u8) bool {
    return (b >= 'A' and b <= 'Z') or (b >= 'a' and b <= 'z');
}

fn isAsciiDigit(b: u8) bool {
    return b >= '0' and b <= '9';
}

fn isAsciiWhitespace(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\n' or b == '\r' or b == 0x0B or b == 0x0C;
}

/// Returns true if codepoint `cp` is a letter under our partial coverage:
/// ASCII letters + Latin-1 supplement letter ranges + Latin Extended-A
/// (most entries).
fn isUnicodeLetter(cp: u21) bool {
    return switch (cp) {
        'A'...'Z', 'a'...'z' => true,
        0x00C0...0x00D6 => true, // Latin-1 uppercase
        0x00D8...0x00F6 => true, // Latin-1 lowercase
        0x00F8...0x02AF => true, // Latin Extended
        0x0370...0x03FF => true, // Greek and Coptic
        0x0400...0x04FF => true, // Cyrillic
        0x0590...0x05FF => true, // Hebrew
        0x0600...0x06FF => true, // Arabic
        0x3040...0x309F => true, // Hiragana
        0x30A0...0x30FF => true, // Katakana
        0x4E00...0x9FFF => true, // CJK Unified Ideographs (Common)
        0xAC00...0xD7AF => true, // Hangul Syllables
        else => false,
    };
}

fn isUnicodeDigit(cp: u21) bool {
    return cp >= '0' and cp <= '9';
}

fn isUnicodeWhitespace(cp: u21) bool {
    return switch (cp) {
        0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20 => true,
        0x00A0 => true, // no-break space
        0x2028, 0x2029 => true, // line/paragraph separators
        0x3000 => true, // ideographic space
        else => false,
    };
}

/// Decode the codepoint at byte position `i` in `text`. Returns
/// (codepoint, byte_length). On invalid UTF-8 returns (b, 1) treating
/// the byte as a single-byte non-class codepoint.
fn decodeAt(text: []const u8, i: usize) struct { cp: u21, len: usize } {
    const n = std.unicode.utf8ByteSequenceLength(text[i]) catch return .{ .cp = text[i], .len = 1 };
    if (i + n > text.len) return .{ .cp = text[i], .len = 1 };
    const cp = std.unicode.utf8Decode(text[i..][0..n]) catch return .{ .cp = text[i], .len = 1 };
    return .{ .cp = cp, .len = n };
}

/// Split `text` into GPT-2 word-boundary chunks. Each chunk is a slice
/// into `text` (no allocation per chunk). Caller frees the outer slice.
pub fn split(alloc: std.mem.Allocator, text: []const u8) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(alloc);

    var pos: usize = 0;
    while (pos < text.len) {
        // Contractions (case-sensitive, ASCII).
        var matched: bool = false;
        for (contractions) |c| {
            if (pos + c.len <= text.len and std.mem.eql(u8, text[pos .. pos + c.len], c)) {
                try out.append(alloc, text[pos .. pos + c.len]);
                pos += c.len;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        const start = pos;
        const has_space = text[pos] == ' ';
        const after_space = if (has_space) pos + 1 else pos;
        if (after_space >= text.len) {
            // Lone leading space (or text ends in just a space).
            try out.append(alloc, text[pos..text.len]);
            pos = text.len;
            continue;
        }

        // Decide which run-class the codepoint after the optional space falls into.
        const first = decodeAt(text, after_space);
        if (isUnicodeLetter(first.cp)) {
            var p = after_space;
            while (p < text.len) {
                const d = decodeAt(text, p);
                if (!isUnicodeLetter(d.cp)) break;
                p += d.len;
            }
            try out.append(alloc, text[start..p]);
            pos = p;
            continue;
        }
        if (isUnicodeDigit(first.cp)) {
            var p = after_space;
            while (p < text.len) {
                const d = decodeAt(text, p);
                if (!isUnicodeDigit(d.cp)) break;
                p += d.len;
            }
            try out.append(alloc, text[start..p]);
            pos = p;
            continue;
        }
        if (!isUnicodeWhitespace(first.cp)) {
            // "Other" run — non-letter, non-digit, non-whitespace.
            var p = after_space;
            while (p < text.len) {
                const d = decodeAt(text, p);
                if (isUnicodeLetter(d.cp) or isUnicodeDigit(d.cp) or isUnicodeWhitespace(d.cp)) break;
                p += d.len;
            }
            try out.append(alloc, text[start..p]);
            pos = p;
            continue;
        }

        // First (non-leading-space-eaten) codepoint IS whitespace.
        // We took has_space into account; that means we have
        // "space + whitespace+", which collapses to a whitespace run.
        var p = pos;
        while (p < text.len) {
            const d = decodeAt(text, p);
            if (!isUnicodeWhitespace(d.cp)) break;
            p += d.len;
        }
        try out.append(alloc, text[pos..p]);
        pos = p;
    }

    return out.toOwnedSlice(alloc);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "split — simple ASCII words emit ' word' chunks" {
    const chunks = try split(testing.allocator, "hello world");
    defer testing.allocator.free(chunks);
    try testing.expectEqual(@as(usize, 2), chunks.len);
    try testing.expectEqualStrings("hello", chunks[0]);
    try testing.expectEqualStrings(" world", chunks[1]);
}

test "split — contractions split on apostrophe" {
    const chunks = try split(testing.allocator, "don't");
    defer testing.allocator.free(chunks);
    try testing.expectEqual(@as(usize, 2), chunks.len);
    try testing.expectEqualStrings("don", chunks[0]);
    try testing.expectEqualStrings("'t", chunks[1]);
}

test "split — multiple contractions in one phrase" {
    const chunks = try split(testing.allocator, "we're going");
    defer testing.allocator.free(chunks);
    try testing.expectEqual(@as(usize, 3), chunks.len);
    try testing.expectEqualStrings("we", chunks[0]);
    try testing.expectEqualStrings("'re", chunks[1]);
    try testing.expectEqualStrings(" going", chunks[2]);
}

test "split — digits separated from letters" {
    const chunks = try split(testing.allocator, "abc123");
    defer testing.allocator.free(chunks);
    try testing.expectEqual(@as(usize, 2), chunks.len);
    try testing.expectEqualStrings("abc", chunks[0]);
    try testing.expectEqualStrings("123", chunks[1]);
}

test "split — punctuation is its own 'other' chunk" {
    const chunks = try split(testing.allocator, "hi!");
    defer testing.allocator.free(chunks);
    try testing.expectEqual(@as(usize, 2), chunks.len);
    try testing.expectEqualStrings("hi", chunks[0]);
    try testing.expectEqualStrings("!", chunks[1]);
}

test "split — adjacent punctuation merges into one 'other' run" {
    const chunks = try split(testing.allocator, "??!");
    defer testing.allocator.free(chunks);
    try testing.expectEqual(@as(usize, 1), chunks.len);
    try testing.expectEqualStrings("??!", chunks[0]);
}

test "split — trailing whitespace becomes a chunk" {
    const chunks = try split(testing.allocator, "hi   ");
    defer testing.allocator.free(chunks);
    try testing.expectEqual(@as(usize, 2), chunks.len);
    try testing.expectEqualStrings("hi", chunks[0]);
    try testing.expectEqualStrings("   ", chunks[1]);
}

test "split — leading space attaches to next word" {
    const chunks = try split(testing.allocator, " hello");
    defer testing.allocator.free(chunks);
    try testing.expectEqual(@as(usize, 1), chunks.len);
    try testing.expectEqualStrings(" hello", chunks[0]);
}

test "split — Latin-1 letters classified correctly" {
    const chunks = try split(testing.allocator, "café résumé");
    defer testing.allocator.free(chunks);
    try testing.expectEqual(@as(usize, 2), chunks.len);
    try testing.expectEqualStrings("café", chunks[0]);
    try testing.expectEqualStrings(" résumé", chunks[1]);
}

test "split — CJK and Cyrillic characters classified correctly" {
    const chunks = try split(testing.allocator, "你好 mir привет");
    defer testing.allocator.free(chunks);
    try testing.expectEqual(@as(usize, 3), chunks.len);
    try testing.expectEqualStrings("你好", chunks[0]);
    try testing.expectEqualStrings(" mir", chunks[1]);
    try testing.expectEqualStrings(" привет", chunks[2]);
}

test "split — empty input returns no chunks" {
    const chunks = try split(testing.allocator, "");
    defer testing.allocator.free(chunks);
    try testing.expectEqual(@as(usize, 0), chunks.len);
}
