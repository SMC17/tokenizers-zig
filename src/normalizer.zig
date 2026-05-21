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

pub const Error = error{
    OutOfMemory,
    InvalidNormalizerJson,
};

pub const NormalizerKind = union(enum) {
    none: void,
    prepend: []const u8,
    replace: struct { pattern: []const u8, content: []const u8 },
    sequence: []const NormalizerKind,
};

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
    };
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
