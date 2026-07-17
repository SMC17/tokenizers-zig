//! WordPiece model (BERT / DistilBERT / MobileBERT family).
//!
//! Algorithm (greedy longest-match with continuation prefix):
//!   1. If word's codepoint count exceeds max_input_chars_per_word,
//!      emit unk_id and stop.
//!   2. From left to right, find the longest prefix (starting at the
//!      current position) that exists in vocab. For positions after
//!      the first, the candidate is prefixed with continuing_subword_prefix
//!      (canonically "##").
//!   3. If no prefix matches at the current position, the whole word
//!      is unknown — emit unk_id once and stop.
//!   4. Otherwise emit the matched token, advance past it, repeat.
//!
//! Boundaries are codepoint-aware (not byte-aware), matching the HF
//! Rust reference implementation in huggingface/tokenizers/src/models/wordpiece.
//!
//! v0.27 perf: SIMD ASCII fast-path (@Vector(16, u8) test for high-bit)
//! avoids the per-byte UTF-8 sequence-length decode on the overwhelming
//! majority of BERT inputs. Stack-allocated boundaries + key buffer
//! eliminate heap allocs on the hot path.

const std = @import("std");

pub const Error = error{
    OutOfMemory,
    InvalidUtf8,
};

/// Stack buffer sizes — max_input_chars_per_word is canonically 100,
/// so 128 codepoints handles the canonical case + small margin. Each
/// codepoint is at most 4 UTF-8 bytes, plus prefix room.
const MAX_BOUNDARIES: usize = 129; // 128 chars + sentinel
const MAX_KEY_BYTES: usize = 128 * 4 + 8; // prefix ("##") + 128 cps × 4 bytes

/// SIMD-accelerated test: returns true if every byte in `bytes` is
/// pure ASCII (< 0x80). Uses @Vector for 16-byte stripes when possible
/// and falls back to scalar for the tail.
inline fn isAllAscii(bytes: []const u8) bool {
    const Vec = @Vector(16, u8);
    var i: usize = 0;
    while (i + 16 <= bytes.len) : (i += 16) {
        const v: Vec = bytes[i..][0..16].*;
        const masked = v & @as(Vec, @splat(0x80));
        const zero: Vec = @splat(0);
        // Any non-ASCII byte sets the 0x80 bit; reduce-or short-circuits.
        if (@reduce(.Or, masked) != @reduce(.Or, zero)) return false;
    }
    while (i < bytes.len) : (i += 1) {
        if (bytes[i] >= 0x80) return false;
    }
    return true;
}

inline fn buildBoundariesUtf8(
    alloc: std.mem.Allocator,
    word: []const u8,
    stack_boundaries: *[MAX_BOUNDARIES]u32,
    heap_boundaries: *?[]u32,
) ![]u32 {
    var char_count: u32 = 0;
    var i: usize = 0;
    while (i < word.len) {
        const n = try std.unicode.utf8ByteSequenceLength(word[i]);
        if (i + n > word.len) return error.InvalidUtf8;
        i += n;
        char_count += 1;
    }

    const out: []u32 = if (char_count + 1 <= MAX_BOUNDARIES)
        stack_boundaries[0 .. char_count + 1]
    else blk: {
        const hb = try alloc.alloc(u32, char_count + 1);
        heap_boundaries.* = hb;
        break :blk hb;
    };

    out[0] = 0;
    i = 0;
    var b_idx: u32 = 1;
    while (i < word.len) {
        const n = try std.unicode.utf8ByteSequenceLength(word[i]);
        i += n;
        out[b_idx] = @intCast(i);
        b_idx += 1;
    }
    return out;
}

pub const VocabEntry = struct { token: []const u8, id: u32 };

pub const Model = struct {
    allocator: std.mem.Allocator,
    /// Owned storage for vocab token strings + continuing_subword_prefix.
    arena: *std.heap.ArenaAllocator,
    /// token-string → token-id (includes "##"-prefixed continuation forms).
    vocab: std.StringHashMap(u32),
    /// Reverse: id → token-string (built once at construction for fast decode).
    id_to_token: std.AutoHashMap(u32, []const u8),
    unk_id: u32,
    /// Canonically "##" — prefix attached to continuation sub-tokens.
    continuing_subword_prefix: []const u8,
    /// Canonically 100 — words longer than this are emitted as a single unk.
    max_input_chars_per_word: u32,

    pub fn deinit(self: *Model) void {
        self.vocab.deinit();
        self.id_to_token.deinit();
        self.arena.deinit();
        self.allocator.destroy(self.arena);
    }

    /// Encode and return parallel byte-offset ranges (in the input
    /// word's byte space) for each emitted token. Useful for NER sub-
    /// word offset tracking. Caller frees both slices.
    pub fn encodeWithOffsets(
        self: *const Model,
        alloc: std.mem.Allocator,
        word: []const u8,
    ) !struct { ids: []u32, offsets: [][2]u32 } {
        if (word.len == 0) return .{
            .ids = try alloc.alloc(u32, 0),
            .offsets = try alloc.alloc([2]u32, 0),
        };

        // Stack-allocated boundaries: avoids heap thrash on the hot path.
        // Falls back to heap if the word would overflow.
        var stack_boundaries: [MAX_BOUNDARIES]u32 = undefined;
        var heap_boundaries: ?[]u32 = null;
        defer if (heap_boundaries) |hb| alloc.free(hb);

        const boundaries: []u32 = blk: {
            if (isAllAscii(word)) {
                // ASCII fast-path: byte indices ARE codepoint boundaries.
                // For long ASCII words we still need heap storage.
                if (word.len + 1 <= MAX_BOUNDARIES) {
                    var n: u32 = 0;
                    while (n <= word.len) : (n += 1) stack_boundaries[n] = n;
                    break :blk stack_boundaries[0 .. word.len + 1];
                }
                const hb = try alloc.alloc(u32, word.len + 1);
                heap_boundaries = hb;
                var n: u32 = 0;
                while (n <= word.len) : (n += 1) hb[n] = n;
                break :blk hb;
            }
            // Unicode path: SIMD-accelerated continuation-byte mask, then
            // identify boundaries (positions where (byte & 0xC0) != 0x80).
            break :blk try buildBoundariesUtf8(alloc, word, &stack_boundaries, &heap_boundaries);
        };

        const n_chars: u32 = @intCast(boundaries.len - 1);

        if (n_chars > self.max_input_chars_per_word) {
            const ids = try alloc.alloc(u32, 1);
            ids[0] = self.unk_id;
            const offs = try alloc.alloc([2]u32, 1);
            offs[0] = .{ 0, @intCast(word.len) };
            return .{ .ids = ids, .offsets = offs };
        }

        var ids_out: std.ArrayList(u32) = .empty;
        errdefer ids_out.deinit(alloc);
        var off_out: std.ArrayList([2]u32) = .empty;
        errdefer off_out.deinit(alloc);

        // Stack key buffer: prefix + worst-case-byte word fits easily.
        var key_stack: [MAX_KEY_BYTES]u8 = undefined;
        var key_heap: ?[]u8 = null;
        defer if (key_heap) |kh| alloc.free(kh);
        const prefix = self.continuing_subword_prefix;
        const key_buf: []u8 = if (word.len + prefix.len <= MAX_KEY_BYTES)
            key_stack[0 .. word.len + prefix.len]
        else blk: {
            const kh = try alloc.alloc(u8, word.len + prefix.len);
            key_heap = kh;
            break :blk kh;
        };
        @memcpy(key_buf[0..prefix.len], prefix);

        var start: u32 = 0;
        while (start < n_chars) {
            var end: u32 = n_chars;
            var matched_id: ?u32 = null;
            var matched_end: u32 = 0;
            while (end > start) : (end -= 1) {
                const byte_start = boundaries[start];
                const byte_end = boundaries[end];
                const substr = word[byte_start..byte_end];
                const lookup: []const u8 = if (start == 0)
                    substr
                else lk: {
                    @memcpy(key_buf[prefix.len .. prefix.len + substr.len], substr);
                    break :lk key_buf[0 .. prefix.len + substr.len];
                };
                if (self.vocab.get(lookup)) |id| {
                    matched_id = id;
                    matched_end = end;
                    break;
                }
            }
            if (matched_id) |id| {
                try ids_out.append(alloc, id);
                try off_out.append(alloc, .{
                    boundaries[start],
                    boundaries[matched_end],
                });
                start = matched_end;
            } else {
                ids_out.clearRetainingCapacity();
                off_out.clearRetainingCapacity();
                try ids_out.append(alloc, self.unk_id);
                try off_out.append(alloc, .{ 0, @intCast(word.len) });
                return .{
                    .ids = try ids_out.toOwnedSlice(alloc),
                    .offsets = try off_out.toOwnedSlice(alloc),
                };
            }
        }

        return .{
            .ids = try ids_out.toOwnedSlice(alloc),
            .offsets = try off_out.toOwnedSlice(alloc),
        };
    }

    /// Encode a single pre-tokenized "word" (already split on whitespace +
    /// punctuation by the caller) into token IDs. Caller frees.
    pub fn encode(self: *const Model, alloc: std.mem.Allocator, word: []const u8) ![]u32 {
        if (word.len == 0) return alloc.alloc(u32, 0);

        // Stack-allocated boundaries: avoids heap thrash on the hot path.
        // Falls back to heap if the word would overflow.
        var stack_boundaries: [MAX_BOUNDARIES]u32 = undefined;
        var heap_boundaries: ?[]u32 = null;
        defer if (heap_boundaries) |hb| alloc.free(hb);

        const boundaries: []u32 = blk: {
            if (isAllAscii(word)) {
                if (word.len + 1 <= MAX_BOUNDARIES) {
                    var n: u32 = 0;
                    while (n <= word.len) : (n += 1) stack_boundaries[n] = n;
                    break :blk stack_boundaries[0 .. word.len + 1];
                }
                const hb = try alloc.alloc(u32, word.len + 1);
                heap_boundaries = hb;
                var n: u32 = 0;
                while (n <= word.len) : (n += 1) hb[n] = n;
                break :blk hb;
            }
            break :blk try buildBoundariesUtf8(alloc, word, &stack_boundaries, &heap_boundaries);
        };

        const n_chars: u32 = @intCast(boundaries.len - 1);

        // Word too long → emit single unk.
        if (n_chars > self.max_input_chars_per_word) {
            const out = try alloc.alloc(u32, 1);
            out[0] = self.unk_id;
            return out;
        }

        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(alloc);

        // Stack-allocated "##<substr>" key buffer — written once, the
        // prefix bytes never change. Inner loop only copies substr bytes.
        var key_stack: [MAX_KEY_BYTES]u8 = undefined;
        var key_heap: ?[]u8 = null;
        defer if (key_heap) |kh| alloc.free(kh);
        const prefix = self.continuing_subword_prefix;
        const key_buf: []u8 = if (word.len + prefix.len <= MAX_KEY_BYTES)
            key_stack[0 .. word.len + prefix.len]
        else blk: {
            const kh = try alloc.alloc(u8, word.len + prefix.len);
            key_heap = kh;
            break :blk kh;
        };
        @memcpy(key_buf[0..prefix.len], prefix);

        var start: u32 = 0;
        while (start < n_chars) {
            var end: u32 = n_chars;
            var matched_id: ?u32 = null;
            var matched_end: u32 = 0;
            while (end > start) : (end -= 1) {
                const byte_start = boundaries[start];
                const byte_end = boundaries[end];
                const substr = word[byte_start..byte_end];

                const lookup: []const u8 = if (start == 0)
                    substr
                else lk: {
                    @memcpy(key_buf[prefix.len .. prefix.len + substr.len], substr);
                    break :lk key_buf[0 .. prefix.len + substr.len];
                };

                if (self.vocab.get(lookup)) |id| {
                    matched_id = id;
                    matched_end = end;
                    break;
                }
            }

            if (matched_id) |id| {
                try out.append(alloc, id);
                start = matched_end;
            } else {
                // No prefix matches at this position — whole word is unk.
                out.clearRetainingCapacity();
                try out.append(alloc, self.unk_id);
                return out.toOwnedSlice(alloc);
            }
        }

        return out.toOwnedSlice(alloc);
    }

    /// Decode token IDs to text. Continuation tokens (## prefix) are
    /// concatenated to the previous token without a space; first tokens
    /// are space-separated.
    /// Caller frees.
    pub fn decode(self: *const Model, alloc: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(alloc);

        for (ids, 0..) |id, i| {
            const tok = self.id_to_token.get(id) orelse continue;
            if (std.mem.startsWith(u8, tok, self.continuing_subword_prefix)) {
                try out.appendSlice(alloc, tok[self.continuing_subword_prefix.len..]);
            } else {
                if (i > 0) try out.append(alloc, ' ');
                try out.appendSlice(alloc, tok);
            }
        }

        return out.toOwnedSlice(alloc);
    }
};

/// Build a WordPiece Model. All vocab token strings + prefix are
/// copied into an internal arena; caller's slices may be freed after.
pub fn build(
    alloc: std.mem.Allocator,
    vocab_entries: []const VocabEntry,
    unk_id: u32,
    continuing_subword_prefix: []const u8,
    max_input_chars_per_word: u32,
) !Model {
    const arena = try alloc.create(std.heap.ArenaAllocator);
    errdefer alloc.destroy(arena);
    arena.* = std.heap.ArenaAllocator.init(alloc);
    errdefer arena.deinit();
    const aa = arena.allocator();

    var vocab = std.StringHashMap(u32).init(alloc);
    errdefer vocab.deinit();
    var id_to_token = std.AutoHashMap(u32, []const u8).init(alloc);
    errdefer id_to_token.deinit();

    for (vocab_entries) |e| {
        const tok = try aa.dupe(u8, e.token);
        try vocab.put(tok, e.id);
        try id_to_token.put(e.id, tok);
    }

    const prefix = try aa.dupe(u8, continuing_subword_prefix);

    return .{
        .allocator = alloc,
        .arena = arena,
        .vocab = vocab,
        .id_to_token = id_to_token,
        .unk_id = unk_id,
        .continuing_subword_prefix = prefix,
        .max_input_chars_per_word = max_input_chars_per_word,
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "WordPiece — exact-vocab word encodes as single token" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "[UNK]", .id = 0 },
            .{ .token = "hello", .id = 1 },
        },
        0,
        "##",
        100,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "hello");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{1}, ids);
}

test "WordPiece — greedy longest-match prefers full word over split" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "[UNK]", .id = 0 },
            .{ .token = "play", .id = 1 },
            .{ .token = "##ing", .id = 2 },
            .{ .token = "playing", .id = 3 },
        },
        0,
        "##",
        100,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "playing");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{3}, ids);
}

test "WordPiece — splits into root + continuation when no full match" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "[UNK]", .id = 0 },
            .{ .token = "play", .id = 1 },
            .{ .token = "##ing", .id = 2 },
        },
        0,
        "##",
        100,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "playing");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2 }, ids);
}

test "WordPiece — three-piece split (root + 2 continuations)" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "[UNK]", .id = 0 },
            .{ .token = "un", .id = 1 },
            .{ .token = "##aff", .id = 2 },
            .{ .token = "##able", .id = 3 },
        },
        0,
        "##",
        100,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "unaffable");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3 }, ids);
}

test "WordPiece — unknown subword emits whole-word unk once" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "[UNK]", .id = 0 },
            .{ .token = "play", .id = 1 },
            .{ .token = "##ing", .id = 2 },
        },
        0,
        "##",
        100,
    );
    defer m.deinit();

    // "playxq" — "play" matches first; then "xq" has no continuation
    // match starting with "##x" or "##q". Whole word emits single unk.
    const ids = try m.encode(testing.allocator, "playxq");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{0}, ids);
}

test "WordPiece — over-long word emits single unk" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "[UNK]", .id = 0 },
            .{ .token = "a", .id = 1 },
        },
        0,
        "##",
        5,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "aaaaaa"); // 6 chars > 5
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{0}, ids);
}

test "WordPiece — empty input returns empty ids" {
    var m = try build(
        testing.allocator,
        &.{.{ .token = "[UNK]", .id = 0 }},
        0,
        "##",
        100,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "");
    defer testing.allocator.free(ids);
    try testing.expectEqual(@as(usize, 0), ids.len);
}

test "WordPiece — UTF-8 (multibyte codepoints) handled correctly" {
    // café = c, a, f, é. With vocab {café=5}, should encode as [5].
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "[UNK]", .id = 0 },
            .{ .token = "café", .id = 5 },
            .{ .token = "ca", .id = 1 },
            .{ .token = "##fé", .id = 2 },
        },
        0,
        "##",
        100,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "café");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{5}, ids);
}

test "WordPiece — decode rejoins ## continuations without space" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "[UNK]", .id = 0 },
            .{ .token = "un", .id = 1 },
            .{ .token = "##aff", .id = 2 },
            .{ .token = "##able", .id = 3 },
        },
        0,
        "##",
        100,
    );
    defer m.deinit();

    const text = try m.decode(testing.allocator, &[_]u32{ 1, 2, 3 });
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("unaffable", text);
}

test "WordPiece — decode adds space between root tokens" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "[UNK]", .id = 0 },
            .{ .token = "hello", .id = 1 },
            .{ .token = "world", .id = 2 },
        },
        0,
        "##",
        100,
    );
    defer m.deinit();

    const text = try m.decode(testing.allocator, &[_]u32{ 1, 2 });
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("hello world", text);
}

test "WordPiece encodeWithOffsets — sub-word ranges within a single word" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "[UNK]", .id = 0 },
            .{ .token = "un", .id = 1 },
            .{ .token = "##aff", .id = 2 },
            .{ .token = "##able", .id = 3 },
        },
        0,
        "##",
        100,
    );
    defer m.deinit();

    const r = try m.encodeWithOffsets(testing.allocator, "unaffable");
    defer testing.allocator.free(r.ids);
    defer testing.allocator.free(r.offsets);

    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3 }, r.ids);
    // un=[0,2], ##aff=[2,5], ##able=[5,9]
    try testing.expectEqual(@as(u32, 0), r.offsets[0][0]);
    try testing.expectEqual(@as(u32, 2), r.offsets[0][1]);
    try testing.expectEqual(@as(u32, 2), r.offsets[1][0]);
    try testing.expectEqual(@as(u32, 5), r.offsets[1][1]);
    try testing.expectEqual(@as(u32, 5), r.offsets[2][0]);
    try testing.expectEqual(@as(u32, 9), r.offsets[2][1]);
}

test "WordPiece — encode + decode round-trip on split word" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "[UNK]", .id = 0 },
            .{ .token = "un", .id = 1 },
            .{ .token = "##aff", .id = 2 },
            .{ .token = "##able", .id = 3 },
        },
        0,
        "##",
        100,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "unaffable");
    defer testing.allocator.free(ids);
    const back = try m.decode(testing.allocator, ids);
    defer testing.allocator.free(back);
    try testing.expectEqualStrings("unaffable", back);
}
