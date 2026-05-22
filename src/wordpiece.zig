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

const std = @import("std");

pub const Error = error{
    OutOfMemory,
    InvalidUtf8,
};

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

    /// Encode a single pre-tokenized "word" (already split on whitespace +
    /// punctuation by the caller) into token IDs. Caller frees.
    pub fn encode(self: *const Model, alloc: std.mem.Allocator, word: []const u8) ![]u32 {
        if (word.len == 0) return alloc.alloc(u32, 0);

        // Compute codepoint boundaries in byte-index space.
        // boundaries[k] = byte index where codepoint k starts;
        // boundaries[n_chars] = word.len.
        var boundaries: std.ArrayList(usize) = .empty;
        defer boundaries.deinit(alloc);
        try boundaries.append(alloc, 0);
        var i: usize = 0;
        while (i < word.len) {
            const n = try std.unicode.utf8ByteSequenceLength(word[i]);
            if (i + n > word.len) return Error.InvalidUtf8;
            i += n;
            try boundaries.append(alloc, i);
        }
        const n_chars: u32 = @intCast(boundaries.items.len - 1);

        // Word too long → emit single unk.
        if (n_chars > self.max_input_chars_per_word) {
            const out = try alloc.alloc(u32, 1);
            out[0] = self.unk_id;
            return out;
        }

        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(alloc);

        // Buffer for "##<substr>" key construction. Sized once.
        var key_buf: std.ArrayList(u8) = .empty;
        defer key_buf.deinit(alloc);

        var start: u32 = 0;
        while (start < n_chars) {
            var end: u32 = n_chars;
            var matched_id: ?u32 = null;
            var matched_end: u32 = 0;
            while (end > start) : (end -= 1) {
                const byte_start = boundaries.items[start];
                const byte_end = boundaries.items[end];
                const substr = word[byte_start..byte_end];

                const lookup: []const u8 = if (start == 0)
                    substr
                else blk: {
                    key_buf.clearRetainingCapacity();
                    try key_buf.appendSlice(alloc, self.continuing_subword_prefix);
                    try key_buf.appendSlice(alloc, substr);
                    break :blk key_buf.items;
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
