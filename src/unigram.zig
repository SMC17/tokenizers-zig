//! Unigram language-model tokenizer (T5 / mBART / XLM-R family,
//! SentencePiece-style).
//!
//! The vocab is a flat list of (token-string, log-probability) entries
//! indexed by token id. Encoding finds the segmentation of the input
//! that maximises the sum of log-probabilities over all chosen pieces,
//! computed via Viterbi dynamic programming on codepoint boundaries.
//!
//! Algorithm:
//!   1. Compute codepoint boundary positions b[0..N+1] over the input.
//!   2. Initialise best[0] = 0.0, best[i>0] = -inf.
//!   3. For each end position i in 1..N+1:
//!        For each start position j in 0..i:
//!          substr = input[b[j]..b[i]]
//!          score  = vocab[substr].score    (if substr ∈ vocab)
//!                   unk_score              (if i-j == 1, single codepoint)
//!                   skip                   (otherwise)
//!          cand = best[j] + score
//!          if cand > best[i]: best[i] = cand; back[i] = (j, id)
//!   4. Trace back from N via back[] to recover segments + ids.
//!
//! Reference:
//!   Kudo, *Subword Regularization* (2018).
//!   https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/models/unigram/

const std = @import("std");

pub const Error = error{
    OutOfMemory,
    InvalidUtf8,
    EmptyVocab,
    UnknownToken,
};

pub const VocabEntry = struct {
    token: []const u8,
    score: f64,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    /// Owned storage for all vocab token strings.
    arena: *std.heap.ArenaAllocator,
    /// vocab indexed by token id. token_strings[i] == vocab_tokens[i].token.
    /// score[i] == vocab[i].score.
    tokens: [][]const u8,
    scores: []f64,
    /// token-string → token-id, populated for all entries.
    str_to_id: std.StringHashMap(u32),
    /// Unknown-token id. Single-codepoint fallback when nothing else matches.
    unk_id: u32,
    /// Score assigned to unk fallback during Viterbi. Conventionally a
    /// large negative number so unk is dispreferred when alternatives exist.
    /// Default = min(scores) - 10.0, set at build time.
    unk_score: f64,

    pub fn deinit(self: *Model) void {
        self.str_to_id.deinit();
        self.allocator.free(self.tokens);
        self.allocator.free(self.scores);
        self.arena.deinit();
        self.allocator.destroy(self.arena);
    }

    /// Encode + return per-token byte-offset ranges (in `text`'s byte
    /// space). Viterbi reconstructs both the id sequence and the byte
    /// boundaries of each segment. Caller frees both slices.
    pub fn encodeWithOffsets(
        self: *const Model,
        alloc: std.mem.Allocator,
        text: []const u8,
    ) !struct { ids: []u32, offsets: [][2]u32 } {
        if (text.len == 0) return .{
            .ids = try alloc.alloc(u32, 0),
            .offsets = try alloc.alloc([2]u32, 0),
        };

        var boundaries: std.ArrayList(usize) = .empty;
        defer boundaries.deinit(alloc);
        try boundaries.append(alloc, 0);
        var i: usize = 0;
        while (i < text.len) {
            const n = try std.unicode.utf8ByteSequenceLength(text[i]);
            if (i + n > text.len) return Error.InvalidUtf8;
            i += n;
            try boundaries.append(alloc, i);
        }
        const n_chars = boundaries.items.len - 1;

        const NEG_INF = -std.math.inf(f64);
        const best = try alloc.alloc(f64, n_chars + 1);
        defer alloc.free(best);
        const back_prev = try alloc.alloc(usize, n_chars + 1);
        defer alloc.free(back_prev);
        const back_id = try alloc.alloc(u32, n_chars + 1);
        defer alloc.free(back_id);

        @memset(best, NEG_INF);
        @memset(back_prev, 0);
        @memset(back_id, 0);
        best[0] = 0.0;

        var end_pos: usize = 1;
        while (end_pos <= n_chars) : (end_pos += 1) {
            const b_end = boundaries.items[end_pos];
            var start_pos: usize = 0;
            while (start_pos < end_pos) : (start_pos += 1) {
                if (best[start_pos] == NEG_INF) continue;
                const b_start = boundaries.items[start_pos];
                const substr = text[b_start..b_end];
                var step_score: f64 = undefined;
                var step_id: u32 = undefined;
                if (self.str_to_id.get(substr)) |id| {
                    step_score = self.scores[id];
                    step_id = id;
                } else if (end_pos - start_pos == 1) {
                    step_score = self.unk_score;
                    step_id = self.unk_id;
                } else continue;
                const cand = best[start_pos] + step_score;
                if (cand > best[end_pos]) {
                    best[end_pos] = cand;
                    back_prev[end_pos] = start_pos;
                    back_id[end_pos] = step_id;
                }
            }
        }

        if (best[n_chars] == NEG_INF) {
            const ids_out = try alloc.alloc(u32, 1);
            ids_out[0] = self.unk_id;
            const off_out = try alloc.alloc([2]u32, 1);
            off_out[0] = .{ 0, @intCast(text.len) };
            return .{ .ids = ids_out, .offsets = off_out };
        }

        // Trace back; reverse to forward order with parallel offsets.
        var rev_ids: std.ArrayList(u32) = .empty;
        defer rev_ids.deinit(alloc);
        var rev_offs: std.ArrayList([2]u32) = .empty;
        defer rev_offs.deinit(alloc);
        var cur: usize = n_chars;
        while (cur > 0) {
            const j = back_prev[cur];
            try rev_ids.append(alloc, back_id[cur]);
            try rev_offs.append(alloc, .{
                @intCast(boundaries.items[j]),
                @intCast(boundaries.items[cur]),
            });
            cur = j;
        }

        const ids_out = try alloc.alloc(u32, rev_ids.items.len);
        const off_out = try alloc.alloc([2]u32, rev_offs.items.len);
        var k: usize = 0;
        while (k < rev_ids.items.len) : (k += 1) {
            ids_out[k] = rev_ids.items[rev_ids.items.len - 1 - k];
            off_out[k] = rev_offs.items[rev_offs.items.len - 1 - k];
        }
        return .{ .ids = ids_out, .offsets = off_out };
    }

    /// Encode an input string into token IDs by Viterbi-best segmentation.
    /// Caller frees the returned slice.
    pub fn encode(self: *const Model, alloc: std.mem.Allocator, text: []const u8) ![]u32 {
        if (text.len == 0) return alloc.alloc(u32, 0);

        // Codepoint boundaries: byte index of each codepoint, plus end.
        var boundaries: std.ArrayList(usize) = .empty;
        defer boundaries.deinit(alloc);
        try boundaries.append(alloc, 0);
        var i: usize = 0;
        while (i < text.len) {
            const n = try std.unicode.utf8ByteSequenceLength(text[i]);
            if (i + n > text.len) return Error.InvalidUtf8;
            i += n;
            try boundaries.append(alloc, i);
        }
        const n_chars = boundaries.items.len - 1;

        // Viterbi tables.
        const NEG_INF = -std.math.inf(f64);
        const best = try alloc.alloc(f64, n_chars + 1);
        defer alloc.free(best);
        // back_prev[i] = j such that best path to i comes from j
        // back_id[i] = id of the piece input[b[j]..b[i]]
        const back_prev = try alloc.alloc(usize, n_chars + 1);
        defer alloc.free(back_prev);
        const back_id = try alloc.alloc(u32, n_chars + 1);
        defer alloc.free(back_id);

        @memset(best, NEG_INF);
        @memset(back_prev, 0);
        @memset(back_id, 0);
        best[0] = 0.0;

        var end_pos: usize = 1;
        while (end_pos <= n_chars) : (end_pos += 1) {
            const b_end = boundaries.items[end_pos];
            var start_pos: usize = 0;
            while (start_pos < end_pos) : (start_pos += 1) {
                if (best[start_pos] == NEG_INF) continue;
                const b_start = boundaries.items[start_pos];
                const substr = text[b_start..b_end];

                var step_score: f64 = undefined;
                var step_id: u32 = undefined;
                if (self.str_to_id.get(substr)) |id| {
                    step_score = self.scores[id];
                    step_id = id;
                } else if (end_pos - start_pos == 1) {
                    // Single-codepoint unknown — fall back to unk.
                    step_score = self.unk_score;
                    step_id = self.unk_id;
                } else {
                    continue;
                }

                const cand = best[start_pos] + step_score;
                if (cand > best[end_pos]) {
                    best[end_pos] = cand;
                    back_prev[end_pos] = start_pos;
                    back_id[end_pos] = step_id;
                }
            }
        }

        if (best[n_chars] == NEG_INF) {
            // No segmentation found at all — pathological vocab. Emit one unk.
            const out = try alloc.alloc(u32, 1);
            out[0] = self.unk_id;
            return out;
        }

        // Trace back to recover ids in reverse, then reverse to forward order.
        var rev: std.ArrayList(u32) = .empty;
        defer rev.deinit(alloc);
        var cur: usize = n_chars;
        while (cur > 0) {
            try rev.append(alloc, back_id[cur]);
            cur = back_prev[cur];
        }

        const out = try alloc.alloc(u32, rev.items.len);
        var k: usize = 0;
        while (k < rev.items.len) : (k += 1) {
            out[k] = rev.items[rev.items.len - 1 - k];
        }
        return out;
    }

    /// Decode token IDs by concatenating token strings.
    /// Caller frees.
    pub fn decode(self: *const Model, alloc: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(alloc);
        for (ids) |id| {
            if (id >= self.tokens.len) return Error.UnknownToken;
            try out.appendSlice(alloc, self.tokens[id]);
        }
        return out.toOwnedSlice(alloc);
    }
};

/// Build a Unigram Model from a vocab list. `unk_id` must be a valid
/// index into the vocab. `unk_score`, if null, defaults to
/// min(scores) - 10.0 so unk fallback is consistently dispreferred.
pub fn build(
    alloc: std.mem.Allocator,
    vocab_entries: []const VocabEntry,
    unk_id: u32,
    unk_score: ?f64,
) !Model {
    if (vocab_entries.len == 0) return Error.EmptyVocab;
    if (unk_id >= vocab_entries.len) return Error.UnknownToken;

    const arena = try alloc.create(std.heap.ArenaAllocator);
    errdefer alloc.destroy(arena);
    arena.* = std.heap.ArenaAllocator.init(alloc);
    errdefer arena.deinit();
    const aa = arena.allocator();

    const tokens = try alloc.alloc([]const u8, vocab_entries.len);
    errdefer alloc.free(tokens);
    const scores = try alloc.alloc(f64, vocab_entries.len);
    errdefer alloc.free(scores);

    var str_to_id = std.StringHashMap(u32).init(alloc);
    errdefer str_to_id.deinit();

    var min_score: f64 = std.math.inf(f64);
    for (vocab_entries, 0..) |e, idx| {
        const tok = try aa.dupe(u8, e.token);
        tokens[idx] = tok;
        scores[idx] = e.score;
        try str_to_id.put(tok, @intCast(idx));
        if (e.score < min_score) min_score = e.score;
    }

    const resolved_unk_score = unk_score orelse (min_score - 10.0);

    return .{
        .allocator = alloc,
        .arena = arena,
        .tokens = tokens,
        .scores = scores,
        .str_to_id = str_to_id,
        .unk_id = unk_id,
        .unk_score = resolved_unk_score,
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "Unigram — single in-vocab token encodes as itself" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "<unk>", .score = 0.0 },
            .{ .token = "hello", .score = -2.0 },
        },
        0,
        null,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "hello");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{1}, ids);
}

test "Unigram — Viterbi prefers higher-score (closer-to-zero) segmentation" {
    // "ab": vocab {a=-1, b=-1, ab=-1.5}. ab=-1.5 wins over a+b=-2.
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "<unk>", .score = -20.0 },
            .{ .token = "a", .score = -1.0 },
            .{ .token = "b", .score = -1.0 },
            .{ .token = "ab", .score = -1.5 },
        },
        0,
        null,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "ab");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{3}, ids);
}

test "Unigram — Viterbi splits when split-sum exceeds single token" {
    // "ab": vocab {a=-1, b=-1, ab=-3}. a+b=-2 wins over ab=-3.
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "<unk>", .score = -20.0 },
            .{ .token = "a", .score = -1.0 },
            .{ .token = "b", .score = -1.0 },
            .{ .token = "ab", .score = -3.0 },
        },
        0,
        null,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "ab");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2 }, ids);
}

test "Unigram — three-piece DP path" {
    // "abc": vocab {a=-1, b=-1, c=-1, ab=-1.4, bc=-1.4}.
    // Paths: a+b+c=-3, ab+c=-2.4, a+bc=-2.4. ab+c chosen on lex tie.
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "<unk>", .score = -20.0 },
            .{ .token = "a", .score = -1.0 },
            .{ .token = "b", .score = -1.0 },
            .{ .token = "c", .score = -1.0 },
            .{ .token = "ab", .score = -1.4 },
            .{ .token = "bc", .score = -1.4 },
        },
        0,
        null,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "abc");
    defer testing.allocator.free(ids);
    // ab + c = -2.4; a + bc = -2.4 (tie); first-cand-wins → ab + c
    // back_id[end] takes the FIRST start_pos that achieves the max.
    // For end=2 best path is ab (id=4) from start=0.
    // For end=3 candidates: best[2]+score(c)=-2.4-1.0=-3.4 (skip)... wait
    // Actually best[2] = max(best[0]+score(ab)=-1.4, best[1]+score(b)=-2.0) = -1.4
    // best[3] = max(best[2]+score(c)=-1.4-1.0=-2.4, best[1]+score(bc)=-1.0-1.4=-2.4, best[0]+score(abc)=-inf)
    // tie at -2.4 between ab+c and a+bc. Our impl picks first (strict >),
    // so first-encountered start_pos=1 (a+bc) reached -2.4 first;
    // start_pos=2 (ab+c) sees cand=-2.4 which is NOT > current best=-2.4 → keeps a+bc.
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 5 }, ids); // a (id=1) + bc (id=5)
}

test "Unigram — single-codepoint unknown falls back to unk" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "<unk>", .score = -20.0 },
            .{ .token = "a", .score = -1.0 },
        },
        0,
        null,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "ax");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 0 }, ids); // a + <unk>
}

test "Unigram — empty input returns empty ids" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "<unk>", .score = -20.0 },
            .{ .token = "a", .score = -1.0 },
        },
        0,
        null,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "");
    defer testing.allocator.free(ids);
    try testing.expectEqual(@as(usize, 0), ids.len);
}

test "Unigram — UTF-8 multibyte vocab tokens work" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "<unk>", .score = -20.0 },
            .{ .token = "▁", .score = -3.0 }, // U+2581, 3-byte UTF-8
            .{ .token = "hello", .score = -2.0 },
        },
        0,
        null,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "▁hello");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2 }, ids);
}

test "Unigram — decode concatenates token strings in order" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "<unk>", .score = -20.0 },
            .{ .token = "hel", .score = -2.0 },
            .{ .token = "lo", .score = -2.0 },
        },
        0,
        null,
    );
    defer m.deinit();

    const text = try m.decode(testing.allocator, &[_]u32{ 1, 2 });
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("hello", text);
}

test "Unigram encodeWithOffsets — Viterbi segments emit byte ranges" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "<unk>", .score = -20.0 },
            .{ .token = "a", .score = -1.0 },
            .{ .token = "b", .score = -1.0 },
            .{ .token = "ab", .score = -1.5 },
        },
        0,
        null,
    );
    defer m.deinit();

    const r = try m.encodeWithOffsets(testing.allocator, "ab");
    defer testing.allocator.free(r.ids);
    defer testing.allocator.free(r.offsets);
    // ab=-1.5 > a+b=-2.0; single segment [0, 2].
    try testing.expectEqualSlices(u32, &[_]u32{3}, r.ids);
    try testing.expectEqual(@as(u32, 0), r.offsets[0][0]);
    try testing.expectEqual(@as(u32, 2), r.offsets[0][1]);
}

test "Unigram encodeWithOffsets — multi-segment offsets" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "<unk>", .score = -20.0 },
            .{ .token = "a", .score = -1.0 },
            .{ .token = "b", .score = -1.0 },
            .{ .token = "c", .score = -1.0 },
            .{ .token = "ab", .score = -3.0 },
            .{ .token = "bc", .score = -3.0 },
        },
        0,
        null,
    );
    defer m.deinit();

    // a+b+c=-3 beats ab+c=-4 and a+bc=-4. Three segments.
    const r = try m.encodeWithOffsets(testing.allocator, "abc");
    defer testing.allocator.free(r.ids);
    defer testing.allocator.free(r.offsets);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3 }, r.ids);
    try testing.expectEqual(@as(u32, 0), r.offsets[0][0]);
    try testing.expectEqual(@as(u32, 1), r.offsets[0][1]);
    try testing.expectEqual(@as(u32, 1), r.offsets[1][0]);
    try testing.expectEqual(@as(u32, 2), r.offsets[1][1]);
    try testing.expectEqual(@as(u32, 2), r.offsets[2][0]);
    try testing.expectEqual(@as(u32, 3), r.offsets[2][1]);
}

test "Unigram — encode + decode round-trip on segmented word" {
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "<unk>", .score = -20.0 },
            .{ .token = "a", .score = -1.0 },
            .{ .token = "b", .score = -1.0 },
            .{ .token = "ab", .score = -3.0 },
        },
        0,
        null,
    );
    defer m.deinit();

    const ids = try m.encode(testing.allocator, "ab");
    defer testing.allocator.free(ids);
    const back = try m.decode(testing.allocator, ids);
    defer testing.allocator.free(back);
    try testing.expectEqualStrings("ab", back);
}

test "Unigram — explicit unk_score overrides default" {
    // With unk_score = -1.5 (higher than score of "a" = -1.0 means unk's prob
    // is lower → still dispreferred). Should still segment "a" correctly.
    var m = try build(
        testing.allocator,
        &.{
            .{ .token = "<unk>", .score = -20.0 },
            .{ .token = "a", .score = -1.0 },
        },
        0,
        -1.5,
    );
    defer m.deinit();

    try testing.expectEqual(@as(f64, -1.5), m.unk_score);
}

test "Unigram — empty vocab errors at build" {
    try testing.expectError(
        Error.EmptyVocab,
        build(testing.allocator, &.{}, 0, null),
    );
}

test "Unigram — out-of-range unk_id errors at build" {
    try testing.expectError(
        Error.UnknownToken,
        build(
            testing.allocator,
            &.{.{ .token = "a", .score = -1.0 }},
            5, // out of range
            null,
        ),
    );
}
