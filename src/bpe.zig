//! Byte-Pair Encoding (BPE) model.
//!
//! Given a vocabulary (token-string → token-id) and an ordered list of
//! merges (pairs of token-strings that combine into a new token), this
//! encodes an input ByteLevel-pre-tokenized string into a sequence of
//! token IDs.
//!
//! Algorithm (HF-compatible greedy lowest-rank merge):
//!   1. Split input into ByteLevel "characters" (each is one UTF-8
//!      codepoint from the ByteLevel domain).
//!   2. While there exists an adjacent pair in current sequence whose
//!      merge rank is the lowest (highest priority): apply that merge.
//!   3. Convert remaining tokens to IDs via vocab lookup.
//!
//! This is the canonical BPE merging algorithm. Performance is
//! intentionally simple O(n^2) per word at v0.0.1; v0.1 will add
//! priority-queue-based O(n log n) merging.

const std = @import("std");

pub const Error = error{
    UnknownToken,
    EmptyMerges,
    OutOfMemory,
};

pub const VocabEntry = struct { token: []const u8, id: u32 };
pub const MergeEntry = struct { left: []const u8, right: []const u8, rank: u32 };

pub const Model = struct {
    allocator: std.mem.Allocator,
    /// Owned storage for all vocab + merge strings. Strings inside
    /// vocab / merges fields are slices into this arena.
    arena: *std.heap.ArenaAllocator,
    /// token-string → token-id
    vocab: std.StringHashMap(u32),
    /// (left, right) → rank. Lower rank = higher priority.
    merges: std.StringHashMap(u32),
    /// unk_token: optional fallback. If null, UnknownToken error.
    unk_token: ?u32,

    pub fn deinit(self: *Model) void {
        self.vocab.deinit();
        self.merges.deinit();
        self.arena.deinit();
        self.allocator.destroy(self.arena);
    }

    /// Encode a ByteLevel-encoded string into token IDs. `word` is
    /// assumed to be a single "word" (no whitespace splitting here;
    /// the caller is responsible for pre-tokenization).
    /// Returns owned slice; caller frees.
    pub fn encode(self: *const Model, alloc: std.mem.Allocator, word: []const u8) ![]u32 {
        if (word.len == 0) return alloc.alloc(u32, 0);

        // Step 1: split into ByteLevel "characters" — each is one UTF-8
        // codepoint. We track byte ranges into `word` so we can slice
        // adjacent pairs without allocating.
        var pieces: std.ArrayList([]const u8) = .empty;
        defer pieces.deinit(alloc);

        var i: usize = 0;
        while (i < word.len) {
            const n = try std.unicode.utf8ByteSequenceLength(word[i]);
            if (i + n > word.len) return error.InvalidUtf8;
            try pieces.append(alloc, word[i .. i + n]);
            i += n;
        }

        // Step 2: greedy lowest-rank merge.
        var merge_key_buf: std.ArrayList(u8) = .empty;
        defer merge_key_buf.deinit(alloc);

        while (pieces.items.len > 1) {
            var best_rank: ?u32 = null;
            var best_idx: usize = 0;

            var k: usize = 0;
            while (k + 1 < pieces.items.len) : (k += 1) {
                merge_key_buf.clearRetainingCapacity();
                try merge_key_buf.appendSlice(alloc, pieces.items[k]);
                try merge_key_buf.append(alloc, ' ');
                try merge_key_buf.appendSlice(alloc, pieces.items[k + 1]);

                if (self.merges.get(merge_key_buf.items)) |rank| {
                    if (best_rank == null or rank < best_rank.?) {
                        best_rank = rank;
                        best_idx = k;
                    }
                }
            }

            if (best_rank == null) break;

            // Merge pieces[best_idx] and pieces[best_idx+1] into
            // pieces[best_idx]. We need a new string covering both,
            // allocated in the arena.
            const a = pieces.items[best_idx];
            const b = pieces.items[best_idx + 1];
            const merged = try self.arena.allocator().alloc(u8, a.len + b.len);
            @memcpy(merged[0..a.len], a);
            @memcpy(merged[a.len..], b);
            pieces.items[best_idx] = merged;
            _ = pieces.orderedRemove(best_idx + 1);
        }

        // Step 3: vocab lookup.
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(alloc);
        for (pieces.items) |p| {
            if (self.vocab.get(p)) |id| {
                try out.append(alloc, id);
            } else if (self.unk_token) |unk| {
                try out.append(alloc, unk);
            } else {
                return Error.UnknownToken;
            }
        }
        return out.toOwnedSlice(alloc);
    }

    /// Decode a sequence of token IDs back to a ByteLevel-encoded string.
    /// Caller frees the result.
    pub fn decode(self: *const Model, alloc: std.mem.Allocator, ids: []const u32) ![]u8 {
        // Reverse lookup: build a small id→token map by iterating vocab.
        // For v0.0.1 we accept the O(N) cost; v0.1 caches an id-indexed array.
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(alloc);

        for (ids) |id| {
            var found: ?[]const u8 = null;
            var it = self.vocab.iterator();
            while (it.next()) |e| {
                if (e.value_ptr.* == id) {
                    found = e.key_ptr.*;
                    break;
                }
            }
            if (found) |s| {
                try out.appendSlice(alloc, s);
            } else {
                return Error.UnknownToken;
            }
        }
        return out.toOwnedSlice(alloc);
    }
};

/// Build a Model from a vocab + merges list. Both are copied into an
/// internal arena; caller's slices may be freed after init returns.
pub fn build(
    alloc: std.mem.Allocator,
    vocab_entries: []const VocabEntry,
    merges_entries: []const MergeEntry,
    unk_token: ?u32,
) !Model {
    const arena = try alloc.create(std.heap.ArenaAllocator);
    errdefer alloc.destroy(arena);
    arena.* = std.heap.ArenaAllocator.init(alloc);
    errdefer arena.deinit();
    const aa = arena.allocator();

    var vocab = std.StringHashMap(u32).init(alloc);
    errdefer vocab.deinit();
    for (vocab_entries) |e| {
        const tok = try aa.dupe(u8, e.token);
        try vocab.put(tok, e.id);
    }

    var merges = std.StringHashMap(u32).init(alloc);
    errdefer merges.deinit();
    for (merges_entries) |e| {
        const key = try std.fmt.allocPrint(aa, "{s} {s}", .{ e.left, e.right });
        try merges.put(key, e.rank);
    }

    return .{
        .allocator = alloc,
        .arena = arena,
        .vocab = vocab,
        .merges = merges,
        .unk_token = unk_token,
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "BPE — single character vocab returns id sequence" {
    var model = try build(
        testing.allocator,
        &.{
            .{ .token = "h", .id = 0 },
            .{ .token = "i", .id = 1 },
        },
        &.{},
        null,
    );
    defer model.deinit();

    const ids = try model.encode(testing.allocator, "hi");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1 }, ids);
}

test "BPE — single merge applies" {
    var model = try build(
        testing.allocator,
        &.{
            .{ .token = "h", .id = 0 },
            .{ .token = "i", .id = 1 },
            .{ .token = "hi", .id = 2 },
        },
        &.{.{ .left = "h", .right = "i", .rank = 0 }},
        null,
    );
    defer model.deinit();

    const ids = try model.encode(testing.allocator, "hi");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{2}, ids);
}

test "BPE — lowest rank merge wins" {
    // "abc": both "a b" and "b c" are merges, but "b c" has lower rank
    // so it merges first. Then "a" + "bc" needs an "a bc" merge to
    // continue; if not, we get [a, bc].
    var model = try build(
        testing.allocator,
        &.{
            .{ .token = "a", .id = 0 },
            .{ .token = "b", .id = 1 },
            .{ .token = "c", .id = 2 },
            .{ .token = "ab", .id = 3 },
            .{ .token = "bc", .id = 4 },
        },
        &.{
            .{ .left = "a", .right = "b", .rank = 10 },
            .{ .left = "b", .right = "c", .rank = 0 },
        },
        null,
    );
    defer model.deinit();

    const ids = try model.encode(testing.allocator, "abc");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 4 }, ids);
}

test "BPE — unknown token without unk_token errors" {
    var model = try build(
        testing.allocator,
        &.{.{ .token = "a", .id = 0 }},
        &.{},
        null,
    );
    defer model.deinit();
    try testing.expectError(
        Error.UnknownToken,
        model.encode(testing.allocator, "b"),
    );
}

test "BPE — unknown token with unk_token returns unk id" {
    var model = try build(
        testing.allocator,
        &.{
            .{ .token = "a", .id = 0 },
            .{ .token = "<unk>", .id = 99 },
        },
        &.{},
        99,
    );
    defer model.deinit();
    const ids = try model.encode(testing.allocator, "ab");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 99 }, ids);
}

test "BPE — encode + decode round-trip with single chars" {
    var model = try build(
        testing.allocator,
        &.{
            .{ .token = "a", .id = 0 },
            .{ .token = "b", .id = 1 },
            .{ .token = "c", .id = 2 },
        },
        &.{},
        null,
    );
    defer model.deinit();
    const ids = try model.encode(testing.allocator, "abc");
    defer testing.allocator.free(ids);
    const back = try model.decode(testing.allocator, ids);
    defer testing.allocator.free(back);
    try testing.expectEqualStrings("abc", back);
}

test "BPE — empty input returns empty ids" {
    var model = try build(
        testing.allocator,
        &.{.{ .token = "a", .id = 0 }},
        &.{},
        null,
    );
    defer model.deinit();
    const ids = try model.encode(testing.allocator, "");
    defer testing.allocator.free(ids);
    try testing.expectEqual(@as(usize, 0), ids.len);
}

test "BPE — chained merges (3 pieces → 1)" {
    var model = try build(
        testing.allocator,
        &.{
            .{ .token = "h", .id = 0 },
            .{ .token = "e", .id = 1 },
            .{ .token = "l", .id = 2 },
            .{ .token = "he", .id = 3 },
            .{ .token = "hel", .id = 4 },
            .{ .token = "ll", .id = 5 },
            .{ .token = "hell", .id = 6 },
        },
        &.{
            .{ .left = "h", .right = "e", .rank = 0 },  // best
            .{ .left = "l", .right = "l", .rank = 1 },
            .{ .left = "he", .right = "ll", .rank = 2 },
        },
        null,
    );
    defer model.deinit();

    const ids = try model.encode(testing.allocator, "hell");
    defer testing.allocator.free(ids);
    // h+e merges first → he, l+l merges → ll, he+ll merges → hell
    try testing.expectEqualSlices(u32, &[_]u32{6}, ids);
}
