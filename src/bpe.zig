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

/// Encoded output with byte-offset ranges per token. Used by
/// offset-tracking callers (e.g. `Tokenizer.encodeWithOffsets`).
pub const EncodedWithOffsets = struct {
    ids: []u32,
    offsets: [][2]u32,
};

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
    /// SentencePiece byte-fallback (tokenizer.json `model.byte_fallback`,
    /// used by Llama/Mistral-family BPE, distinct from GPT-2 ByteLevel
    /// pre-tokenization). When true, a codepoint absent from the vocab
    /// is decomposed into its UTF-8 bytes and each byte is looked up as
    /// a `<0xXX>` vocab entry rather than collapsing to `unk_token`.
    byte_fallback: bool = false,

    pub fn deinit(self: *Model) void {
        self.vocab.deinit();
        self.merges.deinit();
        self.arena.deinit();
        self.allocator.destroy(self.arena);
    }

    /// Look up the vocab id for the `<0xXX>` byte-fallback token
    /// representing `byte`, if present.
    fn byteTokenId(self: *const Model, byte: u8) ?u32 {
        var buf: [6]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{byte}) catch return null;
        return self.vocab.get(s);
    }

    /// Encode a ByteLevel-encoded string into token IDs. `word` is
    /// assumed to be a single "word" (no whitespace splitting here;
    /// the caller is responsible for pre-tokenization).
    /// Returns owned slice; caller frees.
    ///
    /// v0.2 algorithm: priority-queue + doubly-linked list, O(n log n)
    /// per word. See bpeEncodePQ below for the heap-based merge loop.
    pub fn encode(self: *const Model, alloc: std.mem.Allocator, word: []const u8) ![]u32 {
        if (word.len == 0) return alloc.alloc(u32, 0);
        return bpeEncodePQ(self, alloc, word);
    }

    /// Like encode but also returns parallel `[][2]u32 offsets` giving
    /// the byte range of each output token in the BPE input (`word`).
    /// Used by higher-level offset-tracking code; for ByteLevel-encoded
    /// inputs, callers map these offsets through the ByteLevel
    /// `input_index` table to recover original-text byte ranges.
    pub fn encodeWithOffsets(
        self: *const Model,
        alloc: std.mem.Allocator,
        word: []const u8,
    ) !EncodedWithOffsets {
        if (word.len == 0) return .{
            .ids = try alloc.alloc(u32, 0),
            .offsets = try alloc.alloc([2]u32, 0),
        };
        return bpeEncodePQWithOffsets(self, alloc, word);
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
                if (std.mem.startsWith(u8, s, "▁")) {
                    try out.append(alloc, ' ');
                    try out.appendSlice(alloc, s[3..]);
                } else {
                    try out.appendSlice(alloc, s);
                }
            } else {
                return Error.UnknownToken;
            }
        }
        return out.toOwnedSlice(alloc);
    }
};

// ============================================================================
// Priority-queue BPE encode (v0.2).
//
// The v0.1 O(n²) loop scanned all adjacent pairs each iteration. The
// v0.2 algorithm maintains a doubly-linked list of pieces plus a
// min-heap of (rank, left_node_index) candidates:
//
//   1. Split input into ByteLevel codepoint slices, link them into a DLL
//   2. For each adjacent pair, look up the merge rank; if mergeable,
//      push (rank, left_node_idx, snapshot_of_left_piece) onto heap
//   3. Pop best candidate. Skip if:
//        a) The node has been merged away (next pointer invalid)
//        b) The node's current piece doesn't match the snapshot
//           (already absorbed into a later merge)
//   4. Otherwise: combine left.piece + right.piece into a fresh arena
//      slice. Unlink right from DLL. Re-check new adjacencies (left's
//      prev with left, left with left's new next) and push fresh
//      candidates.
//   5. Walk the DLL → output token IDs via vocab lookup.
//
// Stale-entry detection via piece-snapshot is the canonical HF
// huggingface/tokenizers approach. We avoid the version-counter scheme
// because piece contents are immutable arena slices and pointer
// comparison is O(1).
// ============================================================================

const Node = struct {
    piece: []const u8,
    prev: i32, // -1 = head
    next: i32, // -1 = tail
    /// Set when this node is a SentencePiece byte-fallback byte (`piece`
    /// is a raw single byte, not a vocab-lookupable string). `bf_byte`
    /// is the byte to look up via `<0xXX>` at output time.
    is_bf: bool = false,
    bf_byte: u8 = 0,
};

const HeapEntry = struct {
    rank: u32,
    left_idx: u32,
    left_piece_ptr: [*]const u8, // identity snapshot
    left_piece_len: u32,
};

fn heapLess(_: void, a: HeapEntry, b: HeapEntry) std.math.Order {
    if (a.rank != b.rank) return std.math.order(a.rank, b.rank);
    // Tie-break by left_idx for determinism
    return std.math.order(a.left_idx, b.left_idx);
}

fn bpeEncodePQ(self: *const Model, alloc: std.mem.Allocator, word: []const u8) ![]u32 {
    // Step 1: tokenize input into per-codepoint slices, build DLL.
    var nodes: std.ArrayList(Node) = .empty;
    defer nodes.deinit(alloc);

    var i: usize = 0;
    while (i < word.len) {
        const n = try std.unicode.utf8ByteSequenceLength(word[i]);
        if (i + n > word.len) return error.InvalidUtf8;
        const cp = word[i .. i + n];
        if (self.byte_fallback and self.vocab.get(cp) == null) {
            // Byte-fallback: this codepoint isn't a vocab entry on its
            // own — decompose into per-byte nodes so each byte can be
            // looked up as `<0xXX>` at output time (matches HF Rust,
            // which never collapses OOV UTF-8 to a single unk when
            // byte_fallback is enabled).
            for (cp) |b| {
                const node_idx: i32 = @intCast(nodes.items.len);
                try nodes.append(alloc, .{
                    .piece = cp[0..1], // never matches a merge key; harmless
                    .prev = node_idx - 1,
                    .next = node_idx + 1,
                    .is_bf = true,
                    .bf_byte = b,
                });
            }
        } else {
            const node_idx: i32 = @intCast(nodes.items.len);
            try nodes.append(alloc, .{
                .piece = cp,
                .prev = node_idx - 1,
                .next = node_idx + 1, // patched below
            });
        }
        i += n;
    }
    if (nodes.items.len == 0) return alloc.alloc(u32, 0);
    nodes.items[nodes.items.len - 1].next = -1;

    // Step 2: initial heap population.
    var heap: std.PriorityQueue(HeapEntry, void, heapLess) = .empty;
    defer heap.deinit(alloc);

    // Helper: try to push a merge candidate for (left_idx, left.next).
    var key_buf: std.ArrayList(u8) = .empty;
    defer key_buf.deinit(alloc);

    const tryPushMerge = struct {
        fn run(
            allocator: std.mem.Allocator,
            ns: []Node,
            h: *std.PriorityQueue(HeapEntry, void, heapLess),
            merges: *const std.StringHashMap(u32),
            kb: *std.ArrayList(u8),
            left_idx: i32,
        ) !void {
            if (left_idx < 0) return;
            const left = ns[@intCast(left_idx)];
            if (left.next < 0) return;
            const right = ns[@intCast(left.next)];
            kb.clearRetainingCapacity();
            try kb.appendSlice(allocator, left.piece);
            try kb.append(allocator, ' ');
            try kb.appendSlice(allocator, right.piece);
            if (merges.get(kb.items)) |rank| {
                try h.push(allocator, .{
                    .rank = rank,
                    .left_idx = @intCast(left_idx),
                    .left_piece_ptr = left.piece.ptr,
                    .left_piece_len = @intCast(left.piece.len),
                });
            }
        }
    }.run;

    var k: usize = 0;
    while (k < nodes.items.len) : (k += 1) {
        try tryPushMerge(alloc, nodes.items, &heap, &self.merges, &key_buf, @intCast(k));
    }

    // Step 3: process heap.
    while (heap.pop()) |entry| {
        const left_idx_usize: usize = @intCast(entry.left_idx);
        const left = &nodes.items[left_idx_usize];

        // Stale check (dead node): a node absorbed as a "right" in a
        // previous merge gets its prev set to -2 sentinel. Skip if dead.
        if (left.prev == -2) continue;

        // Stale check (left side): identity snapshot must match.
        if (left.piece.ptr != entry.left_piece_ptr or
            left.piece.len != entry.left_piece_len) continue;
        if (left.next < 0) continue;

        const right_idx_usize: usize = @intCast(left.next);
        const right = &nodes.items[right_idx_usize];

        // Stale check (right side): verify the merge "left.piece
        // SPACE right.piece" still has the same rank. If the right
        // neighbor was absorbed into a downstream merge, the key
        // changes and either resolves to a different rank or no rank.
        key_buf.clearRetainingCapacity();
        try key_buf.appendSlice(alloc, left.piece);
        try key_buf.append(alloc, ' ');
        try key_buf.appendSlice(alloc, right.piece);
        const current_rank = self.merges.get(key_buf.items) orelse continue;
        if (current_rank != entry.rank) continue;

        // Merge: allocate combined piece in arena, splice DLL.
        const aa = self.arena.allocator();
        const merged = try aa.alloc(u8, left.piece.len + right.piece.len);
        @memcpy(merged[0 .. left.piece.len], left.piece);
        @memcpy(merged[left.piece.len ..], right.piece);
        left.piece = merged;
        left.next = right.next;
        if (right.next >= 0) {
            nodes.items[@intCast(right.next)].prev = @intCast(entry.left_idx);
        }
        // Mark right as dead so any pending heap entries pointing at it
        // are detected and skipped by the dead-node check above.
        right.prev = -2;
        right.next = -2;

        // Re-check new adjacencies
        try tryPushMerge(alloc, nodes.items, &heap, &self.merges, &key_buf, left.prev);
        try tryPushMerge(alloc, nodes.items, &heap, &self.merges, &key_buf, @intCast(entry.left_idx));
    }

    // Step 4: walk DLL → token IDs.
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(alloc);

    // Find head
    var cur_idx: i32 = 0;
    while (cur_idx >= 0 and nodes.items[@intCast(cur_idx)].prev >= 0) {
        cur_idx = nodes.items[@intCast(cur_idx)].prev;
    }
    // Walk forward
    while (cur_idx >= 0) {
        const node = nodes.items[@intCast(cur_idx)];
        if (node.is_bf) {
            if (self.byteTokenId(node.bf_byte)) |id| {
                try out.append(alloc, id);
            } else if (self.unk_token) |unk| {
                try out.append(alloc, unk);
            } else {
                return Error.UnknownToken;
            }
        } else if (self.vocab.get(node.piece)) |id| {
            try out.append(alloc, id);
        } else if (self.unk_token) |unk| {
            try out.append(alloc, unk);
        } else {
            return Error.UnknownToken;
        }
        cur_idx = node.next;
    }

    return out.toOwnedSlice(alloc);
}

// ============================================================================
// Offset-tracking variant — same algorithm with per-node (start, end) byte
// ranges in the BPE input space carried through merges.
// ============================================================================

const NodeO = struct {
    piece: []const u8,
    /// Byte range [start, end) of this piece in the BPE input `word`.
    start: u32,
    end: u32,
    prev: i32,
    next: i32,
    /// See Node.is_bf / Node.bf_byte.
    is_bf: bool = false,
    bf_byte: u8 = 0,
};

fn bpeEncodePQWithOffsets(
    self: *const Model,
    alloc: std.mem.Allocator,
    word: []const u8,
) !EncodedWithOffsets {
    var nodes: std.ArrayList(NodeO) = .empty;
    defer nodes.deinit(alloc);

    var i: usize = 0;
    while (i < word.len) {
        const n = try std.unicode.utf8ByteSequenceLength(word[i]);
        if (i + n > word.len) return error.InvalidUtf8;
        const cp = word[i .. i + n];
        if (self.byte_fallback and self.vocab.get(cp) == null) {
            // Byte-fallback expansion (see bpeEncodePQ). Each byte gets
            // the *same* enclosing-codepoint offset span, matching
            // SentencePiece convention — the original text has no
            // sub-byte position of its own.
            for (cp) |b| {
                const node_idx: i32 = @intCast(nodes.items.len);
                try nodes.append(alloc, .{
                    .piece = cp[0..1],
                    .start = @intCast(i),
                    .end = @intCast(i + n),
                    .prev = node_idx - 1,
                    .next = node_idx + 1,
                    .is_bf = true,
                    .bf_byte = b,
                });
            }
        } else {
            const node_idx: i32 = @intCast(nodes.items.len);
            try nodes.append(alloc, .{
                .piece = cp,
                .start = @intCast(i),
                .end = @intCast(i + n),
                .prev = node_idx - 1,
                .next = node_idx + 1,
            });
        }
        i += n;
    }
    if (nodes.items.len == 0) return .{
        .ids = try alloc.alloc(u32, 0),
        .offsets = try alloc.alloc([2]u32, 0),
    };
    nodes.items[nodes.items.len - 1].next = -1;

    var heap: std.PriorityQueue(HeapEntry, void, heapLess) = .empty;
    defer heap.deinit(alloc);

    var key_buf: std.ArrayList(u8) = .empty;
    defer key_buf.deinit(alloc);

    const tryPushMergeO = struct {
        fn run(
            allocator: std.mem.Allocator,
            ns: []NodeO,
            h: *std.PriorityQueue(HeapEntry, void, heapLess),
            merges: *const std.StringHashMap(u32),
            kb: *std.ArrayList(u8),
            left_idx: i32,
        ) !void {
            if (left_idx < 0) return;
            const left = ns[@intCast(left_idx)];
            if (left.next < 0) return;
            const right = ns[@intCast(left.next)];
            kb.clearRetainingCapacity();
            try kb.appendSlice(allocator, left.piece);
            try kb.append(allocator, ' ');
            try kb.appendSlice(allocator, right.piece);
            if (merges.get(kb.items)) |rank| {
                try h.push(allocator, .{
                    .rank = rank,
                    .left_idx = @intCast(left_idx),
                    .left_piece_ptr = left.piece.ptr,
                    .left_piece_len = @intCast(left.piece.len),
                });
            }
        }
    }.run;

    var k: usize = 0;
    while (k < nodes.items.len) : (k += 1) {
        try tryPushMergeO(alloc, nodes.items, &heap, &self.merges, &key_buf, @intCast(k));
    }

    while (heap.pop()) |entry| {
        const left_idx_usize: usize = @intCast(entry.left_idx);
        const left = &nodes.items[left_idx_usize];
        if (left.prev == -2) continue;
        if (left.piece.ptr != entry.left_piece_ptr or
            left.piece.len != entry.left_piece_len) continue;
        if (left.next < 0) continue;

        const right_idx_usize: usize = @intCast(left.next);
        const right = &nodes.items[right_idx_usize];

        key_buf.clearRetainingCapacity();
        try key_buf.appendSlice(alloc, left.piece);
        try key_buf.append(alloc, ' ');
        try key_buf.appendSlice(alloc, right.piece);
        const current_rank = self.merges.get(key_buf.items) orelse continue;
        if (current_rank != entry.rank) continue;

        const aa = self.arena.allocator();
        const merged = try aa.alloc(u8, left.piece.len + right.piece.len);
        @memcpy(merged[0 .. left.piece.len], left.piece);
        @memcpy(merged[left.piece.len ..], right.piece);
        left.piece = merged;
        left.end = right.end; // merged range absorbs right's end
        left.next = right.next;
        if (right.next >= 0) {
            nodes.items[@intCast(right.next)].prev = @intCast(entry.left_idx);
        }
        right.prev = -2;
        right.next = -2;

        try tryPushMergeO(alloc, nodes.items, &heap, &self.merges, &key_buf, left.prev);
        try tryPushMergeO(alloc, nodes.items, &heap, &self.merges, &key_buf, @intCast(entry.left_idx));
    }

    var ids_out: std.ArrayList(u32) = .empty;
    errdefer ids_out.deinit(alloc);
    var off_out: std.ArrayList([2]u32) = .empty;
    errdefer off_out.deinit(alloc);

    var cur_idx: i32 = 0;
    while (cur_idx >= 0 and nodes.items[@intCast(cur_idx)].prev >= 0) {
        cur_idx = nodes.items[@intCast(cur_idx)].prev;
    }
    while (cur_idx >= 0) {
        const node = nodes.items[@intCast(cur_idx)];
        const id = if (node.is_bf)
            self.byteTokenId(node.bf_byte) orelse (self.unk_token orelse return Error.UnknownToken)
        else
            self.vocab.get(node.piece) orelse (self.unk_token orelse return Error.UnknownToken);
        try ids_out.append(alloc, id);
        try off_out.append(alloc, .{ node.start, node.end });
        cur_idx = node.next;
    }

    return .{
        .ids = try ids_out.toOwnedSlice(alloc),
        .offsets = try off_out.toOwnedSlice(alloc),
    };
}

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

test "BPE encodeWithOffsets — single chars have unit ranges" {
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

    const r = try model.encodeWithOffsets(testing.allocator, "abc");
    defer testing.allocator.free(r.ids);
    defer testing.allocator.free(r.offsets);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, r.ids);
    try testing.expectEqual(@as(u32, 0), r.offsets[0][0]);
    try testing.expectEqual(@as(u32, 1), r.offsets[0][1]);
    try testing.expectEqual(@as(u32, 1), r.offsets[1][0]);
    try testing.expectEqual(@as(u32, 2), r.offsets[1][1]);
    try testing.expectEqual(@as(u32, 2), r.offsets[2][0]);
    try testing.expectEqual(@as(u32, 3), r.offsets[2][1]);
}

test "BPE encodeWithOffsets — merged tokens span their input range" {
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

    const r = try model.encodeWithOffsets(testing.allocator, "hi");
    defer testing.allocator.free(r.ids);
    defer testing.allocator.free(r.offsets);
    try testing.expectEqualSlices(u32, &[_]u32{2}, r.ids);
    try testing.expectEqual(@as(u32, 0), r.offsets[0][0]);
    try testing.expectEqual(@as(u32, 2), r.offsets[0][1]);
}

test "BPE encodeWithOffsets — three-piece chain offsets" {
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
            .{ .left = "h", .right = "e", .rank = 0 },
            .{ .left = "l", .right = "l", .rank = 1 },
            .{ .left = "he", .right = "ll", .rank = 2 },
        },
        null,
    );
    defer model.deinit();

    const r = try model.encodeWithOffsets(testing.allocator, "hell");
    defer testing.allocator.free(r.ids);
    defer testing.allocator.free(r.offsets);
    try testing.expectEqualSlices(u32, &[_]u32{6}, r.ids);
    try testing.expectEqual(@as(u32, 0), r.offsets[0][0]);
    try testing.expectEqual(@as(u32, 4), r.offsets[0][1]);
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
