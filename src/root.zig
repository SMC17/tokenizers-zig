//! tokenizers-zig — Zig 0.16 BPE tokenizer compatible with the
//! HuggingFace `tokenizer.json` format (ByteLevel + BPE subset).
//!
//! v0.0.1 scope: ByteLevel pre-tokenization + BPE encode/decode for
//! the most common HF tokenizer.json shape used by GPT-2, Llama-3,
//! Qwen-2.5, Mistral families.
//!
//! What's covered:
//!   - ByteLevel pre-tokenizer (canonical GPT-2 byte → unicode map)
//!   - BPE model (vocab + merges, greedy lowest-rank merge)
//!   - HuggingFace tokenizer.json loader for the `BPE` model variant
//!     with `ByteLevel` pre-tokenizer
//!   - Encode (text → token IDs) and decode (token IDs → text)
//!
//! What's not yet covered (v0.1+):
//!   - WordPiece, Unigram, SentencePiece models
//!   - Whitespace / WhitespaceSplit / Regex pre-tokenizers
//!   - Truncation, padding, template post-processors
//!   - added_tokens / special tokens (skeleton present, not exercised)
//!   - Decoder configs beyond ByteLevel-inverse
//!   - Performance work (the encode is O(n^2) per word in v0.0.1)

const std = @import("std");

pub const byte_level = @import("byte_level.zig");
pub const bpe = @import("bpe.zig");

pub const Error = error{
    InvalidTokenizerJson,
    UnsupportedModelType,
    UnsupportedPreTokenizer,
    MissingField,
};

/// High-level tokenizer assembled from a HF tokenizer.json.
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    model: bpe.Model,
    /// True if pre-tokenizer is ByteLevel.
    use_byte_level: bool,

    pub fn deinit(self: *Tokenizer) void {
        self.model.deinit();
    }

    /// Encode a UTF-8 text string into token IDs. Caller frees the
    /// returned slice.
    pub fn encode(self: *Tokenizer, alloc: std.mem.Allocator, text: []const u8) ![]u32 {
        // For v0.0.1 we treat the entire input as a single "word" —
        // ByteLevel BPE in HF actually splits on whitespace+punct
        // boundaries via a regex first. We do the simpler thing and
        // let the ByteLevel mapping handle spaces (they become Ġ).
        if (self.use_byte_level) {
            const encoded = try byte_level.encode(alloc, text);
            defer alloc.free(encoded);
            return self.model.encode(alloc, encoded);
        } else {
            return self.model.encode(alloc, text);
        }
    }

    /// Decode token IDs back to UTF-8 text. Caller frees.
    pub fn decode(self: *Tokenizer, alloc: std.mem.Allocator, ids: []const u32) ![]u8 {
        const encoded = try self.model.decode(alloc, ids);
        defer alloc.free(encoded);
        if (self.use_byte_level) {
            return byte_level.decode(alloc, encoded);
        } else {
            return alloc.dupe(u8, encoded);
        }
    }
};

/// Build a Tokenizer from an in-memory tokenizer.json byte slice.
pub fn fromJson(alloc: std.mem.Allocator, json_bytes: []const u8) !Tokenizer {
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const parsed = std.json.parseFromSliceLeaky(std.json.Value, aa, json_bytes, .{
        .max_value_len = std.json.default_max_value_len,
    }) catch return Error.InvalidTokenizerJson;
    if (parsed != .object) return Error.InvalidTokenizerJson;
    const root = parsed.object;

    // Pre-tokenizer
    var use_byte_level = false;
    if (root.get("pre_tokenizer")) |pt| {
        if (pt == .object) {
            if (pt.object.get("type")) |t| {
                if (t == .string and std.mem.eql(u8, t.string, "ByteLevel")) {
                    use_byte_level = true;
                }
                // Sequence of pre-tokenizers — check if any is ByteLevel
                if (t == .string and std.mem.eql(u8, t.string, "Sequence")) {
                    if (pt.object.get("pretokenizers")) |arr| {
                        if (arr == .array) {
                            for (arr.array.items) |item| {
                                if (item == .object) {
                                    if (item.object.get("type")) |it| {
                                        if (it == .string and std.mem.eql(u8, it.string, "ByteLevel")) {
                                            use_byte_level = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Model
    const model_v = root.get("model") orelse return Error.MissingField;
    if (model_v != .object) return Error.InvalidTokenizerJson;
    const model_o = model_v.object;
    const model_type = model_o.get("type") orelse return Error.MissingField;
    if (model_type != .string) return Error.InvalidTokenizerJson;
    if (!std.mem.eql(u8, model_type.string, "BPE")) return Error.UnsupportedModelType;

    // vocab
    const vocab_v = model_o.get("vocab") orelse return Error.MissingField;
    if (vocab_v != .object) return Error.InvalidTokenizerJson;

    var vocab_entries: std.ArrayList(bpe.VocabEntry) = .empty;
    defer vocab_entries.deinit(aa);
    var vit = vocab_v.object.iterator();
    while (vit.next()) |e| {
        if (e.value_ptr.* != .integer) return Error.InvalidTokenizerJson;
        const id: u32 = @intCast(e.value_ptr.integer);
        try vocab_entries.append(aa, .{ .token = e.key_ptr.*, .id = id });
    }

    // merges
    const merges_v = model_o.get("merges") orelse return Error.MissingField;
    if (merges_v != .array) return Error.InvalidTokenizerJson;
    var merges_entries: std.ArrayList(bpe.MergeEntry) = .empty;
    defer merges_entries.deinit(aa);
    for (merges_v.array.items, 0..) |m, rank| {
        if (m == .string) {
            // "a b" form (HF v1)
            const space = std.mem.indexOf(u8, m.string, " ") orelse continue;
            try merges_entries.append(aa, .{
                .left = m.string[0..space],
                .right = m.string[space + 1 ..],
                .rank = @intCast(rank),
            });
        } else if (m == .array and m.array.items.len == 2) {
            // ["a", "b"] form (HF v2 tokenizers >= 0.15)
            const l = m.array.items[0];
            const r = m.array.items[1];
            if (l != .string or r != .string) return Error.InvalidTokenizerJson;
            try merges_entries.append(aa, .{
                .left = l.string,
                .right = r.string,
                .rank = @intCast(rank),
            });
        } else {
            return Error.InvalidTokenizerJson;
        }
    }

    // unk_token id, if present
    var unk_id: ?u32 = null;
    if (model_o.get("unk_token")) |unk_v| {
        if (unk_v == .string) {
            // Look up the unk token's id in the vocab we just built
            for (vocab_entries.items) |e| {
                if (std.mem.eql(u8, e.token, unk_v.string)) {
                    unk_id = e.id;
                    break;
                }
            }
        }
    }

    const model = try bpe.build(alloc, vocab_entries.items, merges_entries.items, unk_id);

    return .{
        .allocator = alloc,
        .model = model,
        .use_byte_level = use_byte_level,
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "fromJson — minimal BPE + ByteLevel tokenizer" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "Ġ": 4, "Ġh": 5, "el": 6, "llo": 7, "Ġhello": 8},
        \\    "merges": ["e l", "l l o", "h el", "Ġ h", "Ġh ello"]
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    try testing.expect(tok.use_byte_level);
}

test "fromJson — unsupported model type returns error" {
    const json =
        \\{"model": {"type": "Unigram", "vocab": [], "merges": []}}
    ;
    try testing.expectError(Error.UnsupportedModelType, fromJson(testing.allocator, json));
}

test "fromJson — missing model field" {
    const json =
        \\{"version": "1.0"}
    ;
    try testing.expectError(Error.MissingField, fromJson(testing.allocator, json));
}

test "fromJson + encode — simple ByteLevel BPE round-trip" {
    // Tiny tokenizer: vocab covers a-z + Ġ-prefixed variants for "the".
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"t": 0, "h": 1, "e": 2, "th": 3, "the": 4},
        \\    "merges": ["t h", "th e"]
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const ids = try tok.encode(testing.allocator, "the");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{4}, ids);

    const back = try tok.decode(testing.allocator, ids);
    defer testing.allocator.free(back);
    try testing.expectEqualStrings("the", back);
}

test "fromJson + encode — falls back to single chars when no merge applies" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"x": 0, "y": 1, "z": 2},
        \\    "merges": []
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const ids = try tok.encode(testing.allocator, "xyz");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, ids);
}

test "merges-as-array (v2 format)" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1, "ab": 2},
        \\    "merges": [["a", "b"]]
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    const ids = try tok.encode(testing.allocator, "ab");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{2}, ids);
}
