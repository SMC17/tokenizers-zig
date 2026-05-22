//! tokenizers-zig — Zig 0.16 tokenizers compatible with the
//! HuggingFace `tokenizer.json` format.
//!
//! v0.2 scope: BPE + ByteLevel (v0.0.1 — GPT-2 / Llama / Qwen / Mistral)
//!           + WordPiece (v0.2 — BERT / DistilBERT / MobileBERT)
//!           + Unigram (v0.2 — T5 / mBART / XLM-R, SentencePiece-style).
//!
//! What's covered:
//!   - ByteLevel pre-tokenizer (canonical GPT-2 byte → unicode map)
//!   - Whitespace pre-tokenizer (BERT-style word splitting)
//!   - Metaspace pre-tokenizer (SentencePiece ▁ replacement)
//!   - BPE model with priority-queue + DLL merge loop
//!   - WordPiece model with greedy longest-match + ## continuation
//!   - Unigram model with Viterbi-best segmentation
//!   - HuggingFace tokenizer.json loader dispatching on model.type
//!   - Encode (text → token IDs) and decode (token IDs → text)
//!
//! What's not yet covered (v0.3+):
//!   - BertPreTokenizer punctuation splitting beyond whitespace
//!   - NFC / NFD / NFKC / NFKD Unicode normalization
//!   - Truncation, padding, template post-processors
//!   - Special-token end-to-end (added_tokens parsing skeleton only)
//!   - Regex-based pre-tokenizers / Split

const std = @import("std");

pub const byte_level = @import("byte_level.zig");
pub const bpe = @import("bpe.zig");
pub const wordpiece = @import("wordpiece.zig");
pub const unigram = @import("unigram.zig");
pub const normalizer = @import("normalizer.zig");

pub const Error = error{
    InvalidTokenizerJson,
    UnsupportedModelType,
    UnsupportedPreTokenizer,
    MissingField,
};

/// Tagged union over the model backends supported by this library.
pub const Model = union(enum) {
    bpe: bpe.Model,
    wordpiece: wordpiece.Model,
    unigram: unigram.Model,

    pub fn deinit(self: *Model) void {
        switch (self.*) {
            .bpe => |*m| m.deinit(),
            .wordpiece => |*m| m.deinit(),
            .unigram => |*m| m.deinit(),
        }
    }

    /// Vocabulary size (number of distinct token strings).
    pub fn vocabCount(self: *const Model) usize {
        return switch (self.*) {
            .bpe => |*m| m.vocab.count(),
            .wordpiece => |*m| m.vocab.count(),
            .unigram => |*m| m.tokens.len,
        };
    }
};

/// Added-tokens registry. Parsed from the top-level `added_tokens` array
/// of a tokenizer.json. Entries are matched verbatim against input and
/// bypass the model encoder/decoder.
pub const AddedToken = struct {
    id: u32,
    content: []const u8,
};

/// Pipeline configuration assembled from a HF tokenizer.json.
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    /// Owned storage for parsed config string slices.
    arena: *std.heap.ArenaAllocator,
    model: Model,
    /// True if pre-tokenizer is ByteLevel (GPT-2 / Llama-3 style).
    use_byte_level: bool,
    /// True if pre-tokenizer is Whitespace-style (BERT WordPiece).
    use_whitespace_split: bool,
    /// True if pre-tokenizer is BertPreTokenizer — split on whitespace AND
    /// on punctuation (each punctuation char becomes its own token).
    split_on_punctuation: bool,
    /// Metaspace replacement char (SentencePiece "▁"). When set, spaces
    /// in input are replaced by this string before model.encode, and
    /// decode does the inverse.
    metaspace_replacement: ?[]const u8,
    /// Optional pre-BPE normalizer (Llama / Mistral SentencePiece style).
    normalizer: normalizer.NormalizerKind,
    /// SentencePiece space marker — when set, decoder replaces it
    /// back to a literal space. Llama: "▁" (U+2581).
    sp_space_marker: ?[]const u8,
    /// Added-token list, in source order. Lifetime tied to arena.
    added_tokens: []AddedToken,
    /// id → content lookup for decode.
    added_token_by_id: std.AutoHashMap(u32, []const u8),

    pub fn deinit(self: *Tokenizer) void {
        self.model.deinit();
        self.added_token_by_id.deinit();
        self.arena.deinit();
        self.allocator.destroy(self.arena);
    }

    /// Find the longest added-token matching `text` at byte position `pos`.
    /// Returns null if no added token matches. O(N_added) per position; the
    /// first-byte filter cuts most of the constant.
    fn matchAddedToken(self: *const Tokenizer, text: []const u8, pos: usize) ?AddedToken {
        var best: ?AddedToken = null;
        for (self.added_tokens) |at| {
            if (at.content.len == 0) continue;
            if (pos + at.content.len > text.len) continue;
            if (text[pos] != at.content[0]) continue;
            if (!std.mem.eql(u8, text[pos .. pos + at.content.len], at.content)) continue;
            if (best == null or at.content.len > best.?.content.len) best = at;
        }
        return best;
    }

    /// Encode UTF-8 text into token IDs. Caller frees.
    pub fn encode(self: *Tokenizer, alloc: std.mem.Allocator, text: []const u8) ![]u32 {
        // Step 0: split on added tokens. Each non-added chunk goes through
        // the full normalize → metaspace → model pipeline; each added
        // token emits its id directly.
        if (self.added_tokens.len > 0) {
            var out: std.ArrayList(u32) = .empty;
            errdefer out.deinit(alloc);
            var pos: usize = 0;
            var chunk_start: usize = 0;
            while (pos < text.len) {
                if (self.matchAddedToken(text, pos)) |at| {
                    if (pos > chunk_start) {
                        const ids = try self.encodeChunk(alloc, text[chunk_start..pos]);
                        defer alloc.free(ids);
                        try out.appendSlice(alloc, ids);
                    }
                    try out.append(alloc, at.id);
                    pos += at.content.len;
                    chunk_start = pos;
                } else {
                    pos += 1;
                }
            }
            if (chunk_start < text.len) {
                const ids = try self.encodeChunk(alloc, text[chunk_start..]);
                defer alloc.free(ids);
                try out.appendSlice(alloc, ids);
            }
            return out.toOwnedSlice(alloc);
        }
        return self.encodeChunk(alloc, text);
    }

    /// Encode a single chunk of text (no added-token splitting) through the
    /// full normalize → (metaspace) → model pipeline.
    fn encodeChunk(self: *Tokenizer, alloc: std.mem.Allocator, text: []const u8) ![]u32 {
        // Step 1: normalize.
        const normalized = try normalizer.apply(alloc, self.normalizer, text);
        defer alloc.free(normalized);

        // Unigram + WhitespaceSplit + Metaspace (T5 Sequence pattern): split
        // on whitespace BEFORE metaspace; prepend metaspace to each piece;
        // Viterbi each piece. This matches HF's per-piece pre-tokenization.
        if (self.model == .unigram and self.use_whitespace_split and self.metaspace_replacement != null) {
            const rep = self.metaspace_replacement.?;
            var out: std.ArrayList(u32) = .empty;
            errdefer out.deinit(alloc);
            var piece_buf: std.ArrayList(u8) = .empty;
            defer piece_buf.deinit(alloc);
            var it = std.mem.tokenizeAny(u8, normalized, " \t\r\n");
            while (it.next()) |piece| {
                piece_buf.clearRetainingCapacity();
                try piece_buf.appendSlice(alloc, rep);
                try piece_buf.appendSlice(alloc, piece);
                const ids = try self.model.unigram.encode(alloc, piece_buf.items);
                defer alloc.free(ids);
                try out.appendSlice(alloc, ids);
            }
            return out.toOwnedSlice(alloc);
        }

        // Otherwise: apply Metaspace (if any) to the whole text and proceed.
        const metaspaced: []u8 = if (self.metaspace_replacement) |rep| blk: {
            const replaced = try normalizer.apply(
                alloc,
                .{ .replace = .{ .pattern = " ", .content = rep } },
                normalized,
            );
            defer alloc.free(replaced);
            if (std.mem.startsWith(u8, replaced, rep)) {
                break :blk try alloc.dupe(u8, replaced);
            }
            const out = try alloc.alloc(u8, rep.len + replaced.len);
            @memcpy(out[0..rep.len], rep);
            @memcpy(out[rep.len..], replaced);
            break :blk out;
        } else try alloc.dupe(u8, normalized);
        defer alloc.free(metaspaced);

        switch (self.model) {
            .bpe => |*m| {
                if (self.use_byte_level) {
                    const encoded = try byte_level.encode(alloc, metaspaced);
                    defer alloc.free(encoded);
                    return m.encode(alloc, encoded);
                }
                return m.encode(alloc, metaspaced);
            },
            .wordpiece => |*m| {
                var out: std.ArrayList(u32) = .empty;
                errdefer out.deinit(alloc);
                var it = std.mem.tokenizeAny(u8, metaspaced, " \t\r\n");
                while (it.next()) |chunk| {
                    if (self.split_on_punctuation) {
                        try encodePunctSplit(alloc, m, &out, chunk);
                    } else {
                        const ids = try m.encode(alloc, chunk);
                        defer alloc.free(ids);
                        try out.appendSlice(alloc, ids);
                    }
                }
                return out.toOwnedSlice(alloc);
            },
            .unigram => |*m| {
                return m.encode(alloc, metaspaced);
            },
        }
    }

    /// Decode token IDs back to UTF-8 text. Caller frees.
    ///
    /// Added tokens (parsed from `added_tokens`) are emitted verbatim from
    /// their content string; everything else is routed through the model
    /// decoder + post-processing for the configured pipeline.
    pub fn decode(self: *Tokenizer, alloc: std.mem.Allocator, ids: []const u32) ![]u8 {
        if (self.added_token_by_id.count() > 0) {
            var out: std.ArrayList(u8) = .empty;
            errdefer out.deinit(alloc);
            // Decode runs of non-added IDs through the model in a single
            // batch (preserves model-internal pair-joining semantics);
            // added IDs interrupt the run.
            var run_start: usize = 0;
            var i: usize = 0;
            while (i < ids.len) {
                if (self.added_token_by_id.get(ids[i])) |content| {
                    if (i > run_start) {
                        const piece = try self.decodeChunk(alloc, ids[run_start..i]);
                        defer alloc.free(piece);
                        try out.appendSlice(alloc, piece);
                    }
                    try out.appendSlice(alloc, content);
                    i += 1;
                    run_start = i;
                } else {
                    i += 1;
                }
            }
            if (run_start < ids.len) {
                const piece = try self.decodeChunk(alloc, ids[run_start..]);
                defer alloc.free(piece);
                try out.appendSlice(alloc, piece);
            }
            return out.toOwnedSlice(alloc);
        }
        return self.decodeChunk(alloc, ids);
    }

    /// Decode a contiguous run of model-side IDs (no added tokens in the
    /// slice). Applies pipeline-specific reverse mapping (ByteLevel,
    /// SentencePiece marker, Metaspace).
    fn decodeChunk(self: *Tokenizer, alloc: std.mem.Allocator, ids: []const u32) ![]u8 {
        switch (self.model) {
            .bpe => |*m| {
                const encoded = try m.decode(alloc, ids);
                defer alloc.free(encoded);
                if (self.use_byte_level) {
                    return byte_level.decode(alloc, encoded);
                } else if (self.sp_space_marker) |marker| {
                    const with_spaces = try normalizer.apply(alloc, .{
                        .replace = .{ .pattern = marker, .content = " " },
                    }, encoded);
                    if (with_spaces.len > 0 and with_spaces[0] == ' ') {
                        const trimmed = try alloc.dupe(u8, with_spaces[1..]);
                        alloc.free(with_spaces);
                        return trimmed;
                    }
                    return with_spaces;
                }
                return alloc.dupe(u8, encoded);
            },
            .wordpiece => |*m| {
                // WordPiece's own decode already handles ## continuation joining.
                return m.decode(alloc, ids);
            },
            .unigram => |*m| {
                const raw = try m.decode(alloc, ids);
                if (self.metaspace_replacement) |rep| {
                    defer alloc.free(raw);
                    const with_spaces = try normalizer.apply(alloc, .{
                        .replace = .{ .pattern = rep, .content = " " },
                    }, raw);
                    if (with_spaces.len > 0 and with_spaces[0] == ' ') {
                        const trimmed = try alloc.dupe(u8, with_spaces[1..]);
                        alloc.free(with_spaces);
                        return trimmed;
                    }
                    return with_spaces;
                }
                return raw;
            },
        }
    }
};

/// Build a Tokenizer from an in-memory tokenizer.json byte slice.
pub fn fromJson(alloc: std.mem.Allocator, json_bytes: []const u8) !Tokenizer {
    const own_arena = try alloc.create(std.heap.ArenaAllocator);
    errdefer alloc.destroy(own_arena);
    own_arena.* = std.heap.ArenaAllocator.init(alloc);
    errdefer own_arena.deinit();
    const aa = own_arena.allocator();

    const parsed = std.json.parseFromSliceLeaky(std.json.Value, aa, json_bytes, .{
        .max_value_len = std.json.default_max_value_len,
    }) catch return Error.InvalidTokenizerJson;
    if (parsed != .object) return Error.InvalidTokenizerJson;
    const root = parsed.object;

    // Pre-tokenizer (ByteLevel / Whitespace / Bert / Metaspace / Sequence)
    var use_byte_level = false;
    var use_whitespace_split = false;
    var split_on_punctuation = false;
    var metaspace_replacement: ?[]const u8 = null;
    if (root.get("pre_tokenizer")) |pt| {
        if (pt == .object) {
            try parsePreTokenizer(aa, pt.object, &use_byte_level, &use_whitespace_split, &metaspace_replacement, &split_on_punctuation);
        }
    }

    // Model. If `model.type` is missing (older HF format, e.g. BERT-base),
    // infer from structural fields:
    //   - `merges` present                       → BPE
    //   - `continuing_subword_prefix` present    → WordPiece
    //   - `vocab` is an array                    → Unigram
    const model_v = root.get("model") orelse return Error.MissingField;
    if (model_v != .object) return Error.InvalidTokenizerJson;
    const model_o = model_v.object;

    const inferred_type: []const u8 = blk: {
        if (model_o.get("type")) |tv| {
            if (tv == .string) break :blk tv.string;
        }
        if (model_o.contains("merges")) break :blk "BPE";
        if (model_o.contains("continuing_subword_prefix")) break :blk "WordPiece";
        if (model_o.get("vocab")) |vv| {
            if (vv == .array) break :blk "Unigram";
        }
        return Error.MissingField;
    };

    var model: Model = undefined;
    if (std.mem.eql(u8, inferred_type, "BPE")) {
        model = .{ .bpe = try parseBpe(alloc, aa, model_o) };
    } else if (std.mem.eql(u8, inferred_type, "WordPiece")) {
        model = .{ .wordpiece = try parseWordPiece(alloc, aa, model_o) };
    } else if (std.mem.eql(u8, inferred_type, "Unigram")) {
        model = .{ .unigram = try parseUnigram(alloc, aa, model_o) };
    } else {
        return Error.UnsupportedModelType;
    }

    // Normalizer (optional)
    var norm_kind: normalizer.NormalizerKind = .none;
    var sp_marker: ?[]const u8 = null;
    if (root.get("normalizer")) |nv| {
        if (nv == .object) {
            norm_kind = try parseNormalizer(aa, nv.object, &sp_marker);
        }
    }

    // added_tokens (optional). Each entry is an object with at least
    // `id` (integer) and `content` (string). Other fields (`special`,
    // `single_word`, `lstrip`, `rstrip`, `normalized`) are ignored at v0.2.
    var added_list: std.ArrayList(AddedToken) = .empty;
    defer added_list.deinit(aa);
    var added_by_id = std.AutoHashMap(u32, []const u8).init(alloc);
    errdefer added_by_id.deinit();
    if (root.get("added_tokens")) |atv| {
        if (atv == .array) {
            for (atv.array.items) |item| {
                if (item != .object) continue;
                const id_v = item.object.get("id") orelse continue;
                const content_v = item.object.get("content") orelse continue;
                if (id_v != .integer or content_v != .string) continue;
                const id: u32 = @intCast(id_v.integer);
                const content = try aa.dupe(u8, content_v.string);
                try added_list.append(aa, .{ .id = id, .content = content });
                try added_by_id.put(id, content);
            }
        }
    }
    const added_tokens = try aa.dupe(AddedToken, added_list.items);

    return .{
        .allocator = alloc,
        .arena = own_arena,
        .model = model,
        .use_byte_level = use_byte_level,
        .use_whitespace_split = use_whitespace_split,
        .split_on_punctuation = split_on_punctuation,
        .metaspace_replacement = metaspace_replacement,
        .normalizer = norm_kind,
        .sp_space_marker = sp_marker,
        .added_tokens = added_tokens,
        .added_token_by_id = added_by_id,
    };
}

fn parsePreTokenizer(
    aa: std.mem.Allocator,
    obj: std.json.ObjectMap,
    byte_level_out: *bool,
    whitespace_out: *bool,
    metaspace_out: *?[]const u8,
    punct_split_out: *bool,
) !void {
    const tv = obj.get("type") orelse return;
    if (tv != .string) return;
    const t = tv.string;

    if (std.mem.eql(u8, t, "ByteLevel")) {
        byte_level_out.* = true;
        return;
    }
    if (std.mem.eql(u8, t, "Whitespace") or std.mem.eql(u8, t, "WhitespaceSplit")) {
        whitespace_out.* = true;
        return;
    }
    if (std.mem.eql(u8, t, "BertPreTokenizer")) {
        whitespace_out.* = true;
        punct_split_out.* = true;
        return;
    }
    if (std.mem.eql(u8, t, "Metaspace")) {
        // Default replacement is "▁" if not specified; HF default.
        if (obj.get("replacement")) |rv| {
            if (rv == .string) {
                metaspace_out.* = try aa.dupe(u8, rv.string);
                return;
            }
        }
        metaspace_out.* = try aa.dupe(u8, "\u{2581}");
        return;
    }
    if (std.mem.eql(u8, t, "Sequence")) {
        if (obj.get("pretokenizers")) |arr| {
            if (arr == .array) {
                for (arr.array.items) |item| {
                    if (item == .object) {
                        try parsePreTokenizer(aa, item.object, byte_level_out, whitespace_out, metaspace_out, punct_split_out);
                    }
                }
            }
        }
    }
    // Unknown pre-tokenizer types are silently skipped (degrade gracefully).
}

/// ASCII punctuation classifier matching BertPreTokenizer behaviour for
/// the ASCII range. Full Unicode "P" category is v0.4 work.
fn isAsciiPunct(b: u8) bool {
    return switch (b) {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/' => true,
        ':', ';', '<', '=', '>', '?', '@' => true,
        '[', '\\', ']', '^', '_', '`' => true,
        '{', '|', '}', '~' => true,
        else => false,
    };
}

/// Encode a single whitespace-chunk into ids, splitting on punctuation.
/// Each maximal run of non-punctuation ASCII (and any UTF-8 byte ≥ 0x80)
/// becomes a "word" passed to WordPiece; each ASCII punctuation byte
/// becomes its own single-char word.
fn encodePunctSplit(
    alloc: std.mem.Allocator,
    m: *const wordpiece.Model,
    out: *std.ArrayList(u32),
    chunk: []const u8,
) !void {
    var word_start: usize = 0;
    var i: usize = 0;
    while (i < chunk.len) : (i += 1) {
        if (isAsciiPunct(chunk[i])) {
            if (i > word_start) {
                const ids = try m.encode(alloc, chunk[word_start..i]);
                defer alloc.free(ids);
                try out.appendSlice(alloc, ids);
            }
            const ids = try m.encode(alloc, chunk[i .. i + 1]);
            defer alloc.free(ids);
            try out.appendSlice(alloc, ids);
            word_start = i + 1;
        }
    }
    if (word_start < chunk.len) {
        const ids = try m.encode(alloc, chunk[word_start..]);
        defer alloc.free(ids);
        try out.appendSlice(alloc, ids);
    }
}

fn parseBpe(
    alloc: std.mem.Allocator,
    aa: std.mem.Allocator,
    model_o: std.json.ObjectMap,
) !bpe.Model {
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
            const space = std.mem.indexOf(u8, m.string, " ") orelse continue;
            try merges_entries.append(aa, .{
                .left = m.string[0..space],
                .right = m.string[space + 1 ..],
                .rank = @intCast(rank),
            });
        } else if (m == .array and m.array.items.len == 2) {
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

    var unk_id: ?u32 = null;
    if (model_o.get("unk_token")) |unk_v| {
        if (unk_v == .string) {
            for (vocab_entries.items) |e| {
                if (std.mem.eql(u8, e.token, unk_v.string)) {
                    unk_id = e.id;
                    break;
                }
            }
        }
    }

    return bpe.build(alloc, vocab_entries.items, merges_entries.items, unk_id);
}

fn parseWordPiece(
    alloc: std.mem.Allocator,
    aa: std.mem.Allocator,
    model_o: std.json.ObjectMap,
) !wordpiece.Model {
    const vocab_v = model_o.get("vocab") orelse return Error.MissingField;
    if (vocab_v != .object) return Error.InvalidTokenizerJson;

    var vocab_entries: std.ArrayList(wordpiece.VocabEntry) = .empty;
    defer vocab_entries.deinit(aa);
    var vit = vocab_v.object.iterator();
    while (vit.next()) |e| {
        if (e.value_ptr.* != .integer) return Error.InvalidTokenizerJson;
        const id: u32 = @intCast(e.value_ptr.integer);
        try vocab_entries.append(aa, .{ .token = e.key_ptr.*, .id = id });
    }

    // unk_token (string) → look up its id in the vocab.
    var unk_id: u32 = 0;
    if (model_o.get("unk_token")) |unk_v| {
        if (unk_v == .string) {
            for (vocab_entries.items) |e| {
                if (std.mem.eql(u8, e.token, unk_v.string)) {
                    unk_id = e.id;
                    break;
                }
            }
        }
    }

    var prefix: []const u8 = "##";
    if (model_o.get("continuing_subword_prefix")) |pv| {
        if (pv == .string) prefix = pv.string;
    }

    var max_chars: u32 = 100;
    if (model_o.get("max_input_chars_per_word")) |mv| {
        if (mv == .integer) max_chars = @intCast(mv.integer);
    }

    return wordpiece.build(alloc, vocab_entries.items, unk_id, prefix, max_chars);
}

fn parseUnigram(
    alloc: std.mem.Allocator,
    aa: std.mem.Allocator,
    model_o: std.json.ObjectMap,
) !unigram.Model {
    // vocab: array of [string, number]
    const vocab_v = model_o.get("vocab") orelse return Error.MissingField;
    if (vocab_v != .array) return Error.InvalidTokenizerJson;

    var vocab_entries: std.ArrayList(unigram.VocabEntry) = .empty;
    defer vocab_entries.deinit(aa);
    for (vocab_v.array.items) |e| {
        if (e != .array or e.array.items.len != 2) return Error.InvalidTokenizerJson;
        const tok_v = e.array.items[0];
        const score_v = e.array.items[1];
        if (tok_v != .string) return Error.InvalidTokenizerJson;
        const score: f64 = switch (score_v) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            .number_string => |s| std.fmt.parseFloat(f64, s) catch return Error.InvalidTokenizerJson,
            else => return Error.InvalidTokenizerJson,
        };
        try vocab_entries.append(aa, .{ .token = tok_v.string, .score = score });
    }

    var unk_id: u32 = 0;
    if (model_o.get("unk_id")) |unk_v| {
        if (unk_v == .integer) unk_id = @intCast(unk_v.integer);
    }

    return unigram.build(alloc, vocab_entries.items, unk_id, null);
}

fn parseNormalizer(
    aa: std.mem.Allocator,
    obj: std.json.ObjectMap,
    sp_marker_out: *?[]const u8,
) !normalizer.NormalizerKind {
    const tv = obj.get("type") orelse return Error.InvalidTokenizerJson;
    if (tv != .string) return Error.InvalidTokenizerJson;
    const t = tv.string;

    if (std.mem.eql(u8, t, "Prepend")) {
        const pv = obj.get("prepend") orelse return Error.InvalidTokenizerJson;
        if (pv != .string) return Error.InvalidTokenizerJson;
        return .{ .prepend = try aa.dupe(u8, pv.string) };
    }
    if (std.mem.eql(u8, t, "Replace")) {
        const pat_v = obj.get("pattern") orelse return Error.InvalidTokenizerJson;
        const content_v = obj.get("content") orelse return Error.InvalidTokenizerJson;
        if (content_v != .string) return Error.InvalidTokenizerJson;
        var pattern: []const u8 = "";
        if (pat_v == .object) {
            if (pat_v.object.get("String")) |sv| {
                if (sv == .string) pattern = sv.string;
            }
        } else if (pat_v == .string) {
            pattern = pat_v.string;
        }
        const content = content_v.string;
        if (std.mem.eql(u8, pattern, " ") and std.mem.eql(u8, content, "\xE2\x96\x81")) {
            sp_marker_out.* = try aa.dupe(u8, content);
        }
        return .{ .replace = .{
            .pattern = try aa.dupe(u8, pattern),
            .content = try aa.dupe(u8, content),
        } };
    }
    if (std.mem.eql(u8, t, "Sequence")) {
        const arr_v = obj.get("normalizers") orelse return Error.InvalidTokenizerJson;
        if (arr_v != .array) return Error.InvalidTokenizerJson;
        const sub_norms = try aa.alloc(normalizer.NormalizerKind, arr_v.array.items.len);
        for (arr_v.array.items, 0..) |item, i| {
            if (item != .object) return Error.InvalidTokenizerJson;
            sub_norms[i] = try parseNormalizer(aa, item.object, sp_marker_out);
        }
        return .{ .sequence = sub_norms };
    }
    if (std.mem.eql(u8, t, "Lowercase")) {
        return .lowercase;
    }
    if (std.mem.eql(u8, t, "BertNormalizer")) {
        // BertNormalizer config: clean_text, handle_chinese_chars,
        // strip_accents, lowercase. v0.3 supports lowercase only;
        // clean_text + chinese_chars + strip_accents are no-ops.
        var do_lower = true; // HF default
        if (obj.get("lowercase")) |lv| {
            if (lv == .bool) do_lower = lv.bool;
        }
        if (do_lower) return .lowercase;
        return .none;
    }
    return .none;
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
    try testing.expect(tok.model == .bpe);
}

test "fromJson — unsupported model type returns error" {
    const json =
        \\{"model": {"type": "Phantom", "vocab": [], "merges": []}}
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

test "fromJson — WordPiece minimal BERT-style tokenizer parses + encodes" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "BertPreTokenizer"},
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0, "play": 1, "##ing": 2, "the": 3, "game": 4},
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##",
        \\    "max_input_chars_per_word": 100
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    try testing.expect(tok.use_whitespace_split);
    try testing.expect(tok.model == .wordpiece);

    const ids = try tok.encode(testing.allocator, "playing the game");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4 }, ids);
}

test "fromJson — WordPiece decode rejoins ## continuations" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "Whitespace"},
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0, "un": 1, "##aff": 2, "##able": 3},
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##",
        \\    "max_input_chars_per_word": 100
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const ids = try tok.encode(testing.allocator, "unaffable");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3 }, ids);

    const back = try tok.decode(testing.allocator, ids);
    defer testing.allocator.free(back);
    try testing.expectEqualStrings("unaffable", back);
}

test "fromJson — Unigram minimal T5-style tokenizer parses + encodes" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "Metaspace", "replacement": "▁"},
        \\  "model": {
        \\    "type": "Unigram",
        \\    "vocab": [["<unk>", 0.0], ["▁", -3.0], ["▁hello", -2.0], ["▁world", -2.0]],
        \\    "unk_id": 0
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    try testing.expect(tok.model == .unigram);
    try testing.expect(tok.metaspace_replacement != null);

    const ids = try tok.encode(testing.allocator, "hello world");
    defer testing.allocator.free(ids);
    // Pipeline: " hello world" → metaspace → "▁hello▁world"
    // Viterbi: best is "▁hello" + "▁world".
    try testing.expectEqualSlices(u32, &[_]u32{ 2, 3 }, ids);
}

test "fromJson — Unigram decode reverses Metaspace" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "Metaspace"},
        \\  "model": {
        \\    "type": "Unigram",
        \\    "vocab": [["<unk>", 0.0], ["▁", -3.0], ["▁hi", -2.0]],
        \\    "unk_id": 0
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const ids = try tok.encode(testing.allocator, "hi");
    defer testing.allocator.free(ids);
    const back = try tok.decode(testing.allocator, ids);
    defer testing.allocator.free(back);
    // ▁hi → decode → "hi" (leading ▁ → leading space → stripped)
    try testing.expectEqualStrings("hi", back);
}

test "added_tokens — single added token at start of text emits directly" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [{"id": 99, "content": "<s>", "special": true}],
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1, "Ġ": 2},
        \\    "merges": []
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    try testing.expectEqual(@as(usize, 1), tok.added_tokens.len);

    const ids = try tok.encode(testing.allocator, "<s>ab");
    defer testing.allocator.free(ids);
    // <s> → 99; "ab" → ByteLevel("ab") → [0, 1]
    try testing.expectEqualSlices(u32, &[_]u32{ 99, 0, 1 }, ids);
}

test "added_tokens — added tokens interrupt model chunks correctly" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [
        \\    {"id": 100, "content": "<s>", "special": true},
        \\    {"id": 101, "content": "</s>", "special": true}
        \\  ],
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1, "c": 2, "Ġ": 3},
        \\    "merges": []
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const ids = try tok.encode(testing.allocator, "<s>ab</s>c");
    defer testing.allocator.free(ids);
    // <s>=100 | "ab" → [0,1] | </s>=101 | "c" → [2]
    try testing.expectEqualSlices(u32, &[_]u32{ 100, 0, 1, 101, 2 }, ids);
}

test "added_tokens — longest match wins on overlap" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [
        \\    {"id": 1, "content": "<"},
        \\    {"id": 2, "content": "<eos>"}
        \\  ],
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0},
        \\    "merges": []
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    // Both "<" and "<eos>" start with "<". Longest-match wins → emit 2.
    const ids = try tok.encode(testing.allocator, "<eos>");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{2}, ids);
}

test "added_tokens — decode emits content verbatim" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [{"id": 99, "content": "<s>", "special": true}],
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1},
        \\    "merges": []
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const back = try tok.decode(testing.allocator, &[_]u32{ 99, 0, 1 });
    defer testing.allocator.free(back);
    try testing.expectEqualStrings("<s>ab", back);
}

test "added_tokens — empty registry is fast path (BPE-only)" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1},
        \\    "merges": []
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    try testing.expectEqual(@as(usize, 0), tok.added_tokens.len);
    try testing.expectEqual(@as(usize, 0), tok.added_token_by_id.count());

    const ids = try tok.encode(testing.allocator, "ab");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1 }, ids);
}

test "Model — vocabCount works across all variants" {
    const bpe_json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1, "c": 2},
        \\    "merges": []
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, bpe_json);
    defer tok.deinit();
    try testing.expectEqual(@as(usize, 3), tok.model.vocabCount());
}
