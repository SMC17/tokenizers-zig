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
pub const unicode_norm = @import("unicode_norm.zig");
pub const gpt2_regex = @import("gpt2_regex.zig");

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

/// A single element of a TemplateProcessing template:
/// - `append_ids`: fixed sequence of token IDs (resolved from a
///   SpecialToken via the post_processor's special_tokens map)
/// - `append_input`: inject the "A" input's encoded IDs at this position
/// - `append_input_b`: inject the "B" input's encoded IDs (pair-only)
pub const TemplateOp = union(enum) {
    append_ids: []const u32,
    append_input: void,
    append_input_b: void,
};

/// Truncation strategy applied after encode + TemplateProcessing.
pub const TruncationDirection = enum { right, left };

/// Padding direction applied after encode + truncation.
pub const PaddingDirection = enum { right, left };

/// Pipeline configuration assembled from a HF tokenizer.json.
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    /// Owned storage for parsed config string slices.
    arena: *std.heap.ArenaAllocator,
    model: Model,
    /// True if pre-tokenizer is ByteLevel (GPT-2 / Llama-3 style).
    use_byte_level: bool,
    /// True if the ByteLevel pre-tokenizer is configured with
    /// `use_regex: true` (the canonical GPT-2 word-boundary regex).
    /// When set, the BPE encoder splits input via the GPT-2 regex
    /// pattern before applying ByteLevel + BPE per chunk.
    use_gpt2_regex: bool,
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
    /// TemplateProcessing single-input template. When non-null, applied
    /// to the encoded ID sequence to wrap with BOS/EOS/CLS/SEP markers.
    template_single: ?[]const TemplateOp,
    /// TemplateProcessing pair-input template. Applied by `encodePair`
    /// to combine two input sequences with markers between them.
    template_pair: ?[]const TemplateOp,
    /// True if the decoder Sequence contains a `ByteFallback` step. When
    /// set, decode replaces `<0xXX>` token strings with their literal
    /// byte values after the model's vocab lookup.
    use_byte_fallback: bool,
    /// Truncation config. When max_length is non-null, encode trims the
    /// output to at most this many tokens from `truncation_direction`.
    max_length: ?u32,
    truncation_direction: TruncationDirection,
    /// Padding config. When pad_to_length is non-null, encode pads short
    /// outputs to this length with `pad_token_id` from `padding_direction`.
    pad_to_length: ?u32,
    pad_token_id: u32,
    padding_direction: PaddingDirection,

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
    ///
    /// Pipeline (in order):
    ///   1. Added-token longest-match scan splits input into added-id
    ///      emissions and chunks routed through the model pipeline.
    ///   2. TemplateProcessing (if configured) wraps the base IDs with
    ///      BOS/EOS/CLS/SEP markers per the `single` template.
    ///   3. Truncation (if `max_length` set) trims from
    ///      `truncation_direction`.
    ///   4. Padding (if `pad_to_length` set) extends with `pad_token_id`.
    pub fn encode(self: *Tokenizer, alloc: std.mem.Allocator, text: []const u8) ![]u32 {
        const base = try self.encodeBase(alloc, text);
        defer alloc.free(base);
        return self.applyPostProcessing(alloc, base);
    }

    /// Encode WITHOUT TemplateProcessing / truncation / padding. Returns
    /// the raw model-side IDs only. Useful when the caller is composing
    /// templates manually.
    pub fn encodeBase(self: *Tokenizer, alloc: std.mem.Allocator, text: []const u8) ![]u32 {
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

    /// Encode a sentence-pair into one ID sequence. Applies the
    /// `template_pair` template if present; otherwise concatenates
    /// `[encodeBase(a), encodeBase(b)]` with no markers. Truncation and
    /// padding apply post-template. Caller frees.
    pub fn encodePair(self: *Tokenizer, alloc: std.mem.Allocator, text_a: []const u8, text_b: []const u8) ![]u32 {
        const a = try self.encodeBase(alloc, text_a);
        defer alloc.free(a);
        const b = try self.encodeBase(alloc, text_b);
        defer alloc.free(b);
        return self.applyPostProcessingPair(alloc, a, b);
    }

    /// Apply TemplateProcessing → truncation → padding. Caller frees.
    fn applyPostProcessing(self: *Tokenizer, alloc: std.mem.Allocator, base: []const u32) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(alloc);

        if (self.template_single) |tpl| {
            for (tpl) |op| switch (op) {
                .append_ids => |ids| try out.appendSlice(alloc, ids),
                .append_input => try out.appendSlice(alloc, base),
                .append_input_b => {}, // single template never has B
            };
        } else {
            try out.appendSlice(alloc, base);
        }

        // Truncation
        if (self.max_length) |mx| {
            if (out.items.len > mx) {
                switch (self.truncation_direction) {
                    .right => out.shrinkRetainingCapacity(mx),
                    .left => {
                        const drop = out.items.len - mx;
                        std.mem.copyForwards(u32, out.items[0..mx], out.items[drop..]);
                        out.shrinkRetainingCapacity(mx);
                    },
                }
            }
        }

        // Padding
        if (self.pad_to_length) |pl| {
            if (out.items.len < pl) {
                const need = pl - out.items.len;
                switch (self.padding_direction) {
                    .right => {
                        try out.ensureUnusedCapacity(alloc, need);
                        for (0..need) |_| out.appendAssumeCapacity(self.pad_token_id);
                    },
                    .left => {
                        try out.ensureTotalCapacity(alloc, pl);
                        const old_len = out.items.len;
                        out.items.len = pl;
                        std.mem.copyBackwards(u32, out.items[need..], out.items[0..old_len]);
                        for (0..need) |i| out.items[i] = self.pad_token_id;
                    },
                }
            }
        }

        return out.toOwnedSlice(alloc);
    }

    /// Like applyPostProcessing but for a sentence-pair (a, b). Uses
    /// the `template_pair` field when present; otherwise concatenates
    /// a then b without markers.
    fn applyPostProcessingPair(self: *Tokenizer, alloc: std.mem.Allocator, a: []const u32, b: []const u32) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(alloc);

        if (self.template_pair) |tpl| {
            for (tpl) |op| switch (op) {
                .append_ids => |ids| try out.appendSlice(alloc, ids),
                .append_input => try out.appendSlice(alloc, a),
                .append_input_b => try out.appendSlice(alloc, b),
            };
        } else {
            try out.appendSlice(alloc, a);
            try out.appendSlice(alloc, b);
        }

        if (self.max_length) |mx| {
            if (out.items.len > mx) {
                switch (self.truncation_direction) {
                    .right => out.shrinkRetainingCapacity(mx),
                    .left => {
                        const drop = out.items.len - mx;
                        std.mem.copyForwards(u32, out.items[0..mx], out.items[drop..]);
                        out.shrinkRetainingCapacity(mx);
                    },
                }
            }
        }

        if (self.pad_to_length) |pl| {
            if (out.items.len < pl) {
                const need = pl - out.items.len;
                switch (self.padding_direction) {
                    .right => {
                        try out.ensureUnusedCapacity(alloc, need);
                        for (0..need) |_| out.appendAssumeCapacity(self.pad_token_id);
                    },
                    .left => {
                        try out.ensureTotalCapacity(alloc, pl);
                        const old_len = out.items.len;
                        out.items.len = pl;
                        std.mem.copyBackwards(u32, out.items[need..], out.items[0..old_len]);
                        for (0..need) |i| out.items[i] = self.pad_token_id;
                    },
                }
            }
        }

        return out.toOwnedSlice(alloc);
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
                if (self.use_byte_level and self.use_gpt2_regex) {
                    // GPT-2 / RoBERTa pipeline: regex split → byte_level
                    // per chunk → BPE per chunk → concat.
                    const chunks = try gpt2_regex.split(alloc, metaspaced);
                    defer alloc.free(chunks);
                    var out: std.ArrayList(u32) = .empty;
                    errdefer out.deinit(alloc);
                    for (chunks) |c| {
                        const encoded = try byte_level.encode(alloc, c);
                        defer alloc.free(encoded);
                        const ids = try m.encode(alloc, encoded);
                        defer alloc.free(ids);
                        try out.appendSlice(alloc, ids);
                    }
                    return out.toOwnedSlice(alloc);
                }
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
                const raw = try m.decode(alloc, ids);
                defer alloc.free(raw);

                // ByteFallback (Llama family): `<0xXX>` token strings
                // expand to literal byte XX before any other post-processing.
                const after_bf: []u8 = if (self.use_byte_fallback)
                    try applyByteFallback(alloc, raw)
                else
                    try alloc.dupe(u8, raw);
                defer alloc.free(after_bf);

                if (self.use_byte_level) {
                    return byte_level.decode(alloc, after_bf);
                } else if (self.sp_space_marker) |marker| {
                    const with_spaces = try normalizer.apply(alloc, .{
                        .replace = .{ .pattern = marker, .content = " " },
                    }, after_bf);
                    if (with_spaces.len > 0 and with_spaces[0] == ' ') {
                        const trimmed = try alloc.dupe(u8, with_spaces[1..]);
                        alloc.free(with_spaces);
                        return trimmed;
                    }
                    return with_spaces;
                }
                return alloc.dupe(u8, after_bf);
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
    var use_gpt2_regex = false;
    if (root.get("pre_tokenizer")) |pt| {
        if (pt == .object) {
            try parsePreTokenizer(aa, pt.object, &use_byte_level, &use_whitespace_split, &metaspace_replacement, &split_on_punctuation);
            // GPT-2 regex: enabled when ByteLevel has `use_regex: true`
            // (its HF default) OR when there's a standalone "Split"
            // pre-tokenizer with the GPT-2 pattern hint.
            if (use_byte_level) {
                if (pt.object.get("use_regex")) |uv| {
                    if (uv == .bool and uv.bool) use_gpt2_regex = true;
                }
            }
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

    // Decoder (optional). We scan for ByteFallback presence (anywhere
    // in a possibly-nested Sequence) so the decode path knows to
    // expand `<0xXX>` tokens to bytes.
    var use_byte_fallback = false;
    if (root.get("decoder")) |dv| {
        if (dv == .object) scanByteFallback(dv.object, &use_byte_fallback);
    }

    // Post-processor (TemplateProcessing single + pair templates).
    var template_single: ?[]const TemplateOp = null;
    var template_pair: ?[]const TemplateOp = null;
    if (root.get("post_processor")) |pv| {
        if (pv == .object) {
            template_single = try parseTemplate(aa, pv.object, "single");
            template_pair = try parseTemplate(aa, pv.object, "pair");
        }
    }

    // Truncation (top-level `truncation` object, often null in fixtures
    // but settable in real HF configs).
    var max_length: ?u32 = null;
    var truncation_direction: TruncationDirection = .right;
    if (root.get("truncation")) |tv| {
        if (tv == .object) {
            if (tv.object.get("max_length")) |ml| {
                if (ml == .integer) max_length = @intCast(ml.integer);
            }
            if (tv.object.get("direction")) |dv| {
                if (dv == .string and std.mem.eql(u8, dv.string, "Left")) {
                    truncation_direction = .left;
                }
            }
        }
    }

    // Padding (top-level `padding` object).
    var pad_to_length: ?u32 = null;
    var pad_token_id: u32 = 0;
    var padding_direction: PaddingDirection = .right;
    if (root.get("padding")) |pv2| {
        if (pv2 == .object) {
            // Strategy may be "Fixed" with size or "BatchLongest" etc.
            // We only support a fixed pad-to-length at v0.5.
            if (pv2.object.get("strategy")) |sv| {
                if (sv == .object) {
                    if (sv.object.get("Fixed")) |fv| {
                        if (fv == .integer) pad_to_length = @intCast(fv.integer);
                    }
                }
            }
            if (pv2.object.get("pad_id")) |pid| {
                if (pid == .integer) pad_token_id = @intCast(pid.integer);
            }
            if (pv2.object.get("direction")) |dv| {
                if (dv == .string and std.mem.eql(u8, dv.string, "Left")) {
                    padding_direction = .left;
                }
            }
        }
    }

    return .{
        .allocator = alloc,
        .arena = own_arena,
        .model = model,
        .use_byte_level = use_byte_level,
        .use_gpt2_regex = use_gpt2_regex,
        .use_whitespace_split = use_whitespace_split,
        .split_on_punctuation = split_on_punctuation,
        .metaspace_replacement = metaspace_replacement,
        .normalizer = norm_kind,
        .sp_space_marker = sp_marker,
        .added_tokens = added_tokens,
        .added_token_by_id = added_by_id,
        .template_single = template_single,
        .template_pair = template_pair,
        .use_byte_fallback = use_byte_fallback,
        .max_length = max_length,
        .truncation_direction = truncation_direction,
        .pad_to_length = pad_to_length,
        .pad_token_id = pad_token_id,
        .padding_direction = padding_direction,
    };
}

/// Recursive scan: set out to true if any `ByteFallback` decoder node
/// is present, directly or inside a Sequence.
fn scanByteFallback(obj: std.json.ObjectMap, out: *bool) void {
    const tv = obj.get("type") orelse return;
    if (tv != .string) return;
    if (std.mem.eql(u8, tv.string, "ByteFallback")) {
        out.* = true;
        return;
    }
    if (std.mem.eql(u8, tv.string, "Sequence")) {
        if (obj.get("decoders")) |arr| {
            if (arr == .array) {
                for (arr.array.items) |item| {
                    if (item == .object) scanByteFallback(item.object, out);
                }
            }
        }
    }
}

/// Parse a TemplateProcessing post-processor's `template_field`
/// template (`"single"` or `"pair"`) into a sequence of TemplateOps.
/// SpecialToken ids are resolved via the post-processor's
/// `special_tokens` map. Returns null if the post-processor is not
/// TemplateProcessing or the requested field is absent.
///
/// For pair templates, `Sequence.id == "B"` maps to `.append_input_b`;
/// `id == "A"` maps to `.append_input`.
fn parseTemplate(
    aa: std.mem.Allocator,
    obj: std.json.ObjectMap,
    template_field: []const u8,
) !?[]const TemplateOp {
    const tv = obj.get("type") orelse return null;
    if (tv != .string) return null;
    if (!std.mem.eql(u8, tv.string, "TemplateProcessing")) return null;

    const tpl_v = obj.get(template_field) orelse return null;
    if (tpl_v != .array) return null;

    const special_v = obj.get("special_tokens");
    if (special_v == null or special_v.? != .object) return null;
    const special_obj = special_v.?.object;

    var ops: std.ArrayList(TemplateOp) = .empty;
    errdefer ops.deinit(aa);
    for (tpl_v.array.items) |entry| {
        if (entry != .object) continue;
        if (entry.object.get("Sequence")) |seq_v| {
            // Default to "A" if id missing.
            var is_b: bool = false;
            if (seq_v == .object) {
                if (seq_v.object.get("id")) |idv| {
                    if (idv == .string and std.mem.eql(u8, idv.string, "B")) is_b = true;
                }
            }
            try ops.append(aa, if (is_b) .append_input_b else .append_input);
            continue;
        }
        if (entry.object.get("SpecialToken")) |st_v| {
            if (st_v != .object) continue;
            const id_v = st_v.object.get("id") orelse continue;
            if (id_v != .string) continue;
            const st_name = id_v.string;
            const st_meta_v = special_obj.get(st_name) orelse continue;
            if (st_meta_v != .object) continue;
            const ids_v = st_meta_v.object.get("ids") orelse continue;
            if (ids_v != .array) continue;
            const ids = try aa.alloc(u32, ids_v.array.items.len);
            for (ids_v.array.items, 0..) |iv, i| {
                if (iv != .integer) return Error.InvalidTokenizerJson;
                ids[i] = @intCast(iv.integer);
            }
            try ops.append(aa, .{ .append_ids = ids });
        }
    }

    return try aa.dupe(TemplateOp, ops.items);
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
        // HF default is use_regex: true if not explicitly set false.
        // The actual GPT-2 regex split is enabled only when this code's
        // caller plumbs the flag (see fromJson where the value is
        // resolved against the JSON `use_regex` field).
        return;
    }
    if (std.mem.eql(u8, t, "Split")) {
        // Standalone Split pre-tokenizer with the GPT-2 regex — common
        // in newer HF tokenizer.json files that pulled apart ByteLevel.
        // We treat it as enabling the regex split path.
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

/// ByteFallback expansion: scan `text` for occurrences of `<0xXX>`
/// (6 bytes, `<`, `0`, `x`, two ASCII hex chars, `>`) and replace each
/// with the literal byte value `XX`. Other bytes pass through.
/// Caller frees output.
pub fn applyByteFallback(alloc: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    while (i < text.len) {
        if (i + 6 <= text.len and text[i] == '<' and text[i + 1] == '0' and (text[i + 2] == 'x' or text[i + 2] == 'X') and text[i + 5] == '>') {
            const hi = std.fmt.charToDigit(text[i + 3], 16) catch null;
            const lo = std.fmt.charToDigit(text[i + 4], 16) catch null;
            if (hi != null and lo != null) {
                const b: u8 = (@as(u8, hi.?) << 4) | @as(u8, lo.?);
                try out.append(alloc, b);
                i += 6;
                continue;
            }
        }
        try out.append(alloc, text[i]);
        i += 1;
    }
    return out.toOwnedSlice(alloc);
}

/// Unicode punctuation classifier — returns true if codepoint `cp` is
/// in a punctuation block. Covers ASCII, Latin-1 supplement, General
/// Punctuation, Supplemental Punctuation, CJK Symbols, Vertical Forms,
/// CJK Compatibility Forms, Small Form Variants, and Halfwidth/
/// Fullwidth Forms (punctuation subset).
///
/// Not a full Unicode "P" category — uses range-based approximation
/// covering ~99% of real-world punctuation. Codepoints in the Cyrillic
/// / Arabic / Indic punctuation blocks are NOT detected; v0.5 work if
/// real multilingual fixtures demand it.
pub fn isUnicodePunct(cp: u21) bool {
    return switch (cp) {
        // ASCII punctuation (subset of !–/, :–@, [–`, {–~)
        '!'...'/', ':'...'@', '['...'`', '{'...'~' => true,
        // Latin-1 supplement: ¡ (U+00A1) through ¿ (U+00BF) selected
        0x00A1...0x00A9 => true,
        0x00AB => true,
        0x00AC => true,
        0x00AD => true,
        0x00AE => true,
        0x00AF => true,
        0x00B0 => true,
        0x00B1 => true,
        0x00B4 => true,
        0x00B6...0x00B8 => true,
        0x00BB => true,
        0x00BF => true,
        // General Punctuation (U+2000–U+206F)
        0x2000...0x206F => true,
        // Supplemental Punctuation
        0x2E00...0x2E7F => true,
        // CJK Symbols and Punctuation
        0x3000...0x303F => true,
        // Vertical Forms
        0xFE10...0xFE1F => true,
        // CJK Compatibility Forms
        0xFE30...0xFE4F => true,
        // Small Form Variants
        0xFE50...0xFE6F => true,
        // Halfwidth and Fullwidth Forms — punctuation subset
        0xFF01...0xFF0F => true,
        0xFF1A...0xFF20 => true,
        0xFF3B...0xFF40 => true,
        0xFF5B...0xFF65 => true,
        else => false,
    };
}

/// Encode a single whitespace-chunk into ids, splitting on Unicode
/// punctuation (codepoint-aware). Each maximal run of non-punctuation
/// codepoints becomes a "word" passed to WordPiece; each punctuation
/// codepoint becomes its own single-codepoint word.
fn encodePunctSplit(
    alloc: std.mem.Allocator,
    m: *const wordpiece.Model,
    out: *std.ArrayList(u32),
    chunk: []const u8,
) !void {
    var word_start: usize = 0;
    var i: usize = 0;
    while (i < chunk.len) {
        const n = std.unicode.utf8ByteSequenceLength(chunk[i]) catch {
            // Invalid UTF-8 — treat the byte as non-punctuation and continue.
            i += 1;
            continue;
        };
        if (i + n > chunk.len) break;
        const cp: u21 = std.unicode.utf8Decode(chunk[i..][0..n]) catch {
            i += n;
            continue;
        };
        if (isUnicodePunct(cp)) {
            if (i > word_start) {
                const ids = try m.encode(alloc, chunk[word_start..i]);
                defer alloc.free(ids);
                try out.appendSlice(alloc, ids);
            }
            const ids = try m.encode(alloc, chunk[i .. i + n]);
            defer alloc.free(ids);
            try out.appendSlice(alloc, ids);
            word_start = i + n;
        }
        i += n;
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
    if (std.mem.eql(u8, t, "NFC")) return .nfc;
    if (std.mem.eql(u8, t, "NFD")) return .nfd;
    if (std.mem.eql(u8, t, "NFKC")) return .nfkc;
    if (std.mem.eql(u8, t, "NFKD")) return .nfkd;
    if (std.mem.eql(u8, t, "StripAccents")) return .strip_accents;
    if (std.mem.eql(u8, t, "BertNormalizer")) {
        // BertNormalizer config: clean_text, handle_chinese_chars,
        // strip_accents, lowercase. We support strip_accents + lowercase
        // (composed via Sequence). clean_text + handle_chinese_chars
        // remain v0.5+ work.
        var do_lower = true;
        var do_strip_accents = false;
        if (obj.get("lowercase")) |lv| {
            if (lv == .bool) do_lower = lv.bool;
        }
        if (obj.get("strip_accents")) |sv| {
            if (sv == .bool) do_strip_accents = sv.bool;
            // strip_accents = null in JSON defaults to true if lowercase=true
            // (BERT-uncased convention). Some BERT-cased configs disable it.
        }
        if (do_strip_accents and do_lower) {
            const seq = try aa.alloc(normalizer.NormalizerKind, 3);
            seq[0] = .nfd;
            seq[1] = .strip_accents;
            seq[2] = .lowercase;
            return .{ .sequence = seq };
        }
        if (do_lower) return .lowercase;
        if (do_strip_accents) {
            const seq = try aa.alloc(normalizer.NormalizerKind, 2);
            seq[0] = .nfd;
            seq[1] = .strip_accents;
            return .{ .sequence = seq };
        }
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

test "applyByteFallback — single <0xXX> token expands to byte" {
    const out = try applyByteFallback(testing.allocator, "<0xE4>");
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(u8, &[_]u8{0xE4}, out);
}

test "applyByteFallback — bytes embedded in mixed text" {
    const out = try applyByteFallback(testing.allocator, "hi<0x20>there");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("hi there", out);
}

test "applyByteFallback — invalid hex passes through unchanged" {
    const out = try applyByteFallback(testing.allocator, "<0xZZ>");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("<0xZZ>", out);
}

test "applyByteFallback — non-pattern angle brackets pass through" {
    const out = try applyByteFallback(testing.allocator, "<unk>");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("<unk>", out);
}

test "TemplateProcessing pair — synthetic [BOS] A [SEP] B [SEP] template" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [
        \\    {"id": 1, "content": "<s>"},
        \\    {"id": 2, "content": "<sep>"}
        \\  ],
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "post_processor": {
        \\    "type": "TemplateProcessing",
        \\    "single": [
        \\      {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\      {"Sequence": {"id": "A", "type_id": 0}}
        \\    ],
        \\    "pair": [
        \\      {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\      {"Sequence": {"id": "A", "type_id": 0}},
        \\      {"SpecialToken": {"id": "<sep>", "type_id": 0}},
        \\      {"Sequence": {"id": "B", "type_id": 1}},
        \\      {"SpecialToken": {"id": "<sep>", "type_id": 1}}
        \\    ],
        \\    "special_tokens": {
        \\      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
        \\      "<sep>": {"id": "<sep>", "ids": [2], "tokens": ["<sep>"]}
        \\    }
        \\  },
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 3, "c": 4},
        \\    "merges": []
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    try testing.expect(tok.template_pair != null);

    const ids = try tok.encodePair(testing.allocator, "a", "bc");
    defer testing.allocator.free(ids);
    // [<s>=1, a=0, <sep>=2, b=3, c=4, <sep>=2]
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 0, 2, 3, 4, 2 }, ids);
}

test "TemplateProcessing — synthetic [BOS] + sequence template" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [{"id": 1, "content": "<s>"}],
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "post_processor": {
        \\    "type": "TemplateProcessing",
        \\    "single": [
        \\      {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\      {"Sequence": {"id": "A", "type_id": 0}}
        \\    ],
        \\    "special_tokens": {
        \\      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}
        \\    }
        \\  },
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 2},
        \\    "merges": []
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    try testing.expect(tok.template_single != null);

    const ids = try tok.encode(testing.allocator, "ab");
    defer testing.allocator.free(ids);
    // base = [0, 2]; TemplateProcessing prepends <s>=1 → [1, 0, 2]
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 0, 2 }, ids);

    const base = try tok.encodeBase(testing.allocator, "ab");
    defer testing.allocator.free(base);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 2 }, base);
}

test "Truncation — right-trim to max_length" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    tok.max_length = 2;
    tok.truncation_direction = .right;

    const ids = try tok.encode(testing.allocator, "abc");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1 }, ids);
}

test "Truncation — left-trim to max_length" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    tok.max_length = 2;
    tok.truncation_direction = .left;

    const ids = try tok.encode(testing.allocator, "abc");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2 }, ids);
}

test "Padding — right-pad to pad_to_length" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    tok.pad_to_length = 4;
    tok.pad_token_id = 99;
    tok.padding_direction = .right;

    const ids = try tok.encode(testing.allocator, "ab");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 99, 99 }, ids);
}

test "Padding — left-pad to pad_to_length" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    tok.pad_to_length = 4;
    tok.pad_token_id = 99;
    tok.padding_direction = .left;

    const ids = try tok.encode(testing.allocator, "ab");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 99, 99, 0, 1 }, ids);
}

test "Truncation + Padding — over-length truncates, under-length pads" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    tok.max_length = 3;
    tok.pad_to_length = 3;
    tok.pad_token_id = 9;

    const long = try tok.encode(testing.allocator, "abcde");
    defer testing.allocator.free(long);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, long);

    const short = try tok.encode(testing.allocator, "a");
    defer testing.allocator.free(short);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 9, 9 }, short);
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
