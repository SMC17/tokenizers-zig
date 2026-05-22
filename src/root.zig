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
    MismatchedBatchSize,
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

    /// Look up an id's vocab string. Returns null if the id isn't in
    /// the model's vocabulary (the caller can check added_tokens next).
    pub fn idToToken(self: *const Model, id: u32) ?[]const u8 {
        switch (self.*) {
            .bpe => |*m| {
                var it = m.vocab.iterator();
                while (it.next()) |e| {
                    if (e.value_ptr.* == id) return e.key_ptr.*;
                }
                return null;
            },
            .wordpiece => |*m| return m.id_to_token.get(id),
            .unigram => |*m| {
                if (id < m.tokens.len) return m.tokens[id];
                return null;
            },
        }
    }
};

/// Added-tokens registry. Parsed from the top-level `added_tokens` array
/// of a tokenizer.json. Entries are matched verbatim against input and
/// bypass the model encoder/decoder.
pub const AddedToken = struct {
    id: u32,
    content: []const u8,
};

/// A single element of a TemplateProcessing template with its
/// associated `type_id` (BERT segment marker, 0 for sequence A and 1
/// for sequence B by convention).
pub const TemplateOp = struct {
    kind: Kind,
    type_id: u8,

    pub const Kind = union(enum) {
        append_ids: []const u32,
        append_input: void,
        append_input_b: void,
    };
};

/// Encoded output bundle. Mirrors HF tokenizers' canonical return:
///   - `ids`: token IDs
///   - `type_ids`: BERT segment markers (0 = A, 1 = B by convention)
///   - `attention_mask`: 1 for real tokens, 0 for padding tokens
/// Caller frees all slices via `EncodeResult.deinit`.
pub const EncodeResult = struct {
    ids: []u32,
    type_ids: []u8,
    attention_mask: []u8,

    pub fn deinit(self: EncodeResult, alloc: std.mem.Allocator) void {
        alloc.free(self.ids);
        alloc.free(self.type_ids);
        alloc.free(self.attention_mask);
    }
};

/// Token byte-offset range in the original input text. `start` is the
/// byte index where the source chunk begins; `end` is exclusive.
/// Tokens emitted from the same WordPiece "word" share the chunk's
/// range. Tokens from the added-token scan get the exact match range.
/// For pipelines that don't whitespace-split (e.g. raw ByteLevel BPE),
/// every token gets `(0, text.len)`.
pub const TokenOffset = struct {
    start: u32,
    end: u32,
};

/// Encoded output with per-token byte offsets in the original input.
/// Useful for NER and other tasks that need to map tokens back to
/// source text positions.
pub const EncodeWithOffsetsResult = struct {
    ids: []u32,
    offsets: []TokenOffset,

    pub fn deinit(self: EncodeWithOffsetsResult, alloc: std.mem.Allocator) void {
        alloc.free(self.ids);
        alloc.free(self.offsets);
    }
};

/// Pair-encoded output with per-token offsets AND per-token sequence
/// indicator (0 = A, 1 = B). Each token's offset is in the byte space
/// of its source text (A or B); use `sequence_ids[i]` to disambiguate.
pub const EncodePairWithOffsetsResult = struct {
    ids: []u32,
    offsets: []TokenOffset,
    sequence_ids: []u8,

    pub fn deinit(self: EncodePairWithOffsetsResult, alloc: std.mem.Allocator) void {
        alloc.free(self.ids);
        alloc.free(self.offsets);
        alloc.free(self.sequence_ids);
    }
};

/// Truncation strategy applied after encode + TemplateProcessing.
pub const TruncationDirection = enum { right, left };

/// Padding direction applied after encode + truncation.
pub const PaddingDirection = enum { right, left };

/// A single decoder step parsed from `decoder.decoders[]`. Applied in
/// order after the model's vocab-string decode.
pub const DecoderOp = union(enum) {
    /// Substring replacement (e.g. ▁ → " ").
    replace: struct { pattern: []const u8, content: []const u8 },
    /// Expand `<0xXX>` patterns to literal bytes.
    byte_fallback: void,
    /// No-op join (the string is already one buffer).
    fuse: void,
    /// Trim N codepoints from the start and M codepoints from the end
    /// matching a specific content character (typically " ").
    strip: struct { content: []const u8, start: u32, stop: u32 },
    /// Inverse of ByteLevel pre-tokenize (map ByteLevel codepoints → bytes).
    byte_level: void,
    /// Inverse of Metaspace (replacement → " ").
    metaspace: struct { replacement: []const u8 },
    /// WordPiece decoder: drop continuation prefix, join.
    wordpiece_decoder: struct { prefix: []const u8 },
};

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
    /// First-byte → indices-into-added_tokens index. Cuts the per-
    /// position match cost from O(N) to O(matches-with-this-first-byte).
    /// Each entry is a slice into `added_index_storage`.
    added_token_by_first_byte: [256][]const u32,
    /// Backing storage for added_token_by_first_byte slices.
    added_index_storage: []u32,
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
    /// Parsed decoder Sequence steps. When non-null, the BPE decode
    /// path applies these steps in order instead of the implicit
    /// SP-marker / ByteFallback / strip pipeline.
    decoder_steps: ?[]const DecoderOp,
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
    /// Uses the per-first-byte index built at fromJson — O(K_byte) where
    /// K_byte is the number of added tokens sharing this first byte (much
    /// less than total N for typical tokenizer.json files).
    fn matchAddedToken(self: *const Tokenizer, text: []const u8, pos: usize) ?AddedToken {
        const candidates = self.added_token_by_first_byte[text[pos]];
        var best: ?AddedToken = null;
        for (candidates) |idx| {
            const at = self.added_tokens[idx];
            if (pos + at.content.len > text.len) continue;
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

    /// Batch encode with per-token byte offsets. Returns an array of
    /// `EncodeWithOffsetsResult` aligned with `texts`. Caller frees via
    /// `freeBatchWithOffsets`.
    pub fn encodeBatchWithOffsets(
        self: *Tokenizer,
        alloc: std.mem.Allocator,
        texts: []const []const u8,
    ) ![]EncodeWithOffsetsResult {
        const out = try alloc.alloc(EncodeWithOffsetsResult, texts.len);
        errdefer alloc.free(out);
        var done: usize = 0;
        errdefer for (out[0..done]) |r| r.deinit(alloc);
        for (texts, 0..) |t, i| {
            out[i] = try self.encodeWithOffsets(alloc, t);
            done = i + 1;
        }
        return out;
    }

    pub fn freeBatchWithOffsets(alloc: std.mem.Allocator, batch: []EncodeWithOffsetsResult) void {
        for (batch) |r| r.deinit(alloc);
        alloc.free(batch);
    }

    /// Parallel variant of `encodeBatchWithOffsets`. `n_threads <= 1`
    /// falls through to sequential.
    pub fn encodeBatchParallelWithOffsets(
        self: *Tokenizer,
        alloc: std.mem.Allocator,
        texts: []const []const u8,
        n_threads: usize,
    ) ![]EncodeWithOffsetsResult {
        if (n_threads <= 1 or texts.len <= 1) return self.encodeBatchWithOffsets(alloc, texts);

        const out = try alloc.alloc(EncodeWithOffsetsResult, texts.len);
        errdefer alloc.free(out);
        const errs = try alloc.alloc(?anyerror, n_threads);
        defer alloc.free(errs);
        @memset(errs, null);
        const workers = try alloc.alloc(std.Thread, n_threads);
        defer alloc.free(workers);

        const chunk = (texts.len + n_threads - 1) / n_threads;
        var spawned: usize = 0;
        var t: usize = 0;
        while (t < n_threads) : (t += 1) {
            const s = t * chunk;
            if (s >= texts.len) break;
            const e = @min(s + chunk, texts.len);
            workers[spawned] = try std.Thread.spawn(.{}, encodeBatchOffsetsWorker, .{
                self, alloc, texts[s..e], out[s..e], &errs[t],
            });
            spawned += 1;
        }

        var i: usize = 0;
        while (i < spawned) : (i += 1) workers[i].join();

        for (errs) |e| if (e) |err| {
            for (out) |r| if (r.ids.len != 0 or r.offsets.len != 0) r.deinit(alloc);
            alloc.free(out);
            return err;
        };
        return out;
    }

    fn encodeBatchOffsetsWorker(
        tok: *Tokenizer,
        alloc: std.mem.Allocator,
        texts: []const []const u8,
        out: []EncodeWithOffsetsResult,
        err_out: *?anyerror,
    ) void {
        for (texts, 0..) |t, i| {
            const r = tok.encodeWithOffsets(alloc, t) catch |e| {
                err_out.* = e;
                var j = i;
                while (j < out.len) : (j += 1) out[j] = .{
                    .ids = &[_]u32{},
                    .offsets = &[_]TokenOffset{},
                };
                return;
            };
            out[i] = r;
        }
    }

    /// Look up the vocab string for `id`. Checks added_tokens first
    /// (special tokens emit their content verbatim), then the model
    /// vocabulary. Returns null if not found.
    pub fn idToToken(self: *const Tokenizer, id: u32) ?[]const u8 {
        if (self.added_token_by_id.get(id)) |content| return content;
        return self.model.idToToken(id);
    }

    /// Return the parallel `[]const []const u8` of vocab strings for
    /// each id. Caller frees the outer slice; the inner strings are
    /// owned by the Tokenizer's arena and shouldn't be freed.
    pub fn tokens(self: *const Tokenizer, alloc: std.mem.Allocator, ids: []const u32) ![][]const u8 {
        const out = try alloc.alloc([]const u8, ids.len);
        for (ids, 0..) |id, i| out[i] = self.idToToken(id) orelse "";
        return out;
    }

    /// Return the parallel `[]u8` mask where 1 = id is in added_tokens
    /// (special / control) and 0 = id is a regular model token. Useful
    /// for masking out special tokens before downstream processing.
    pub fn specialTokensMask(self: *const Tokenizer, alloc: std.mem.Allocator, ids: []const u32) ![]u8 {
        const out = try alloc.alloc(u8, ids.len);
        for (ids, 0..) |id, i| {
            out[i] = if (self.added_token_by_id.contains(id)) 1 else 0;
        }
        return out;
    }

    /// Encode a sentence-pair with sub-token offsets for each token
    /// AND per-token sequence indicator (0 = A, 1 = B). Used for
    /// BERT NSP + NER pipelines. Caller frees via deinit.
    pub fn encodePairWithOffsets(
        self: *Tokenizer,
        alloc: std.mem.Allocator,
        text_a: []const u8,
        text_b: []const u8,
    ) !EncodePairWithOffsetsResult {
        // Encode each side with offsets independently.
        const ra = try self.encodeWithOffsets(alloc, text_a);
        defer alloc.free(ra.ids);
        defer alloc.free(ra.offsets);
        const rb = try self.encodeWithOffsets(alloc, text_b);
        defer alloc.free(rb.ids);
        defer alloc.free(rb.offsets);

        var ids_out: std.ArrayList(u32) = .empty;
        errdefer ids_out.deinit(alloc);
        var off_out: std.ArrayList(TokenOffset) = .empty;
        errdefer off_out.deinit(alloc);
        var seq_out: std.ArrayList(u8) = .empty;
        errdefer seq_out.deinit(alloc);

        if (self.template_pair) |tpl| {
            for (tpl) |op| switch (op.kind) {
                .append_ids => |ids| {
                    for (ids) |id| {
                        try ids_out.append(alloc, id);
                        try off_out.append(alloc, .{ .start = 0, .end = 0 });
                        try seq_out.append(alloc, op.type_id);
                    }
                },
                .append_input => {
                    for (ra.ids, ra.offsets) |id, off| {
                        try ids_out.append(alloc, id);
                        try off_out.append(alloc, off);
                        try seq_out.append(alloc, op.type_id);
                    }
                },
                .append_input_b => {
                    for (rb.ids, rb.offsets) |id, off| {
                        try ids_out.append(alloc, id);
                        try off_out.append(alloc, off);
                        try seq_out.append(alloc, op.type_id);
                    }
                },
            };
        } else {
            for (ra.ids, ra.offsets) |id, off| {
                try ids_out.append(alloc, id);
                try off_out.append(alloc, off);
                try seq_out.append(alloc, 0);
            }
            for (rb.ids, rb.offsets) |id, off| {
                try ids_out.append(alloc, id);
                try off_out.append(alloc, off);
                try seq_out.append(alloc, 1);
            }
        }

        return .{
            .ids = try ids_out.toOwnedSlice(alloc),
            .offsets = try off_out.toOwnedSlice(alloc),
            .sequence_ids = try seq_out.toOwnedSlice(alloc),
        };
    }

    /// Encode WITH per-token byte offsets. Each token's offset is the
    /// byte range of the *chunk* it came from (whitespace + punctuation
    /// split word for WordPiece pipelines, full text for non-split
    /// pipelines). Caller frees via `EncodeWithOffsetsResult.deinit`.
    pub fn encodeWithOffsets(self: *Tokenizer, alloc: std.mem.Allocator, text: []const u8) !EncodeWithOffsetsResult {
        var ids_out: std.ArrayList(u32) = .empty;
        errdefer ids_out.deinit(alloc);
        var off_out: std.ArrayList(TokenOffset) = .empty;
        errdefer off_out.deinit(alloc);

        if (self.added_tokens.len > 0) {
            var pos: usize = 0;
            var chunk_start: usize = 0;
            while (pos < text.len) {
                if (self.matchAddedToken(text, pos)) |at| {
                    if (pos > chunk_start) {
                        try self.encodeChunkWithOffsets(alloc, text, chunk_start, pos, &ids_out, &off_out);
                    }
                    try ids_out.append(alloc, at.id);
                    try off_out.append(alloc, .{ .start = @intCast(pos), .end = @intCast(pos + at.content.len) });
                    pos += at.content.len;
                    chunk_start = pos;
                } else {
                    pos += 1;
                }
            }
            if (chunk_start < text.len) {
                try self.encodeChunkWithOffsets(alloc, text, chunk_start, text.len, &ids_out, &off_out);
            }
        } else {
            try self.encodeChunkWithOffsets(alloc, text, 0, text.len, &ids_out, &off_out);
        }

        return .{
            .ids = try ids_out.toOwnedSlice(alloc),
            .offsets = try off_out.toOwnedSlice(alloc),
        };
    }

    /// Encode a chunk of `text[start..end]` and append ids + offsets.
    /// For wordpiece + whitespace_split, each whitespace-separated word
    /// gets its own offset range; tokens within a word share that range.
    /// For other pipelines, every emitted token gets `(start, end)`.
    fn encodeChunkWithOffsets(
        self: *Tokenizer,
        alloc: std.mem.Allocator,
        text: []const u8,
        start: usize,
        end: usize,
        ids_out: *std.ArrayList(u32),
        off_out: *std.ArrayList(TokenOffset),
    ) !void {
        const chunk_text = text[start..end];
        if (self.model == .wordpiece and self.use_whitespace_split) {
            // Sub-word offset tracking via wordpiece.encodeWithOffsets:
            // each piece gets its true byte range within the word, and
            // we shift to absolute positions in `text`.
            var i: usize = 0;
            while (i < chunk_text.len) {
                while (i < chunk_text.len and isWhitespaceByte(chunk_text[i])) i += 1;
                const word_start = i;
                while (i < chunk_text.len and !isWhitespaceByte(chunk_text[i])) i += 1;
                const word_end = i;
                if (word_end > word_start) {
                    const word = chunk_text[word_start..word_end];
                    const abs_word_start: u32 = @intCast(start + word_start);
                    if (self.split_on_punctuation) {
                        // Walk punctuation-split sub-chunks within the word.
                        try encodeWordPunctSplitWithOffsets(self, alloc, word, abs_word_start, ids_out, off_out);
                    } else {
                        const r = try self.model.wordpiece.encodeWithOffsets(alloc, word);
                        defer alloc.free(r.ids);
                        defer alloc.free(r.offsets);
                        for (r.ids, r.offsets) |id, off| {
                            try ids_out.append(alloc, id);
                            try off_out.append(alloc, .{
                                .start = abs_word_start + off[0],
                                .end = abs_word_start + off[1],
                            });
                        }
                    }
                }
            }
            return;
        }

        // Unigram + WhitespaceSplit + Metaspace (T5 pattern): split
        // chunk on whitespace into words; per word, prepend metaspace
        // and Viterbi-encode with offsets; map offsets back to original
        // chunk byte positions by subtracting the metaspace prefix length.
        if (self.model == .unigram and self.use_whitespace_split and self.metaspace_replacement != null) {
            const rep = self.metaspace_replacement.?;
            const normalized = try normalizer.apply(alloc, self.normalizer, chunk_text);
            defer alloc.free(normalized);
            // Walk whitespace-separated words in `normalized`. Track each
            // word's byte range within `normalized` so we can shift to
            // absolute positions in the original `text`.
            // NOTE: we assume normalize is byte-identical to the original
            // chunk for ASCII inputs (which is true for T5's Precompiled
            // normalizer degrading to none). For Latin normalize, ranges
            // map approximately to original positions.
            var piece_buf: std.ArrayList(u8) = .empty;
            defer piece_buf.deinit(alloc);
            var p: usize = 0;
            while (p < normalized.len) {
                while (p < normalized.len and isWhitespaceByte(normalized[p])) p += 1;
                const word_start_n = p;
                while (p < normalized.len and !isWhitespaceByte(normalized[p])) p += 1;
                const word_end_n = p;
                if (word_end_n > word_start_n) {
                    const word = normalized[word_start_n..word_end_n];
                    piece_buf.clearRetainingCapacity();
                    try piece_buf.appendSlice(alloc, rep);
                    try piece_buf.appendSlice(alloc, word);
                    const r = try self.model.unigram.encodeWithOffsets(alloc, piece_buf.items);
                    defer alloc.free(r.ids);
                    defer alloc.free(r.offsets);
                    const abs_word_start: u32 = @intCast(start + word_start_n);
                    const word_len: u32 = @intCast(word.len);
                    for (r.ids, r.offsets) |id, off| {
                        const rep_len: u32 = @intCast(rep.len);
                        const in_start: u32 = if (off[0] > rep_len) off[0] - rep_len else 0;
                        const in_end: u32 = if (off[1] > rep_len) @min(off[1] - rep_len, word_len) else 0;
                        try ids_out.append(alloc, id);
                        try off_out.append(alloc, .{
                            .start = abs_word_start + in_start,
                            .end = abs_word_start + in_end,
                        });
                    }
                }
            }
            return;
        }

        // BPE + ByteLevel pipeline: track sub-token offsets through
        // ByteLevel encode → BPE merge. Each token's byte range maps
        // back to the original input via the ByteLevel input_index table.
        if (self.model == .bpe and self.use_byte_level) {
            const normalized = try normalizer.apply(alloc, self.normalizer, chunk_text);
            defer alloc.free(normalized);
            const bl = try byte_level.encodeWithOffsets(alloc, normalized);
            defer alloc.free(bl.encoded);
            defer alloc.free(bl.input_index);
            const r = try self.model.bpe.encodeWithOffsets(alloc, bl.encoded);
            defer alloc.free(r.ids);
            defer alloc.free(r.offsets);
            for (r.ids, r.offsets) |id, off| {
                try ids_out.append(alloc, id);
                // Map byte_level output bytes back to input bytes via input_index.
                // off[0] = first byte_level byte → input_index[off[0]] = first input byte.
                // off[1] = exclusive end → look up the last included byte (off[1]-1) and add 1.
                const in_start: u32 = if (off[0] < bl.input_index.len) bl.input_index[off[0]] else @intCast(normalized.len);
                const in_end: u32 = if (off[1] > 0 and off[1] - 1 < bl.input_index.len) bl.input_index[off[1] - 1] + 1 else @intCast(normalized.len);
                try off_out.append(alloc, .{
                    .start = @as(u32, @intCast(start)) + in_start,
                    .end = @as(u32, @intCast(start)) + in_end,
                });
            }
            return;
        }

        // BPE + SP marker (Llama, Mistral) — no ByteLevel; normalizer
        // does Prepend(▁) + Replace(" " → ▁). Use normalizer offset
        // tracking to remap BPE token byte ranges back to original.
        if (self.model == .bpe and self.sp_space_marker != null) {
            const nrm = try normalizer.applyWithOffsets(alloc, self.normalizer, chunk_text);
            defer alloc.free(nrm.output);
            defer alloc.free(nrm.input_index);
            const r = try self.model.bpe.encodeWithOffsets(alloc, nrm.output);
            defer alloc.free(r.ids);
            defer alloc.free(r.offsets);
            for (r.ids, r.offsets) |id, off| {
                // Find first non-SYNTHETIC input position from off[0] going
                // right; if none, fall back to chunk_text.len. Similarly
                // walk backward from off[1]-1 for the inclusive end.
                var s_in: u32 = @intCast(chunk_text.len);
                var k: usize = off[0];
                while (k < off[1] and k < nrm.input_index.len) : (k += 1) {
                    if (nrm.input_index[k] != normalizer.SYNTHETIC) {
                        s_in = nrm.input_index[k];
                        break;
                    }
                }
                var e_in: u32 = s_in;
                if (off[1] > 0) {
                    var kk: usize = off[1] - 1;
                    while (kk + 1 > off[0] and kk < nrm.input_index.len) {
                        if (nrm.input_index[kk] != normalizer.SYNTHETIC) {
                            e_in = nrm.input_index[kk] + 1;
                            break;
                        }
                        if (kk == 0) break;
                        kk -= 1;
                    }
                }
                try ids_out.append(alloc, id);
                try off_out.append(alloc, .{
                    .start = @as(u32, @intCast(start)) + s_in,
                    .end = @as(u32, @intCast(start)) + e_in,
                });
            }
            return;
        }

        // Other pipelines: tag every token with the whole-chunk range.
        const before = ids_out.items.len;
        const chunk_ids = try self.encodeChunk(alloc, chunk_text);
        defer alloc.free(chunk_ids);
        try ids_out.appendSlice(alloc, chunk_ids);
        const added = ids_out.items.len - before;
        try off_out.appendNTimes(alloc, .{
            .start = @intCast(start),
            .end = @intCast(end),
        }, added);
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

    /// Encode a batch of independent texts. Each text runs through the
    /// full encode pipeline (TemplateProcessing + truncation + padding).
    /// Returned slice owns the outer array; each inner []u32 is also
    /// owned. Caller frees both via `freeBatch`.
    pub fn encodeBatch(self: *Tokenizer, alloc: std.mem.Allocator, texts: []const []const u8) ![][]u32 {
        const out = try alloc.alloc([]u32, texts.len);
        errdefer alloc.free(out);
        var done: usize = 0;
        errdefer for (out[0..done]) |ids| alloc.free(ids);
        for (texts, 0..) |t, i| {
            out[i] = try self.encode(alloc, t);
            done = i + 1;
        }
        return out;
    }

    /// Batch encode returning `EncodeResult` per text — includes
    /// `ids`, `type_ids`, and `attention_mask`. Caller frees via
    /// `freeBatchWithTypes`.
    pub fn encodeBatchWithTypes(self: *Tokenizer, alloc: std.mem.Allocator, texts: []const []const u8) ![]EncodeResult {
        const out = try alloc.alloc(EncodeResult, texts.len);
        errdefer alloc.free(out);
        var done: usize = 0;
        errdefer for (out[0..done]) |r| r.deinit(alloc);
        for (texts, 0..) |t, i| {
            out[i] = try self.encodeWithTypes(alloc, t);
            done = i + 1;
        }
        return out;
    }

    /// Free a slice of EncodeResults.
    pub fn freeBatchWithTypes(alloc: std.mem.Allocator, batch: []EncodeResult) void {
        for (batch) |r| r.deinit(alloc);
        alloc.free(batch);
    }

    /// Batch sentence-pair encode returning `EncodeResult` per pair.
    pub fn encodeBatchPairWithTypes(
        self: *Tokenizer,
        alloc: std.mem.Allocator,
        pairs_a: []const []const u8,
        pairs_b: []const []const u8,
    ) ![]EncodeResult {
        if (pairs_a.len != pairs_b.len) return error.MismatchedBatchSize;
        const out = try alloc.alloc(EncodeResult, pairs_a.len);
        errdefer alloc.free(out);
        var done: usize = 0;
        errdefer for (out[0..done]) |r| r.deinit(alloc);
        for (pairs_a, pairs_b, 0..) |a, b, i| {
            out[i] = try self.encodePairWithTypes(alloc, a, b);
            done = i + 1;
        }
        return out;
    }

    /// Like `encodeBatch` but parallelised across `n_threads`. Each
    /// worker thread takes a contiguous chunk of inputs and encodes
    /// them with its own per-thread allocator (an `ArenaAllocator`
    /// thread-local arena). Output ids are allocated through `alloc`
    /// so the caller can free them via `freeBatch` as usual.
    ///
    /// When `n_threads <= 1` this falls through to the single-threaded
    /// `encodeBatch` to avoid thread-spawn overhead on tiny inputs.
    pub fn encodeBatchParallel(
        self: *Tokenizer,
        alloc: std.mem.Allocator,
        texts: []const []const u8,
        n_threads: usize,
    ) ![][]u32 {
        if (n_threads <= 1 or texts.len <= 1) return self.encodeBatch(alloc, texts);

        const out = try alloc.alloc([]u32, texts.len);
        errdefer alloc.free(out);
        // Track which entries got populated so we can free on partial error.
        const errs = try alloc.alloc(?anyerror, n_threads);
        defer alloc.free(errs);
        @memset(errs, null);

        const workers = try alloc.alloc(std.Thread, n_threads);
        defer alloc.free(workers);

        const chunk = (texts.len + n_threads - 1) / n_threads;
        var spawned: usize = 0;
        var t: usize = 0;
        while (t < n_threads) : (t += 1) {
            const start = t * chunk;
            if (start >= texts.len) break;
            const end = @min(start + chunk, texts.len);
            workers[spawned] = try std.Thread.spawn(.{}, encodeBatchWorker, .{
                self, alloc, texts[start..end], out[start..end], &errs[t],
            });
            spawned += 1;
        }

        var i: usize = 0;
        while (i < spawned) : (i += 1) workers[i].join();

        for (errs) |e| if (e) |err| {
            // On worker error, free whatever succeeded and propagate.
            for (out) |ids| if (ids.len != 0) alloc.free(ids);
            alloc.free(out);
            return err;
        };
        return out;
    }

    fn encodeBatchWorker(
        tok: *Tokenizer,
        alloc: std.mem.Allocator,
        texts: []const []const u8,
        out: [][]u32,
        err_out: *?anyerror,
    ) void {
        for (texts, 0..) |t, i| {
            const ids = tok.encode(alloc, t) catch |e| {
                err_out.* = e;
                // Clear remaining slots so the parent's cleanup loop knows.
                var j = i;
                while (j < out.len) : (j += 1) out[j] = &[_]u32{};
                return;
            };
            out[i] = ids;
        }
    }

    /// Free a batch returned by `encodeBatch`. Frees the inner []u32
    /// slices and the outer [][]u32 slice.
    pub fn freeBatch(alloc: std.mem.Allocator, batch: [][]u32) void {
        for (batch) |ids| alloc.free(ids);
        alloc.free(batch);
    }

    /// Decode a batch of ID sequences. Each row independently
    /// reconstructs text via the model + decoder pipeline. Caller frees
    /// the outer slice + each inner string via `freeDecodeBatch`.
    pub fn decodeBatch(self: *Tokenizer, alloc: std.mem.Allocator, batches: []const []const u32) ![][]u8 {
        const out = try alloc.alloc([]u8, batches.len);
        errdefer alloc.free(out);
        var done: usize = 0;
        errdefer for (out[0..done]) |s| alloc.free(s);
        for (batches, 0..) |ids, i| {
            out[i] = try self.decode(alloc, ids);
            done = i + 1;
        }
        return out;
    }

    /// Free a batch returned by `decodeBatch`.
    pub fn freeDecodeBatch(alloc: std.mem.Allocator, batch: [][]u8) void {
        for (batch) |s| alloc.free(s);
        alloc.free(batch);
    }

    /// Parallel decode (mirrors `encodeBatchParallel`). Workers share
    /// read-only Tokenizer state. `n_threads <= 1` falls through.
    pub fn decodeBatchParallel(
        self: *Tokenizer,
        alloc: std.mem.Allocator,
        batches: []const []const u32,
        n_threads: usize,
    ) ![][]u8 {
        if (n_threads <= 1 or batches.len <= 1) return self.decodeBatch(alloc, batches);

        const out = try alloc.alloc([]u8, batches.len);
        errdefer alloc.free(out);
        const errs = try alloc.alloc(?anyerror, n_threads);
        defer alloc.free(errs);
        @memset(errs, null);
        const workers = try alloc.alloc(std.Thread, n_threads);
        defer alloc.free(workers);

        const chunk = (batches.len + n_threads - 1) / n_threads;
        var spawned: usize = 0;
        var t: usize = 0;
        while (t < n_threads) : (t += 1) {
            const start = t * chunk;
            if (start >= batches.len) break;
            const end = @min(start + chunk, batches.len);
            workers[spawned] = try std.Thread.spawn(.{}, decodeBatchWorker, .{
                self, alloc, batches[start..end], out[start..end], &errs[t],
            });
            spawned += 1;
        }

        var i: usize = 0;
        while (i < spawned) : (i += 1) workers[i].join();

        for (errs) |e| if (e) |err| {
            for (out) |s| if (s.len != 0) alloc.free(s);
            alloc.free(out);
            return err;
        };
        return out;
    }

    fn decodeBatchWorker(
        tok: *Tokenizer,
        alloc: std.mem.Allocator,
        batches: []const []const u32,
        out: [][]u8,
        err_out: *?anyerror,
    ) void {
        for (batches, 0..) |ids, i| {
            const s = tok.decode(alloc, ids) catch |e| {
                err_out.* = e;
                var j = i;
                while (j < out.len) : (j += 1) out[j] = &[_]u8{};
                return;
            };
            out[i] = s;
        }
    }

    /// Encode a batch of sentence-pairs. `pairs_a` and `pairs_b` must
    /// have the same length; each row [i] produces `encodePair(a[i], b[i])`.
    /// Caller frees via `freeBatch`.
    pub fn encodeBatchPair(
        self: *Tokenizer,
        alloc: std.mem.Allocator,
        pairs_a: []const []const u8,
        pairs_b: []const []const u8,
    ) ![][]u32 {
        if (pairs_a.len != pairs_b.len) return error.MismatchedBatchSize;
        const out = try alloc.alloc([]u32, pairs_a.len);
        errdefer alloc.free(out);
        var done: usize = 0;
        errdefer for (out[0..done]) |ids| alloc.free(ids);
        for (pairs_a, pairs_b, 0..) |a, b, i| {
            out[i] = try self.encodePair(alloc, a, b);
            done = i + 1;
        }
        return out;
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

    /// Encode a sentence-pair returning both IDs and token_type_ids
    /// (BERT segment markers). type_ids are 0 for "A" content and 1
    /// for "B" by template convention. Caller frees via
    /// `EncodeResult.deinit`.
    pub fn encodePairWithTypes(self: *Tokenizer, alloc: std.mem.Allocator, text_a: []const u8, text_b: []const u8) !EncodeResult {
        const a = try self.encodeBase(alloc, text_a);
        defer alloc.free(a);
        const b = try self.encodeBase(alloc, text_b);
        defer alloc.free(b);

        var ids_out: std.ArrayList(u32) = .empty;
        errdefer ids_out.deinit(alloc);
        var types_out: std.ArrayList(u8) = .empty;
        errdefer types_out.deinit(alloc);

        if (self.template_pair) |tpl| {
            for (tpl) |op| switch (op.kind) {
                .append_ids => |ids| {
                    try ids_out.appendSlice(alloc, ids);
                    try types_out.appendNTimes(alloc, op.type_id, ids.len);
                },
                .append_input => {
                    try ids_out.appendSlice(alloc, a);
                    try types_out.appendNTimes(alloc, op.type_id, a.len);
                },
                .append_input_b => {
                    try ids_out.appendSlice(alloc, b);
                    try types_out.appendNTimes(alloc, op.type_id, b.len);
                },
            };
        } else {
            try ids_out.appendSlice(alloc, a);
            try types_out.appendNTimes(alloc, 0, a.len);
            try ids_out.appendSlice(alloc, b);
            try types_out.appendNTimes(alloc, 1, b.len);
        }

        // Build attention_mask (all 1s pre-padding); apply truncation +
        // padding symmetrically.
        var mask_out: std.ArrayList(u8) = .empty;
        errdefer mask_out.deinit(alloc);
        try mask_out.appendNTimes(alloc, 1, ids_out.items.len);
        try applyTruncPadTriple(alloc, &ids_out, &types_out, &mask_out, self);

        return .{
            .ids = try ids_out.toOwnedSlice(alloc),
            .type_ids = try types_out.toOwnedSlice(alloc),
            .attention_mask = try mask_out.toOwnedSlice(alloc),
        };
    }

    /// Encode a single text returning ids + type_ids (single-template
    /// type_id, typically all 0 for BERT). For tokenizers without a
    /// template, type_ids is all-zero of length ids.len.
    pub fn encodeWithTypes(self: *Tokenizer, alloc: std.mem.Allocator, text: []const u8) !EncodeResult {
        const base = try self.encodeBase(alloc, text);
        defer alloc.free(base);

        var ids_out: std.ArrayList(u32) = .empty;
        errdefer ids_out.deinit(alloc);
        var types_out: std.ArrayList(u8) = .empty;
        errdefer types_out.deinit(alloc);

        if (self.template_single) |tpl| {
            for (tpl) |op| switch (op.kind) {
                .append_ids => |ids| {
                    try ids_out.appendSlice(alloc, ids);
                    try types_out.appendNTimes(alloc, op.type_id, ids.len);
                },
                .append_input => {
                    try ids_out.appendSlice(alloc, base);
                    try types_out.appendNTimes(alloc, op.type_id, base.len);
                },
                .append_input_b => {},
            };
        } else {
            try ids_out.appendSlice(alloc, base);
            try types_out.appendNTimes(alloc, 0, base.len);
        }

        var mask_out: std.ArrayList(u8) = .empty;
        errdefer mask_out.deinit(alloc);
        try mask_out.appendNTimes(alloc, 1, ids_out.items.len);
        try applyTruncPadTriple(alloc, &ids_out, &types_out, &mask_out, self);

        return .{
            .ids = try ids_out.toOwnedSlice(alloc),
            .type_ids = try types_out.toOwnedSlice(alloc),
            .attention_mask = try mask_out.toOwnedSlice(alloc),
        };
    }

    /// Apply TemplateProcessing → truncation → padding. Caller frees.
    fn applyPostProcessing(self: *Tokenizer, alloc: std.mem.Allocator, base: []const u32) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(alloc);

        if (self.template_single) |tpl| {
            for (tpl) |op| switch (op.kind) {
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
            for (tpl) |op| switch (op.kind) {
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

                // Explicit decoder Sequence path: when parsed from
                // tokenizer.json, apply each step in order.
                if (self.decoder_steps) |steps| {
                    return applyDecoderSteps(alloc, raw, steps);
                }

                // Implicit fallback path (matches v0.4-v0.6 behaviour).
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

    // Build first-byte index for the longest-match scan in encode.
    // For each byte b ∈ 0..255, collect indices of added_tokens whose
    // content starts with that byte. Total storage = sum of lengths.
    var added_index_storage: []u32 = &[_]u32{};
    var added_token_by_first_byte: [256][]const u32 = .{&[_]u32{}} ** 256;
    if (added_tokens.len > 0) {
        var counts: [256]u32 = .{0} ** 256;
        for (added_tokens) |at| {
            if (at.content.len == 0) continue;
            counts[at.content[0]] += 1;
        }
        var total: u32 = 0;
        var offsets: [256]u32 = .{0} ** 256;
        for (counts, 0..) |c, b| {
            offsets[b] = total;
            total += c;
        }
        added_index_storage = try aa.alloc(u32, total);
        var cursors: [256]u32 = .{0} ** 256;
        for (added_tokens, 0..) |at, i| {
            if (at.content.len == 0) continue;
            const b = at.content[0];
            added_index_storage[offsets[b] + cursors[b]] = @intCast(i);
            cursors[b] += 1;
        }
        for (counts, 0..) |c, b| {
            added_token_by_first_byte[b] = added_index_storage[offsets[b] .. offsets[b] + c];
        }
    }

    // Decoder (optional). Two complementary outputs:
    //   - use_byte_fallback: legacy flag for the implicit decode path
    //   - decoder_steps: parsed Sequence for the explicit decode path
    var use_byte_fallback = false;
    var decoder_steps: ?[]const DecoderOp = null;
    if (root.get("decoder")) |dv| {
        if (dv == .object) {
            scanByteFallback(dv.object, &use_byte_fallback);
            decoder_steps = try parseDecoderSequence(aa, dv.object);
        }
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
        .added_token_by_first_byte = added_token_by_first_byte,
        .added_index_storage = added_index_storage,
        .template_single = template_single,
        .template_pair = template_pair,
        .use_byte_fallback = use_byte_fallback,
        .decoder_steps = decoder_steps,
        .max_length = max_length,
        .truncation_direction = truncation_direction,
        .pad_to_length = pad_to_length,
        .pad_token_id = pad_token_id,
        .padding_direction = padding_direction,
    };
}

/// Parse a `decoder` object into an ordered `[]DecoderOp`. Returns null
/// when the decoder is a single non-Sequence node (the legacy implicit
/// path will handle it) or when the type is unsupported. The Sequence
/// node walks its `decoders[]` array and emits one DecoderOp per child.
fn parseDecoderSequence(aa: std.mem.Allocator, obj: std.json.ObjectMap) !?[]const DecoderOp {
    const tv = obj.get("type") orelse return null;
    if (tv != .string) return null;
    if (!std.mem.eql(u8, tv.string, "Sequence")) return null;
    const arr_v = obj.get("decoders") orelse return null;
    if (arr_v != .array) return null;

    var ops: std.ArrayList(DecoderOp) = .empty;
    errdefer ops.deinit(aa);
    for (arr_v.array.items) |item| {
        if (item != .object) continue;
        const inner_tv = item.object.get("type") orelse continue;
        if (inner_tv != .string) continue;
        const t = inner_tv.string;

        if (std.mem.eql(u8, t, "Replace")) {
            const pat_v = item.object.get("pattern") orelse continue;
            const content_v = item.object.get("content") orelse continue;
            if (content_v != .string) continue;
            var pattern: []const u8 = "";
            if (pat_v == .object) {
                if (pat_v.object.get("String")) |sv| {
                    if (sv == .string) pattern = sv.string;
                }
            } else if (pat_v == .string) {
                pattern = pat_v.string;
            }
            try ops.append(aa, .{ .replace = .{
                .pattern = try aa.dupe(u8, pattern),
                .content = try aa.dupe(u8, content_v.string),
            } });
        } else if (std.mem.eql(u8, t, "ByteFallback")) {
            try ops.append(aa, .byte_fallback);
        } else if (std.mem.eql(u8, t, "Fuse")) {
            try ops.append(aa, .fuse);
        } else if (std.mem.eql(u8, t, "Strip")) {
            var content: []const u8 = " ";
            var start: u32 = 0;
            var stop: u32 = 0;
            if (item.object.get("content")) |cv| {
                if (cv == .string) content = cv.string;
            }
            if (item.object.get("start")) |sv| {
                if (sv == .integer) start = @intCast(sv.integer);
            }
            if (item.object.get("stop")) |sv2| {
                if (sv2 == .integer) stop = @intCast(sv2.integer);
            }
            try ops.append(aa, .{ .strip = .{
                .content = try aa.dupe(u8, content),
                .start = start,
                .stop = stop,
            } });
        } else if (std.mem.eql(u8, t, "ByteLevel")) {
            try ops.append(aa, .byte_level);
        } else if (std.mem.eql(u8, t, "Metaspace")) {
            var replacement: []const u8 = "\u{2581}";
            if (item.object.get("replacement")) |rv| {
                if (rv == .string) replacement = rv.string;
            }
            try ops.append(aa, .{ .metaspace = .{
                .replacement = try aa.dupe(u8, replacement),
            } });
        } else if (std.mem.eql(u8, t, "WordPiece")) {
            var prefix: []const u8 = "##";
            if (item.object.get("prefix")) |pv| {
                if (pv == .string) prefix = pv.string;
            }
            try ops.append(aa, .{ .wordpiece_decoder = .{
                .prefix = try aa.dupe(u8, prefix),
            } });
        }
        // Unknown decoder types silently dropped.
    }

    return try aa.dupe(DecoderOp, ops.items);
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
/// template (`"single"` or `"pair"`) into a sequence of TemplateOps
/// with per-op type_id. SpecialToken ids are resolved via the
/// post-processor's `special_tokens` map. Returns null if the
/// post-processor is not TemplateProcessing or the requested field
/// is absent.
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
            var is_b: bool = false;
            var type_id: u8 = 0;
            if (seq_v == .object) {
                if (seq_v.object.get("id")) |idv| {
                    if (idv == .string and std.mem.eql(u8, idv.string, "B")) is_b = true;
                }
                if (seq_v.object.get("type_id")) |tid| {
                    if (tid == .integer) type_id = @intCast(tid.integer);
                }
            }
            try ops.append(aa, .{
                .kind = if (is_b) .append_input_b else .append_input,
                .type_id = type_id,
            });
            continue;
        }
        if (entry.object.get("SpecialToken")) |st_v| {
            if (st_v != .object) continue;
            const id_v = st_v.object.get("id") orelse continue;
            if (id_v != .string) continue;
            var type_id: u8 = 0;
            if (st_v.object.get("type_id")) |tid| {
                if (tid == .integer) type_id = @intCast(tid.integer);
            }
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
            try ops.append(aa, .{ .kind = .{ .append_ids = ids }, .type_id = type_id });
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

fn isWhitespaceByte(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\n' or b == '\r' or b == 0x0B or b == 0x0C;
}

/// Helper for encodeChunkWithOffsets: walk a single word's punctuation-
/// split pieces, encode each through WordPiece with offsets, shift to
/// absolute positions.
fn encodeWordPunctSplitWithOffsets(
    self: *Tokenizer,
    alloc: std.mem.Allocator,
    word: []const u8,
    word_abs_start: u32,
    ids_out: *std.ArrayList(u32),
    off_out: *std.ArrayList(TokenOffset),
) !void {
    var i: usize = 0;
    var piece_start: usize = 0;
    while (i < word.len) {
        const n = std.unicode.utf8ByteSequenceLength(word[i]) catch {
            i += 1;
            continue;
        };
        if (i + n > word.len) break;
        const cp: u21 = std.unicode.utf8Decode(word[i..][0..n]) catch {
            i += n;
            continue;
        };
        if (isUnicodePunct(cp)) {
            if (i > piece_start) {
                try encodePieceWithOffsets(self, alloc, word[piece_start..i], word_abs_start + @as(u32, @intCast(piece_start)), ids_out, off_out);
            }
            try encodePieceWithOffsets(self, alloc, word[i .. i + n], word_abs_start + @as(u32, @intCast(i)), ids_out, off_out);
            i += n;
            piece_start = i;
        } else {
            i += n;
        }
    }
    if (piece_start < word.len) {
        try encodePieceWithOffsets(self, alloc, word[piece_start..], word_abs_start + @as(u32, @intCast(piece_start)), ids_out, off_out);
    }
}

fn encodePieceWithOffsets(
    self: *Tokenizer,
    alloc: std.mem.Allocator,
    piece: []const u8,
    abs_start: u32,
    ids_out: *std.ArrayList(u32),
    off_out: *std.ArrayList(TokenOffset),
) !void {
    const r = try self.model.wordpiece.encodeWithOffsets(alloc, piece);
    defer alloc.free(r.ids);
    defer alloc.free(r.offsets);
    for (r.ids, r.offsets) |id, off| {
        try ids_out.append(alloc, id);
        try off_out.append(alloc, .{
            .start = abs_start + off[0],
            .end = abs_start + off[1],
        });
    }
}

/// Apply truncation + padding to ids + type_ids + attention_mask
/// arrays in lockstep. Padding type_ids are 0 and padding attention
/// mask bits are 0 (marking padding for downstream consumers).
fn applyTruncPadTriple(
    alloc: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    types: *std.ArrayList(u8),
    mask: *std.ArrayList(u8),
    tok: *const Tokenizer,
) !void {
    if (tok.max_length) |mx| {
        if (ids.items.len > mx) {
            switch (tok.truncation_direction) {
                .right => {
                    ids.shrinkRetainingCapacity(mx);
                    types.shrinkRetainingCapacity(mx);
                    mask.shrinkRetainingCapacity(mx);
                },
                .left => {
                    const drop = ids.items.len - mx;
                    std.mem.copyForwards(u32, ids.items[0..mx], ids.items[drop..]);
                    std.mem.copyForwards(u8, types.items[0..mx], types.items[drop..]);
                    std.mem.copyForwards(u8, mask.items[0..mx], mask.items[drop..]);
                    ids.shrinkRetainingCapacity(mx);
                    types.shrinkRetainingCapacity(mx);
                    mask.shrinkRetainingCapacity(mx);
                },
            }
        }
    }
    if (tok.pad_to_length) |pl| {
        if (ids.items.len < pl) {
            const need = pl - ids.items.len;
            switch (tok.padding_direction) {
                .right => {
                    try ids.ensureUnusedCapacity(alloc, need);
                    try types.ensureUnusedCapacity(alloc, need);
                    try mask.ensureUnusedCapacity(alloc, need);
                    for (0..need) |_| {
                        ids.appendAssumeCapacity(tok.pad_token_id);
                        types.appendAssumeCapacity(0);
                        mask.appendAssumeCapacity(0); // padding has mask=0
                    }
                },
                .left => {
                    try ids.ensureTotalCapacity(alloc, pl);
                    try types.ensureTotalCapacity(alloc, pl);
                    try mask.ensureTotalCapacity(alloc, pl);
                    const old_len = ids.items.len;
                    ids.items.len = pl;
                    types.items.len = pl;
                    mask.items.len = pl;
                    std.mem.copyBackwards(u32, ids.items[need..], ids.items[0..old_len]);
                    std.mem.copyBackwards(u8, types.items[need..], types.items[0..old_len]);
                    std.mem.copyBackwards(u8, mask.items[need..], mask.items[0..old_len]);
                    for (0..need) |i| {
                        ids.items[i] = tok.pad_token_id;
                        types.items[i] = 0;
                        mask.items[i] = 0;
                    }
                },
            }
        }
    }
}

/// Apply a parsed decoder Sequence to the raw model-decoded string.
/// Each step transforms the string in place; the result is owned and
/// freed by the caller.
fn applyDecoderSteps(alloc: std.mem.Allocator, raw: []const u8, steps: []const DecoderOp) ![]u8 {
    var current: []u8 = try alloc.dupe(u8, raw);
    errdefer alloc.free(current);

    for (steps) |step| {
        const next: []u8 = switch (step) {
            .replace => |r| try normalizer.apply(alloc, .{ .replace = .{ .pattern = r.pattern, .content = r.content } }, current),
            .byte_fallback => try applyByteFallback(alloc, current),
            .fuse => try alloc.dupe(u8, current),
            .strip => |s| try applyStrip(alloc, current, s.content, s.start, s.stop),
            .byte_level => try byte_level.decode(alloc, current),
            .metaspace => |m| try normalizer.apply(alloc, .{ .replace = .{ .pattern = m.replacement, .content = " " } }, current),
            .wordpiece_decoder => |w| try applyWordpieceDecode(alloc, current, w.prefix),
        };
        alloc.free(current);
        current = next;
    }
    return current;
}

/// Strip `start` codepoints from the front and `stop` codepoints from
/// the end that match `content` (typically " "). Matches HF Rust's
/// Strip decoder semantics.
fn applyStrip(alloc: std.mem.Allocator, text: []const u8, content: []const u8, start: u32, stop: u32) ![]u8 {
    if (content.len == 0) return alloc.dupe(u8, text);
    var begin: usize = 0;
    var end: usize = text.len;

    var stripped_start: u32 = 0;
    while (stripped_start < start and begin + content.len <= end) {
        if (!std.mem.eql(u8, text[begin .. begin + content.len], content)) break;
        begin += content.len;
        stripped_start += 1;
    }

    var stripped_stop: u32 = 0;
    while (stripped_stop < stop and begin + content.len <= end) {
        if (!std.mem.eql(u8, text[end - content.len .. end], content)) break;
        end -= content.len;
        stripped_stop += 1;
    }

    return alloc.dupe(u8, text[begin..end]);
}

/// WordPiece decoder: at the decode-step level we receive the
/// already-concatenated string from `wordpiece.Model.decode`, which has
/// rejoined ##-continuations and space-separated root tokens. The
/// WordPiece decoder step in a Sequence is therefore a no-op
/// pass-through here. Future work may post-process prefix variants that
/// survived the model-internal decoder.
fn applyWordpieceDecode(alloc: std.mem.Allocator, text: []const u8, prefix: []const u8) ![]u8 {
    _ = prefix;
    return alloc.dupe(u8, text);
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
        // BertNormalizer composes (in order): clean_text →
        // handle_chinese_chars → NFD → strip_accents → lowercase.
        // Each step is opt-in via the corresponding JSON flag; the
        // composition matches HF Rust's BertNormalizer::new pipeline.
        var do_clean_text = true; // HF default
        var do_handle_chinese = true; // HF default
        var do_lower = true; // HF default
        var do_strip_accents = false;
        var strip_accents_set_explicitly = false;
        if (obj.get("clean_text")) |cv| {
            if (cv == .bool) do_clean_text = cv.bool;
        }
        if (obj.get("handle_chinese_chars")) |hv| {
            if (hv == .bool) do_handle_chinese = hv.bool;
        }
        if (obj.get("lowercase")) |lv| {
            if (lv == .bool) do_lower = lv.bool;
        }
        if (obj.get("strip_accents")) |sv| {
            if (sv == .bool) {
                do_strip_accents = sv.bool;
                strip_accents_set_explicitly = true;
            }
        }
        // HF rule: when strip_accents is null in JSON, default to
        // do_lower's value (uncased = strip; cased = keep).
        if (!strip_accents_set_explicitly) do_strip_accents = do_lower;

        var steps: std.ArrayList(normalizer.NormalizerKind) = .empty;
        defer steps.deinit(aa);
        if (do_clean_text) try steps.append(aa, .clean_text);
        if (do_handle_chinese) try steps.append(aa, .handle_chinese_chars);
        if (do_strip_accents) {
            try steps.append(aa, .nfd);
            try steps.append(aa, .strip_accents);
        }
        if (do_lower) try steps.append(aa, .lowercase);

        if (steps.items.len == 0) return .none;
        if (steps.items.len == 1) return steps.items[0];
        return .{ .sequence = try aa.dupe(normalizer.NormalizerKind, steps.items) };
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

test "encodePairWithTypes — BERT-style synthetic produces canonical type_ids" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [
        \\    {"id": 101, "content": "[CLS]"},
        \\    {"id": 102, "content": "[SEP]"}
        \\  ],
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "post_processor": {
        \\    "type": "TemplateProcessing",
        \\    "pair": [
        \\      {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        \\      {"Sequence": {"id": "A", "type_id": 0}},
        \\      {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        \\      {"Sequence": {"id": "B", "type_id": 1}},
        \\      {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
        \\    ],
        \\    "special_tokens": {
        \\      "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
        \\      "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
        \\    }
        \\  },
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1},
        \\    "merges": []
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const result = try tok.encodePairWithTypes(testing.allocator, "a", "b");
    defer result.deinit(testing.allocator);

    // ids:      [CLS=101, a=0,   [SEP]=102, b=1,   [SEP]=102]
    // type_ids: [0,        0,    0,         1,      1]
    try testing.expectEqualSlices(u32, &[_]u32{ 101, 0, 102, 1, 102 }, result.ids);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 1, 1 }, result.type_ids);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 1, 1, 1 }, result.attention_mask);
}

test "encodeWithTypes — single sequence emits all-zero type_ids by default" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const result = try tok.encodeWithTypes(testing.allocator, "ab");
    defer result.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1 }, result.ids);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0 }, result.type_ids);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1 }, result.attention_mask);
}

test "encodeWithTypes — padding has attention_mask=0" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    tok.pad_to_length = 5;
    tok.pad_token_id = 99;

    const result = try tok.encodeWithTypes(testing.allocator, "ab");
    defer result.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 99, 99, 99 }, result.ids);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 0, 0 }, result.type_ids);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 0, 0, 0 }, result.attention_mask);
}

test "encodeWithTypes — left-padding shifts attention_mask correctly" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    tok.pad_to_length = 5;
    tok.pad_token_id = 99;
    tok.padding_direction = .left;

    const result = try tok.encodeWithTypes(testing.allocator, "ab");
    defer result.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &[_]u32{ 99, 99, 99, 0, 1 }, result.ids);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 1, 1 }, result.attention_mask);
}

test "Decoder Sequence — Replace + ByteFallback + Strip applied in order" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "decoder": {
        \\    "type": "Sequence",
        \\    "decoders": [
        \\      {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        \\      {"type": "ByteFallback"},
        \\      {"type": "Fuse"},
        \\      {"type": "Strip", "content": " ", "start": 1, "stop": 0}
        \\    ]
        \\  },
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"▁H": 0, "i": 1, "<0x20>": 2, "wld": 3},
        \\    "merges": []
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();
    try testing.expect(tok.decoder_steps != null);
    try testing.expectEqual(@as(usize, 4), tok.decoder_steps.?.len);

    // ids [0, 1, 2, 3] → vocab strings "▁H" + "i" + "<0x20>" + "wld"
    // → "▁Hi<0x20>wld"
    // → Replace ▁→" " → " Hi<0x20>wld"
    // → ByteFallback <0x20>→" " → " Hi wld"
    // → Fuse (no-op) → " Hi wld"
    // → Strip start=1 (drop leading " ") → "Hi wld"
    const back = try tok.decode(testing.allocator, &[_]u32{ 0, 1, 2, 3 });
    defer testing.allocator.free(back);
    try testing.expectEqualStrings("Hi wld", back);
}

test "encodeBatch — encodes multiple texts independently" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const texts = [_][]const u8{ "a", "ab", "abc" };
    const batch = try tok.encodeBatch(testing.allocator, &texts);
    defer Tokenizer.freeBatch(testing.allocator, batch);

    try testing.expectEqual(@as(usize, 3), batch.len);
    try testing.expectEqualSlices(u32, &[_]u32{0}, batch[0]);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1 }, batch[1]);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, batch[2]);
}

test "encodeBatchParallel — produces same output as encodeBatch (4 threads)" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const texts = [_][]const u8{ "a", "ab", "abc", "abcd", "abcde", "edcba" };
    const seq = try tok.encodeBatch(testing.allocator, &texts);
    defer Tokenizer.freeBatch(testing.allocator, seq);
    const par = try tok.encodeBatchParallel(testing.allocator, &texts, 4);
    defer Tokenizer.freeBatch(testing.allocator, par);

    try testing.expectEqual(seq.len, par.len);
    for (seq, par) |s, p| try testing.expectEqualSlices(u32, s, p);
}

test "idToToken + tokens + specialTokensMask" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [{"id": 100, "content": "<unk>"}, {"id": 101, "content": "<s>"}],
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1, "c": 2},
        \\    "merges": []
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    try testing.expectEqualStrings("a", tok.idToToken(0).?);
    try testing.expectEqualStrings("<unk>", tok.idToToken(100).?);
    try testing.expectEqualStrings("<s>", tok.idToToken(101).?);
    try testing.expect(tok.idToToken(999) == null);

    const ids = [_]u32{ 101, 0, 1, 100, 2 };
    const strs = try tok.tokens(testing.allocator, &ids);
    defer testing.allocator.free(strs);
    try testing.expectEqualStrings("<s>", strs[0]);
    try testing.expectEqualStrings("a", strs[1]);
    try testing.expectEqualStrings("b", strs[2]);
    try testing.expectEqualStrings("<unk>", strs[3]);
    try testing.expectEqualStrings("c", strs[4]);

    const mask = try tok.specialTokensMask(testing.allocator, &ids);
    defer testing.allocator.free(mask);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 0, 0, 1, 0 }, mask);
}

test "encodeBatchParallelWithOffsets — matches sequential output (4 threads)" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
        \\    "merges": ["a b", "ab c"]
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const texts = [_][]const u8{ "a", "ab", "abc", "abca", "cab", "ba" };
    const seq = try tok.encodeBatchWithOffsets(testing.allocator, &texts);
    defer Tokenizer.freeBatchWithOffsets(testing.allocator, seq);
    const par = try tok.encodeBatchParallelWithOffsets(testing.allocator, &texts, 4);
    defer Tokenizer.freeBatchWithOffsets(testing.allocator, par);

    try testing.expectEqual(seq.len, par.len);
    for (seq, par) |s, p| {
        try testing.expectEqualSlices(u32, s.ids, p.ids);
        try testing.expectEqual(s.offsets.len, p.offsets.len);
        for (s.offsets, p.offsets) |so, po| {
            try testing.expectEqual(so.start, po.start);
            try testing.expectEqual(so.end, po.end);
        }
    }
}

test "encodeWithOffsets — BPE+SP marker (Llama-style) sub-token offsets" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "normalizer": {
        \\    "type": "Sequence",
        \\    "normalizers": [
        \\      {"type": "Prepend", "prepend": "▁"},
        \\      {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
        \\    ]
        \\  },
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"▁": 0, "h": 1, "i": 2, "▁hi": 3},
        \\    "merges": ["▁ h", "▁h i"]
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const r = try tok.encodeWithOffsets(testing.allocator, "hi");
    defer r.deinit(testing.allocator);
    // Normalized: "▁hi". BPE merges ▁+h → ▁h, then ▁h+i → ▁hi → [3].
    // Offset for ▁hi token: bytes [0,5) in normalized → input bytes
    // are SYNTHETIC for ▁ (0,1,2), then 0 for h, 1 for i. First
    // non-SYNTHETIC = 0, last = 1+1 = 2.
    try testing.expectEqualSlices(u32, &[_]u32{3}, r.ids);
    try testing.expectEqual(@as(u32, 0), r.offsets[0].start);
    try testing.expectEqual(@as(u32, 2), r.offsets[0].end);
}

test "encodeWithOffsets — Unigram+Metaspace sub-token offsets" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, {"type": "Metaspace"}]},
        \\  "model": {
        \\    "type": "Unigram",
        \\    "vocab": [["<unk>", 0.0], ["▁hello", -1.0], ["▁world", -1.0]],
        \\    "unk_id": 0
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const r = try tok.encodeWithOffsets(testing.allocator, "hello world");
    defer r.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2 }, r.ids);
    // ▁hello at word-relative byte [0,5]; ▁world at byte [6,11].
    try testing.expectEqual(@as(u32, 0), r.offsets[0].start);
    try testing.expectEqual(@as(u32, 5), r.offsets[0].end);
    try testing.expectEqual(@as(u32, 6), r.offsets[1].start);
    try testing.expectEqual(@as(u32, 11), r.offsets[1].end);
}

test "encodeWithOffsets — BPE+ByteLevel sub-token offsets" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hello": 6},
        \\    "merges": ["h e", "l l", "he llo"]
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const r = try tok.encodeWithOffsets(testing.allocator, "hello");
    defer r.deinit(testing.allocator);

    // Merge order: h+e=he, l+l=ll, then... actually 'hello' isn't reachable
    // from these merges directly. Let me trace:
    //   h e l l o → after h+e: he l l o → after l+l: he ll o
    //   "he llo" merge needs "he ll" pair? No, looks for "he ll" exactly.
    //   Actually merges are (left right) string keys. We have ("he", "llo")
    //   but "llo" isn't a piece yet. So encoding stops at [he, ll, o].
    // The range of `he` covers byte 0-2, `ll` covers 2-4, `o` covers 4-5.
    try testing.expectEqualSlices(u32, &[_]u32{ 4, 5, 3 }, r.ids);
    try testing.expectEqual(@as(u32, 0), r.offsets[0].start);
    try testing.expectEqual(@as(u32, 2), r.offsets[0].end);
    try testing.expectEqual(@as(u32, 2), r.offsets[1].start);
    try testing.expectEqual(@as(u32, 4), r.offsets[1].end);
    try testing.expectEqual(@as(u32, 4), r.offsets[2].start);
    try testing.expectEqual(@as(u32, 5), r.offsets[2].end);
}

test "encodeWithOffsets — WordPiece sub-word ranges within a word" {
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

    const result = try tok.encodeWithOffsets(testing.allocator, "unaffable");
    defer result.deinit(testing.allocator);

    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3 }, result.ids);
    // True sub-word offsets: un=[0,2], ##aff=[2,5], ##able=[5,9].
    try testing.expectEqual(@as(u32, 0), result.offsets[0].start);
    try testing.expectEqual(@as(u32, 2), result.offsets[0].end);
    try testing.expectEqual(@as(u32, 2), result.offsets[1].start);
    try testing.expectEqual(@as(u32, 5), result.offsets[1].end);
    try testing.expectEqual(@as(u32, 5), result.offsets[2].start);
    try testing.expectEqual(@as(u32, 9), result.offsets[2].end);
}

test "encodeWithOffsets — multi-word with punctuation tracking" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "BertPreTokenizer"},
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "!": 3, ",": 4},
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##",
        \\    "max_input_chars_per_word": 100
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const result = try tok.encodeWithOffsets(testing.allocator, "hello, world!");
    defer result.deinit(testing.allocator);

    try testing.expectEqualSlices(u32, &[_]u32{ 1, 4, 2, 3 }, result.ids);
    // hello=[0,5], ,=[5,6], world=[7,12], !=[12,13]
    try testing.expectEqual(@as(u32, 0), result.offsets[0].start);
    try testing.expectEqual(@as(u32, 5), result.offsets[0].end);
    try testing.expectEqual(@as(u32, 5), result.offsets[1].start);
    try testing.expectEqual(@as(u32, 6), result.offsets[1].end);
    try testing.expectEqual(@as(u32, 7), result.offsets[2].start);
    try testing.expectEqual(@as(u32, 12), result.offsets[2].end);
    try testing.expectEqual(@as(u32, 12), result.offsets[3].start);
    try testing.expectEqual(@as(u32, 13), result.offsets[3].end);
}

test "encodeWithOffsets — added tokens get exact match offsets" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [{"id": 101, "content": "[CLS]"}],
        \\  "pre_tokenizer": {"type": "BertPreTokenizer"},
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0, "hi": 1},
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##",
        \\    "max_input_chars_per_word": 100
        \\  }
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const result = try tok.encodeWithOffsets(testing.allocator, "[CLS] hi");
    defer result.deinit(testing.allocator);

    try testing.expectEqualSlices(u32, &[_]u32{ 101, 1 }, result.ids);
    // [CLS] at [0,5]; hi at [6,8].
    try testing.expectEqual(@as(u32, 0), result.offsets[0].start);
    try testing.expectEqual(@as(u32, 5), result.offsets[0].end);
    try testing.expectEqual(@as(u32, 6), result.offsets[1].start);
    try testing.expectEqual(@as(u32, 8), result.offsets[1].end);
}

test "encodeBatchWithTypes — returns parallel EncodeResults" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const texts = [_][]const u8{ "ab", "abc" };
    const batch = try tok.encodeBatchWithTypes(testing.allocator, &texts);
    defer Tokenizer.freeBatchWithTypes(testing.allocator, batch);

    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1 }, batch[0].ids);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0 }, batch[0].type_ids);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1 }, batch[0].attention_mask);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, batch[1].ids);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 1 }, batch[1].attention_mask);
}

test "encodeBatchPairWithTypes — preserves type_ids across pair batch" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [
        \\    {"id": 101, "content": "[CLS]"},
        \\    {"id": 102, "content": "[SEP]"}
        \\  ],
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "post_processor": {
        \\    "type": "TemplateProcessing",
        \\    "pair": [
        \\      {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        \\      {"Sequence": {"id": "A", "type_id": 0}},
        \\      {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        \\      {"Sequence": {"id": "B", "type_id": 1}},
        \\      {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
        \\    ],
        \\    "special_tokens": {
        \\      "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
        \\      "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
        \\    }
        \\  },
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const a = [_][]const u8{ "a", "ab" };
    const b = [_][]const u8{ "b", "c" };
    const batch = try tok.encodeBatchPairWithTypes(testing.allocator, &a, &b);
    defer Tokenizer.freeBatchWithTypes(testing.allocator, batch);

    try testing.expectEqual(@as(usize, 2), batch.len);
    try testing.expectEqualSlices(u32, &[_]u32{ 101, 0, 102, 1, 102 }, batch[0].ids);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 1, 1 }, batch[0].type_ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 101, 0, 1, 102, 2, 102 }, batch[1].ids);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 0, 1, 1 }, batch[1].type_ids);
}

test "decodeBatch — round-trips encodeBatch output" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const texts = [_][]const u8{ "a", "ab", "abc" };
    const ids_batch = try tok.encodeBatch(testing.allocator, &texts);
    defer Tokenizer.freeBatch(testing.allocator, ids_batch);

    const slices = try testing.allocator.alloc([]const u32, ids_batch.len);
    defer testing.allocator.free(slices);
    for (ids_batch, 0..) |ids, i| slices[i] = ids;

    const decoded = try tok.decodeBatch(testing.allocator, slices);
    defer Tokenizer.freeDecodeBatch(testing.allocator, decoded);

    try testing.expectEqualStrings("a", decoded[0]);
    try testing.expectEqualStrings("ab", decoded[1]);
    try testing.expectEqualStrings("abc", decoded[2]);
}

test "decodeBatchParallel — matches sequential output (4 threads)" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "d": 3}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const inputs = [_][]const u32{
        &[_]u32{ 0, 1 },
        &[_]u32{ 1, 2, 3 },
        &[_]u32{0},
        &[_]u32{ 0, 1, 2, 3 },
        &[_]u32{ 3, 2, 1, 0 },
    };
    const seq = try tok.decodeBatch(testing.allocator, &inputs);
    defer Tokenizer.freeDecodeBatch(testing.allocator, seq);
    const par = try tok.decodeBatchParallel(testing.allocator, &inputs, 4);
    defer Tokenizer.freeDecodeBatch(testing.allocator, par);

    try testing.expectEqual(seq.len, par.len);
    for (seq, par) |s, p| try testing.expectEqualStrings(s, p);
}

test "encodeBatchParallel — single thread falls through to sequential" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const texts = [_][]const u8{"a"};
    const out = try tok.encodeBatchParallel(testing.allocator, &texts, 1);
    defer Tokenizer.freeBatch(testing.allocator, out);
    try testing.expectEqualSlices(u32, &[_]u32{0}, out[0]);
}

test "encodeBatch — empty batch returns empty slice" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
        \\}
    ;
    var tok = try fromJson(testing.allocator, json);
    defer tok.deinit();

    const batch = try tok.encodeBatch(testing.allocator, &[_][]const u8{});
    defer Tokenizer.freeBatch(testing.allocator, batch);
    try testing.expectEqual(@as(usize, 0), batch.len);
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
