//! tokenize — encode / decode / inspect CLI.
//!
//! Subcommands:
//!   tokenize encode <tokenizer.json> "<text>"
//!       Encode text → list of token IDs (default: with TemplateProcessing).
//!   tokenize encode-base <tokenizer.json> "<text>"
//!       Encode without TemplateProcessing wrap (raw model output).
//!   tokenize encode-pair <tokenizer.json> "<text_a>" "<text_b>"
//!       Sentence-pair encode (BERT NSP style). Returns ids + type_ids.
//!   tokenize encode-full <tokenizer.json> "<text>"
//!       Encode and dump ids + type_ids + attention_mask.
//!   tokenize decode <tokenizer.json> "<comma-separated-ids>"
//!       Decode IDs → text. E.g. `tokenize decode bert.json 101,7592,102`
//!   tokenize info <tokenizer.json>
//!       Print model type, vocab size, added-token count, post-processor.

const std = @import("std");
const tokenizers = @import("tokenizers");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer iter.deinit();
    _ = iter.next();

    const cmd = iter.next() orelse {
        printUsage();
        std.process.exit(2);
    };

    if (std.mem.eql(u8, cmd, "encode")) {
        try runEncode(alloc, io, &iter, .full);
    } else if (std.mem.eql(u8, cmd, "encode-base")) {
        try runEncode(alloc, io, &iter, .base);
    } else if (std.mem.eql(u8, cmd, "encode-pair")) {
        try runEncodePair(alloc, io, &iter);
    } else if (std.mem.eql(u8, cmd, "encode-full")) {
        try runEncodeFull(alloc, io, &iter);
    } else if (std.mem.eql(u8, cmd, "decode")) {
        try runDecode(alloc, io, &iter);
    } else if (std.mem.eql(u8, cmd, "info")) {
        try runInfo(alloc, io, &iter);
    } else {
        std.debug.print("unknown subcommand: {s}\n", .{cmd});
        printUsage();
        std.process.exit(2);
    }
}

fn printUsage() void {
    std.debug.print(
        \\usage:
        \\  tokenize encode      <tokenizer.json> "<text>"
        \\  tokenize encode-base <tokenizer.json> "<text>"
        \\  tokenize encode-pair <tokenizer.json> "<text_a>" "<text_b>"
        \\  tokenize encode-full <tokenizer.json> "<text>"
        \\  tokenize decode      <tokenizer.json> "<ids,comma,sep>"
        \\  tokenize info        <tokenizer.json>
        \\
    , .{});
}

const EncodeMode = enum { full, base };

fn loadTokenizer(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !tokenizers.Tokenizer {
    const json_bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, std.Io.Limit.limited(256 * 1024 * 1024));
    defer alloc.free(json_bytes);
    return tokenizers.fromJson(alloc, json_bytes);
}

fn runEncode(alloc: std.mem.Allocator, io: std.Io, iter: *std.process.Args.Iterator, mode: EncodeMode) !void {
    const tok_path = iter.next() orelse {
        printUsage();
        std.process.exit(2);
    };
    const text = iter.next() orelse {
        printUsage();
        std.process.exit(2);
    };
    var tok = try loadTokenizer(alloc, io, tok_path);
    defer tok.deinit();
    const ids = switch (mode) {
        .full => try tok.encode(alloc, text),
        .base => try tok.encodeBase(alloc, text),
    };
    defer alloc.free(ids);
    std.debug.print("input:  {s}\n", .{text});
    std.debug.print("ids:    [", .{});
    for (ids, 0..) |id, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d}", .{id});
    }
    std.debug.print("]\n", .{});
    std.debug.print("count:  {d}\n", .{ids.len});
}

fn runEncodePair(alloc: std.mem.Allocator, io: std.Io, iter: *std.process.Args.Iterator) !void {
    const tok_path = iter.next() orelse {
        printUsage();
        std.process.exit(2);
    };
    const text_a = iter.next() orelse {
        printUsage();
        std.process.exit(2);
    };
    const text_b = iter.next() orelse {
        printUsage();
        std.process.exit(2);
    };
    var tok = try loadTokenizer(alloc, io, tok_path);
    defer tok.deinit();
    const result = try tok.encodePairWithTypes(alloc, text_a, text_b);
    defer result.deinit(alloc);
    std.debug.print("A:        {s}\n", .{text_a});
    std.debug.print("B:        {s}\n", .{text_b});
    printArray(u32, "ids", result.ids);
    printArray(u8, "type_ids", result.type_ids);
    printArray(u8, "mask", result.attention_mask);
}

fn runEncodeFull(alloc: std.mem.Allocator, io: std.Io, iter: *std.process.Args.Iterator) !void {
    const tok_path = iter.next() orelse {
        printUsage();
        std.process.exit(2);
    };
    const text = iter.next() orelse {
        printUsage();
        std.process.exit(2);
    };
    var tok = try loadTokenizer(alloc, io, tok_path);
    defer tok.deinit();
    const result = try tok.encodeWithTypes(alloc, text);
    defer result.deinit(alloc);
    std.debug.print("input:    {s}\n", .{text});
    printArray(u32, "ids", result.ids);
    printArray(u8, "type_ids", result.type_ids);
    printArray(u8, "mask", result.attention_mask);
}

fn runDecode(alloc: std.mem.Allocator, io: std.Io, iter: *std.process.Args.Iterator) !void {
    const tok_path = iter.next() orelse {
        printUsage();
        std.process.exit(2);
    };
    const ids_str = iter.next() orelse {
        printUsage();
        std.process.exit(2);
    };
    var tok = try loadTokenizer(alloc, io, tok_path);
    defer tok.deinit();

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(alloc);
    var it = std.mem.tokenizeAny(u8, ids_str, ", ");
    while (it.next()) |part| {
        const id = std.fmt.parseInt(u32, part, 10) catch {
            std.debug.print("invalid id: {s}\n", .{part});
            std.process.exit(2);
        };
        try ids.append(alloc, id);
    }
    const text = try tok.decode(alloc, ids.items);
    defer alloc.free(text);
    printArray(u32, "ids", ids.items);
    std.debug.print("decoded:  {s}\n", .{text});
}

fn runInfo(alloc: std.mem.Allocator, io: std.Io, iter: *std.process.Args.Iterator) !void {
    const tok_path = iter.next() orelse {
        printUsage();
        std.process.exit(2);
    };
    var tok = try loadTokenizer(alloc, io, tok_path);
    defer tok.deinit();
    const model_name = switch (tok.model) {
        .bpe => "BPE",
        .wordpiece => "WordPiece",
        .unigram => "Unigram",
    };
    std.debug.print("path:               {s}\n", .{tok_path});
    std.debug.print("model:              {s}\n", .{model_name});
    std.debug.print("vocab_size:         {d}\n", .{tok.model.vocabCount()});
    std.debug.print("added_tokens:       {d}\n", .{tok.added_tokens.len});
    std.debug.print("use_byte_level:     {}\n", .{tok.use_byte_level});
    std.debug.print("use_gpt2_regex:     {}\n", .{tok.use_gpt2_regex});
    std.debug.print("whitespace_split:   {}\n", .{tok.use_whitespace_split});
    std.debug.print("punct_split:        {}\n", .{tok.split_on_punctuation});
    std.debug.print("byte_fallback:      {}\n", .{tok.use_byte_fallback});
    std.debug.print("metaspace_marker:   {s}\n", .{if (tok.metaspace_replacement) |m| m else "none"});
    std.debug.print("template_single:    {}\n", .{tok.template_single != null});
    std.debug.print("template_pair:      {}\n", .{tok.template_pair != null});
    std.debug.print("decoder_steps:      {d}\n", .{if (tok.decoder_steps) |s| s.len else 0});
}

fn printArray(comptime T: type, name: []const u8, slice: []const T) void {
    std.debug.print("{s:<10}[", .{name});
    for (slice, 0..) |x, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d}", .{x});
    }
    std.debug.print("]\n", .{});
}
