//! tokenize — encode + decode demo CLI.
//!
//! Usage:
//!   tokenize <tokenizer.json> "text to tokenize"

const std = @import("std");
const tokenizers = @import("tokenizers");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer iter.deinit();
    _ = iter.next();

    const tok_path = iter.next() orelse {
        std.debug.print("usage: tokenize <tokenizer.json> \"<text>\"\n", .{});
        std.process.exit(2);
    };
    const text = iter.next() orelse {
        std.debug.print("usage: tokenize <tokenizer.json> \"<text>\"\n", .{});
        std.process.exit(2);
    };

    const json_bytes = try std.Io.Dir.cwd().readFileAlloc(
        io,
        tok_path,
        alloc,
        std.Io.Limit.limited(256 * 1024 * 1024),
    );
    defer alloc.free(json_bytes);

    var tok = try tokenizers.fromJson(alloc, json_bytes);
    defer tok.deinit();

    const ids = try tok.encode(alloc, text);
    defer alloc.free(ids);

    std.debug.print("input:  {s}\n", .{text});
    std.debug.print("tokens: {d}\n", .{ids.len});
    std.debug.print("ids:    [", .{});
    for (ids, 0..) |id, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d}", .{id});
    }
    std.debug.print("]\n", .{});

    const back = try tok.decode(alloc, ids);
    defer alloc.free(back);
    std.debug.print("decode: {s}\n", .{back});
}
