//! Differential dump (Zig side): reads a JSONL corpus (one JSON-encoded
//! string per line) and a tokenizer.json fixture, encodes every line with
//! Tokenizer.encode (which always applies TemplateProcessing — matches
//! HF Rust's add_special_tokens=true), and prints one JSON object per line:
//!   {"line": N, "ok": true, "ids": [...]}
//!   {"line": N, "ok": false, "error": "..."}
//! Paired with bench/diff_rust — exact token-ID equality across every
//! corpus line is the differential-correctness pass criterion, replacing
//! the single-fixture timing-only comparison.

const std = @import("std");
const tokenizers = @import("tokenizers");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer iter.deinit();
    _ = iter.next();
    const fixture_path = iter.next() orelse {
        std.debug.print("usage: diff_dump <fixture.json> <corpus.jsonl>\n", .{});
        std.process.exit(2);
    };
    const corpus_path = iter.next() orelse {
        std.debug.print("usage: diff_dump <fixture.json> <corpus.jsonl>\n", .{});
        std.process.exit(2);
    };

    const fixture_json = try std.Io.Dir.cwd().readFileAlloc(io, fixture_path, alloc, std.Io.Limit.limited(64 * 1024 * 1024));
    defer alloc.free(fixture_json);

    var tok = try tokenizers.fromJson(alloc, fixture_json);
    defer tok.deinit();

    const corpus = try std.Io.Dir.cwd().readFileAlloc(io, corpus_path, alloc, std.Io.Limit.limited(16 * 1024 * 1024));
    defer alloc.free(corpus);

    const stdout_file = std.Io.File.stdout();
    var out_buf: [8192]u8 = undefined;
    var writer = stdout_file.writer(io, &out_buf);

    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, corpus, '\n');
    while (lines.next()) |raw_line| : (line_no += 1) {
        const trimmed = std.mem.trim(u8, raw_line, " \t\r");
        if (trimmed.len == 0) continue;

        var arena = std.heap.ArenaAllocator.init(alloc);
        defer arena.deinit();
        const aa = arena.allocator();

        const parsed = std.json.parseFromSliceLeaky([]const u8, aa, trimmed, .{}) catch {
            try writer.interface.print("{{\"line\": {d}, \"ok\": false, \"error\": \"corpus line is not a valid JSON string\"}}\n", .{line_no});
            continue;
        };

        const ids = tok.encode(alloc, parsed) catch |err| {
            try writer.interface.print("{{\"line\": {d}, \"ok\": false, \"error\": \"{s}\"}}\n", .{ line_no, @errorName(err) });
            continue;
        };
        defer alloc.free(ids);

        try writer.interface.print("{{\"line\": {d}, \"ok\": true, \"ids\": [", .{line_no});
        for (ids, 0..) |id, i| {
            if (i != 0) try writer.interface.writeAll(",");
            try writer.interface.print("{d}", .{id});
        }
        try writer.interface.writeAll("]}\n");
    }
    try writer.interface.flush();
}
