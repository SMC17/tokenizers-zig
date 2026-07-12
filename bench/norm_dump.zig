//! Standalone normalizer dump (Zig side): applies just the parsed
//! normalizer chain (no pre-tokenization, no model encode) so it can
//! be diffed directly against bench/diff_rust/src/bin/norm_dump.rs to
//! isolate whether a differential-correctness mismatch comes from
//! normalization or from the model's subword matching.

const std = @import("std");
const tokenizers = @import("tokenizers");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer iter.deinit();
    _ = iter.next();
    const fixture_path = iter.next() orelse {
        std.debug.print("usage: norm_dump <fixture.json> <corpus.jsonl>\n", .{});
        std.process.exit(2);
    };
    const corpus_path = iter.next() orelse {
        std.debug.print("usage: norm_dump <fixture.json> <corpus.jsonl>\n", .{});
        std.process.exit(2);
    };

    const fixture_json = try std.Io.Dir.cwd().readFileAlloc(io, fixture_path, alloc, std.Io.Limit.limited(64 * 1024 * 1024));
    defer alloc.free(fixture_json);

    var tok = try tokenizers.fromJson(alloc, fixture_json);
    defer tok.deinit();

    const corpus = try std.Io.Dir.cwd().readFileAlloc(io, corpus_path, alloc, std.Io.Limit.limited(16 * 1024 * 1024));
    defer alloc.free(corpus);

    const stdout_file = std.Io.File.stdout();
    var out_buf: [16384]u8 = undefined;
    var writer = stdout_file.writer(io, &out_buf);

    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, corpus, '\n');
    while (lines.next()) |raw_line| : (line_no += 1) {
        const trimmed = std.mem.trim(u8, raw_line, " \t\r");
        if (trimmed.len == 0) continue;

        var arena = std.heap.ArenaAllocator.init(alloc);
        defer arena.deinit();
        const aa = arena.allocator();

        const parsed = std.json.parseFromSliceLeaky([]const u8, aa, trimmed, .{}) catch continue;

        const normalized = tokenizers.normalizer.apply(alloc, tok.normalizer, parsed) catch |err| {
            try writer.interface.print("{{\"line\": {d}, \"error\": \"{s}\"}}\n", .{ line_no, @errorName(err) });
            continue;
        };
        defer alloc.free(normalized);

        var w = std.json.Stringify{ .writer = &writer.interface, .options = .{} };
        try writer.interface.print("{{\"line\": {d}, \"normalized\": ", .{line_no});
        try w.write(normalized);
        try writer.interface.writeAll("}\n");
    }
    try writer.interface.flush();
}
