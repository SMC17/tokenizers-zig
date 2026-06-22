const std = @import("std");
const normalizer = @import("src/normalizer.zig");
const tokenizers = @import("src/root.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    
    const r = tokenizers.NormalizerOp{
        .replace = .{
            .pattern = "▁",
            .content = " ",
        }
    };
    
    const input = "▁10▁mg▁twice▁daily";
    const out = try normalizer.apply(alloc, r, input);
    defer alloc.free(out);
    
    std.debug.print("Input: {s}\n", .{input});
    std.debug.print("Output: {s}\n", .{out});
}
