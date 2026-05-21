//! Encode throughput bench against a synthetic ByteLevel BPE
//! tokenizer. Builds a small in-memory tokenizer and encodes a fixed
//! text repeatedly.
//!
//! Honest non-claim: this is a synthetic bench. A real tokenizer.json
//! comparison against HuggingFace tokenizers (Rust) is v0.1 work.

const std = @import("std");
const tokenizers = @import("tokenizers");

const TEXT =
    \\The quick brown fox jumps over the lazy dog.
    \\The five boxing wizards jump quickly.
    \\Pack my box with five dozen liquor jugs.
;

const ITERS = 1000;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    const json =
        \\{
        \\  "version": "1.0",
        \\  "pre_tokenizer": {"type": "ByteLevel"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "T": 0, "h": 1, "e": 2, "Ġ": 3, "q": 4, "u": 5, "i": 6, "c": 7, "k": 8,
        \\      "b": 9, "r": 10, "o": 11, "w": 12, "n": 13, "f": 14, "x": 15,
        \\      "j": 16, "m": 17, "p": 18, "s": 19, "v": 20, "t": 21, "l": 22,
        \\      "a": 23, "z": 24, "y": 25, "d": 26, "g": 27, ".": 28, "P": 29,
        \\      "Ċ": 30,
        \\      "Th": 31, "The": 32, "ĠT": 33, "Ġq": 34, "Ġu": 35, "Ġqu": 36,
        \\      "Ġqui": 37, "Ġquick": 38, "Ġbr": 39, "Ġbrown": 40, "Ġfox": 41,
        \\      "Ġjumps": 42, "Ġover": 43, "Ġthe": 44, "Ġlazy": 45, "Ġdog": 46,
        \\      "<unk>": 99
        \\    },
        \\    "merges": [
        \\      "T h", "Th e", "Ġ T", "ĠT he", "Ġ q", "Ġq u", "Ġqu i",
        \\      "Ġqui ck", "Ġ b", "Ġb r", "Ġbr o", "Ġbro wn", "Ġ f",
        \\      "Ġf o", "Ġfo x", "Ġ j", "Ġj u", "Ġju mps", "Ġ o",
        \\      "Ġo ver", "Ġt h", "Ġth e", "Ġ l", "Ġl a", "Ġla zy",
        \\      "Ġ d", "Ġd o", "Ġdo g"
        \\    ],
        \\    "unk_token": "<unk>"
        \\  }
        \\}
    ;

    var tok = try tokenizers.fromJson(alloc, json);
    defer tok.deinit();

    // Warmup
    {
        const ids = try tok.encode(alloc, TEXT);
        alloc.free(ids);
    }

    const t0 = std.Io.Clock.awake.now(io);
    var sink: u32 = 0;
    var i: usize = 0;
    while (i < ITERS) : (i += 1) {
        const ids = try tok.encode(alloc, TEXT);
        sink ^= @intCast(ids.len);
        alloc.free(ids);
    }
    const t1 = std.Io.Clock.awake.now(io);
    const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);

    const ns_per_encode = ns / ITERS;
    const text_bytes = TEXT.len;
    const bytes_per_sec = @as(u64, ITERS) * text_bytes * std.time.ns_per_s / ns;

    std.debug.print("config\n", .{});
    std.debug.print("  text_bytes      = {d}\n", .{text_bytes});
    std.debug.print("  iters           = {d}\n", .{ITERS});
    std.debug.print("encode\n", .{});
    std.debug.print("  ns_per_encode   = {d}\n", .{ns_per_encode});
    std.debug.print("  us_per_encode   = {d}\n", .{ns_per_encode / 1000});
    std.debug.print("  bytes_per_sec   = {d}\n", .{bytes_per_sec});
    std.debug.print("  MB_per_sec      = {d}\n", .{bytes_per_sec / 1_000_000});
    std.debug.print("  sink            = {d}\n", .{sink}); // anti-DCE
}
