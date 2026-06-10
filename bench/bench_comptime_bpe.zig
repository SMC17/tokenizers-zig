//! Comptime-baked BPE encoder vs runtime-hashmap BPE encoder.
//!
//! Both paths implement the same greedy lowest-rank BPE merge over a
//! fixed 64-merge vocabulary on a 256-byte test string. The comptime
//! path generates a switch expression at compile time so pair lookups
//! become branch-tree dispatches rather than hashmap probes.
//!
//! Pre-reg: exp-1780016950-857326811. Falsifier: speedup <5x or
//! token-ID divergence between paths.
//!
//! The vocab here is deliberately a TOY domain — single-letter base
//! tokens plus 64 hand-picked merges — so the kernel is the same
//! greedy O(n*m) merge loop on both paths. The win is purely the
//! lookup-path shape: hashmap vs comptime-switch.

const std = @import("std");

// ------------------------------------------------------------------
// Shared vocab: base tokens 0..25 = letters 'a'..'z'. Merges 26..89
// = 64 hand-picked pairs by rank. Lower rank = applied first.
// ------------------------------------------------------------------

const BaseToken = u32;

/// Each merge entry: (left_id, right_id, new_id). Sorted by rank
/// in the array index (idx 0 = rank 0 = highest priority).
const Merge = struct { left: BaseToken, right: BaseToken, new_id: BaseToken };

const MERGES = [_]Merge{
    .{ .left = 't', .right = 'h', .new_id = 26 }, // "th"
    .{ .left = 'h', .right = 'e', .new_id = 27 }, // "he"
    .{ .left = 26, .right = 'e', .new_id = 28 }, // "the"
    .{ .left = 'a', .right = 'n', .new_id = 29 }, // "an"
    .{ .left = 29, .right = 'd', .new_id = 30 }, // "and"
    .{ .left = 'i', .right = 'n', .new_id = 31 }, // "in"
    .{ .left = 'e', .right = 'r', .new_id = 32 }, // "er"
    .{ .left = 'r', .right = 'e', .new_id = 33 }, // "re"
    .{ .left = 'o', .right = 'n', .new_id = 34 }, // "on"
    .{ .left = 'a', .right = 't', .new_id = 35 }, // "at"
    .{ .left = 's', .right = 't', .new_id = 36 }, // "st"
    .{ .left = 'o', .right = 'r', .new_id = 37 }, // "or"
    .{ .left = 'e', .right = 'n', .new_id = 38 }, // "en"
    .{ .left = 'n', .right = 'd', .new_id = 39 }, // "nd"
    .{ .left = 'o', .right = 'u', .new_id = 40 }, // "ou"
    .{ .left = 'i', .right = 's', .new_id = 41 }, // "is"
    .{ .left = 'a', .right = 's', .new_id = 42 }, // "as"
    .{ .left = 'i', .right = 't', .new_id = 43 }, // "it"
    .{ .left = 'a', .right = 'r', .new_id = 44 }, // "ar"
    .{ .left = 'a', .right = 'l', .new_id = 45 }, // "al"
    .{ .left = 'e', .right = 'd', .new_id = 46 }, // "ed"
    .{ .left = 'l', .right = 'e', .new_id = 47 }, // "le"
    .{ .left = 'c', .right = 'o', .new_id = 48 }, // "co"
    .{ .left = 'o', .right = 'f', .new_id = 49 }, // "of"
    .{ .left = 't', .right = 'o', .new_id = 50 }, // "to"
    .{ .left = 'b', .right = 'e', .new_id = 51 }, // "be"
    .{ .left = 'v', .right = 'e', .new_id = 52 }, // "ve"
    .{ .left = 's', .right = 'e', .new_id = 53 }, // "se"
    .{ .left = 'n', .right = 'e', .new_id = 54 }, // "ne"
    .{ .left = 'm', .right = 'e', .new_id = 55 }, // "me"
    .{ .left = 'd', .right = 'e', .new_id = 56 }, // "de"
    .{ .left = 'r', .right = 'i', .new_id = 57 }, // "ri"
    .{ .left = 'l', .right = 'i', .new_id = 58 }, // "li"
    .{ .left = 't', .right = 'i', .new_id = 59 }, // "ti"
    .{ .left = 'n', .right = 'i', .new_id = 60 }, // "ni"
    .{ .left = 's', .right = 'i', .new_id = 61 }, // "si"
    .{ .left = 'r', .right = 'o', .new_id = 62 }, // "ro"
    .{ .left = 'l', .right = 'o', .new_id = 63 }, // "lo"
    .{ .left = 'm', .right = 'o', .new_id = 64 }, // "mo"
    .{ .left = 'n', .right = 'o', .new_id = 65 }, // "no"
    .{ .left = 'p', .right = 'o', .new_id = 66 }, // "po"
    .{ .left = 'c', .right = 'a', .new_id = 67 }, // "ca"
    .{ .left = 'l', .right = 'a', .new_id = 68 }, // "la"
    .{ .left = 'm', .right = 'a', .new_id = 69 }, // "ma"
    .{ .left = 'n', .right = 'a', .new_id = 70 }, // "na"
    .{ .left = 'p', .right = 'a', .new_id = 71 }, // "pa"
    .{ .left = 'r', .right = 'a', .new_id = 72 }, // "ra"
    .{ .left = 's', .right = 'a', .new_id = 73 }, // "sa"
    .{ .left = 'c', .right = 'e', .new_id = 74 }, // "ce"
    .{ .left = 'g', .right = 'e', .new_id = 75 }, // "ge"
    .{ .left = 'k', .right = 'e', .new_id = 76 }, // "ke"
    .{ .left = 'b', .right = 'a', .new_id = 77 }, // "ba"
    .{ .left = 'g', .right = 'a', .new_id = 78 }, // "ga"
    .{ .left = 'b', .right = 'o', .new_id = 79 }, // "bo"
    .{ .left = 'f', .right = 'o', .new_id = 80 }, // "fo"
    .{ .left = 'g', .right = 'o', .new_id = 81 }, // "go"
    .{ .left = 's', .right = 'o', .new_id = 82 }, // "so"
    .{ .left = 'h', .right = 'a', .new_id = 83 }, // "ha"
    .{ .left = 'j', .right = 'a', .new_id = 84 }, // "ja"
    .{ .left = 't', .right = 'a', .new_id = 85 }, // "ta"
    .{ .left = 'w', .right = 'a', .new_id = 86 }, // "wa"
    .{ .left = 'y', .right = 'a', .new_id = 87 }, // "ya"
    .{ .left = 'd', .right = 'i', .new_id = 88 }, // "di"
    .{ .left = 'p', .right = 'i', .new_id = 89 }, // "pi"
};

const N_MERGES = MERGES.len;

// ------------------------------------------------------------------
// Runtime path: hashmap of packed-pair (u64) → rank, plus rank-table
// indexed lookup for new_id. Greedy lowest-rank merge.
// ------------------------------------------------------------------

inline fn packPair(l: BaseToken, r: BaseToken) u64 {
    return (@as(u64, l) << 32) | @as(u64, r);
}

fn encodeRuntime(
    alloc: std.mem.Allocator,
    rank_map: *const std.AutoHashMap(u64, u32),
    out_tokens: []BaseToken,
    input: []const u8,
) usize {
    var seq = std.array_list.Managed(BaseToken).init(alloc);
    defer seq.deinit();
    seq.ensureTotalCapacity(input.len) catch unreachable;
    for (input) |b| seq.append(@as(BaseToken, b)) catch unreachable;

    while (true) {
        var best_rank: u32 = std.math.maxInt(u32);
        var best_idx: usize = 0;
        var found = false;
        for (0..seq.items.len - 1) |i| {
            const k = packPair(seq.items[i], seq.items[i + 1]);
            if (rank_map.get(k)) |r| {
                if (r < best_rank) {
                    best_rank = r;
                    best_idx = i;
                    found = true;
                }
            }
        }
        if (!found) break;
        const m = MERGES[best_rank];
        seq.items[best_idx] = m.new_id;
        _ = seq.orderedRemove(best_idx + 1);
        if (seq.items.len < 2) break;
    }

    @memcpy(out_tokens[0..seq.items.len], seq.items);
    return seq.items.len;
}

// ------------------------------------------------------------------
// Comptime path: pair → rank lookup is generated as an inline switch
// at comptime. Each entry becomes a single compare-and-return; LLVM
// turns this into a balanced compare tree (much better cache locality
// than the runtime hashmap which probes a heap-allocated buckets array).
// ------------------------------------------------------------------

inline fn pairRankComptime(l: BaseToken, r: BaseToken) ?u32 {
    const k = packPair(l, r);
    inline for (MERGES, 0..) |m, idx| {
        if (k == packPair(m.left, m.right)) return @intCast(idx);
    }
    return null;
}

fn encodeComptime(
    alloc: std.mem.Allocator,
    out_tokens: []BaseToken,
    input: []const u8,
) usize {
    var seq = std.array_list.Managed(BaseToken).init(alloc);
    defer seq.deinit();
    seq.ensureTotalCapacity(input.len) catch unreachable;
    for (input) |b| seq.append(@as(BaseToken, b)) catch unreachable;

    while (true) {
        var best_rank: u32 = std.math.maxInt(u32);
        var best_idx: usize = 0;
        var found = false;
        for (0..seq.items.len - 1) |i| {
            if (pairRankComptime(seq.items[i], seq.items[i + 1])) |r| {
                if (r < best_rank) {
                    best_rank = r;
                    best_idx = i;
                    found = true;
                }
            }
        }
        if (!found) break;
        const m = MERGES[best_rank];
        seq.items[best_idx] = m.new_id;
        _ = seq.orderedRemove(best_idx + 1);
        if (seq.items.len < 2) break;
    }

    @memcpy(out_tokens[0..seq.items.len], seq.items);
    return seq.items.len;
}

// ------------------------------------------------------------------
// Driver.
// ------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    const INPUT_TEXT =
        "thequickbrownfoxjumpsoverthelazydogtheendisnearmakehasteandbe" ++
        "soundofthemorningsongandtheriverandtherainandthestonethesound" ++
        "ofthecanyonandthecallofthewildtheoceanandtheskyandthelandof" ++
        "thenorthandthethought"; // 240 bytes
    var input: [256]u8 = undefined;
    @memcpy(input[0..INPUT_TEXT.len], INPUT_TEXT);
    for (INPUT_TEXT.len..256) |i| input[i] = 'a';

    // Build hashmap for runtime path
    var rank_map = std.AutoHashMap(u64, u32).init(alloc);
    defer rank_map.deinit();
    try rank_map.ensureTotalCapacity(N_MERGES);
    for (MERGES, 0..) |m, idx| {
        try rank_map.put(packPair(m.left, m.right), @intCast(idx));
    }

    var out_runtime: [256]BaseToken = undefined;
    var out_comptime: [256]BaseToken = undefined;

    // Correctness gate first.
    const n_runtime = encodeRuntime(alloc, &rank_map, &out_runtime, &input);
    const n_comptime = encodeComptime(alloc, &out_comptime, &input);
    if (n_runtime != n_comptime) {
        std.debug.print("DIVERGENCE: runtime n={d} comptime n={d}\n", .{ n_runtime, n_comptime });
        return error.TokenCountMismatch;
    }
    for (0..n_runtime) |i| {
        if (out_runtime[i] != out_comptime[i]) {
            std.debug.print("DIVERGENCE at {d}: runtime={d} comptime={d}\n", .{ i, out_runtime[i], out_comptime[i] });
            return error.TokenIdMismatch;
        }
    }
    std.debug.print("correctness: PASS ({d} tokens, both paths)\n", .{n_runtime});

    const ITERS = 1000;

    // Warmup
    for (0..50) |_| {
        _ = encodeRuntime(alloc, &rank_map, &out_runtime, &input);
        _ = encodeComptime(alloc, &out_comptime, &input);
    }

    // Bench runtime
    const t0 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| {
        _ = encodeRuntime(alloc, &rank_map, &out_runtime, &input);
    }
    const t1 = std.Io.Clock.awake.now(io);

    // Bench comptime
    for (0..ITERS) |_| {
        _ = encodeComptime(alloc, &out_comptime, &input);
    }
    const t2 = std.Io.Clock.awake.now(io);

    const ns_runtime: u64 = @intCast(t0.durationTo(t1).nanoseconds);
    const ns_comptime: u64 = @intCast(t1.durationTo(t2).nanoseconds);
    const ns_per_runtime = ns_runtime / ITERS;
    const ns_per_comptime = ns_comptime / ITERS;
    const speedup_x100 = (ns_runtime * 100) / ns_comptime;

    std.debug.print("\n", .{});
    std.debug.print("input_bytes      = 256\n", .{});
    std.debug.print("n_merges_baked   = {d}\n", .{N_MERGES});
    std.debug.print("iters            = {d}\n", .{ITERS});
    std.debug.print("ns_per_runtime   = {d}\n", .{ns_per_runtime});
    std.debug.print("ns_per_comptime  = {d}\n", .{ns_per_comptime});
    std.debug.print("us_per_runtime   = {d}\n", .{ns_per_runtime / 1000});
    std.debug.print("us_per_comptime  = {d}\n", .{ns_per_comptime / 1000});
    std.debug.print("speedup_x100     = {d}   (>= 500 = >=5x, >= 1000 = >=10x)\n", .{speedup_x100});
}
