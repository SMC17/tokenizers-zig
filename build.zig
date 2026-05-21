const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("tokenizers", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tests = b.addTest(.{ .root_module = mod });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // Demo CLI: tokenize <tokenizer.json> "<text>"
    const tok_mod = b.createModule(.{
        .root_source_file = b.path("examples/tokenize.zig"),
        .target = target,
        .optimize = optimize,
    });
    tok_mod.addImport("tokenizers", mod);
    const tok_exe = b.addExecutable(.{
        .name = "tokenize",
        .root_module = tok_mod,
    });
    b.installArtifact(tok_exe);
    const run_tok = b.addRunArtifact(tok_exe);
    if (b.args) |args| run_tok.addArgs(args);
    const tok_step = b.step("tokenize", "Tokenize text with a tokenizer.json");
    tok_step.dependOn(&run_tok.step);

    // Bench
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_encode.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mod.addImport("tokenizers", mod);
    const bench_exe = b.addExecutable(.{
        .name = "bench-encode",
        .root_module = bench_mod,
    });
    const run_bench = b.addRunArtifact(bench_exe);
    run_bench.has_side_effects = true;
    const bench_step = b.step("bench", "Encode throughput bench");
    bench_step.dependOn(&run_bench.step);
}
