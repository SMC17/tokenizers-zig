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

    // gpt2_regex needs its own test runner because it's exported as
    // its own module-internal file.
    const gpt2_mod = b.createModule(.{
        .root_source_file = b.path("src/gpt2_regex.zig"),
        .target = target,
        .optimize = optimize,
    });
    const gpt2_tests = b.addTest(.{ .root_module = gpt2_mod });
    const run_gpt2 = b.addRunArtifact(gpt2_tests);
    test_step.dependOn(&run_gpt2.step);

    const unorm_mod = b.createModule(.{
        .root_source_file = b.path("src/unicode_norm.zig"),
        .target = target,
        .optimize = optimize,
    });
    const unorm_tests = b.addTest(.{ .root_module = unorm_mod });
    const run_unorm = b.addRunArtifact(unorm_tests);
    test_step.dependOn(&run_unorm.step);

    // Real TinyLlama tokenizer integration test
    const real_mod = b.createModule(.{
        .root_source_file = b.path("tests/test_real_tinyllama.zig"),
        .target = target,
        .optimize = optimize,
    });
    real_mod.addImport("tokenizers", mod);
    const real_tests = b.addTest(.{ .root_module = real_mod });
    const run_real = b.addRunArtifact(real_tests);
    test_step.dependOn(&run_real.step);
    const real_step = b.step("test-real-tinyllama", "Real TinyLlama tokenizer test");
    real_step.dependOn(&run_real.step);

    // Real bert-base-uncased tokenizer integration test
    const bert_mod = b.createModule(.{
        .root_source_file = b.path("tests/test_real_bert.zig"),
        .target = target,
        .optimize = optimize,
    });
    bert_mod.addImport("tokenizers", mod);
    const bert_tests = b.addTest(.{ .root_module = bert_mod });
    const run_bert = b.addRunArtifact(bert_tests);
    test_step.dependOn(&run_bert.step);
    const bert_step = b.step("test-real-bert", "Real BERT-base-uncased tokenizer test");
    bert_step.dependOn(&run_bert.step);

    // Real t5-base tokenizer integration test
    const t5_mod = b.createModule(.{
        .root_source_file = b.path("tests/test_real_t5.zig"),
        .target = target,
        .optimize = optimize,
    });
    t5_mod.addImport("tokenizers", mod);
    const t5_tests = b.addTest(.{ .root_module = t5_mod });
    const run_t5 = b.addRunArtifact(t5_tests);
    test_step.dependOn(&run_t5.step);
    const t5_step = b.step("test-real-t5", "Real T5-base tokenizer test");
    t5_step.dependOn(&run_t5.step);

    // Property-based fuzz invariants over all three real fixtures.
    const fuzz_mod = b.createModule(.{
        .root_source_file = b.path("tests/test_fuzz_invariants.zig"),
        .target = target,
        .optimize = optimize,
    });
    fuzz_mod.addImport("tokenizers", mod);
    const fuzz_tests = b.addTest(.{ .root_module = fuzz_mod });
    const run_fuzz = b.addRunArtifact(fuzz_tests);
    test_step.dependOn(&run_fuzz.step);
    const fuzz_step = b.step("test-fuzz", "Property-based invariant fuzzer");
    fuzz_step.dependOn(&run_fuzz.step);

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
    
    if (optimize != .Debug) {
        tok_exe.root_module.strip = true;
    }

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

    // Real-tokenizer benches (Zig side of Zig-vs-Rust head-to-head)
    inline for (.{
        .{ "bench/bench_real_tinyllama.zig", "bench-real-tinyllama", "Bench real TinyLlama encode" },
        .{ "bench/bench_real_bert.zig", "bench-real-bert", "Bench real BERT-base-uncased encode" },
        .{ "bench/bench_real_t5.zig", "bench-real-t5", "Bench real T5-base encode" },
        .{ "bench/bench_parallel_bert.zig", "bench-parallel-bert", "Bench BERT parallel encodeBatch scaling" },
        .{ "bench/bench_offsets_bert.zig", "bench-offsets-bert", "Bench BERT encodeWithOffsets" },
    }) |bench_spec| {
        const bm = b.createModule(.{
            .root_source_file = b.path(bench_spec[0]),
            .target = target,
            .optimize = .ReleaseFast,
        });
        bm.addImport("tokenizers", mod);
        const be = b.addExecutable(.{
            .name = bench_spec[1],
            .root_module = bm,
        });
        const re = b.addRunArtifact(be);
        re.has_side_effects = true;
        const bs = b.step(bench_spec[1], bench_spec[2]);
        bs.dependOn(&re.step);
    }
}
