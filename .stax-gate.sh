# stax-gate config for tokenizers-zig — generated 2026-06-15. Edit the bench line.
GATE_BUILD="zig build"
GATE_TEST="zig build test"
GATE_BENCH="zig build bench 2>&1"   # auto-detected step:bench — VERIFY emits one metric number
GATE_THRESHOLD_PCT=3
GATE_BENCH_RUNS=5
GATE_BENCH_METRIC="ns_per_encode"
GATE_LOWER_IS_BETTER=1
GATE_LANE="gate:tokenizers-zig"
