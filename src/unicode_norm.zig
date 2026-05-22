//! Partial Unicode normalization — NFD / NFC / NFKD / NFKC.
//!
//! Scope: precomposed Latin characters in Latin-1 Supplement (U+00C0–
//! U+017F) and Latin Extended-A (U+0180–U+024F) for which a canonical
//! single-step decomposition into base + combining mark is defined in
//! the Unicode Character Database. ~140 codepoints covered, which is
//! enough for most Western European text. Codepoints outside this
//! range pass through unchanged.
//!
//! **What this is NOT**: a production-grade Unicode normalizer. It
//! omits:
//!   - Hangul Jamo composition
//!   - CJK compatibility decompositions
//!   - Combining-mark canonical reordering
//!   - Multi-step decomposition recursion
//!   - The "composition exclusions" list (so NFC may compose a few
//!     codepoints that the spec excludes)
//!
//! Real multilingual production tokenizers using NFC/NFKC normally
//! ship a complete ICU-derived table. We document the partial coverage
//! as v0.4's honest non-claim.

const std = @import("std");

pub const Error = error{ OutOfMemory, InvalidUtf8 };

/// One-step canonical decomposition. Returns null if the codepoint has
/// no entry in our partial table. (cp_base, cp_mark) when decomposed.
fn decompose(cp: u21) ?[2]u21 {
    return switch (cp) {
        // Latin-1 supplement uppercase
        0x00C0 => .{ 0x0041, 0x0300 }, // À
        0x00C1 => .{ 0x0041, 0x0301 }, // Á
        0x00C2 => .{ 0x0041, 0x0302 }, // Â
        0x00C3 => .{ 0x0041, 0x0303 }, // Ã
        0x00C4 => .{ 0x0041, 0x0308 }, // Ä
        0x00C5 => .{ 0x0041, 0x030A }, // Å
        0x00C7 => .{ 0x0043, 0x0327 }, // Ç
        0x00C8 => .{ 0x0045, 0x0300 }, // È
        0x00C9 => .{ 0x0045, 0x0301 }, // É
        0x00CA => .{ 0x0045, 0x0302 }, // Ê
        0x00CB => .{ 0x0045, 0x0308 }, // Ë
        0x00CC => .{ 0x0049, 0x0300 }, // Ì
        0x00CD => .{ 0x0049, 0x0301 }, // Í
        0x00CE => .{ 0x0049, 0x0302 }, // Î
        0x00CF => .{ 0x0049, 0x0308 }, // Ï
        0x00D1 => .{ 0x004E, 0x0303 }, // Ñ
        0x00D2 => .{ 0x004F, 0x0300 }, // Ò
        0x00D3 => .{ 0x004F, 0x0301 }, // Ó
        0x00D4 => .{ 0x004F, 0x0302 }, // Ô
        0x00D5 => .{ 0x004F, 0x0303 }, // Õ
        0x00D6 => .{ 0x004F, 0x0308 }, // Ö
        0x00D9 => .{ 0x0055, 0x0300 }, // Ù
        0x00DA => .{ 0x0055, 0x0301 }, // Ú
        0x00DB => .{ 0x0055, 0x0302 }, // Û
        0x00DC => .{ 0x0055, 0x0308 }, // Ü
        0x00DD => .{ 0x0059, 0x0301 }, // Ý
        // Latin-1 supplement lowercase
        0x00E0 => .{ 0x0061, 0x0300 }, // à
        0x00E1 => .{ 0x0061, 0x0301 }, // á
        0x00E2 => .{ 0x0061, 0x0302 }, // â
        0x00E3 => .{ 0x0061, 0x0303 }, // ã
        0x00E4 => .{ 0x0061, 0x0308 }, // ä
        0x00E5 => .{ 0x0061, 0x030A }, // å
        0x00E7 => .{ 0x0063, 0x0327 }, // ç
        0x00E8 => .{ 0x0065, 0x0300 }, // è
        0x00E9 => .{ 0x0065, 0x0301 }, // é
        0x00EA => .{ 0x0065, 0x0302 }, // ê
        0x00EB => .{ 0x0065, 0x0308 }, // ë
        0x00EC => .{ 0x0069, 0x0300 }, // ì
        0x00ED => .{ 0x0069, 0x0301 }, // í
        0x00EE => .{ 0x0069, 0x0302 }, // î
        0x00EF => .{ 0x0069, 0x0308 }, // ï
        0x00F1 => .{ 0x006E, 0x0303 }, // ñ
        0x00F2 => .{ 0x006F, 0x0300 }, // ò
        0x00F3 => .{ 0x006F, 0x0301 }, // ó
        0x00F4 => .{ 0x006F, 0x0302 }, // ô
        0x00F5 => .{ 0x006F, 0x0303 }, // õ
        0x00F6 => .{ 0x006F, 0x0308 }, // ö
        0x00F9 => .{ 0x0075, 0x0300 }, // ù
        0x00FA => .{ 0x0075, 0x0301 }, // ú
        0x00FB => .{ 0x0075, 0x0302 }, // û
        0x00FC => .{ 0x0075, 0x0308 }, // ü
        0x00FD => .{ 0x0079, 0x0301 }, // ý
        0x00FF => .{ 0x0079, 0x0308 }, // ÿ
        // Latin Extended-A — broader coverage (Central + Eastern European)
        0x0100 => .{ 0x0041, 0x0304 }, // Ā
        0x0101 => .{ 0x0061, 0x0304 }, // ā
        0x0102 => .{ 0x0041, 0x0306 }, // Ă
        0x0103 => .{ 0x0061, 0x0306 }, // ă
        0x0104 => .{ 0x0041, 0x0328 }, // Ą
        0x0105 => .{ 0x0061, 0x0328 }, // ą
        0x0106 => .{ 0x0043, 0x0301 }, // Ć
        0x0107 => .{ 0x0063, 0x0301 }, // ć
        0x0108 => .{ 0x0043, 0x0302 }, // Ĉ
        0x0109 => .{ 0x0063, 0x0302 }, // ĉ
        0x010A => .{ 0x0043, 0x0307 }, // Ċ
        0x010B => .{ 0x0063, 0x0307 }, // ċ
        0x010C => .{ 0x0043, 0x030C }, // Č
        0x010D => .{ 0x0063, 0x030C }, // č
        0x010E => .{ 0x0044, 0x030C }, // Ď
        0x010F => .{ 0x0064, 0x030C }, // ď
        0x0112 => .{ 0x0045, 0x0304 }, // Ē
        0x0113 => .{ 0x0065, 0x0304 }, // ē
        0x0114 => .{ 0x0045, 0x0306 }, // Ĕ
        0x0115 => .{ 0x0065, 0x0306 }, // ĕ
        0x0116 => .{ 0x0045, 0x0307 }, // Ė
        0x0117 => .{ 0x0065, 0x0307 }, // ė
        0x0118 => .{ 0x0045, 0x0328 }, // Ę
        0x0119 => .{ 0x0065, 0x0328 }, // ę
        0x011A => .{ 0x0045, 0x030C }, // Ě
        0x011B => .{ 0x0065, 0x030C }, // ě
        0x011C => .{ 0x0047, 0x0302 }, // Ĝ
        0x011D => .{ 0x0067, 0x0302 }, // ĝ
        0x011E => .{ 0x0047, 0x0306 }, // Ğ
        0x011F => .{ 0x0067, 0x0306 }, // ğ
        0x0120 => .{ 0x0047, 0x0307 }, // Ġ
        0x0121 => .{ 0x0067, 0x0307 }, // ġ
        0x0122 => .{ 0x0047, 0x0327 }, // Ģ
        0x0123 => .{ 0x0067, 0x0327 }, // ģ
        0x0124 => .{ 0x0048, 0x0302 }, // Ĥ
        0x0125 => .{ 0x0068, 0x0302 }, // ĥ
        0x0128 => .{ 0x0049, 0x0303 }, // Ĩ
        0x0129 => .{ 0x0069, 0x0303 }, // ĩ
        0x012A => .{ 0x0049, 0x0304 }, // Ī
        0x012B => .{ 0x0069, 0x0304 }, // ī
        0x012C => .{ 0x0049, 0x0306 }, // Ĭ
        0x012D => .{ 0x0069, 0x0306 }, // ĭ
        0x012E => .{ 0x0049, 0x0328 }, // Į
        0x012F => .{ 0x0069, 0x0328 }, // į
        0x0130 => .{ 0x0049, 0x0307 }, // İ
        0x0134 => .{ 0x004A, 0x0302 }, // Ĵ
        0x0135 => .{ 0x006A, 0x0302 }, // ĵ
        0x0136 => .{ 0x004B, 0x0327 }, // Ķ
        0x0137 => .{ 0x006B, 0x0327 }, // ķ
        0x0139 => .{ 0x004C, 0x0301 }, // Ĺ
        0x013A => .{ 0x006C, 0x0301 }, // ĺ
        0x013B => .{ 0x004C, 0x0327 }, // Ļ
        0x013C => .{ 0x006C, 0x0327 }, // ļ
        0x013D => .{ 0x004C, 0x030C }, // Ľ
        0x013E => .{ 0x006C, 0x030C }, // ľ
        0x0143 => .{ 0x004E, 0x0301 }, // Ń
        0x0144 => .{ 0x006E, 0x0301 }, // ń
        0x0145 => .{ 0x004E, 0x0327 }, // Ņ
        0x0146 => .{ 0x006E, 0x0327 }, // ņ
        0x0147 => .{ 0x004E, 0x030C }, // Ň
        0x0148 => .{ 0x006E, 0x030C }, // ň
        0x014C => .{ 0x004F, 0x0304 }, // Ō
        0x014D => .{ 0x006F, 0x0304 }, // ō
        0x014E => .{ 0x004F, 0x0306 }, // Ŏ
        0x014F => .{ 0x006F, 0x0306 }, // ŏ
        0x0150 => .{ 0x004F, 0x030B }, // Ő
        0x0151 => .{ 0x006F, 0x030B }, // ő
        0x0154 => .{ 0x0052, 0x0301 }, // Ŕ
        0x0155 => .{ 0x0072, 0x0301 }, // ŕ
        0x0156 => .{ 0x0052, 0x0327 }, // Ŗ
        0x0157 => .{ 0x0072, 0x0327 }, // ŗ
        0x0158 => .{ 0x0052, 0x030C }, // Ř
        0x0159 => .{ 0x0072, 0x030C }, // ř
        0x015A => .{ 0x0053, 0x0301 }, // Ś
        0x015B => .{ 0x0073, 0x0301 }, // ś
        0x015C => .{ 0x0053, 0x0302 }, // Ŝ
        0x015D => .{ 0x0073, 0x0302 }, // ŝ
        0x015E => .{ 0x0053, 0x0327 }, // Ş
        0x015F => .{ 0x0073, 0x0327 }, // ş
        0x0160 => .{ 0x0053, 0x030C }, // Š
        0x0161 => .{ 0x0073, 0x030C }, // š
        0x0162 => .{ 0x0054, 0x0327 }, // Ţ
        0x0163 => .{ 0x0074, 0x0327 }, // ţ
        0x0164 => .{ 0x0054, 0x030C }, // Ť
        0x0165 => .{ 0x0074, 0x030C }, // ť
        0x0168 => .{ 0x0055, 0x0303 }, // Ũ
        0x0169 => .{ 0x0075, 0x0303 }, // ũ
        0x016A => .{ 0x0055, 0x0304 }, // Ū
        0x016B => .{ 0x0075, 0x0304 }, // ū
        0x016C => .{ 0x0055, 0x0306 }, // Ŭ
        0x016D => .{ 0x0075, 0x0306 }, // ŭ
        0x016E => .{ 0x0055, 0x030A }, // Ů
        0x016F => .{ 0x0075, 0x030A }, // ů
        0x0170 => .{ 0x0055, 0x030B }, // Ű
        0x0171 => .{ 0x0075, 0x030B }, // ű
        0x0172 => .{ 0x0055, 0x0328 }, // Ų
        0x0173 => .{ 0x0075, 0x0328 }, // ų
        0x0174 => .{ 0x0057, 0x0302 }, // Ŵ
        0x0175 => .{ 0x0077, 0x0302 }, // ŵ
        0x0176 => .{ 0x0059, 0x0302 }, // Ŷ
        0x0177 => .{ 0x0079, 0x0302 }, // ŷ
        0x0178 => .{ 0x0059, 0x0308 }, // Ÿ
        0x0179 => .{ 0x005A, 0x0301 }, // Ź
        0x017A => .{ 0x007A, 0x0301 }, // ź
        0x017B => .{ 0x005A, 0x0307 }, // Ż
        0x017C => .{ 0x007A, 0x0307 }, // ż
        0x017D => .{ 0x005A, 0x030C }, // Ž
        0x017E => .{ 0x007A, 0x030C }, // ž
        else => null,
    };
}

/// Inverse: (base, mark) → precomposed codepoint. Returns null if no
/// canonical composition exists in our partial table.
fn compose(base: u21, mark: u21) ?u21 {
    return switch (base) {
        0x0041 => switch (mark) { // A
            0x0300 => 0x00C0, 0x0301 => 0x00C1, 0x0302 => 0x00C2,
            0x0303 => 0x00C3, 0x0304 => 0x0100, 0x0306 => 0x0102,
            0x0308 => 0x00C4, 0x030A => 0x00C5, 0x0328 => 0x0104,
            else => null,
        },
        0x0043 => switch (mark) { // C
            0x0301 => 0x0106, 0x0302 => 0x0108, 0x0307 => 0x010A,
            0x0327 => 0x00C7, 0x030C => 0x010C,
            else => null,
        },
        0x0044 => switch (mark) { 0x030C => 0x010E, else => null }, // D
        0x0045 => switch (mark) { // E
            0x0300 => 0x00C8, 0x0301 => 0x00C9, 0x0302 => 0x00CA,
            0x0304 => 0x0112, 0x0306 => 0x0114, 0x0307 => 0x0116,
            0x0308 => 0x00CB, 0x0328 => 0x0118, 0x030C => 0x011A,
            else => null,
        },
        0x0047 => switch (mark) { // G
            0x0302 => 0x011C, 0x0306 => 0x011E, 0x0307 => 0x0120,
            0x0327 => 0x0122,
            else => null,
        },
        0x0048 => switch (mark) { 0x0302 => 0x0124, else => null }, // H
        0x0049 => switch (mark) { // I
            0x0300 => 0x00CC, 0x0301 => 0x00CD, 0x0302 => 0x00CE,
            0x0303 => 0x0128, 0x0304 => 0x012A, 0x0306 => 0x012C,
            0x0307 => 0x0130, 0x0308 => 0x00CF, 0x0328 => 0x012E,
            else => null,
        },
        0x004A => switch (mark) { 0x0302 => 0x0134, else => null }, // J
        0x004B => switch (mark) { 0x0327 => 0x0136, else => null }, // K
        0x004C => switch (mark) { // L
            0x0301 => 0x0139, 0x0327 => 0x013B, 0x030C => 0x013D,
            else => null,
        },
        0x004E => switch (mark) { // N
            0x0301 => 0x0143, 0x0303 => 0x00D1, 0x0327 => 0x0145,
            0x030C => 0x0147,
            else => null,
        },
        0x004F => switch (mark) { // O
            0x0300 => 0x00D2, 0x0301 => 0x00D3, 0x0302 => 0x00D4,
            0x0303 => 0x00D5, 0x0304 => 0x014C, 0x0306 => 0x014E,
            0x0308 => 0x00D6, 0x030B => 0x0150,
            else => null,
        },
        0x0052 => switch (mark) { // R
            0x0301 => 0x0154, 0x0327 => 0x0156, 0x030C => 0x0158,
            else => null,
        },
        0x0053 => switch (mark) { // S
            0x0301 => 0x015A, 0x0302 => 0x015C, 0x0327 => 0x015E,
            0x030C => 0x0160,
            else => null,
        },
        0x0054 => switch (mark) { // T
            0x0327 => 0x0162, 0x030C => 0x0164,
            else => null,
        },
        0x0055 => switch (mark) { // U
            0x0300 => 0x00D9, 0x0301 => 0x00DA, 0x0302 => 0x00DB,
            0x0303 => 0x0168, 0x0304 => 0x016A, 0x0306 => 0x016C,
            0x0308 => 0x00DC, 0x030A => 0x016E, 0x030B => 0x0170,
            0x0328 => 0x0172,
            else => null,
        },
        0x0057 => switch (mark) { 0x0302 => 0x0174, else => null }, // W
        0x0059 => switch (mark) { // Y
            0x0301 => 0x00DD, 0x0302 => 0x0176, 0x0308 => 0x0178,
            else => null,
        },
        0x005A => switch (mark) { // Z
            0x0301 => 0x0179, 0x0307 => 0x017B, 0x030C => 0x017D,
            else => null,
        },
        0x0061 => switch (mark) { // a
            0x0300 => 0x00E0, 0x0301 => 0x00E1, 0x0302 => 0x00E2,
            0x0303 => 0x00E3, 0x0304 => 0x0101, 0x0306 => 0x0103,
            0x0308 => 0x00E4, 0x030A => 0x00E5, 0x0328 => 0x0105,
            else => null,
        },
        0x0063 => switch (mark) { // c
            0x0301 => 0x0107, 0x0302 => 0x0109, 0x0307 => 0x010B,
            0x0327 => 0x00E7, 0x030C => 0x010D,
            else => null,
        },
        0x0064 => switch (mark) { 0x030C => 0x010F, else => null }, // d
        0x0065 => switch (mark) { // e
            0x0300 => 0x00E8, 0x0301 => 0x00E9, 0x0302 => 0x00EA,
            0x0304 => 0x0113, 0x0306 => 0x0115, 0x0307 => 0x0117,
            0x0308 => 0x00EB, 0x0328 => 0x0119, 0x030C => 0x011B,
            else => null,
        },
        0x0067 => switch (mark) { // g
            0x0302 => 0x011D, 0x0306 => 0x011F, 0x0307 => 0x0121,
            0x0327 => 0x0123,
            else => null,
        },
        0x0068 => switch (mark) { 0x0302 => 0x0125, else => null }, // h
        0x0069 => switch (mark) { // i
            0x0300 => 0x00EC, 0x0301 => 0x00ED, 0x0302 => 0x00EE,
            0x0303 => 0x0129, 0x0304 => 0x012B, 0x0306 => 0x012D,
            0x0308 => 0x00EF, 0x0328 => 0x012F,
            else => null,
        },
        0x006A => switch (mark) { 0x0302 => 0x0135, else => null }, // j
        0x006B => switch (mark) { 0x0327 => 0x0137, else => null }, // k
        0x006C => switch (mark) { // l
            0x0301 => 0x013A, 0x0327 => 0x013C, 0x030C => 0x013E,
            else => null,
        },
        0x006E => switch (mark) { // n
            0x0301 => 0x0144, 0x0303 => 0x00F1, 0x0327 => 0x0146,
            0x030C => 0x0148,
            else => null,
        },
        0x006F => switch (mark) { // o
            0x0300 => 0x00F2, 0x0301 => 0x00F3, 0x0302 => 0x00F4,
            0x0303 => 0x00F5, 0x0304 => 0x014D, 0x0306 => 0x014F,
            0x0308 => 0x00F6, 0x030B => 0x0151,
            else => null,
        },
        0x0072 => switch (mark) { // r
            0x0301 => 0x0155, 0x0327 => 0x0157, 0x030C => 0x0159,
            else => null,
        },
        0x0073 => switch (mark) { // s
            0x0301 => 0x015B, 0x0302 => 0x015D, 0x0327 => 0x015F,
            0x030C => 0x0161,
            else => null,
        },
        0x0074 => switch (mark) { // t
            0x0327 => 0x0163, 0x030C => 0x0165,
            else => null,
        },
        0x0075 => switch (mark) { // u
            0x0300 => 0x00F9, 0x0301 => 0x00FA, 0x0302 => 0x00FB,
            0x0303 => 0x0169, 0x0304 => 0x016B, 0x0306 => 0x016D,
            0x0308 => 0x00FC, 0x030A => 0x016F, 0x030B => 0x0171,
            0x0328 => 0x0173,
            else => null,
        },
        0x0077 => switch (mark) { 0x0302 => 0x0175, else => null }, // w
        0x0079 => switch (mark) { // y
            0x0301 => 0x00FD, 0x0302 => 0x0177, 0x0308 => 0x00FF,
            else => null,
        },
        0x007A => switch (mark) { // z
            0x0301 => 0x017A, 0x0307 => 0x017C, 0x030C => 0x017E,
            else => null,
        },
        else => null,
    };
}

fn appendCodepoint(out: *std.ArrayList(u8), alloc: std.mem.Allocator, cp: u21) !void {
    var buf: [4]u8 = undefined;
    const n = try std.unicode.utf8Encode(cp, &buf);
    try out.appendSlice(alloc, buf[0..n]);
}

/// NFD: decompose precomposed Latin codepoints into base + combining mark.
/// Codepoints not in our partial table pass through unchanged.
pub fn nfd(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    while (i < input.len) {
        const n = std.unicode.utf8ByteSequenceLength(input[i]) catch {
            try out.append(alloc, input[i]);
            i += 1;
            continue;
        };
        if (i + n > input.len) return Error.InvalidUtf8;
        const cp = std.unicode.utf8Decode(input[i..][0..n]) catch {
            try out.appendSlice(alloc, input[i..][0..n]);
            i += n;
            continue;
        };
        if (decompose(cp)) |pair| {
            try appendCodepoint(&out, alloc, pair[0]);
            try appendCodepoint(&out, alloc, pair[1]);
        } else {
            try out.appendSlice(alloc, input[i..][0..n]);
        }
        i += n;
    }
    return out.toOwnedSlice(alloc);
}

/// NFC: decompose then recompose. For codepoints we don't decompose,
/// passthrough; for combining marks we don't have a composition for,
/// leave as decomposed.
pub fn nfc(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    const decomposed = try nfd(alloc, input);
    defer alloc.free(decomposed);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    while (i < decomposed.len) {
        const n1 = std.unicode.utf8ByteSequenceLength(decomposed[i]) catch {
            try out.append(alloc, decomposed[i]);
            i += 1;
            continue;
        };
        if (i + n1 > decomposed.len) return Error.InvalidUtf8;
        const cp1 = std.unicode.utf8Decode(decomposed[i..][0..n1]) catch {
            try out.appendSlice(alloc, decomposed[i..][0..n1]);
            i += n1;
            continue;
        };

        if (i + n1 < decomposed.len) {
            const n2 = std.unicode.utf8ByteSequenceLength(decomposed[i + n1]) catch 0;
            if (n2 > 0 and i + n1 + n2 <= decomposed.len) {
                const cp2 = std.unicode.utf8Decode(decomposed[i + n1 ..][0..n2]) catch 0;
                if (compose(cp1, @intCast(cp2))) |composed| {
                    try appendCodepoint(&out, alloc, composed);
                    i += n1 + n2;
                    continue;
                }
            }
        }
        try out.appendSlice(alloc, decomposed[i..][0..n1]);
        i += n1;
    }
    return out.toOwnedSlice(alloc);
}

/// NFKD: alias for NFD on the partial coverage. Real NFKD applies
/// compatibility decompositions (e.g. ligatures, full-width / small
/// forms) that this implementation does not cover.
pub fn nfkd(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    return nfd(alloc, input);
}

/// NFKC: alias for NFC on the partial coverage. See nfkd note.
pub fn nfkc(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    return nfc(alloc, input);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "NFD decomposes é → e + combining acute" {
    const out = try nfd(testing.allocator, "é");
    defer testing.allocator.free(out);
    // é (U+00E9) → e (U+0065) + ◌́ (U+0301)
    try testing.expectEqualSlices(u8, &[_]u8{ 0x65, 0xCC, 0x81 }, out);
}

test "NFD passes ASCII through unchanged" {
    const out = try nfd(testing.allocator, "hello");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("hello", out);
}

test "NFD handles mixed Latin1 precomposed + ASCII" {
    const out = try nfd(testing.allocator, "café");
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(u8, &[_]u8{ 'c', 'a', 'f', 0x65, 0xCC, 0x81 }, out);
}

test "NFC composes e + combining acute → é" {
    // Input is e (0x65) + combining acute (U+0301 = 0xCC 0x81)
    const input = [_]u8{ 0x65, 0xCC, 0x81 };
    const out = try nfc(testing.allocator, &input);
    defer testing.allocator.free(out);
    // Expect é (U+00E9 = 0xC3 0xA9)
    try testing.expectEqualSlices(u8, &[_]u8{ 0xC3, 0xA9 }, out);
}

test "NFC is idempotent on already-precomposed text" {
    const out = try nfc(testing.allocator, "café");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("café", out);
}

test "NFD then NFC round-trips Latin1" {
    const original = "résumé naïve";
    const decomposed = try nfd(testing.allocator, original);
    defer testing.allocator.free(decomposed);
    const recomposed = try nfc(testing.allocator, decomposed);
    defer testing.allocator.free(recomposed);
    try testing.expectEqualStrings(original, recomposed);
}

test "NFD then NFC round-trips Latin Extended-A (Polish + Czech samples)" {
    const original = "Łódź Praha ČĎŇŘŤŽ Český"; // mix Polish + Czech precomposed
    const decomposed = try nfd(testing.allocator, original);
    defer testing.allocator.free(decomposed);
    const recomposed = try nfc(testing.allocator, decomposed);
    defer testing.allocator.free(recomposed);
    try testing.expectEqualStrings(original, recomposed);
}

test "NFD decomposes Polish ć into c + combining acute" {
    const out = try nfd(testing.allocator, "ć");
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(u8, &[_]u8{ 0x63, 0xCC, 0x81 }, out);
}

test "NFC composes Czech č from c + combining caron" {
    // Input: c (0x63) + combining caron (U+030C = 0xCC 0x8C)
    const input = [_]u8{ 0x63, 0xCC, 0x8C };
    const out = try nfc(testing.allocator, &input);
    defer testing.allocator.free(out);
    // Expected č (U+010D = 0xC4 0x8D)
    try testing.expectEqualSlices(u8, &[_]u8{ 0xC4, 0x8D }, out);
}

test "NFD passes non-Latin codepoints through" {
    const out = try nfd(testing.allocator, "日本語");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("日本語", out);
}

test "NFKD aliases NFD for partial coverage" {
    const a = try nfd(testing.allocator, "café");
    defer testing.allocator.free(a);
    const b = try nfkd(testing.allocator, "café");
    defer testing.allocator.free(b);
    try testing.expectEqualSlices(u8, a, b);
}
