// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2019-2026 Binomial LLC; modifications Copyright (c) lapoule.dev
//
// Portions of this file are ported from Basis Universal v2.50 (tag `v2_50`,
// commit 9bebe16726b3a61c8c213eeee3b7cffb462ef34e),
// `transcoder/basisu_transcoder.cpp` and `transcoder/basisu_transcoder_uastc.h`,
// Copyright (C) 2019-2026 Binomial LLC, licensed under the Apache License,
// Version 2.0 (see LICENSE-APACHE-UPSTREAM and NOTICE). Each ported item says
// so at its head. Modified by lapoule.dev: translated from C++ to Rust, tables
// made `const`, the endpoint unquantisation table computed at compile time.

//! The UASTC LDR 4×4 format's tables: per-mode layout, the partition patterns
//! UASTC shares with ASTC and BC7, and ASTC's integer-sequence tables.

/// Number of UASTC modes, the solid-colour mode included.
pub(crate) const MODES: usize = 19;

/// The solid-colour mode.
pub(crate) const SOLID: usize = 8;

// Ported from Basis Universal v2.50, `g_uastc_mode_huff_codes` (code lengths
// only) and `g_uastc_huff_modes`. Apache-2.0, (C) Binomial LLC. Modified.
/// Bits each mode's prefix code takes; the 20th is reserved.
pub(crate) const MODE_CODE_BITS: [u8; MODES + 1] =
    [4, 6, 5, 5, 5, 5, 5, 5, 5, 5, 3, 2, 3, 5, 5, 7, 6, 6, 4, 7];

/// The mode the block's low seven bits decode to; 19 is reserved.
pub(crate) const MODE_FROM_CODE: [u8; 128] = [
    11, 0, 10, 3, 11, 15, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9, 11, 0, 10, 4, 11, 16, 12, 8, 11, 18,
    10, 6, 11, 2, 12, 13, 11, 0, 10, 3, 11, 17, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9, 11, 0, 10, 4,
    11, 1, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13, 11, 0, 10, 3, 11, 19, 12, 7, 11, 18, 10, 5, 11, 14,
    12, 9, 11, 0, 10, 4, 11, 16, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13, 11, 0, 10, 3, 11, 17, 12, 7,
    11, 18, 10, 5, 11, 14, 12, 9, 11, 0, 10, 4, 11, 1, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13,
];

// Ported from Basis Universal v2.50, the `g_uastc_mode_*` tables.
// Apache-2.0, (C) Binomial LLC. Modified.
/// Bits per weight.
pub(crate) const WEIGHT_BITS: [u8; MODES] =
    [4, 2, 3, 2, 2, 3, 2, 2, 0, 2, 4, 2, 3, 1, 2, 4, 2, 2, 5];
/// Weight BISE range.
pub(crate) const WEIGHT_RANGES: [u8; MODES] =
    [8, 2, 5, 2, 2, 5, 2, 2, 0, 2, 8, 2, 5, 0, 2, 8, 2, 2, 11];
/// Endpoint BISE range.
pub(crate) const ENDPOINT_RANGES: [u8; MODES] = [
    19, 20, 8, 7, 12, 20, 18, 12, 0, 8, 13, 13, 19, 20, 20, 20, 20, 20, 11,
];
/// Partitions.
pub(crate) const SUBSETS: [u8; MODES] = [1, 1, 2, 3, 2, 1, 1, 2, 0, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1];
/// Weight planes.
pub(crate) const PLANES: [u8; MODES] = [1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1];
/// Endpoint components: 2 luminance–alpha, 3 RGB, 4 RGBA.
pub(crate) const COMPS: [u8; MODES] = [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 3];
/// ASTC colour endpoint mode.
pub(crate) const CEM: [u8; MODES] = [
    8, 8, 8, 8, 8, 8, 8, 8, 0, 12, 12, 12, 12, 12, 12, 4, 4, 4, 8,
];
/// Transcoding-hint bits that follow the mode code.
pub(crate) const HINT_BITS: [u8; MODES] = [
    15, 15, 15, 15, 15, 15, 15, 15, 0, 23, 17, 17, 17, 23, 23, 23, 23, 23, 15,
];

// Ported from Basis Universal v2.50, `g_uastc_mode_astc_block_mode`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// The 11-bit ASTC block mode each UASTC mode transcodes to.
pub(crate) const ASTC_BLOCK_MODE: [u16; MODES] = [
    0x242, 0x42, 0x53, 0x42, 0x42, 0x53, 0x442, 0x42, 0, 0x42, 0x242, 0x442, 0x53, 0x441, 0x42,
    0x242, 0x42, 0x442, 0x253,
];

// Ported from Basis Universal v2.50, `g_astc_bise_range_table`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// ASTC's 21 integer-sequence ranges: (bits, trits, quints).
pub(crate) const BISE_RANGES: [[u8; 3]; 21] = [
    [1, 0, 0],
    [0, 1, 0],
    [2, 0, 0],
    [0, 0, 1],
    [1, 1, 0],
    [3, 0, 0],
    [1, 0, 1],
    [2, 1, 0],
    [4, 0, 0],
    [2, 0, 1],
    [3, 1, 0],
    [5, 0, 0],
    [3, 0, 1],
    [4, 1, 0],
    [6, 0, 0],
    [4, 0, 1],
    [5, 1, 0],
    [7, 0, 0],
    [5, 0, 1],
    [6, 1, 0],
    [8, 0, 0],
];

// Ported from Basis Universal v2.50, the ASTC seeds of
// `g_astc_bc7_common_partitions2`, `g_astc_bc7_common_partitions3` and
// `g_bc7_3_astc2_common_partitions`. Apache-2.0, (C) Binomial LLC. Modified.
/// ASTC partition seeds of the 30 two-subset patterns (modes 2, 4, 9, 16).
pub(crate) const SEEDS2: [u16; 30] = [
    28, 20, 16, 29, 91, 9, 107, 72, 149, 204, 50, 114, 496, 17, 78, 39, 252, 828, 43, 156, 116,
    210, 476, 273, 684, 359, 246, 195, 694, 524,
];
/// ASTC partition seeds of the 11 three-subset patterns (mode 3).
pub(crate) const SEEDS3: [u16; 11] = [260, 74, 32, 156, 183, 15, 745, 0, 335, 902, 254];
/// ASTC partition seeds of mode 7's 19 two-subset patterns.
pub(crate) const SEEDS2_MODE7: [u16; 19] = [
    36, 48, 61, 137, 161, 183, 226, 281, 302, 307, 479, 495, 593, 594, 605, 799, 812, 988, 993,
];

// Ported from Basis Universal v2.50, `g_astc_bc7_patterns2`,
// `g_astc_bc7_patterns3` and `g_bc7_3_astc2_patterns2`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// Subset of each texel, for the 30 two-subset patterns.
pub(crate) const PATTERNS2: [[u8; 16]; 30] = [
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
];

/// Subset of each texel, for the 11 three-subset patterns.
pub(crate) const PATTERNS3: [[u8; 16]; 11] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0],
    [0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2],
    [0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1],
    [2, 0, 0, 0, 2, 0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1],
    [2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2],
    [1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1],
    [0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2],
];

/// Subset of each texel, for mode 7's 19 two-subset patterns.
pub(crate) const PATTERNS2_MODE7: [[u8; 16]; 19] = [
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
];

// Ported from Basis Universal v2.50, `g_astc_bc7_pattern2_anchors`,
// `g_astc_bc7_pattern3_anchors` and `g_bc7_3_astc2_patterns2_anchors`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// First texel of each subset (its weight's top bit is implied 0).
pub(crate) const ANCHORS2: [[u8; 3]; 30] = [
    [0, 2, 0],
    [0, 3, 0],
    [1, 0, 0],
    [0, 3, 0],
    [7, 0, 0],
    [0, 2, 0],
    [3, 0, 0],
    [7, 0, 0],
    [0, 11, 0],
    [2, 0, 0],
    [0, 7, 0],
    [11, 0, 0],
    [3, 0, 0],
    [8, 0, 0],
    [0, 4, 0],
    [12, 0, 0],
    [1, 0, 0],
    [8, 0, 0],
    [0, 1, 0],
    [0, 2, 0],
    [0, 4, 0],
    [8, 0, 0],
    [1, 0, 0],
    [0, 2, 0],
    [4, 0, 0],
    [0, 1, 0],
    [4, 0, 0],
    [1, 0, 0],
    [4, 0, 0],
    [1, 0, 0],
];
/// First texel of each subset, three-subset patterns.
pub(crate) const ANCHORS3: [[u8; 3]; 11] = [
    [0, 8, 10],
    [8, 0, 12],
    [4, 0, 12],
    [8, 0, 4],
    [3, 0, 2],
    [0, 1, 3],
    [0, 2, 1],
    [1, 9, 0],
    [1, 2, 0],
    [4, 0, 8],
    [0, 6, 2],
];
/// First texel of each subset, mode 7's patterns.
pub(crate) const ANCHORS2_MODE7: [[u8; 3]; 19] = [
    [0, 4, 0],
    [0, 2, 0],
    [2, 0, 0],
    [0, 7, 0],
    [8, 0, 0],
    [0, 1, 0],
    [0, 3, 0],
    [0, 1, 0],
    [2, 0, 0],
    [0, 1, 0],
    [0, 8, 0],
    [2, 0, 0],
    [0, 1, 0],
    [0, 7, 0],
    [12, 0, 0],
    [2, 0, 0],
    [9, 0, 0],
    [0, 2, 0],
    [4, 0, 0],
];

// Ported from Basis Universal v2.50, `g_bc7_weights1..3`, `g_astc_weights4`
// and `g_astc_weights5` (as `g_astc_weight_tables`). Apache-2.0,
// (C) Binomial LLC. Modified.
/// Interpolation weight (0..=64) of each weight value, by bits per weight.
pub(crate) const WEIGHTS: [&[u8]; 6] = [
    &[],
    &[0, 64],
    &[0, 21, 43, 64],
    &[0, 9, 18, 27, 37, 46, 55, 64],
    &[0, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64],
    &[
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34, 36, 38, 40, 42, 44, 46, 48,
        50, 52, 54, 56, 58, 60, 62, 64,
    ],
];

// Ported from Basis Universal v2.50, `g_astc_trit_encode`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// The byte five trits (`t0 + 3·t1 + … + 81·t4`) encode to.
pub(crate) const TRIT_ENCODE: [u8; 243] = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 20, 21, 22, 24, 25, 26, 3, 7, 11, 19, 23, 27, 12, 13,
    14, 32, 33, 34, 36, 37, 38, 40, 41, 42, 48, 49, 50, 52, 53, 54, 56, 57, 58, 35, 39, 43, 51, 55,
    59, 44, 45, 46, 64, 65, 66, 68, 69, 70, 72, 73, 74, 80, 81, 82, 84, 85, 86, 88, 89, 90, 67, 71,
    75, 83, 87, 91, 76, 77, 78, 128, 129, 130, 132, 133, 134, 136, 137, 138, 144, 145, 146, 148,
    149, 150, 152, 153, 154, 131, 135, 139, 147, 151, 155, 140, 141, 142, 160, 161, 162, 164, 165,
    166, 168, 169, 170, 176, 177, 178, 180, 181, 182, 184, 185, 186, 163, 167, 171, 179, 183, 187,
    172, 173, 174, 192, 193, 194, 196, 197, 198, 200, 201, 202, 208, 209, 210, 212, 213, 214, 216,
    217, 218, 195, 199, 203, 211, 215, 219, 204, 205, 206, 96, 97, 98, 100, 101, 102, 104, 105,
    106, 112, 113, 114, 116, 117, 118, 120, 121, 122, 99, 103, 107, 115, 119, 123, 108, 109, 110,
    224, 225, 226, 228, 229, 230, 232, 233, 234, 240, 241, 242, 244, 245, 246, 248, 249, 250, 227,
    231, 235, 243, 247, 251, 236, 237, 238, 28, 29, 30, 60, 61, 62, 92, 93, 94, 156, 157, 158, 188,
    189, 190, 220, 221, 222, 31, 63, 95, 159, 191, 223, 124, 125, 126,
];

// Ported from Basis Universal v2.50, `g_astc_quint_encode`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// The seven bits three quints (`q0 + 5·q1 + 25·q2`) encode to.
pub(crate) const QUINT_ENCODE: [u8; 125] = [
    0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 5, 13, 21, 29, 6, 32,
    33, 34, 35, 36, 40, 41, 42, 43, 44, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 37, 45, 53, 61, 14,
    64, 65, 66, 67, 68, 72, 73, 74, 75, 76, 80, 81, 82, 83, 84, 88, 89, 90, 91, 92, 69, 77, 85, 93,
    22, 96, 97, 98, 99, 100, 104, 105, 106, 107, 108, 112, 113, 114, 115, 116, 120, 121, 122, 123,
    124, 101, 109, 117, 125, 30, 102, 103, 70, 71, 38, 110, 111, 78, 79, 46, 118, 119, 86, 87, 54,
    126, 127, 94, 95, 62, 39, 47, 55, 63, 31,
];

// Ported from Basis Universal v2.50, `g_astc_endpoint_unquant_params`
// (itself from the ASTC specification). Apache-2.0, (C) Binomial LLC.
// Modified.
/// Per range with trits or quints: the bit-swizzle string `B` and the
/// constant `C` of ASTC's endpoint unquantisation (Khronos Data Format 1.3,
/// §23.18).
const UNQUANT_PARAMS: [(&[u8; 9], u32); 21] = [
    (b"000000000", 0),
    (b"000000000", 0),
    (b"000000000", 0),
    (b"000000000", 0),
    (b"000000000", 204),
    (b"000000000", 0),
    (b"000000000", 113),
    (b"b000b0bb0", 93),
    (b"000000000", 0),
    (b"b0000bb00", 54),
    (b"cb000cbcb", 44),
    (b"000000000", 0),
    (b"cb0000cbc", 26),
    (b"dcb000dcb", 22),
    (b"000000000", 0),
    (b"dcb0000dc", 13),
    (b"edcb000ed", 11),
    (b"000000000", 0),
    (b"edcb0000e", 6),
    (b"fedcb000f", 5),
    (b"000000000", 0),
];

// Ported from Basis Universal v2.50, `unquant_astc_endpoint` and
// `unquant_astc_endpoint_val`. Apache-2.0, (C) Binomial LLC. Modified:
// `const fn`, no assertions.
/// The 0..=255 value the stored endpoint `packed` decodes to in `range`.
const fn unquant_endpoint(packed: u32, range: usize) -> u32 {
    let [bits, trits, quints] = BISE_RANGES[range];
    let bits = bits as u32;
    if trits == 0 && quints == 0 {
        // Bit replication up to 8 bits.
        let mut val = 0u32;
        let mut left = 8i32;
        while left > 0 {
            let n = if left < bits as i32 {
                left
            } else {
                bits as i32
            };
            let mut v = packed;
            if n < bits as i32 {
                v >>= bits as i32 - n;
            }
            val |= v << (left - n);
            left -= n;
        }
        val
    } else {
        let low = packed & ((1 << bits) - 1);
        let d = packed >> bits;
        let a = if low & 1 != 0 { 511 } else { 0 };
        let (swizzle, c) = UNQUANT_PARAMS[range];
        let mut b = 0u32;
        let mut i = 0;
        while i < 9 {
            b <<= 1;
            let ch = swizzle[i];
            if ch != b'0' {
                b |= (low >> (ch - b'a')) & 1;
            }
            i += 1;
        }
        let mut val = d * c + b;
        val ^= a;
        (a & 0x80) | (val >> 2)
    }
}

/// Levels in `range` (§23.12).
pub(crate) const fn levels(range: usize) -> usize {
    let [bits, trits, quints] = BISE_RANGES[range];
    (1 + 2 * trits as usize + 4 * quints as usize) << bits
}

const fn unquant_table() -> [[u8; 256]; 21] {
    let mut table = [[0u8; 256]; 21];
    let mut range = 0;
    while range < 21 {
        let [_, trits, quints] = BISE_RANGES[range];
        // Ranges with trits or quints but no constant are not used by UASTC
        // (upstream leaves them zero too).
        if (trits == 0 && quints == 0) || UNQUANT_PARAMS[range].1 != 0 {
            let mut v = 0;
            while v < levels(range) {
                table[range][v] = unquant_endpoint(v as u32, range) as u8;
                v += 1;
            }
        }
        range += 1;
    }
    table
}

/// `UNQUANT[range][stored]`: an endpoint's 0..=255 value.
pub(crate) const UNQUANT: [[u8; 256]; 21] = unquant_table();

#[cfg(test)]
mod tests {
    use super::*;

    /// ASTC's partition function (Khronos Data Format 1.3, §23.21), for a
    /// 4×4 block, written from the specification.
    fn astc_partition(seed: u32, x: u32, y: u32, count: u32) -> u8 {
        let (x, y) = (x << 1, y << 1); // small block
        let seed = seed + (count - 1) * 1024;
        let mut p = seed;
        p ^= p >> 15;
        p = p.wrapping_sub(p << 17);
        p = p.wrapping_add(p << 7);
        p = p.wrapping_add(p << 4);
        p ^= p >> 5;
        p = p.wrapping_add(p << 16);
        p ^= p >> 7;
        p ^= p >> 3;
        p ^= p << 6;
        p ^= p >> 17;
        let r = p;
        let mut s = [0u32; 12];
        for (i, shift) in [0, 4, 8, 12, 16, 20, 24, 28, 18, 22, 26].iter().enumerate() {
            s[i] = (r >> shift) & 0xf;
        }
        s[11] = r.rotate_right(30) & 0xf;
        for v in &mut s {
            *v *= *v;
        }
        let (sh1, sh2) = if seed & 1 != 0 {
            (
                if seed & 2 != 0 { 4 } else { 5 },
                if count == 3 { 6 } else { 5 },
            )
        } else {
            (
                if count == 3 { 6 } else { 5 },
                if seed & 2 != 0 { 4 } else { 5 },
            )
        };
        let sh3 = if seed & 0x10 != 0 { sh1 } else { sh2 };
        for (i, v) in s.iter_mut().enumerate() {
            *v >>= match i {
                0 | 2 | 4 | 6 => sh1,
                1 | 3 | 5 | 7 => sh2,
                _ => sh3,
            };
        }
        let a = (s[0] * x + s[1] * y + (r >> 14)) & 0x3f;
        let b = (s[2] * x + s[3] * y + (r >> 10)) & 0x3f;
        let mut c = (s[4] * x + s[5] * y + (r >> 6)) & 0x3f;
        if count < 3 {
            c = 0;
        }
        if a >= b && a >= c {
            0
        } else if b >= c {
            1
        } else {
            2
        }
    }

    /// **Every pattern is the ASTC partition its seed generates, and every
    /// anchor is its subset's first texel**: the tables agree with the
    /// specification, independently of upstream.
    #[test]
    fn patterns_and_anchors_follow_the_astc_partition_function() {
        type Set = (&'static [u16], &'static [[u8; 16]], &'static [[u8; 3]], u32);
        let sets: [Set; 3] = [
            (&SEEDS2, &PATTERNS2, &ANCHORS2, 2),
            (&SEEDS3, &PATTERNS3, &ANCHORS3, 3),
            (&SEEDS2_MODE7, &PATTERNS2_MODE7, &ANCHORS2_MODE7, 2),
        ];
        for (seeds, patterns, anchors, count) in sets {
            for ((&seed, pattern), anchor) in seeds.iter().zip(patterns).zip(anchors) {
                for (i, &subset) in pattern.iter().enumerate() {
                    let i = i as u32;
                    assert_eq!(
                        astc_partition(seed as u32, i % 4, i / 4, count),
                        subset,
                        "seed {seed}"
                    );
                }
                for s in 0..count as u8 {
                    let first = pattern.iter().position(|&p| p == s).expect("subset used");
                    assert_eq!(anchor[s as usize] as usize, first, "seed {seed} subset {s}");
                }
            }
        }
    }

    /// **The mode decode table inverts the prefix codes**: every 7-bit value
    /// decodes to the mode whose code it starts with.
    #[test]
    fn mode_codes_are_a_prefix_code() {
        const CODES: [u8; MODES + 1] = [
            0x1, 0x35, 0x1D, 0x3, 0x13, 0xB, 0x1B, 0x7, 0x17, 0xF, 0x2, 0x0, 0x6, 0x1F, 0xD, 0x5,
            0x15, 0x25, 0x9, 0x45,
        ];
        for v in 0..128u8 {
            let mode = MODE_FROM_CODE[v as usize] as usize;
            let bits = MODE_CODE_BITS[mode];
            assert_eq!(
                v & ((1 << bits) - 1),
                CODES[mode],
                "value {v:#x} → mode {mode}"
            );
        }
    }

    /// **Endpoint unquantisation matches the specification's tables** at a
    /// few known points, and is monotonic in the decoded order.
    #[test]
    fn endpoint_unquantisation() {
        // Bit ranges replicate: 5 bits 0b10101 → 0b10101101.
        assert_eq!(UNQUANT[11][0b10101], 0b1010_1101);
        assert_eq!(UNQUANT[20][200], 200);
        // Range 0..=5 (one bit + a trit): Table 168 of the specification.
        let want = [0, 255, 51, 204, 102, 153];
        for (v, &w) in want.iter().enumerate() {
            assert_eq!(UNQUANT[4][v], w, "range 4 value {v}");
        }
        for &range in &ENDPOINT_RANGES {
            let range = range as usize;
            let mut values: Vec<u8> = UNQUANT[range][..levels(range)].to_vec();
            assert_eq!(values[0], 0, "range {range}");
            values.sort_unstable();
            values.dedup();
            assert_eq!(values.len(), levels(range), "range {range} is injective");
            assert_eq!(*values.last().expect("levels"), 255, "range {range}");
        }
    }
}
