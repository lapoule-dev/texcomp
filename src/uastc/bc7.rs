// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2019-2026 Binomial LLC; modifications Copyright (c) lapoule.dev
//
// Ported from Basis Universal v2.50 (tag `v2_50`, commit
// 9bebe16726b3a61c8c213eeee3b7cffb462ef34e), `transcoder/basisu_transcoder.cpp`
// and `transcoder/basisu_transcoder_uastc.h`, Copyright (C) 2019-2026 Binomial
// LLC, licensed under the Apache License, Version 2.0 (see
// LICENSE-APACHE-UPSTREAM and NOTICE). Modified by lapoule.dev: translated
// from C++ to Rust, the solid-colour endpoint tables built once on first use,
// the index-selection bit (always 0 here) dropped. Each ported item says so at
// its head.

//! Writing the BC7 block a UASTC block transcodes to.

use std::sync::OnceLock;

use super::tables::{COMPS, ENDPOINT_RANGES, SOLID, UNQUANT, WEIGHTS};
use super::unpack::Unpacked;

// Ported from Basis Universal v2.50, the `m_bc7`/`m_invert`,
// `m_bc7`/`m_astc_to_bc7_perm` and `m_bc73`/`k` fields of
// `g_astc_bc7_common_partitions2`, `g_astc_bc7_common_partitions3` and
// `g_bc7_3_astc2_common_partitions`, and `g_astc_to_bc7_partition_index_perm_tables`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// For each two-subset UASTC pattern: the BC7 partition, and whether its
/// subsets are swapped.
pub(crate) const COMMON2: [(u8, bool); 30] = [
    (0, false),
    (1, false),
    (2, true),
    (3, false),
    (4, true),
    (5, false),
    (6, true),
    (7, true),
    (8, false),
    (9, true),
    (10, false),
    (11, true),
    (12, true),
    (13, true),
    (14, false),
    (15, true),
    (17, true),
    (18, true),
    (19, false),
    (20, false),
    (21, false),
    (22, true),
    (23, true),
    (24, false),
    (25, true),
    (26, false),
    (29, true),
    (32, true),
    (33, true),
    (52, true),
];
/// For each three-subset UASTC pattern: the BC7 partition, and the
/// permutation of its subsets.
const COMMON3: [(u8, u8); 11] = [
    (4, 0),
    (8, 5),
    (9, 5),
    (10, 2),
    (11, 2),
    (12, 0),
    (13, 4),
    (20, 1),
    (35, 1),
    (36, 5),
    (57, 0),
];
/// For each of mode 7's patterns: the three-subset BC7 partition, and how
/// its three subsets fold onto the two ASTC ones.
const COMMON2_MODE7: [(u8, u8); 19] = [
    (10, 4),
    (11, 4),
    (0, 3),
    (2, 4),
    (8, 5),
    (13, 4),
    (1, 2),
    (33, 2),
    (40, 3),
    (20, 4),
    (21, 0),
    (58, 3),
    (3, 0),
    (32, 2),
    (59, 1),
    (34, 3),
    (20, 1),
    (14, 4),
    (31, 3),
];
const ASTC_TO_BC7_PERM: [[u8; 3]; 6] = [
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
    [0, 2, 1],
    [1, 0, 2],
];

// Ported from Basis Universal v2.50, `bc7_convert_partition_index_3_to_2`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// The ASTC subset (0 or 1) that BC7 subset `p` (0..3) folds onto.
fn bc7_3_to_2(p: u32, k: u32) -> u32 {
    let p = match k >> 1 {
        0 => u32::from(p > 1),
        1 => u32::from(p != 0),
        _ => u32::from(p != 0 && p != 2),
    };
    if k & 1 != 0 {
        1 - p
    } else {
        p
    }
}

// Ported from Basis Universal v2.50, the BC7 tables (`g_bc7_partition2`,
// `g_bc7_partition3`, the anchor tables and the per-mode tables).
// Apache-2.0, (C) Binomial LLC. Modified.
pub(crate) const PARTITION2: [[u8; 16]; 64] = [
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
];
const PARTITION3: [[u8; 16]; 64] = [
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 2],
    [0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 2, 0, 0, 1, 2, 2, 1, 1, 2, 2, 1, 1],
    [0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2],
    [0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2],
    [0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2],
    [0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2],
    [0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2],
    [0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 2, 2, 2, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2],
    [0, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0],
    [0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
    [0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2],
    [0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2],
    [0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 0, 2, 2, 1, 0],
    [0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2],
    [0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1],
    [0, 0, 2, 2, 1, 1, 0, 2, 1, 1, 0, 2, 0, 0, 2, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 2, 2, 2, 2],
    [0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1],
    [0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 2, 2, 2],
    [0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 2],
    [0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
    [0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0],
    [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2],
    [0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1],
    [0, 2, 2, 0, 1, 2, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1],
    [0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1],
    [0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2],
    [0, 2, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1],
    [0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 2],
    [0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
    [0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2],
    [0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 0, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2],
    [0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2],
    [0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1],
    [0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2],
    [0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0],
];
const ANCHOR2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];
const ANCHOR3_1: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5,
    15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5,
    10, 8, 13, 15, 12, 3, 3,
];
const ANCHOR3_2: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6,
    10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];
const NUM_SUBSETS: [u8; 8] = [3, 2, 3, 2, 1, 1, 1, 2];
const PARTITION_BITS: [u8; 8] = [4, 6, 6, 6, 0, 0, 0, 6];
const COLOR_INDEX_BITS: [u8; 8] = [3, 3, 2, 2, 2, 2, 4, 2];
const HAS_P_BITS: [bool; 8] = [true, true, false, true, false, false, true, true];
const SHARED_P_BITS: [bool; 8] = [false, true, false, false, false, false, false, false];
const COLOR_PRECISION: [u8; 8] = [4, 6, 5, 7, 5, 7, 7, 5];
const ALPHA_PRECISION: [u8; 8] = [0, 0, 0, 0, 6, 8, 7, 5];
const ALPHA_INDEX_BITS: [u8; 8] = [0, 0, 0, 0, 3, 2, 4, 2];

/// A BC7 block before packing (upstream's `bc7_optimization_results`).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Bc7 {
    mode: u32,
    partition: u32,
    selectors: [u8; 16],
    alpha_selectors: [u8; 16],
    low: [[u8; 4]; 3],
    high: [[u8; 4]; 3],
    pbits: [[u32; 2]; 3],
    rotation: u32,
}

/// Best `(error, low, high)` endpoint pair for one 8-bit value.
#[derive(Clone, Copy, Debug)]
struct EndpointErr {
    error: u16,
    lo: u8,
    hi: u8,
}

/// BC7 mode 6's index 5 (weight 21) and mode 5's index 1 (weight 21).
const MODE_6_OPTIMAL_INDEX: u8 = 5;
const MODE_5_OPTIMAL_INDEX: u8 = 1;

struct SolidTables {
    /// `[c][pbit]`: mode 6 endpoints (7 bits + p-bit) hitting `c` at index 5.
    mode6: [[EndpointErr; 2]; 256],
    /// `[c]`: mode 5 endpoints (7 bits) hitting `c` at index 1.
    mode5: [EndpointErr; 256],
}

// Ported from Basis Universal v2.50, the BC7 part of `uastc_init`.
// Apache-2.0, (C) Binomial LLC. Modified: computed once, on first use.
fn solid_tables() -> &'static SolidTables {
    static TABLES: OnceLock<Box<SolidTables>> = OnceLock::new();
    TABLES.get_or_init(|| {
        let none = EndpointErr {
            error: u16::MAX,
            lo: 0,
            hi: 0,
        };
        let mut t = Box::new(SolidTables {
            mode6: [[none; 2]; 256],
            mode5: [none; 256],
        });
        let best = |c: i32, expand: &dyn Fn(u32) -> u32, w: u32| {
            let mut best = none;
            for l in 0..128u32 {
                let low = expand(l);
                for h in 0..128u32 {
                    let high = expand(h);
                    let k = ((low * (64 - w) + high * w + 32) >> 6) as i32;
                    let err = (k - c) * (k - c);
                    if err < best.error as i32 {
                        best = EndpointErr {
                            error: err as u16,
                            lo: l as u8,
                            hi: h as u8,
                        };
                    }
                }
            }
            best
        };
        // `g_bc7_weights4[5]` and `g_bc7_weights2[1]`.
        let (w6, w5) = (21, 21);
        for c in 0..256 {
            for lp in 0..2u32 {
                t.mode6[c][lp as usize] = best(c as i32, &|v| (v << 1) | lp, w6);
            }
            t.mode5[c] = best(c as i32, &|v| (v << 1) | (v >> 6), w5);
        }
        t
    })
}

// Ported from Basis Universal v2.50, `determine_shared_pbits` and
// `determine_unique_pbits`. Apache-2.0, (C) Binomial LLC. Modified: one
// function; the float arithmetic is kept operation for operation.
/// The endpoints (`comp_bits` bits) and p-bits nearest to `xl`/`xh` in
/// 0..=1; with `shared`, one p-bit for both endpoints.
fn pbits(
    shared: bool,
    comps: usize,
    comp_bits: u32,
    xl: &[f32; 4],
    xh: &[f32; 4],
) -> ([u8; 4], [u8; 4], [u32; 2]) {
    let total_bits = comp_bits + 1;
    let iscalep = (1i32 << total_bits) - 1;
    let scalep = iscalep as f32;
    let (mut best_min, mut best_max, mut best_p) = ([0u8; 4], [0u8; 4], [0u32; 2]);
    let (mut best_err, mut best_err0, mut best_err1) = (1e9f32, 1e9f32, 1e9f32);
    for p in 0..2i32 {
        let quant = |x: f32| -> u8 {
            let v = ((x * scalep - p as f32) / 2.0f32 + 0.5f32) as i32;
            (v * 2 + p).clamp(p, iscalep - 1 + p) as u8
        };
        let min = xl.map(quant);
        let max = xh.map(quant);
        // As upstream: stored as `u8`, widened to `int` for the shift.
        let scale = |v: u8| -> u8 {
            let s = ((v as u32) << (8 - total_bits)) as u8 as u32;
            (s | (s >> total_bits)) as u8
        };
        let (lo, hi) = (min.map(scale), max.map(scale));
        if shared {
            let mut err = 0f32;
            for i in 0..comps {
                let a = lo[i] as f32 / 255.0f32 - xl[i];
                let b = hi[i] as f32 / 255.0f32 - xh[i];
                err += a * a + b * b;
            }
            if err < best_err {
                best_err = err;
                best_p = [p as u32; 2];
                best_min = min.map(|v| v >> 1);
                best_max = max.map(|v| v >> 1);
            }
        } else {
            let (mut err0, mut err1) = (0f32, 0f32);
            for i in 0..comps {
                let a = lo[i] as f32 - xl[i] * 255.0f32;
                let b = hi[i] as f32 - xh[i] * 255.0f32;
                err0 += a * a;
                err1 += b * b;
            }
            if err0 < best_err0 {
                best_err0 = err0;
                best_p[0] = p as u32;
                best_min = min.map(|v| v >> 1);
            }
            if err1 < best_err1 {
                best_err1 = err1;
                best_p[1] = p as u32;
                best_max = max.map(|v| v >> 1);
            }
        }
    }
    (best_min, best_max, best_p)
}

// Ported from Basis Universal v2.50, `transcode_uastc_to_bc7(const
// unpacked_uastc_block&, bc7_optimization_results&)`. Apache-2.0,
// (C) Binomial LLC. Modified.
/// The unpacked BC7 block for a block read without undoing blue contraction.
pub(crate) fn transcode(u: &Unpacked) -> Bc7 {
    let mut d = Bc7::default();
    let mode = u.mode;
    let unq = &UNQUANT[ENDPOINT_RANGES[mode] as usize];
    let comps = COMPS[mode] as usize;
    let e = &u.endpoints;
    let f = |i: usize| unq[e[i] as usize] as f32 / 255.0f32;
    let q = |i: usize| unq[e[i] as usize] as u32;
    match mode {
        0 | 5 | 10 | 12 | 14 | 15 | 18 => {
            d.mode = 6;
            let (xl, xh) = if comps == 2 {
                ([f(0), f(0), f(0), f(2)], [f(1), f(1), f(1), f(3)])
            } else if comps == 4 {
                ([f(0), f(2), f(4), f(6)], [f(1), f(3), f(5), f(7)])
            } else {
                ([f(0), f(2), f(4), 1.0], [f(1), f(3), f(5), 1.0])
            };
            let (lo, hi, p) = pbits(false, if comps == 2 { 4 } else { comps }, 7, &xl, &xh);
            d.low[0] = lo;
            d.high[0] = hi;
            if comps == 3 {
                d.low[0][3] = 127;
                d.high[0][3] = 127;
            }
            d.pbits[0] = p;
            const B5_TO_4: [u8; 32] = [
                0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12,
                13, 13, 14, 14, 15, 15,
            ];
            for (s, &w) in d.selectors.iter_mut().zip(&u.weights[..16]) {
                *s = match mode {
                    18 => B5_TO_4[w as usize],
                    14 => [0, 5, 10, 15][w as usize],
                    5 | 12 => [0, 2, 4, 6, 9, 11, 13, 15][w as usize],
                    _ => w,
                };
            }
        }
        1 => {
            // Endpoint range 20: the stored values are the 8-bit values.
            d.mode = 3;
            let g = |i: usize| e[i] as f32 / 255.0f32;
            let xl = [g(0), g(2), g(4), 1.0];
            let xh = [g(1), g(3), g(5), 1.0];
            let (lo, hi, p) = pbits(false, 3, 7, &xl, &xh);
            for s in 0..2 {
                d.low[s][..3].copy_from_slice(&lo[..3]);
                d.high[s][..3].copy_from_slice(&hi[..3]);
                d.pbits[s] = p;
            }
            d.selectors.copy_from_slice(&u.weights[..16]);
        }
        2 => {
            d.mode = 1;
            let (part, invert) = COMMON2[u.pattern];
            d.partition = part as u32;
            for subset in 0..2 {
                let (mut xl, mut xh) = ([0f32, 0., 0., 1.], [0f32, 0., 0., 1.]);
                for i in 0..3 {
                    let v = e[i * 2 + subset * 6] as u32;
                    xl[i] = ((v << 4) | v) as f32 / 255.0f32;
                    let v = e[i * 2 + subset * 6 + 1] as u32;
                    xh[i] = ((v << 4) | v) as f32 / 255.0f32;
                }
                let (lo, hi, p) = pbits(true, 3, 6, &xl, &xh);
                let b = if invert { 1 - subset } else { subset };
                d.low[b][..3].copy_from_slice(&lo[..3]);
                d.high[b][..3].copy_from_slice(&hi[..3]);
                d.pbits[b][0] = p[0];
            }
            d.selectors.copy_from_slice(&u.weights[..16]);
        }
        3 => {
            d.mode = 2;
            let (part, perm) = COMMON3[u.pattern];
            d.partition = part as u32;
            for (subset, &b) in ASTC_TO_BC7_PERM[perm as usize].iter().enumerate() {
                let b = b as usize;
                for comp in 0..3 {
                    d.low[b][comp] = ((q(comp * 2 + subset * 6) * 31 + 127) / 255) as u8;
                    d.high[b][comp] = ((q(comp * 2 + 1 + subset * 6) * 31 + 127) / 255) as u8;
                }
            }
            d.selectors.copy_from_slice(&u.weights[..16]);
        }
        4 => {
            d.mode = 3;
            let (part, invert) = COMMON2[u.pattern];
            d.partition = part as u32;
            for subset in 0..2 {
                let (mut xl, mut xh) = ([0f32, 0., 0., 1.], [0f32, 0., 0., 1.]);
                for i in 0..3 {
                    xl[i] = f(i * 2 + subset * 6);
                    xh[i] = f(i * 2 + subset * 6 + 1);
                }
                let (lo, hi, p) = pbits(false, 3, 7, &xl, &xh);
                let b = if invert { 1 - subset } else { subset };
                d.low[b][..3].copy_from_slice(&lo[..3]);
                d.high[b][..3].copy_from_slice(&hi[..3]);
                d.low[b][3] = 127;
                d.high[b][3] = 127;
                d.pbits[b] = p;
            }
            d.selectors.copy_from_slice(&u.weights[..16]);
        }
        6 | 11 | 13 | 17 => {
            d.mode = 5;
            d.rotation = (u.ccs as u32 + 1) & 3;
            if comps == 2 {
                let l = ((q(0) * 127 + 127) / 255) as u8;
                let h = ((q(1) * 127 + 127) / 255) as u8;
                d.low[0] = [l, l, l, q(2) as u8];
                d.high[0] = [h, h, h, q(3) as u8];
            } else {
                for astc_comp in 0..4 {
                    // ASTC's second plane interpolates the CCS channel; BC7's
                    // interpolates alpha, then swaps it with the rotated one.
                    let bc7_comp = if astc_comp == u.ccs as usize {
                        3
                    } else if astc_comp == 3 {
                        u.ccs as usize
                    } else {
                        astc_comp
                    };
                    let (mut l, mut h) = (255, 255);
                    if astc_comp < comps {
                        l = q(astc_comp * 2);
                        h = q(astc_comp * 2 + 1);
                    }
                    if bc7_comp < 3 {
                        l = (l * 127 + 127) / 255;
                        h = (h * 127 + 127) / 255;
                    }
                    d.low[0][bc7_comp] = l as u8;
                    d.high[0][bc7_comp] = h as u8;
                }
            }
            for i in 0..16 {
                let (w0, w1) = (u.weights[i * 2], u.weights[i * 2 + 1]);
                if mode == 13 {
                    d.selectors[i] = if w0 != 0 { 3 } else { 0 };
                    d.alpha_selectors[i] = if w1 != 0 { 3 } else { 0 };
                } else {
                    d.selectors[i] = w0;
                    d.alpha_selectors[i] = w1;
                }
            }
        }
        7 => {
            d.mode = 2;
            let (part, k) = COMMON2_MODE7[u.pattern];
            d.partition = part as u32;
            for bc7_part in 0..3 {
                let astc_part = bc7_3_to_2(bc7_part as u32, k as u32) as usize;
                for c in 0..3 {
                    d.low[bc7_part][c] = ((q(c * 2 + astc_part * 6) * 31 + 127) / 255) as u8;
                    d.high[bc7_part][c] = ((q(c * 2 + 1 + astc_part * 6) * 31 + 127) / 255) as u8;
                }
            }
            d.selectors.copy_from_slice(&u.weights[..16]);
        }
        SOLID => {
            let t = solid_tables();
            let c = u.solid;
            let err =
                |p: usize| -> u32 { c.iter().map(|&v| t.mode6[v as usize][p].error as u32).sum() };
            let (err0, err1) = (err(0), err(1));
            if err0 > 0 && err1 > 0 {
                d.mode = 5;
                for (i, &v) in c.iter().take(3).enumerate() {
                    d.low[0][i] = t.mode5[v as usize].lo;
                    d.high[0][i] = t.mode5[v as usize].hi;
                }
                d.selectors = [MODE_5_OPTIMAL_INDEX; 16];
                d.low[0][3] = c[3];
                d.high[0][3] = c[3];
            } else {
                d.mode = 6;
                let p = usize::from(err1 < err0);
                for (i, &v) in c.iter().enumerate() {
                    d.low[0][i] = t.mode6[v as usize][p].lo;
                    d.high[0][i] = t.mode6[v as usize][p].hi;
                }
                d.pbits[0] = [p as u32; 2];
                d.selectors = [MODE_6_OPTIMAL_INDEX; 16];
            }
        }
        9 | 16 => {
            d.mode = 7;
            let (part, invert) = COMMON2[u.pattern];
            d.partition = part as u32;
            for subset in 0..2 {
                let (xl, xh) = if comps == 2 {
                    let o = subset * 4;
                    (
                        [f(o), f(o), f(o), f(o + 2)],
                        [f(o + 1), f(o + 1), f(o + 1), f(o + 3)],
                    )
                } else {
                    let o = subset * 8;
                    (
                        [f(o), f(o + 2), f(o + 4), f(o + 6)],
                        [f(o + 1), f(o + 3), f(o + 5), f(o + 7)],
                    )
                };
                let (lo, hi, p) = pbits(false, 4, 5, &xl, &xh);
                let b = if invert { 1 - subset } else { subset };
                d.low[b] = lo;
                d.high[b] = hi;
                d.pbits[b] = p;
            }
            d.selectors.copy_from_slice(&u.weights[..16]);
        }
        _ => unreachable!("every mode below MODES is handled"),
    }
    d
}

// Ported from Basis Universal v2.50, `bc7_set_block_bits`.
// Apache-2.0, (C) Binomial LLC. Modified.
fn set_bits(out: &mut [u8; 16], pos: &mut u32, mut val: u32, mut count: u32) {
    while count > 0 {
        let n = (8 - (*pos & 7)).min(count);
        out[(*pos >> 3) as usize] |= (val << (*pos & 7)) as u8;
        val >>= n;
        count -= n;
        *pos += n;
    }
}

// Ported from Basis Universal v2.50, `encode_bc7_block`.
// Apache-2.0, (C) Binomial LLC. Modified: the index-selection bit is 0.
/// Packs a BC7 block, flipping each subset whose anchor index has its top
/// bit set (BC7 stores anchors one bit short).
pub(crate) fn encode(r: &Bc7) -> [u8; 16] {
    let mode = r.mode as usize;
    let subsets = NUM_SUBSETS[mode] as usize;
    let partitions = 1u32 << PARTITION_BITS[mode];
    let part = r.partition as usize;
    let pattern: &[u8; 16] = match subsets {
        1 => &[0; 16],
        2 => &PARTITION2[part],
        _ => &PARTITION3[part],
    };
    let separate_alpha = mode == 4 || mode == 5;
    let (mut color, mut alpha) = (r.selectors, r.alpha_selectors);
    let (mut low, mut high, mut pbits) = (r.low, r.high, r.pbits);
    let mut anchor = [usize::MAX; 3];

    for k in 0..subsets {
        let a = match (subsets, k) {
            (_, 0) => 0,
            (3, 1) => ANCHOR3_1[part],
            (3, 2) => ANCHOR3_2[part],
            _ => ANCHOR2[part],
        } as usize;
        anchor[k] = a;

        let indices = 1u8 << COLOR_INDEX_BITS[mode];
        if color[a] & (indices >> 1) != 0 {
            for (s, &p) in color.iter_mut().zip(pattern) {
                if p as usize == k {
                    *s = (indices - 1) - *s;
                }
            }
            if separate_alpha {
                for q in 0..3 {
                    std::mem::swap(&mut low[k][q], &mut high[k][q]);
                }
            } else {
                std::mem::swap(&mut low[k], &mut high[k]);
            }
            if !SHARED_P_BITS[mode] {
                pbits[k].swap(0, 1);
            }
        }

        if separate_alpha {
            let indices = 1u8 << ALPHA_INDEX_BITS[mode];
            if alpha[a] & (indices >> 1) != 0 {
                for (s, &p) in alpha.iter_mut().zip(pattern) {
                    if p as usize == k {
                        *s = (indices - 1) - *s;
                    }
                }
                std::mem::swap(&mut low[k][3], &mut high[k][3]);
            }
        }
    }

    let mut out = [0u8; 16];
    let mut pos = 0;
    set_bits(&mut out, &mut pos, 1 << mode, mode as u32 + 1);
    if separate_alpha {
        set_bits(&mut out, &mut pos, r.rotation, 2);
    }
    if mode == 4 {
        set_bits(&mut out, &mut pos, 0, 1); // index selection
    }
    if partitions > 1 {
        set_bits(
            &mut out,
            &mut pos,
            r.partition,
            if partitions == 64 { 6 } else { 4 },
        );
    }
    let comps = if mode >= 4 { 4 } else { 3 };
    for comp in 0..comps {
        let bits = if comp == 3 {
            ALPHA_PRECISION[mode]
        } else {
            COLOR_PRECISION[mode]
        } as u32;
        for s in 0..subsets {
            set_bits(&mut out, &mut pos, low[s][comp] as u32, bits);
            set_bits(&mut out, &mut pos, high[s][comp] as u32, bits);
        }
    }
    if HAS_P_BITS[mode] {
        for p in pbits.iter().take(subsets) {
            set_bits(&mut out, &mut pos, p[0], 1);
            if !SHARED_P_BITS[mode] {
                set_bits(&mut out, &mut pos, p[1], 1);
            }
        }
    }
    let planes: &[(&[u8; 16], u8)] = if separate_alpha {
        &[
            (&color, COLOR_INDEX_BITS[mode]),
            (&alpha, ALPHA_INDEX_BITS[mode]),
        ]
    } else {
        &[(&color, COLOR_INDEX_BITS[mode])]
    };
    for &(sel, bits) in planes {
        for (i, &s) in sel.iter().enumerate() {
            let n = bits as u32 - u32::from(anchor.contains(&i));
            set_bits(&mut out, &mut pos, s as u32, n);
        }
    }
    debug_assert_eq!(pos, 128);
    out
}

// Ported from Basis Universal v2.50, `transcode_uastc_to_bc7(const
// uastc_block&, void*)`. Apache-2.0, (C) Binomial LLC. Modified.
/// The BC7 block of a block read without undoing blue contraction.
pub(crate) fn pack(u: &Unpacked) -> [u8; 16] {
    encode(&transcode(u))
}

// Ported from Basis Universal v2.50, `bc7u::unpack_bc7` and its per-mode
// helpers (`unpack_bc7_mode0_2`, `_mode1_3_7`, `_mode4_5`, `_mode6`).
// Apache-2.0, (C) Binomial LLC. Modified: one routine for all modes.
/// Decodes a BC7 block to its 16 texels, row-major. `None` for the reserved
/// mode (first byte zero).
pub(crate) fn decode(block: &[u8; 16]) -> Option<[[u8; 4]; 16]> {
    let mode = block[0].trailing_zeros() as usize;
    if mode > 7 {
        return None;
    }
    let mut pos = 0u32;
    let mut read = |n: u32| -> u32 {
        let mut v = 0u32;
        for i in 0..n {
            let bit = pos + i;
            v |= (((block[(bit >> 3) as usize] >> (bit & 7)) & 1) as u32) << i;
        }
        pos += n;
        v
    };
    read(mode as u32 + 1);
    let dequant = |v: u32, bits: u32| -> u32 {
        let v = v << (8 - bits);
        v | (v >> bits)
    };
    // BC7's weights: ASTC's 4-bit table differs from BC7's.
    const BC7_W4: [u8; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
    let interp = |l: u32, h: u32, w: u32, bits: u32| -> u8 {
        let wt = if bits == 4 {
            BC7_W4[w as usize]
        } else {
            WEIGHTS[bits as usize][w as usize]
        } as u32;
        ((l * (64 - wt) + h * wt + 32) >> 6) as u8
    };
    let mut px = [[0u8; 4]; 16];
    match mode {
        0 | 1 | 2 | 3 | 7 => {
            let subsets = NUM_SUBSETS[mode] as usize;
            let part = read(PARTITION_BITS[mode] as u32) as usize;
            let comps = if mode == 7 { 4 } else { 3 };
            let ebits = match mode {
                0 => 4,
                1 => 6,
                2 => 5,
                3 => 7,
                _ => 5,
            };
            let wbits = COLOR_INDEX_BITS[mode] as u32;
            let mut e = [[0u32; 4]; 6];
            for c in 0..comps {
                for ep in e.iter_mut().take(subsets * 2) {
                    ep[c] = read(ebits);
                }
            }
            if HAS_P_BITS[mode] {
                let mut p = [0u32; 6];
                let count = if SHARED_P_BITS[mode] {
                    subsets
                } else {
                    subsets * 2
                };
                for v in p.iter_mut().take(count) {
                    *v = read(1);
                }
                for (i, ep) in e.iter_mut().enumerate().take(subsets * 2) {
                    let pb = if SHARED_P_BITS[mode] { p[i >> 1] } else { p[i] };
                    for v in ep.iter_mut().take(comps) {
                        *v = dequant((*v << 1) | pb, ebits + 1);
                    }
                }
            } else {
                for ep in e.iter_mut().take(subsets * 2) {
                    for v in ep.iter_mut().take(comps) {
                        *v = dequant(*v, ebits);
                    }
                }
            }
            let pattern: &[u8; 16] = if subsets == 3 {
                &PARTITION3[part]
            } else {
                &PARTITION2[part]
            };
            let anchors = if subsets == 3 {
                [0, ANCHOR3_1[part] as usize, ANCHOR3_2[part] as usize]
            } else {
                [0, ANCHOR2[part] as usize, 0]
            };
            for (i, t) in px.iter_mut().enumerate() {
                let w = read(wbits - u32::from(anchors.contains(&i)));
                let s = pattern[i] as usize;
                for c in 0..comps {
                    t[c] = interp(e[s * 2][c], e[s * 2 + 1][c], w, wbits);
                }
                if comps == 3 {
                    t[3] = 255;
                }
            }
        }
        4 | 5 => {
            let rot = read(2);
            let index_mode = if mode == 4 { read(1) } else { 0 };
            let (ebits, abits) = if mode == 4 { (5, 6) } else { (7, 8) };
            let mut e = [[0u32; 4]; 2];
            for c in 0..4 {
                for ep in e.iter_mut() {
                    ep[c] = read(if c == 3 { abits } else { ebits });
                }
            }
            for ep in e.iter_mut() {
                for (c, v) in ep.iter_mut().enumerate() {
                    *v = dequant(*v, if c == 3 { abits } else { ebits });
                }
            }
            let (cw, aw) = (2u32, if mode == 4 { 3u32 } else { 2 });
            let wb = if index_mode != 0 { [aw, cw] } else { [cw, aw] };
            let mut first = [0u32; 16];
            let mut second = [0u32; 16];
            let im = index_mode as usize;
            for (i, w) in first.iter_mut().enumerate() {
                *w = read(wb[im] - u32::from(i == 0));
            }
            for (i, w) in second.iter_mut().enumerate() {
                *w = read(wb[1 - im] - u32::from(i == 0));
            }
            let (cwts, awts) = if index_mode != 0 {
                (second, first)
            } else {
                (first, second)
            };
            for (i, t) in px.iter_mut().enumerate() {
                for c in 0..3 {
                    t[c] = interp(e[0][c], e[1][c], cwts[i], wb[0]);
                }
                t[3] = interp(e[0][3], e[1][3], awts[i], wb[1]);
                if rot >= 1 {
                    t.swap(3, rot as usize - 1);
                }
            }
        }
        _ => {
            let mut e = [[0u32; 4]; 2];
            for c in 0..4 {
                for ep in e.iter_mut() {
                    ep[c] = read(7);
                }
            }
            let (p0, p1) = (read(1), read(1));
            for (ep, p) in e.iter_mut().zip([p0, p1]) {
                for v in ep.iter_mut() {
                    *v = (*v << 1) | p;
                }
            }
            for (i, t) in px.iter_mut().enumerate() {
                let w = read(4 - u32::from(i == 0));
                for c in 0..4 {
                    t[c] = interp(e[0][c], e[1][c], w, 4);
                }
            }
        }
    }
    Some(px)
}

#[cfg(test)]
mod tests {
    use super::super::tables::{PATTERNS2, PATTERNS2_MODE7, PATTERNS3};
    use super::*;

    /// **Each UASTC pattern maps onto a BC7 partition with the same
    /// texels**: two-subset patterns (inverted where flagged), three-subset
    /// ones through their permutation, and mode 7's three BC7 subsets folded
    /// onto its two. A wrong entry would put texels in the wrong subset.
    #[test]
    fn uastc_patterns_map_onto_bc7_partitions() {
        for (pattern, &(part, invert)) in PATTERNS2.iter().zip(&COMMON2) {
            for (i, &s) in pattern.iter().enumerate() {
                let bc7 = PARTITION2[part as usize][i];
                assert_eq!(if invert { 1 - s } else { s }, bc7, "partition {part}");
            }
        }
        for (pattern, &(part, perm)) in PATTERNS3.iter().zip(&COMMON3) {
            for (i, &s) in pattern.iter().enumerate() {
                let bc7 = PARTITION3[part as usize][i];
                assert_eq!(
                    ASTC_TO_BC7_PERM[perm as usize][s as usize], bc7,
                    "partition {part}"
                );
            }
        }
        for (pattern, &(part, k)) in PATTERNS2_MODE7.iter().zip(&COMMON2_MODE7) {
            for (i, &s) in pattern.iter().enumerate() {
                let bc7 = PARTITION3[part as usize][i];
                assert_eq!(
                    bc7_3_to_2(bc7 as u32, k as u32),
                    s as u32,
                    "partition {part}"
                );
            }
        }
    }

    /// **The solid-colour tables reach every value exactly where BC7 can**:
    /// mode 5's 7-bit endpoints at weight 21/64 hit all 256 values, and each
    /// entry's error is what its endpoints give.
    #[test]
    fn solid_tables_are_consistent() {
        let t = solid_tables();
        for c in 0..256usize {
            let e = t.mode5[c];
            let (l, h) = (e.lo as i32, e.hi as i32);
            let (l, h) = ((l << 1) | (l >> 6), (h << 1) | (h >> 6));
            let k = (l * 43 + h * 21 + 32) >> 6;
            assert_eq!(
                ((k - c as i32) * (k - c as i32)) as u16,
                e.error,
                "mode 5, {c}"
            );
            for p in 0..2 {
                let e = t.mode6[c][p];
                let (l, h) = (
                    ((e.lo as i32) << 1) | p as i32,
                    ((e.hi as i32) << 1) | p as i32,
                );
                let k = (l * 43 + h * 21 + 32) >> 6;
                assert_eq!(
                    ((k - c as i32) * (k - c as i32)) as u16,
                    e.error,
                    "mode 6, {c}"
                );
            }
        }
        assert!(t.mode5.iter().all(|e| e.error == 0));
    }
}
