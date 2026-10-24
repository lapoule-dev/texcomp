// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2019-2026 Binomial LLC; modifications Copyright (c) lapoule.dev
//
// Ported from Basis Universal v2.50 (tag `v2_50`, commit
// 9bebe16726b3a61c8c213eeee3b7cffb462ef34e), `transcoder/basisu_transcoder.cpp`
// and `transcoder/basisu_transcoder_uastc.h`, Copyright (C) 2019-2026 Binomial
// LLC, licensed under the Apache License, Version 2.0 (see
// LICENSE-APACHE-UPSTREAM and NOTICE). Modified by lapoule.dev: translated
// from C++ to Rust, the transcoding hints skipped rather than read, the
// endian-specific word reads replaced by one byte-order-independent read.

//! Reading a UASTC block into its ASTC description, and the reference decode
//! of that description to texels.

use super::tables::{
    ANCHORS2, ANCHORS2_MODE7, ANCHORS3, BISE_RANGES, CEM, COMPS, ENDPOINT_RANGES, HINT_BITS, MODES,
    MODE_CODE_BITS, MODE_FROM_CODE, PATTERNS2, PATTERNS2_MODE7, PATTERNS3, PLANES, SEEDS2,
    SEEDS2_MODE7, SEEDS3, SOLID, SUBSETS, UNQUANT, WEIGHTS, WEIGHT_BITS, WEIGHT_RANGES,
};
use super::InvalidBlock;

/// A UASTC block read into the ASTC block it stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Unpacked {
    pub(crate) mode: usize,
    /// Index into the mode's pattern table (0 for one subset).
    pub(crate) pattern: usize,
    /// The colour of a solid block.
    pub(crate) solid: [u8; 4],
    pub(crate) subsets: usize,
    pub(crate) seed: u16,
    pub(crate) cem: u8,
    /// Colour component selector of the second weight plane.
    pub(crate) ccs: u8,
    pub(crate) dual_plane: bool,
    pub(crate) weight_range: u8,
    /// Stored (not unquantised) endpoint values, low/high per component.
    pub(crate) endpoints: [u8; 18],
    /// Weight values in texel order; dual plane interleaves the planes.
    pub(crate) weights: [u8; 32],
}

// Ported from Basis Universal v2.50, `read_bits1_to_9`, `read_bits1_to_9_fst`
// and `read_bit`. Apache-2.0, (C) Binomial LLC. Modified: one byte-wise read
// for every platform; a read past the block's end reads zeros.
fn read_bits(bytes: &[u8; 16], offset: &mut u32, count: u32) -> u32 {
    if count == 0 {
        return 0;
    }
    let at = (*offset >> 3) as usize;
    let shift = *offset & 7;
    let lo = bytes.get(at).copied().unwrap_or(0) as u32;
    let hi = bytes.get(at + 1).copied().unwrap_or(0) as u32;
    *offset += count;
    ((lo | (hi << 8)) >> shift) & ((1 << count) - 1)
}

/// The block's mode, or the reserved code.
pub(crate) fn mode(bytes: &[u8; 16]) -> Result<usize, InvalidBlock> {
    let mode = MODE_FROM_CODE[(bytes[0] & 127) as usize] as usize;
    if mode >= MODES {
        return Err(InvalidBlock::ReservedMode);
    }
    Ok(mode)
}

// Ported from Basis Universal v2.50, `get_anchor_indices`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// The block's partition pattern and the anchor texel of each subset.
pub(crate) fn pattern_and_anchors(
    subsets: usize,
    mode: usize,
    pattern: usize,
) -> (&'static [u8; 16], &'static [u8; 3]) {
    const ZERO_PATTERN: [u8; 16] = [0; 16];
    const ZERO_ANCHORS: [u8; 3] = [0; 3];
    match subsets {
        3 => (&PATTERNS3[pattern], &ANCHORS3[pattern]),
        2 if mode == 7 => (&PATTERNS2_MODE7[pattern], &ANCHORS2_MODE7[pattern]),
        2 => (&PATTERNS2[pattern], &ANCHORS2[pattern]),
        _ => (&ZERO_PATTERN, &ZERO_ANCHORS),
    }
}

// Ported from Basis Universal v2.50, `unpack_uastc(const uastc_block&,
// unpacked_uastc_block&, bool blue_contract_check, bool read_hints)`, with
// `read_hints` false. Apache-2.0, (C) Binomial LLC. Modified: hints are
// skipped, errors are typed, the weight bits are read as one 128-bit
// little-endian integer.
/// Reads a block. With `undo_blue_contraction`, swaps the endpoints (and
/// inverts the weights) of every RGB(A) subset whose high endpoint sums
/// below its low one, which ASTC would otherwise decode as blue-contracted.
pub(crate) fn unpack(
    bytes: &[u8; 16],
    undo_blue_contraction: bool,
) -> Result<Unpacked, InvalidBlock> {
    let mode = mode(bytes)?;
    let mut u = Unpacked {
        mode,
        pattern: 0,
        solid: [0; 4],
        subsets: 1,
        seed: 0,
        cem: 0,
        ccs: 0,
        dual_plane: false,
        weight_range: 0,
        endpoints: [0; 18],
        weights: [0; 32],
    };
    let mut offset = MODE_CODE_BITS[mode] as u32;

    if mode == SOLID {
        for c in &mut u.solid {
            *c = read_bits(bytes, &mut offset, 8) as u8;
        }
        return Ok(u);
    }

    offset += HINT_BITS[mode] as u32;

    match mode {
        2 | 4 | 7 | 9 | 16 => {
            u.pattern = read_bits(bytes, &mut offset, 5) as usize;
            u.subsets = 2;
        }
        3 => {
            u.pattern = read_bits(bytes, &mut offset, 4) as usize;
            u.subsets = 3;
        }
        _ => {}
    }

    u.seed = match mode {
        2 | 4 | 9 | 16 => *SEEDS2.get(u.pattern).ok_or(InvalidBlock::Pattern)?,
        3 => *SEEDS3.get(u.pattern).ok_or(InvalidBlock::Pattern)?,
        7 => *SEEDS2_MODE7.get(u.pattern).ok_or(InvalidBlock::Pattern)?,
        _ => 0,
    };

    let planes = match mode {
        6 | 11 | 13 => {
            u.ccs = read_bits(bytes, &mut offset, 2) as u8;
            2
        }
        17 => {
            u.ccs = 3;
            2
        }
        _ => 1,
    };
    debug_assert_eq!(planes, PLANES[mode] as usize);
    debug_assert_eq!(u.subsets, SUBSETS[mode] as usize);
    u.dual_plane = planes == 2;

    let comps = COMPS[mode] as usize;
    let weight_bits = WEIGHT_BITS[mode] as u32;
    u.weight_range = WEIGHT_RANGES[mode];
    u.cem = CEM[mode];

    // Endpoints: the trits or quints of all values first, packed as base-3
    // or base-5 bundles, then each value's low bits.
    let values = comps * 2 * u.subsets;
    let range = ENDPOINT_RANGES[mode] as usize;
    let [ep_bits, ep_trits, ep_quints] = BISE_RANGES[range];
    let (bundles, bundle, base) = if ep_trits != 0 {
        (values.div_ceil(5), 5, 3)
    } else if ep_quints != 0 {
        (values.div_ceil(3), 3, 5)
    } else {
        (0, 0, 0)
    };
    let mut tq = [0u32; 8];
    for (i, t) in tq.iter_mut().enumerate().take(bundles) {
        let mut bits = if ep_trits != 0 { 8 } else { 7 };
        if i == bundles - 1 {
            let remaining = values - (bundles - 1) * bundle;
            bits = match (ep_trits != 0, remaining) {
                (true, 1) => 2,
                (true, 2) => 4,
                (true, 3) => 5,
                (true, 4) => 7,
                (false, 1) => 3,
                (false, 2) => 5,
                _ => bits,
            };
        }
        *t = read_bits(bytes, &mut offset, bits);
    }
    let (mut accum, mut left, mut next) = (0u32, 0, 0);
    for e in u.endpoints.iter_mut().take(values) {
        let mut value = read_bits(bytes, &mut offset, ep_bits as u32);
        if bundles != 0 {
            if left == 0 {
                accum = tq[next];
                next += 1;
                left = bundle;
            }
            value |= (accum % base) << ep_bits;
            accum /= base;
            left -= 1;
        }
        *e = value as u8;
    }

    let (pattern, anchors) = pattern_and_anchors(u.subsets, mode, u.pattern);

    if mode == 18 {
        // The only mode with more than 64 weight bits.
        for (i, w) in u.weights.iter_mut().take(16).enumerate() {
            let bits = if i == 0 { weight_bits - 1 } else { weight_bits };
            *w = read_bits(bytes, &mut offset, bits) as u8;
        }
    } else {
        // The weights are the last ≤ 64 bits of the block.
        let bits = (u128::from_le_bytes(*bytes) >> offset) as u64;
        let mask = (1u64 << weight_bits) - 1;
        let anchor_mask = (1u64 << (weight_bits - 1)) - 1;
        let mut at = 0;
        let mut read = |anchor: bool| {
            let (m, n) = if anchor {
                (anchor_mask, weight_bits - 1)
            } else {
                (mask, weight_bits)
            };
            let v = ((bits >> at) & m) as u8;
            at += n;
            v
        };
        if planes == 2 {
            // One subset; the first texel's two weights are anchors.
            for (i, w) in u.weights.iter_mut().enumerate() {
                *w = read(i < 2);
            }
        } else {
            let [a0, a1, a2] = anchors.map(usize::from);
            for (i, w) in u.weights.iter_mut().take(16).enumerate() {
                let anchor = i == a0 || (u.subsets > 1 && (i == a1 || i == a2));
                *w = read(anchor);
            }
        }
    }

    if undo_blue_contraction && comps >= 3 {
        let unq = &UNQUANT[range];
        let mut invert = [false; 3];
        for (s, inv) in invert.iter_mut().enumerate().take(u.subsets) {
            let e = &mut u.endpoints[s * comps * 2..];
            let sum = |o: usize| -> i32 { (0..3).map(|c| unq[e[c * 2 + o] as usize] as i32).sum() };
            if sum(1) < sum(0) {
                for c in 0..comps {
                    e.swap(c * 2, c * 2 + 1);
                }
                *inv = true;
            }
        }
        if invert.iter().any(|&i| i) {
            let max = ((1u32 << weight_bits) - 1) as u8;
            for (i, &subset) in pattern.iter().enumerate() {
                if invert[subset as usize] {
                    for p in 0..planes {
                        let w = &mut u.weights[i * planes + p];
                        *w = max - *w;
                    }
                }
            }
        }
    }

    Ok(u)
}

// Ported from Basis Universal v2.50, `astc_interpolate`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// ASTC's LDR interpolation of two 8-bit endpoints by a 0..=64 weight, with
/// the sRGB or the linear endpoint expansion (Khronos Data Format 1.3,
/// §23.19).
fn interpolate(l: u32, h: u32, w: u32, srgb: bool) -> u8 {
    let (l, h) = if srgb {
        ((l << 8) | 0x80, (h << 8) | 0x80)
    } else {
        ((l << 8) | l, (h << 8) | h)
    };
    (((l * (64 - w) + h * w + 32) >> 6) >> 8) as u8
}

// Ported from Basis Universal v2.50, `unpack_uastc(uint32_t mode, uint32_t
// common_pattern, const color32& solid_color, const astc_block_desc& astc,
// color32* pPixels, bool srgb)`. Apache-2.0, (C) Binomial LLC. Modified.
/// The 16 texels, row-major, of a block read without undoing blue
/// contraction.
pub(crate) fn decode(u: &Unpacked, srgb: bool) -> [[u8; 4]; 16] {
    if u.mode == SOLID {
        return [u.solid; 16];
    }
    let comps = (COMPS[u.mode] as usize).min(4);
    let unq = &UNQUANT[ENDPOINT_RANGES[u.mode] as usize];
    let weight_bits = WEIGHT_BITS[u.mode] as usize;
    let weights = WEIGHTS[weight_bits];

    let mut ends = [[[0u32; 4]; 2]; 3];
    for (s, end) in ends.iter_mut().enumerate().take(u.subsets) {
        let e = &u.endpoints[s * comps * 2..];
        let v = |i: usize| unq[e[i] as usize] as u32;
        if comps == 2 {
            *end = [[v(0), v(0), v(0), v(2)], [v(1), v(1), v(1), v(3)]];
        } else {
            let [lo, hi] = end;
            for (c, (l, h)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                (*l, *h) = if c < comps {
                    (v(c * 2), v(c * 2 + 1))
                } else {
                    (255, 255)
                };
            }
        }
    }

    let mut colours = [[[0u8; 4]; 32]; 3];
    for (s, row) in colours.iter_mut().enumerate().take(u.subsets) {
        let [lo, hi] = ends[s];
        for (colour, &w) in row.iter_mut().zip(weights) {
            let w = w as u32;
            if comps == 2 {
                let l = interpolate(lo[0], hi[0], w, srgb);
                *colour = [l, l, l, interpolate(lo[3], hi[3], w, srgb)];
            } else {
                for c in 0..4 {
                    colour[c] = if c < comps {
                        interpolate(lo[c], hi[c], w, srgb)
                    } else {
                        255
                    };
                }
            }
        }
    }

    let (pattern, _) = pattern_and_anchors(u.subsets, u.mode, u.pattern);
    let mut texels = [[0u8; 4]; 16];
    if u.dual_plane {
        for (i, t) in texels.iter_mut().enumerate() {
            let (w0, w1) = (u.weights[i * 2] as usize, u.weights[i * 2 + 1] as usize);
            for c in 0..4 {
                t[c] = if c == u.ccs as usize {
                    colours[0][w1][c]
                } else {
                    colours[0][w0][c]
                };
            }
        }
    } else {
        for (i, t) in texels.iter_mut().enumerate() {
            *t = colours[pattern[i] as usize][u.weights[i] as usize];
        }
    }
    texels
}
