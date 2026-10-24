// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2019-2026 Binomial LLC; modifications Copyright (c) lapoule.dev
//
// Ported from Basis Universal v2.50 (tag `v2_50`, commit
// 9bebe16726b3a61c8c213eeee3b7cffb462ef34e), `encoder/basisu_uastc_enc.cpp`,
// Copyright (C) 2019-2026 Binomial LLC, licensed under the Apache License,
// Version 2.0 (see LICENSE-APACHE-UPSTREAM and NOTICE). Modified by
// lapoule.dev: translated from C++ to Rust; only the effort levels 0–2 and
// the modes they try are kept; no RDO. Every floating-point expression keeps
// upstream's types and operation order. Each ported item says so at its head.

//! The UASTC LDR 4×4 encoder.

// The loops keep upstream's indexing, which keeps the port reviewable line
// for line against it.
#![allow(clippy::needless_range_loop)]

use super::bc7;
use super::cell::{self, Effort, Params, Weights, ASTC_4, BC7_2, BC7_3};
use super::etc::{self, EtcBlock, EAC_TABLES};
use super::tables::{
    BISE_RANGES, COMPS, ENDPOINT_RANGES, HAS_ALPHA, HAS_BC1_HINT0, HAS_BC1_HINT1, HAS_ETC1_BIAS,
    IS_LA, MODE_CODES, MODE_CODE_BITS, PLANES, SOLID, SUBSETS, UNQUANT, WEIGHT_BITS, WEIGHT_RANGES,
};
use super::unpack::{self, pattern_and_anchors, Unpacked};
use super::Level;

type Block = [[u8; 4]; 16];

/// One candidate: an ASTC block in UASTC terms and its error.
#[derive(Clone, Copy)]
struct Candidate {
    u: Unpacked,
    err: u64,
}

fn candidate(
    mode: usize,
    pattern: usize,
    endpoints: &[u8],
    weights: &[u8],
    ccs: u8,
    err: u64,
) -> Candidate {
    let mut u = Unpacked {
        mode,
        pattern,
        solid: [0; 4],
        subsets: SUBSETS[mode] as usize,
        seed: 0,
        cem: 0,
        ccs,
        dual_plane: PLANES[mode] == 2,
        weight_range: WEIGHT_RANGES[mode],
        endpoints: [0; 18],
        weights: [0; 32],
    };
    u.endpoints[..endpoints.len()].copy_from_slice(endpoints);
    u.weights[..weights.len()].copy_from_slice(weights);
    Candidate { u, err }
}

/// The per-level search settings (upstream's switch on the level).
struct Config {
    mode_mask: u32,
    uber_level: u32,
    estimate_partition: bool,
    always_try_alpha_modes: bool,
    eac_mul_search_rad: u32,
    eac_table_mask: u32,
    least_squares_passes: u32,
    bc1_hints: bool,
    only_use_la_on_transparent_blocks: bool,
    level: u32,
}

// Ported from Basis Universal v2.50, the level switch of `encode_uastc`.
// Apache-2.0, (C) Binomial LLC. Modified.
fn config(level: Level) -> Config {
    match level {
        Level::Fastest => Config {
            mode_mask: (1 << 0) | (1 << 8) | (1 << 11) | (1 << 12) | (1 << 15),
            uber_level: 0,
            estimate_partition: true,
            always_try_alpha_modes: false,
            eac_mul_search_rad: 0,
            eac_table_mask: (1 << 2) | (1 << 8) | (1 << 11) | (1 << 13),
            least_squares_passes: 1,
            bc1_hints: false,
            only_use_la_on_transparent_blocks: true,
            level: 0,
        },
    }
}

fn unq(range: usize, v: u8) -> i32 {
    UNQUANT[range][v as usize] as i32
}

/// Upstream's endpoint-order check: when the high endpoint's RGB sum is
/// below the low one's, swap every pair in `pairs` and invert the weights.
fn order(endpoints: &mut [u8], range: usize, pairs: usize) -> bool {
    let s0 = unq(range, endpoints[0]) + unq(range, endpoints[2]) + unq(range, endpoints[4]);
    let s1 = unq(range, endpoints[1]) + unq(range, endpoints[3]) + unq(range, endpoints[5]);
    if s1 < s0 {
        for p in 0..pairs {
            endpoints.swap(p * 2, p * 2 + 1);
        }
        true
    } else {
        false
    }
}

fn fit(
    pixels: &[[u8; 4]],
    weights: Weights,
    range: usize,
    has_alpha: bool,
    effort: Effort,
) -> cell::Fit {
    cell::compress(
        &Params {
            pixels,
            weights,
            range,
            has_alpha,
        },
        effort,
    )
}

// Ported from Basis Universal v2.50, `astc_mode0_or_18` (mode 0).
// Apache-2.0, (C) Binomial LLC. Modified.
fn mode0(block: &Block, out: &mut Vec<Candidate>, effort: Effort) {
    let range = 19;
    let f = fit(block, ASTC_4, range, false, effort);
    let mut e = [0u8; 6];
    for c in 0..3 {
        e[c * 2] = f.astc_low[c];
        e[c * 2 + 1] = f.astc_high[c];
    }
    let invert = order(&mut e, range, 3);
    let w = f.selectors.map(|s| if invert { 15 - s } else { s });
    out.push(candidate(0, 0, &e, &w, 0, f.err));
}

// Ported from Basis Universal v2.50, `astc_mode12`. Apache-2.0,
// (C) Binomial LLC. Modified.
fn mode12(block: &Block, out: &mut Vec<Candidate>, effort: Effort) {
    let range = 19;
    let f = fit(block, BC7_3, range, true, effort);
    let mut e = [0u8; 8];
    for c in 0..4 {
        e[c * 2] = f.astc_low[c];
        e[c * 2 + 1] = f.astc_high[c];
    }
    let invert = order(&mut e, range, 4);
    let w = f.selectors.map(|s| if invert { 7 - s } else { s });
    out.push(candidate(12, 0, &e, &w, 0, f.err));
}

// Ported from Basis Universal v2.50, `astc_mode11_or_17` (mode 11).
// Apache-2.0, (C) Binomial LLC. Modified.
fn mode11(block: &Block, out: &mut Vec<Candidate>, effort: Effort) {
    let range = 13;
    for rot in 0..4usize {
        let mut rgb = *block;
        let mut a = *block;
        for i in 0..16 {
            let c = a[i][rot];
            a[i] = [c, c, c, 255];
            rgb[i][rot] = rgb[i][3];
            rgb[i][3] = 255;
        }
        let fr = fit(&rgb, BC7_2, range, false, effort);
        let fa = fit(&a, BC7_2, range, false, effort);
        let total = fr.err + fa.err / 3;
        let pick = |c: usize| if rot == c { &fa } else { &fr };
        let mut e = [0u8; 8];
        for c in 0..3 {
            e[c * 2] = pick(c).astc_low[c];
            e[c * 2 + 1] = pick(c).astc_high[c];
        }
        if rot == 3 {
            e[6] = fa.astc_low[0];
            e[7] = fa.astc_high[0];
        } else {
            e[6] = fr.astc_low[rot];
            e[7] = fr.astc_high[rot];
        }
        let invert = order(&mut e, range, 4);
        let mut w = [0u8; 32];
        for i in 0..16 {
            let (ri, ai) = (fr.selectors[i], fa.selectors[i]);
            w[i * 2] = if invert { 3 - ri } else { ri };
            w[i * 2 + 1] = if invert { 3 - ai } else { ai };
        }
        out.push(candidate(11, 0, &e, &w, rot as u8, total));
    }
}

// Ported from Basis Universal v2.50, `astc_mode15`. Apache-2.0,
// (C) Binomial LLC. Modified.
fn mode15(block: &Block, out: &mut Vec<Candidate>, effort: Effort) {
    let range = 20;
    let la: Block = block.map(|c| [c[0], 0, 0, c[3]]);
    let f = fit(&la, ASTC_4, range, true, effort);
    let e = [f.astc_low[0], f.astc_high[0], f.astc_low[3], f.astc_high[3]];
    let mut colors = [[0u8; 4]; 16];
    for c in 0..4 {
        let k = if c < 3 { 0 } else { 3 };
        colors[0][c] = unq(range, f.astc_low[k]) as u8;
        colors[15][c] = unq(range, f.astc_high[k]) as u8;
    }
    for i in 1..15 {
        for c in 0..4 {
            colors[i][c] = cell::interpolate_linear(
                colors[0][c] as u32,
                colors[15][c] as u32,
                ASTC_4.w[i] as u32,
            ) as u8;
        }
    }
    let err = block
        .iter()
        .zip(&f.selectors)
        .map(|(b, &s)| dist_la(b, &colors[s as usize]))
        .sum();
    out.push(candidate(15, 0, &e, &f.selectors, 0, err));
}

fn dist_la(a: &[u8; 4], b: &[u8; 4]) -> u64 {
    let dl = a[0] as i32 - b[0] as i32;
    let da = a[3] as i32 - b[3] as i32;
    (dl * dl + da * da) as u64
}

fn dist_rgba(a: &[u8; 4], b: &[u8; 4]) -> u64 {
    (0..4)
        .map(|c| (a[c] as i32 - b[c] as i32).pow(2) as u64)
        .sum()
}

/// `compute_block_error`: (RGB, RGBA, LA) squared errors.
fn block_error(a: &Block, b: &Block) -> (u64, u64, u64) {
    let mut e = [0u64; 4];
    for (x, y) in a.iter().zip(b) {
        for c in 0..4 {
            e[c] += (x[c] as i32 - y[c] as i32).pow(2) as u64;
        }
    }
    let rgb = e[0] + e[1] + e[2];
    (rgb, rgb + e[3], e[0] + e[3])
}

// Ported from Basis Universal v2.50, `uastc_pack_eac_a8`. Apache-2.0,
// (C) Binomial LLC. Modified: returns the table and multiplier only (the
// hint), as the encoder uses.
fn pack_eac_a8(alpha: &[u8; 16], base_rad: u32, mul_rad: u32, table_mask: u32) -> (u32, u32) {
    let min = *alpha.iter().min().expect("16 texels") as u32;
    let max = *alpha.iter().max().expect("16 texels") as u32;
    if min == max {
        return (13, 1);
    }
    let range_a = max - min;
    let (mut best_err, mut best) = (u64::MAX, (0u32, 0u32));
    for table in 0..16usize {
        if table_mask & (1 << table) == 0 {
            continue;
        }
        let t = &EAC_TABLES[table];
        let range = (t[7] as i32 - t[3] as i32) as f32;
        let lerp_t = (0 - t[3] as i32) as f32 / range;
        let center = (min as f32 + (max as f32 - min as f32) * lerp_t).round() as i32;
        let base_min = (center - base_rad as i32).clamp(0, 255);
        let base_max = (center + base_rad as i32).clamp(0, 255);
        let mul = (range_a as f32 / range).round() as i32;
        let mul_low = (mul - mul_rad as i32).clamp(1, 15);
        let mul_high = (mul + mul_rad as i32).clamp(1, 15);
        for base in base_min..=base_max {
            for m in mul_low..=mul_high {
                let mut total = 0u64;
                for &a in alpha {
                    let mut best_s = u32::MAX;
                    for &s in t {
                        let v = (m * s as i32 + base).clamp(0, 255);
                        let err = (a as i32 - v).unsigned_abs();
                        if err < best_s {
                            best_s = err;
                        }
                    }
                    total += (best_s * best_s) as u64;
                    if total >= best_err {
                        break;
                    }
                }
                if total < best_err {
                    best_err = total;
                    best = (table as u32, m as u32);
                    if best_err == 0 {
                        return best;
                    }
                }
            }
        }
    }
    best
}

#[derive(Clone, Copy)]
struct Ycbcr {
    y: i32,
    cb: i32,
    cr: i32,
}

fn ycbcr(c: &[u8; 4]) -> Ycbcr {
    let y = c[0] as i32 * 54 + c[1] as i32 * 183 + c[2] as i32 * 19;
    Ycbcr {
        y,
        cb: ((c[2] as i32) << 8) - y,
        cr: ((c[0] as i32) << 8) - y,
    }
}

fn color_diff(a: &Ycbcr, b: &Ycbcr) -> u64 {
    let (dy, dcb, dcr) = (
        (a.y - b.y) as i64,
        (a.cb - b.cb) as i64,
        (a.cr - b.cr) as i64,
    );
    (dy * dy * 4 + dcr * dcr + dcb * dcb) as u64
}

fn gray_distance2(c: &[u8; 4], r: i32, g: i32, b: i32) -> i32 {
    let gd = ((c[0] as i32 - r) + (c[1] as i32 - g) + (c[2] as i32 - b) + 1) / 3;
    let (pr, pg, pb) = (
        (r + gd).clamp(0, 255),
        (g + gd).clamp(0, 255),
        (b + gd).clamp(0, 255),
    );
    let (dr, dg, db) = (c[0] as i32 - pr, c[1] as i32 - pg, c[2] as i32 - pb);
    dr * dr + dg * dg + db * db
}

// Ported from Basis Universal v2.50, `pack_etc1_estimate_flipped`.
// Apache-2.0, (C) Binomial LLC. Modified.
fn estimate_flipped(px: &Block) -> bool {
    let at = |x: usize, y: usize| &px[x + y * 4];
    let mut sums = [[[0i32; 2]; 2]; 3];
    for (c, s) in sums.iter_mut().enumerate() {
        let g = |x, y| at(x, y)[c] as i32;
        s[0][0] = g(0, 0) + g(0, 1) + g(1, 0) + g(1, 1);
        s[1][0] = g(2, 0) + g(2, 1) + g(3, 0) + g(3, 1);
        s[0][1] = g(0, 2) + g(0, 3) + g(1, 2) + g(1, 3);
        s[1][1] = g(2, 2) + g(2, 3) + g(3, 2) + g(3, 3);
    }
    let avg = |f: &dyn Fn(&[[i32; 2]; 2]) -> i32| -> [i32; 3] {
        std::array::from_fn(|c| (f(&sums[c]) + 4) / 8)
    };
    let upper = avg(&|s| s[0][0] + s[1][0]);
    let lower = avg(&|s| s[0][1] + s[1][1]);
    let left = avg(&|s| s[0][0] + s[0][1]);
    let right = avg(&|s| s[1][0] + s[1][1]);
    let gd = |x, y, a: &[i32; 3]| gray_distance2(at(x, y), a[0], a[1], a[2]);
    let (mut u, mut l, mut le, mut ri) = (0, 0, 0, 0);
    for i in 0..4 {
        for j in 0..2 {
            u += gd(i, j, &upper);
            l += gd(i, 2 + j, &lower);
            le += gd(j, i, &left);
            ri += gd(2 + j, i, &right);
        }
    }
    u + l < le + ri
}

/// Texels (x, y) of an ETC1 subblock: the left/right halves, or with `flip`
/// the top/bottom ones.
fn subblock_texels(flip: bool, subset: usize) -> [(usize, usize); 8] {
    std::array::from_fn(|j| {
        if flip {
            (j % 4, subset * 2 + j / 4)
        } else {
            (subset * 2 + j / 4, j % 4)
        }
    })
}

// Ported from Basis Universal v2.50, `compute_etc1_hints`. Apache-2.0,
// (C) Binomial LLC. Modified: returns the block and the bias.
fn etc1_hints(best: &Candidate, block: &Block, decoded: &Block, cfg: &Config) -> (EtcBlock, u32) {
    const SORTED_BIAS_MODES: [u32; 32] = [
        13, 0, 22, 29, 27, 12, 26, 9, 30, 31, 8, 10, 25, 2, 23, 5, 15, 7, 3, 11, 6, 17, 28, 18, 1,
        19, 20, 21, 24, 4, 14, 16,
    ];
    let mode = best.u.mode;
    let has_bias = HAS_ETC1_BIAS[mode];
    // No ETC1 speed flags at these levels: upstream's bias counts per level,
    // searched in its sorted order.
    let (last_bias, faster_table) = if has_bias {
        (
            match cfg.level {
                0 => 2,
                1 => 5,
                _ => 20,
            },
            true,
        )
    } else {
        (1, false)
    };
    let flip_estimate = cfg.level <= 1;

    let mut best_blk = EtcBlock::default();
    let mut best_bias = 0;
    let mut best_err = u64::MAX;
    let mut trial = EtcBlock::default();
    let block_y: [Ycbcr; 16] = std::array::from_fn(|i| ycbcr(&block[i]));
    let dec_y: [Ycbcr; 16] = std::array::from_fn(|i| ycbcr(&decoded[i]));
    let (mut first_flip, mut last_flip) = (0, 2);
    if flip_estimate {
        if estimate_flipped(decoded) {
            first_flip = 1;
        }
        last_flip = first_flip + 1;
    }
    let min4 = |a: u64, b: u64, c: u64, d: u64| a.min(b).min(c).min(d);

    for flip in first_flip..last_flip {
        trial.set_flip(flip != 0);
        for individ in 0..2 {
            let mul: u32 = if individ != 0 { 15 } else { 31 };
            trial.set_diff(individ == 0);
            let mut unbiased = [[0u8; 4]; 2];
            let mut lo = [[255i32; 3]; 2];
            let mut hi = [[0i32; 3]; 2];
            for subset in 0..2 {
                let mut avg = [0u32; 3];
                for (x, y) in subblock_texels(flip != 0, subset) {
                    let p = &decoded[x + y * 4];
                    for c in 0..3 {
                        avg[c] += p[c] as u32;
                        lo[subset][c] = lo[subset][c].min(p[c] as i32);
                        hi[subset][c] = hi[subset][c].max(p[c] as i32);
                    }
                }
                for c in 0..3 {
                    unbiased[subset][c] = ((avg[c] * mul + 1020) / (8 * 255)) as u8;
                }
            }
            for bias_iter in 0..last_bias {
                let bias = if faster_table {
                    SORTED_BIAS_MODES[bias_iter as usize]
                } else {
                    bias_iter
                };
                let colors: [[u8; 4]; 2] = std::array::from_fn(|s| {
                    if has_bias {
                        etc::apply_bias(unbiased[s], bias, mul, s)
                    } else {
                        unbiased[s]
                    }
                });
                if individ != 0 {
                    trial.set_color4(colors[0], colors[1]);
                } else {
                    trial.set_color5_clamp(colors[0], colors[1]);
                }
                let mut range = [0i32; 2];
                for subset in 0..2 {
                    let base = trial.block_color(subset, true);
                    range[subset] = (0..3)
                        .flat_map(|c| {
                            [
                                (hi[subset][c] - base[c] as i32).abs(),
                                (base[c] as i32 - lo[subset][c]).abs(),
                            ]
                        })
                        .max()
                        .unwrap_or(0);
                }
                let mut best_inten = [0u32; 2];
                for subset in 0..2 {
                    let mut best_sub = u64::MAX;
                    let limit = if range[subset] > 51 {
                        8
                    } else if range[subset] >= 7 {
                        4
                    } else {
                        2
                    };
                    for inten in 0..limit {
                        trial.set_inten_table(subset, inten);
                        let table = trial.block_colors(subset).map(|c| ycbcr(&c));
                        let near = |c: &Ycbcr| {
                            min4(
                                color_diff(&table[0], c),
                                color_diff(&table[1], c),
                                color_diff(&table[2], c),
                                color_diff(&table[3], c),
                            )
                        };
                        let mut total = 0u64;
                        if flip != 0 {
                            for y in 0..2 {
                                for x in 0..4 {
                                    total += near(&dec_y[x + (subset * 2 + y) * 4]);
                                }
                                if total >= best_sub {
                                    break;
                                }
                            }
                        } else {
                            for y in 0..4 {
                                for x in 0..2 {
                                    total += near(&dec_y[subset * 2 + x + y * 4]);
                                }
                            }
                            // Upstream's check sits outside the row loop
                            // here: it ends the table search.
                            if total >= best_sub {
                                break;
                            }
                        }
                        if total < best_sub {
                            best_sub = total;
                            best_inten[subset] = inten;
                        }
                    }
                }
                trial.set_inten_table(0, best_inten[0]);
                trial.set_inten_table(1, best_inten[1]);

                let mut err = 0u64;
                for subset in 0..2 {
                    let table = trial.block_colors(subset).map(|c| ycbcr(&c));
                    let cost = |i: usize| {
                        let c = &dec_y[i];
                        let k = min4(
                            color_diff(&table[0], c) << 2,
                            (color_diff(&table[1], c) << 2) + 1,
                            (color_diff(&table[2], c) << 2) + 2,
                            (color_diff(&table[3], c) << 2) + 3,
                        );
                        color_diff(&block_y[i], &table[(k & 3) as usize])
                    };
                    if flip != 0 {
                        for y in 0..2 {
                            for x in 0..4 {
                                err += cost(x + (subset * 2 + y) * 4);
                            }
                            if err >= best_err {
                                break;
                            }
                        }
                    } else {
                        for y in 0..4 {
                            for x in 0..2 {
                                err += cost(subset * 2 + x + y * 4);
                            }
                            if err >= best_err {
                                break;
                            }
                        }
                    }
                }
                if err < best_err {
                    best_err = err;
                    best_blk = trial;
                    best_bias = bias;
                }
            }
        }
    }
    (best_blk, best_bias)
}

// Ported from Basis Universal v2.50, `uastc_write_bits`. Apache-2.0,
// (C) Binomial LLC. Modified.
fn write_bits(buf: &mut [u8; 32], pos: &mut u32, mut code: u64, mut size: u32) {
    while size > 0 {
        let shift = *pos & 7;
        let n = size.min(8 - shift);
        buf[(*pos >> 3) as usize] |= (code << shift) as u8;
        code >>= n;
        size -= n;
        *pos += n;
    }
}

/// The hints `pack_uastc` writes besides the ASTC block.
struct Hints {
    etc1: EtcBlock,
    etc1_bias: u32,
    eac_table: u32,
    eac_multiplier: u32,
    bc1_hint0: bool,
    bc1_hint1: bool,
}

// Ported from Basis Universal v2.50, `pack_uastc`. Apache-2.0,
// (C) Binomial LLC. Modified.
fn pack(c: &Candidate, h: &Hints) -> [u8; 16] {
    let mode = c.u.mode;
    let mut buf = [0u8; 32];
    let mut pos = 0u32;
    write_bits(
        &mut buf,
        &mut pos,
        MODE_CODES[mode] as u64,
        MODE_CODE_BITS[mode] as u32,
    );
    if mode == SOLID {
        for v in c.u.solid {
            write_bits(&mut buf, &mut pos, v as u64, 8);
        }
        write_bits(&mut buf, &mut pos, h.etc1.diff() as u64, 1);
        write_bits(&mut buf, &mut pos, h.etc1.inten_table(0) as u64, 3);
        write_bits(&mut buf, &mut pos, h.etc1.selector(0, 0) as u64, 2);
        let rgb = if h.etc1.diff() {
            etc::unpack_color5(h.etc1.base5(), false)
        } else {
            etc::unpack_color4(h.etc1.base4(0), false)
        };
        for v in &rgb[..3] {
            write_bits(&mut buf, &mut pos, *v as u64, 5);
        }
        let mut out = [0u8; 16];
        out.copy_from_slice(&buf[..16]);
        return out;
    }
    if HAS_BC1_HINT0[mode] {
        write_bits(&mut buf, &mut pos, h.bc1_hint0 as u64, 1);
    }
    if HAS_BC1_HINT1[mode] {
        write_bits(&mut buf, &mut pos, h.bc1_hint1 as u64, 1);
    }
    write_bits(&mut buf, &mut pos, h.etc1.flip() as u64, 1);
    write_bits(&mut buf, &mut pos, h.etc1.diff() as u64, 1);
    write_bits(&mut buf, &mut pos, h.etc1.inten_table(0) as u64, 3);
    write_bits(&mut buf, &mut pos, h.etc1.inten_table(1) as u64, 3);
    if HAS_ETC1_BIAS[mode] {
        write_bits(&mut buf, &mut pos, h.etc1_bias as u64, 5);
    }
    if HAS_ALPHA[mode] {
        write_bits(
            &mut buf,
            &mut pos,
            (h.eac_table | (h.eac_multiplier << 4)) as u64,
            8,
        );
    }
    let subsets = match mode {
        2 | 4 | 7 | 9 | 16 => {
            write_bits(&mut buf, &mut pos, c.u.pattern as u64, 5);
            2
        }
        3 => {
            write_bits(&mut buf, &mut pos, c.u.pattern as u64, 4);
            3
        }
        _ => 1,
    };
    let planes = match mode {
        6 | 11 | 13 => {
            write_bits(&mut buf, &mut pos, c.u.ccs as u64, 2);
            2
        }
        17 => 2,
        _ => 1,
    };
    let mut weights = c.u.weights;
    let mut endpoints = c.u.endpoints;
    let comps = COMPS[mode] as usize;
    let weight_bits = WEIGHT_BITS[mode] as u32;
    let (pattern, anchors) = pattern_and_anchors(subsets, mode, c.u.pattern);
    for plane in 0..planes {
        for subset in 0..subsets {
            let anchor = anchors[subset] as usize;
            if weights[anchor * planes + plane] & (1 << (weight_bits - 1)) != 0 {
                for i in 0..16 {
                    if pattern[i] as usize == subset {
                        let w = &mut weights[i * planes + plane];
                        *w = ((1 << weight_bits) - 1) as u8 - *w;
                    }
                }
                if planes == 2 {
                    for comp in 0..comps {
                        let comp_plane = if comps == 2 {
                            comp
                        } else if comp == c.u.ccs as usize {
                            1
                        } else {
                            0
                        };
                        if comp_plane == plane {
                            endpoints.swap(comp * 2, comp * 2 + 1);
                        }
                    }
                } else {
                    for comp in 0..comps {
                        let at = subset * comps * 2 + comp * 2;
                        endpoints.swap(at, at + 1);
                    }
                }
            }
        }
    }

    let values = comps * 2 * subsets;
    let range = ENDPOINT_RANGES[mode] as usize;
    let [ep_bits, ep_trits, ep_quints] = BISE_RANGES[range];
    let mut low = [0u32; 18];
    let mut tq = Vec::with_capacity(8);
    let (mut accum, mut mulv) = (0u32, 1u32);
    for i in 0..values {
        let v = endpoints[i] as u32;
        low[i] = v & ((1 << ep_bits) - 1);
        let t = v >> ep_bits;
        if ep_trits != 0 {
            accum += t * mulv;
            mulv *= 3;
            if mulv == 243 {
                tq.push(accum);
                accum = 0;
                mulv = 1;
            }
        } else if ep_quints != 0 {
            accum += t * mulv;
            mulv *= 5;
            if mulv == 125 {
                tq.push(accum);
                accum = 0;
                mulv = 1;
            }
        }
    }
    for &v in &tq {
        write_bits(
            &mut buf,
            &mut pos,
            v as u64,
            if ep_trits != 0 { 8 } else { 7 },
        );
    }
    if mulv > 1 {
        let bits = if ep_trits != 0 {
            match mulv {
                3 => 2,
                9 => 4,
                27 => 5,
                _ => 7,
            }
        } else if mulv == 5 {
            3
        } else {
            5
        };
        write_bits(&mut buf, &mut pos, accum as u64, bits);
    }
    for &v in &low[..values] {
        write_bits(&mut buf, &mut pos, v as u64, ep_bits as u32);
    }
    let shift = if planes == 2 { 1 } else { 0 };
    for (i, &w) in weights.iter().enumerate().take(16 * planes) {
        let anchor = anchors[..subsets].iter().any(|&a| a as usize == i >> shift);
        write_bits(
            &mut buf,
            &mut pos,
            w as u64,
            weight_bits - u32::from(anchor),
        );
    }
    let mut out = [0u8; 16];
    out.copy_from_slice(&buf[..16]);
    out
}

/// `DEFAULT_BC7_ERROR_WEIGHT` and `UASTC_ERROR_THRESH`.
const BC7_ERROR_WEIGHT: u64 = 50;
const UASTC_ERROR_THRESH: f32 = 1.3;

// Ported from Basis Universal v2.50, `encode_uastc` (no RDO flags).
// Apache-2.0, (C) Binomial LLC. Modified.
/// Encodes 16 texels, row-major, as upstream does at `level`.
pub(crate) fn encode(block: &Block, level: Level) -> [u8; 16] {
    let first = block[0];
    let solid = block.iter().all(|c| *c == first);
    let has_alpha = block.iter().any(|c| c[3] < 255);
    let mut is_la = block.iter().all(|c| c[0] == c[1] && c[0] == c[2]);
    let cfg = config(level);

    if solid {
        let mut c = candidate(SOLID, 0, &[], &[], 0, 0);
        c.u.solid = first;
        let h = Hints {
            etc1: etc::pack_solid(first),
            etc1_bias: 0,
            eac_table: 0,
            eac_multiplier: 1,
            bc1_hint0: false,
            bc1_hint1: false,
        };
        return pack(&c, &h);
    }

    if cfg.only_use_la_on_transparent_blocks && is_la && !has_alpha {
        is_la = false;
    }
    let try_alpha = has_alpha || cfg.always_try_alpha_modes;
    let effort = Effort {
        least_squares_passes: cfg.least_squares_passes,
        uber_level: cfg.uber_level,
    };
    let _ = cfg.estimate_partition;
    let on = |m: u32| cfg.mode_mask & (1 << m) != 0;

    let mut results: Vec<Candidate> = Vec::with_capacity(32);
    if is_la && on(15) {
        mode15(block, &mut results, effort);
    }
    if !has_alpha && on(0) {
        mode0(block, &mut results, effort);
    }
    if try_alpha {
        if on(11) {
            mode11(block, &mut results, effort);
        }
        if on(12) {
            mode12(block, &mut results, effort);
        }
    }

    // Consistent error: LA for LA blocks, RGBA otherwise.
    for r in &mut results {
        if !is_la {
            if IS_LA[r.u.mode] {
                let d = unpack::decode(&r.u, false);
                r.err = d.iter().zip(block).map(|(a, b)| dist_rgba(a, b)).sum();
            }
        } else if !IS_LA[r.u.mode] {
            let d = unpack::decode(&r.u, false);
            r.err = d.iter().zip(block).map(|(a, b)| dist_la(a, b)).sum();
        }
    }

    let mut best_index: Option<usize> = None;
    if results.len() == 1 {
        best_index = Some(0);
    } else {
        let mut overall = vec![0u64; results.len()];
        let mut uastc_f = vec![0f32; results.len()];
        let mut best_uastc_f = 1e20f32 as f64;
        for (i, r) in results.iter().enumerate() {
            let decoded = unpack::decode(&r.u, false);
            let (u_rgb, u_rgba, u_la) = block_error(block, &decoded);
            let bc7_block = bc7::encode(&bc7::transcode(&r.u));
            let bc7_px = bc7::decode(&bc7_block).unwrap_or([[0; 4]; 16]);
            let (b_rgb, b_rgba, b_la) = block_error(block, &bc7_px);
            let (bc7_err, uastc_err) = if is_la {
                (b_la, u_la)
            } else if has_alpha {
                (b_rgba, u_rgba)
            } else {
                (b_rgb, u_rgb)
            };
            overall[i] = (bc7_err * BC7_ERROR_WEIGHT) / 100 + (uastc_err * 100) / 100;
            if overall[i] == 0 {
                best_index = Some(i);
                break;
            }
            uastc_f[i] = (uastc_err as f32).sqrt();
            if (uastc_f[i] as f64) < best_uastc_f {
                best_uastc_f = uastc_f[i] as f64;
            }
        }
        if best_index.is_none() {
            let mut best_err = u64::MAX;
            for i in 0..results.len() {
                let eligible = best_uastc_f == 0.0
                    || (uastc_f[i] as f64 / best_uastc_f) <= UASTC_ERROR_THRESH as f64;
                if eligible {
                    let w = ((overall[i] as f32) * 1.0f32) as u64;
                    if w < best_err {
                        best_err = w;
                        best_index = Some(i);
                        if best_err == 0 {
                            break;
                        }
                    }
                }
            }
        }
    }
    let best = results[best_index.expect("at least one mode is tried")];
    let decoded = unpack::decode(&best.u, false);

    let (bc1_hint0, bc1_hint1) = (false, false);
    let _ = cfg.bc1_hints;

    let (eac_table, eac_multiplier) = if HAS_ALPHA[best.u.mode] {
        let alpha: [u8; 16] = std::array::from_fn(|i| decoded[i][3]);
        pack_eac_a8(&alpha, 0, cfg.eac_mul_search_rad, cfg.eac_table_mask)
    } else {
        (0, 0)
    };
    let (etc1, etc1_bias) = etc1_hints(&best, block, &decoded, &cfg);
    pack(
        &best,
        &Hints {
            etc1,
            etc1_bias,
            eac_table,
            eac_multiplier,
            bc1_hint0,
            bc1_hint1,
        },
    )
}
