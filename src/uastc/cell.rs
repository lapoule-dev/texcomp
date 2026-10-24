// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2019-2026 Binomial LLC; modifications Copyright (c) lapoule.dev
//
// Ported from Basis Universal v2.50 (tag `v2_50`, commit
// 9bebe16726b3a61c8c213eeee3b7cffb462ef34e), `encoder/basisu_bc7enc.cpp`,
// Copyright (C) 2019-2024 Binomial LLC, licensed under the Apache License,
// Version 2.0 (see LICENSE-APACHE-UPSTREAM and NOTICE). Modified by
// lapoule.dev: translated from C++ to Rust; only the ASTC-endpoint paths the
// UASTC encoder uses are kept (non-perceptual, unit channel weights, no forced
// selectors, no p-bits); the lookup tables are built once on first use. Every
// floating-point expression keeps upstream's types and operation order, so
// the results are bit-identical. Each ported item says so at its head.

//! The endpoint search behind every UASTC mode: fit two endpoints and a
//! selector per texel to a cell of up to 16 texels, for a given ASTC endpoint
//! range and weight table.

// The loops keep upstream's indexing, which keeps the port reviewable line
// for line against it.
#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

use super::tables::{levels, UNQUANT, WEIGHTS};

// Ported from Basis Universal v2.50, `g_bc7_weights1x` … `g_astc_weights5x`
// (for each weight w: w·w, (1−w)·w, (1−w)², w). Apache-2.0,
// (C) Binomial LLC. Modified.
#[allow(dead_code)] // mode 13 (level 2)
const W1X: [[f32; 4]; 2] = [
    [0.000000, 0.000000, 1.000000, 0.000000],
    [1.000000, 0.000000, 0.000000, 1.000000],
];
const W2X: [[f32; 4]; 4] = [
    [0.000000, 0.000000, 1.000000, 0.000000],
    [0.107666, 0.220459, 0.451416, 0.328125],
    [0.451416, 0.220459, 0.107666, 0.671875],
    [1.000000, 0.000000, 0.000000, 1.000000],
];
const W3X: [[f32; 4]; 8] = [
    [0.000000, 0.000000, 1.000000, 0.000000],
    [0.019775, 0.120850, 0.738525, 0.140625],
    [0.079102, 0.202148, 0.516602, 0.281250],
    [0.177979, 0.243896, 0.334229, 0.421875],
    [0.334229, 0.243896, 0.177979, 0.578125],
    [0.516602, 0.202148, 0.079102, 0.718750],
    [0.738525, 0.120850, 0.019775, 0.859375],
    [1.000000, 0.000000, 0.000000, 1.000000],
];
const A4X: [[f32; 4]; 16] = [
    [0.000000, 0.000000, 1.000000, 0.000000],
    [0.003906, 0.058594, 0.878906, 0.062500],
    [0.015625, 0.109375, 0.765625, 0.125000],
    [0.035156, 0.152344, 0.660156, 0.187500],
    [0.070557, 0.195068, 0.539307, 0.265625],
    [0.107666, 0.220459, 0.451416, 0.328125],
    [0.152588, 0.238037, 0.371338, 0.390625],
    [0.205322, 0.247803, 0.299072, 0.453125],
    [0.299072, 0.247803, 0.205322, 0.546875],
    [0.371338, 0.238037, 0.152588, 0.609375],
    [0.451416, 0.220459, 0.107666, 0.671875],
    [0.539307, 0.195068, 0.070557, 0.734375],
    [0.660156, 0.152344, 0.035156, 0.812500],
    [0.765625, 0.109375, 0.015625, 0.875000],
    [0.878906, 0.058594, 0.003906, 0.937500],
    [1.000000, 0.000000, 0.000000, 1.000000],
];
const A5X: [[f32; 4]; 32] = [
    [0.000000, 0.000000, 1.000000, 0.000000],
    [0.000977, 0.030273, 0.938477, 0.031250],
    [0.003906, 0.058594, 0.878906, 0.062500],
    [0.008789, 0.084961, 0.821289, 0.093750],
    [0.015625, 0.109375, 0.765625, 0.125000],
    [0.024414, 0.131836, 0.711914, 0.156250],
    [0.035156, 0.152344, 0.660156, 0.187500],
    [0.047852, 0.170898, 0.610352, 0.218750],
    [0.062500, 0.187500, 0.562500, 0.250000],
    [0.079102, 0.202148, 0.516602, 0.281250],
    [0.097656, 0.214844, 0.472656, 0.312500],
    [0.118164, 0.225586, 0.430664, 0.343750],
    [0.140625, 0.234375, 0.390625, 0.375000],
    [0.165039, 0.241211, 0.352539, 0.406250],
    [0.191406, 0.246094, 0.316406, 0.437500],
    [0.219727, 0.249023, 0.282227, 0.468750],
    [0.282227, 0.249023, 0.219727, 0.531250],
    [0.316406, 0.246094, 0.191406, 0.562500],
    [0.352539, 0.241211, 0.165039, 0.593750],
    [0.390625, 0.234375, 0.140625, 0.625000],
    [0.430664, 0.225586, 0.118164, 0.656250],
    [0.472656, 0.214844, 0.097656, 0.687500],
    [0.516602, 0.202148, 0.079102, 0.718750],
    [0.562500, 0.187500, 0.062500, 0.750000],
    [0.610352, 0.170898, 0.047852, 0.781250],
    [0.660156, 0.152344, 0.035156, 0.812500],
    [0.711914, 0.131836, 0.024414, 0.843750],
    [0.765625, 0.109375, 0.015625, 0.875000],
    [0.821289, 0.084961, 0.008789, 0.906250],
    [0.878906, 0.058594, 0.003906, 0.937500],
    [0.938477, 0.030273, 0.000977, 0.968750],
    [1.000000, 0.000000, 0.000000, 1.000000],
];

/// A selector weight table: the integer weights (0..=64) and their
/// least-squares terms.
#[derive(Clone, Copy)]
pub(crate) struct Weights {
    pub(crate) w: &'static [u8],
    pub(crate) x: &'static [[f32; 4]],
}

/// BC7's 1-, 2- and 3-bit weights and ASTC's 4- and 5-bit ones.
#[allow(dead_code)] // mode 13 (level 2)
pub(crate) const BC7_1: Weights = Weights {
    w: WEIGHTS[1],
    x: &W1X,
};
pub(crate) const BC7_2: Weights = Weights {
    w: WEIGHTS[2],
    x: &W2X,
};
pub(crate) const BC7_3: Weights = Weights {
    w: WEIGHTS[3],
    x: &W3X,
};
pub(crate) const ASTC_4: Weights = Weights {
    w: WEIGHTS[4],
    x: &A4X,
};
#[allow(dead_code)]
pub(crate) const ASTC_5: Weights = Weights {
    w: WEIGHTS[5],
    x: &A5X,
};

/// One endpoint pair hitting a value with the least error.
#[derive(Clone, Copy, Debug)]
struct EndpointErr {
    error: u16,
    lo: u8,
    hi: u8,
}

/// An ASTC value in sorted order: its 0..=255 value and its stored index.
#[derive(Clone, Copy, Debug, Default)]
struct QuantBin {
    unquant: u8,
    index: u8,
}

struct Tables {
    /// `[range][i]`: the range's values in increasing order.
    sorted: [[QuantBin; 256]; 21],
    /// `[range][v]`: the sorted position nearest to `v`.
    nearest: [[u8; 256]; 21],
    /// Range 13 (0..47) endpoints hitting `c` at BC7 2-bit weight 1.
    range13_2bit: [EndpointErr; 256],
    /// 4-bit endpoints hitting `c` at BC7 2-bit weight 1.
    bits4_2bit: [EndpointErr; 256],
}

// Ported from Basis Universal v2.50, `astc_init` and the parts of
// `bc7enc_compress_block_init` the UASTC encoder uses. Apache-2.0,
// (C) Binomial LLC. Modified: built once on first use.
fn tables() -> &'static Tables {
    static TABLES: OnceLock<Box<Tables>> = OnceLock::new();
    TABLES.get_or_init(|| {
        let none = EndpointErr {
            error: u16::MAX,
            lo: 0,
            hi: 0,
        };
        let mut t = Box::new(Tables {
            sorted: [[QuantBin::default(); 256]; 21],
            nearest: [[0; 256]; 21],
            range13_2bit: [none; 256],
            bits4_2bit: [none; 256],
        });
        for range in 0..21 {
            let n = levels(range);
            // Ranges with trits or quints but no unquantisation constant are
            // not valid ASTC endpoint ranges; upstream skips them.
            if n > 1 && UNQUANT[range][1] == 0 {
                continue;
            }
            let mut vals: Vec<u32> = (0..n)
                .map(|i| ((UNQUANT[range][i] as u32) << 8) | i as u32)
                .collect();
            vals.sort_unstable();
            for (i, v) in vals.iter().enumerate() {
                t.sorted[range][i] = QuantBin {
                    unquant: (v >> 8) as u8,
                    index: (v & 0xff) as u8,
                };
            }
            for v in 0..256i32 {
                let (mut best, mut best_err) = (0, i32::MAX);
                for j in 0..n {
                    let err = (t.sorted[range][j].unquant as i32 - v).abs();
                    if err < best_err {
                        best_err = err;
                        best = j;
                    }
                }
                t.nearest[range][v as usize] = best as u8;
            }
        }
        let fit = |c: i32, lows: &dyn Fn(u32) -> u32, n: u32, w: u32| {
            let mut best = none;
            for l in 0..n {
                for h in 0..n {
                    let k = interpolate_linear(lows(l), lows(h), w) as i32;
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
        let w21 = WEIGHTS[2][1] as u32;
        let sorted13: Vec<u32> = (0..48).map(|i| t.sorted[13][i].unquant as u32).collect();
        for c in 0..256 {
            t.bits4_2bit[c] = fit(c as i32, &|l| (l << 4) | l, 16, w21);
            t.range13_2bit[c] = fit(c as i32, &|l| sorted13[l as usize], 48, w21);
        }
        t
    })
}

// Ported from Basis Universal v2.50, `astc_interpolate_linear`.
// Apache-2.0, (C) Binomial LLC. Modified.
pub(crate) fn interpolate_linear(l: u32, h: u32, w: u32) -> u32 {
    let (l, h) = ((l << 8) | l, (h << 8) | h);
    ((l * (64 - w) + h * w + 32) >> 6) >> 8
}

/// A cell to fit.
pub(crate) struct Params<'a> {
    pub(crate) pixels: &'a [[u8; 4]],
    pub(crate) weights: Weights,
    /// The ASTC endpoint range.
    pub(crate) range: usize,
    pub(crate) has_alpha: bool,
}

/// The search's effort (upstream's `bc7enc_compress_block_params`, with
/// unit weights and non-perceptual error).
#[derive(Clone, Copy)]
pub(crate) struct Effort {
    pub(crate) least_squares_passes: u32,
    pub(crate) uber_level: u32,
}

/// A fit: sorted-order endpoints, their stored ASTC values, the selectors.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Fit {
    pub(crate) err: u64,
    low: [u8; 4],
    high: [u8; 4],
    pub(crate) selectors: [u8; 16],
    temp: [u8; 16],
    pub(crate) astc_low: [u8; 4],
    pub(crate) astc_high: [u8; 4],
}

type V4 = [f32; 4];

fn dot(a: &V4, b: &V4) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn mul(a: &V4, s: f32) -> V4 {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

fn add(a: &V4, b: &V4) -> V4 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

fn sub(a: &V4, b: &V4) -> V4 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[allow(clippy::manual_clamp)] // NaN passes through, as upstream's does
fn saturate(x: f32) -> f32 {
    // Upstream's `clampf(x, 0, 1)`.
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

fn normalize(v: &mut V4) {
    let mut s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    if s != 0.0 {
        s = 1.0 / s.sqrt();
        for c in v.iter_mut() {
            *c *= s;
        }
    }
}

fn from_color(c: &[u8; 4]) -> V4 {
    [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32]
}

fn maxf(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

fn minf(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

fn dist_rgb(a: &[u8; 4], b: &[u8; 4]) -> u64 {
    let d = |c: usize| (a[c] as i32 - b[c] as i32).pow(2) as u32;
    (d(0) + d(1) + d(2)) as u64
}

fn dist_rgba(a: &[u8; 4], b: &[u8; 4]) -> u64 {
    let da = (a[3] as i32 - b[3] as i32).pow(2) as u32;
    dist_rgb(a, b) + da as u64
}

// Ported from Basis Universal v2.50, `compute_least_squares_endpoints_rgba`
// and `_rgb`. Apache-2.0, (C) Binomial LLC. Modified: one function.
fn least_squares(p: &Params, selectors: &[u8]) -> (V4, V4) {
    let (mut z00, mut z10, mut z11) = (0f64, 0f64, 0f64);
    let mut q00 = [0f64; 4];
    let mut t = [0f64; 4];
    let comps = if p.has_alpha { 4 } else { 3 };
    for (px, &sel) in p.pixels.iter().zip(selectors) {
        let wx = p.weights.x[sel as usize];
        z00 += wx[0] as f64;
        z10 += wx[1] as f64;
        z11 += wx[2] as f64;
        let w = wx[3];
        for c in 0..comps {
            q00[c] += (w * px[c] as f32) as f64;
            t[c] += px[c] as f64;
        }
    }
    let z01 = z10;
    let mut det = z00 * z11 - z01 * z10;
    if det != 0.0 {
        det = 1.0 / det;
    }
    let (iz00, iz01, iz10, iz11) = (z11 * det, -z01 * det, -z10 * det, z00 * det);
    let (mut xl, mut xh) = ([255f32; 4], [255f32; 4]);
    for c in 0..comps {
        let q10 = t[c] - q00[c];
        xl[c] = (iz00 * q00[c] + iz01 * q10) as f32;
        xh[c] = (iz10 * q00[c] + iz11 * q10) as f32;
    }
    for c in 0..comps {
        if xl[c] < 0.0 || xh[c] > 255.0 {
            let lo = p.pixels.iter().map(|px| px[c]).min().unwrap_or(0);
            let hi = p.pixels.iter().map(|px| px[c]).max().unwrap_or(0);
            if lo == hi {
                xl[c] = lo as f32;
                xh[c] = hi as f32;
            }
        }
    }
    (xl, xh)
}

// Ported from Basis Universal v2.50, `evaluate_solution` (ASTC,
// non-perceptual, no forced selectors). Apache-2.0, (C) Binomial LLC.
// Modified.
fn evaluate(low: [u8; 4], high: [u8; 4], p: &Params, r: &mut Fit) -> u64 {
    let t = tables();
    let sorted = &t.sorted[p.range];
    let min = low.map(|v| sorted[v as usize].unquant);
    let max = high.map(|v| sorted[v as usize].unquant);
    let n = p.weights.w.len();
    let mut colors = [[0u8; 4]; 32];
    colors[0] = min;
    colors[n - 1] = max;
    let nc = if p.has_alpha { 4 } else { 3 };
    for i in 1..n - 1 {
        for j in 0..nc {
            colors[i][j] =
                interpolate_linear(min[j] as u32, max[j] as u32, p.weights.w[i] as u32) as u8;
        }
    }
    let (lr, lg, lb) = (min[0] as i32, min[1] as i32, min[2] as i32);
    let (dr, dg, db) = (max[0] as i32 - lr, max[1] as i32 - lg, max[2] as i32 - lb);
    let mut total = 0u64;
    if p.has_alpha {
        let la = min[3] as i32;
        let da = max[3] as i32 - la;
        let f = n as f32 / ((dr * dr + dg * dg + db * db + da * da) as f32 + 0.00000125f32);
        for (i, c) in p.pixels.iter().enumerate() {
            let (r_, g, b, a) = (c[0] as i32, c[1] as i32, c[2] as i32, c[3] as i32);
            let dotp = (r_ - lr) * dr + (g - lg) * dg + (b - lb) * db + (a - la) * da;
            let mut sel = ((dotp as f32) * f + 0.5f32) as i32;
            sel = sel.clamp(1, n as i32 - 1);
            let err0 = dist_rgba(&colors[sel as usize - 1], c);
            let mut err1 = dist_rgba(&colors[sel as usize], c);
            if err0 == err1 {
                if sel - 1 == 0 {
                    sel = 0;
                }
            } else if err1 > err0 {
                err1 = err0;
                sel -= 1;
            }
            total += err1;
            r.temp[i] = sel as u8;
        }
    } else {
        let f = n as f32 / ((dr * dr + dg * dg + db * db) as f32 + 0.00000125f32);
        for (i, c) in p.pixels.iter().enumerate() {
            let (r_, g, b) = (c[0] as i32, c[1] as i32, c[2] as i32);
            let dotp = (r_ - lr) * dr + (g - lg) * dg + (b - lb) * db;
            let sel = (((dotp as f32) * f + 0.5f32) as i32).clamp(1, n as i32 - 1);
            let err0 = dist_rgb(&colors[sel as usize - 1], c);
            let err1 = dist_rgb(&colors[sel as usize], c);
            let (mut best_sel, mut best_err) = (sel, err1);
            if err0 == err1 {
                if best_sel - 1 == 0 {
                    best_sel = 0;
                }
            } else if err0 < best_err {
                best_err = err0;
                best_sel = sel - 1;
            }
            total += best_err;
            r.temp[i] = best_sel as u8;
        }
    }
    if total < r.err {
        r.err = total;
        r.low = low;
        r.high = high;
        let n = p.pixels.len();
        r.selectors[..n].copy_from_slice(&r.temp[..n]);
    }
    total
}

// Ported from Basis Universal v2.50, `fixDegenerateEndpoints` (mode 255).
// Apache-2.0, (C) Binomial LLC. Modified.
fn fix_degenerate(min: &mut [u8; 4], max: &mut [u8; 4], xl: &V4, xh: &V4, iscale: u32, flags: u32) {
    for i in 0..3 {
        if min[i] == max[i] && (xl[i] - xh[i]).abs() > 0.000125f32 {
            if flags & 1 != 0 && min[i] > 0 {
                min[i] -= 1;
            }
            if flags & 2 != 0 && (max[i] as u32) < iscale {
                max[i] += 1;
            }
        }
    }
}

// Ported from Basis Universal v2.50, `find_optimal_solution` (ASTC branch)
// and `areDegenerateEndpoints`. Apache-2.0, (C) Binomial LLC. Modified.
fn find_optimal(mut xl: V4, mut xh: V4, p: &Params, r: &mut Fit) -> u64 {
    xl = xl.map(saturate);
    xh = xh.map(saturate);
    let t = tables();
    let iscale = levels(p.range) as u32 - 1;
    let to8 = |x: f32| ((x * 255.0f32 + 0.5f32) as i32).clamp(0, 255) as usize;
    let mut min = [0u8; 4];
    let mut max = [0u8; 4];
    for i in 0..4 {
        min[i] = t.nearest[p.range][to8(xl[i])];
        max[i] = t.nearest[p.range][to8(xh[i])];
    }
    let degenerate = (0..3).any(|i| min[i] == max[i] && (xl[i] - xh[i]).abs() > 0.0);
    let try_eval = |lo: [u8; 4], hi: [u8; 4], r: &mut Fit| {
        if r.err == u64::MAX || lo != r.low || hi != r.high {
            evaluate(lo, hi, p, r);
        }
    };
    if degenerate {
        for flags in [1, 0, 2, 3] {
            let (mut lo, mut hi) = (min, max);
            fix_degenerate(&mut lo, &mut hi, &xl, &xh, iscale, flags);
            try_eval(lo, hi, r);
        }
    } else {
        try_eval(min, max, r);
    }
    for i in 0..4 {
        r.astc_low[i] = t.sorted[p.range][r.low[i] as usize].index;
        r.astc_high[i] = t.sorted[p.range][r.high[i] as usize].index;
    }
    r.err
}

/// Which one-colour table, if any, applies to a cell (upstream's chain of
/// `pack_astc_*_to_one_color` conditions, for the ranges UASTC levels 0–2
/// use).
#[derive(Clone, Copy, PartialEq)]
enum OneColor {
    Range13TwoBit,
    FourBitTwoBitRgba,
}

fn one_color_kind(p: &Params) -> Option<OneColor> {
    let n = p.weights.w.len();
    match (p.range, n, p.has_alpha) {
        (13, 4, false) => Some(OneColor::Range13TwoBit),
        (8, 4, true) => Some(OneColor::FourBitTwoBitRgba),
        _ => None,
    }
}

// Ported from Basis Universal v2.50, `pack_astc_range13_2bit_to_one_color`
// and `pack_astc_4bit_2bit_to_one_color_rgba`. Apache-2.0,
// (C) Binomial LLC. Modified: one function, writing the selectors into
// `into_temp` or the result's selectors as upstream's pointer argument does.
fn pack_one_color(kind: OneColor, p: &Params, r: &mut Fit, c: [u32; 4], into_temp: bool) -> u64 {
    let t = tables();
    let (lo, hi, index, rgba) = match kind {
        OneColor::Range13TwoBit => {
            let e = |v: u32| t.range13_2bit[v as usize];
            (
                [e(c[0]).lo, e(c[1]).lo, e(c[2]).lo, 47],
                [e(c[0]).hi, e(c[1]).hi, e(c[2]).hi, 47],
                1u8,
                false,
            )
        }
        OneColor::FourBitTwoBitRgba => {
            let e = |v: u32| t.bits4_2bit[v as usize];
            (
                [e(c[0]).lo, e(c[1]).lo, e(c[2]).lo, e(c[3]).lo],
                [e(c[0]).hi, e(c[1]).hi, e(c[2]).hi, e(c[3]).hi],
                1u8,
                true,
            )
        }
    };
    r.low = lo;
    r.high = hi;
    for i in 0..4 {
        r.astc_low[i] = t.sorted[p.range][lo[i] as usize].index;
        r.astc_high[i] = t.sorted[p.range][hi[i] as usize].index;
    }
    let n = p.pixels.len();
    if into_temp {
        r.temp[..n].fill(index);
    } else {
        r.selectors[..n].fill(index);
    }
    let w = WEIGHTS[2][index as usize] as u32;
    let mut px = [0u8; 4];
    for i in 0..4 {
        let (l, h) = match kind {
            OneColor::Range13TwoBit => (
                t.sorted[13][lo[i] as usize].unquant as u32,
                t.sorted[13][hi[i] as usize].unquant as u32,
            ),
            OneColor::FourBitTwoBitRgba => (
                ((lo[i] as u32) << 4) | lo[i] as u32,
                ((hi[i] as u32) << 4) | hi[i] as u32,
            ),
        };
        px[i] = interpolate_linear(l, h, w) as u8;
    }
    let total: u64 = p
        .pixels
        .iter()
        .map(|c| {
            if rgba {
                dist_rgba(&px, c)
            } else {
                dist_rgb(&px, c)
            }
        })
        .sum();
    r.err = total;
    total
}

// Ported from Basis Universal v2.50, `color_cell_compression` (ASTC
// endpoints, `mode` 255, no forced selectors). Apache-2.0,
// (C) Binomial LLC. Modified.
/// Fits the cell; returns the fit and its error (sum of squared channel
/// differences).
pub(crate) fn compress(p: &Params, effort: Effort) -> Fit {
    let mut r = Fit {
        err: u64::MAX,
        low: [0; 4],
        high: [0; 4],
        selectors: [0; 16],
        temp: [0; 16],
        astc_low: [0; 4],
        astc_high: [0; 4],
    };
    let first = p.pixels[0];
    let kind = one_color_kind(p);
    if let Some(kind) = kind {
        let solid = if kind == OneColor::FourBitTwoBitRgba {
            p.pixels.iter().all(|c| *c == first)
        } else {
            p.pixels.iter().all(|c| c[..3] == first[..3])
        };
        if solid {
            pack_one_color(kind, p, &mut r, first.map(u32::from), false);
            return r;
        }
    }

    let n = p.pixels.len();
    let mut mean = [0f32; 4];
    for c in p.pixels {
        mean = add(&mean, &from_color(c));
    }
    let mean_scaled = mul(&mean, 1.0f32 / (n as f32));
    mean = mul(&mean, 1.0f32 / (n as f32 * 255.0f32));
    mean = mean.map(saturate);

    let mut axis;
    if p.has_alpha {
        axis = [0f32; 4];
        for (i, px) in p.pixels.iter().enumerate() {
            let color = sub(&from_color(px), &mean_scaled);
            let a = mul(&color, color[0]);
            let b = mul(&color, color[1]);
            let c = mul(&color, color[2]);
            let d = mul(&color, color[3]);
            let mut nv = if i != 0 { axis } else { color };
            normalize(&mut nv);
            axis[0] += dot(&a, &nv);
            axis[1] += dot(&b, &nv);
            axis[2] += dot(&c, &nv);
            axis[3] += dot(&d, &nv);
        }
        normalize(&mut axis);
    } else {
        let mut cov = [0f32; 6];
        for px in p.pixels {
            let r_ = px[0] as f32 - mean_scaled[0];
            let g = px[1] as f32 - mean_scaled[1];
            let b = px[2] as f32 - mean_scaled[2];
            cov[0] += r_ * r_;
            cov[1] += r_ * g;
            cov[2] += r_ * b;
            cov[3] += g * g;
            cov[4] += g * b;
            cov[5] += b * b;
        }
        let (mut xr, mut xg, mut xb) = (0.9f32, 1.0f32, 0.7f32);
        for _ in 0..3 {
            let mut r_ = xr * cov[0] + xg * cov[1] + xb * cov[2];
            let mut g = xr * cov[1] + xg * cov[3] + xb * cov[4];
            let mut b = xr * cov[2] + xg * cov[4] + xb * cov[5];
            let mut m = maxf(maxf(r_.abs(), g.abs()), b.abs());
            if m > 1e-10f32 {
                m = 1.0 / m;
                r_ *= m;
                g *= m;
                b *= m;
            }
            xr = r_;
            xg = g;
            xb = b;
        }
        let mut len = xr * xr + xg * xg + xb * xb;
        if len < 1e-10f32 {
            axis = [0f32; 4];
        } else {
            len = 1.0 / len.sqrt();
            axis = [xr * len, xg * len, xb * len, 0.0];
        }
    }
    if dot(&axis, &axis) < 0.5f32 {
        axis = [1.0, 1.0, 1.0, if p.has_alpha { 1.0 } else { 0.0 }];
        normalize(&mut axis);
    }

    let (mut l, mut h) = (1e9f32, -1e9f32);
    for px in p.pixels {
        let q = sub(&from_color(px), &mean_scaled);
        let d = dot(&q, &axis);
        l = minf(l, d);
        h = maxf(h, d);
    }
    l *= 1.0f32 / 255.0f32;
    h *= 1.0f32 / 255.0f32;
    let mut min_color = add(&mean, &mul(&axis, l)).map(saturate);
    let mut max_color = add(&mean, &mul(&axis, h)).map(saturate);
    let white = [1f32; 4];
    if dot(&min_color, &white) > dot(&max_color, &white) {
        std::mem::swap(&mut min_color, &mut max_color);
    }

    if find_optimal(min_color, max_color, p, &mut r) == 0 {
        return r;
    }

    let refit = |sels: &[u8], r: &mut Fit| -> bool {
        let (xl, xh) = least_squares(p, sels);
        let s = 1.0f32 / 255.0f32;
        find_optimal(mul(&xl, s), mul(&xh, s), p, r) != 0
    };

    for _ in 0..effort.least_squares_passes {
        let sels = r.selectors;
        if !refit(&sels[..n], &mut r) {
            return r;
        }
    }

    if effort.uber_level > 0 {
        let sels = r.selectors;
        let top = p.weights.w.len() as u32 - 1;
        let min_sel = sels[..n].iter().copied().min().unwrap_or(0) as u32;
        let max_sel = sels[..n].iter().copied().max().unwrap_or(0) as u32;
        let mut tmp = [0u8; 16];
        for i in 0..n {
            let s = sels[i] as u32;
            tmp[i] = if s == min_sel && s < top { s + 1 } else { s } as u8;
        }
        if !refit(&tmp[..n], &mut r) {
            return r;
        }
        for i in 0..n {
            let s = sels[i] as u32;
            tmp[i] = if s == max_sel && s > 0 { s - 1 } else { s } as u8;
        }
        if !refit(&tmp[..n], &mut r) {
            return r;
        }
        for i in 0..n {
            let s = sels[i] as u32;
            tmp[i] = if s == min_sel && s < top {
                s + 1
            } else if s == max_sel && s > 0 {
                s - 1
            } else {
                s
            } as u8;
        }
        if !refit(&tmp[..n], &mut r) {
            return r;
        }
        // Uber levels 2+ are used only by effort levels 3 and 4.
    }

    if let Some(kind) = kind {
        let c = [
            (0.5f32 + mean[0] * 255.0f32) as i32 as u32,
            (0.5f32 + mean[1] * 255.0f32) as i32 as u32,
            (0.5f32 + mean[2] * 255.0f32) as i32 as u32,
            (0.5f32 + mean[3] * 255.0f32) as i32 as u32,
        ];
        let mut avg = r;
        let avg_err = pack_one_color(kind, p, &mut avg, c, true);
        // Upstream's averaged result shares the caller's scratch selectors.
        r.temp = avg.temp;
        if avg_err < r.err {
            r = avg;
            r.selectors[..n].copy_from_slice(&avg.temp[..n]);
            r.err = avg_err;
        }
    }
    r
}

// Ported from Basis Universal v2.50, `color_cell_compression_est_astc`
// (unit weights). Apache-2.0, (C) Binomial LLC. Modified.
/// A quick error estimate for a cell fitted by its bounding box diagonal.
#[allow(dead_code)]
pub(crate) fn estimate(weights: &[u8], comps: usize, pixels: &[[u8; 4]], best_so_far: u64) -> u64 {
    let n = weights.len();
    let (mut lo, mut hi) = ([255u8; 4], [0u8; 4]);
    for px in pixels {
        for c in 0..comps {
            lo[c] = lo[c].min(px[c]);
            hi[c] = hi[c].max(px[c]);
        }
    }
    if comps == 3 {
        lo[3] = 255;
        hi[3] = 255;
    }
    let mut colors = [[0u8; 4]; 32];
    colors[0] = lo;
    colors[n - 1] = hi;
    for i in 1..n - 1 {
        for c in 0..3 {
            colors[i][c] = interpolate_linear(lo[c] as u32, hi[c] as u32, weights[i] as u32) as u8;
        }
        colors[i][3] = if comps == 4 {
            interpolate_linear(lo[3] as u32, hi[3] as u32, weights[i] as u32) as u8
        } else {
            255
        };
    }
    let a: [i32; 4] = std::array::from_fn(|c| hi[c] as i32 - lo[c] as i32);
    let dots: Vec<i32> = colors[..n]
        .iter()
        .map(|w| {
            let mut d = w[0] as i32 * a[0] + w[1] as i32 * a[1] + w[2] as i32 * a[2];
            if comps == 4 {
                d += w[3] as i32 * a[3];
            }
            d
        })
        .collect();
    let thresh: Vec<i32> = (0..n - 1)
        .map(|i| (dots[i] + dots[i + 1] + 1) >> 1)
        .collect();
    let mut total = 0u64;
    for px in pixels {
        let mut d = a[0] * px[0] as i32 + a[1] * px[1] as i32 + a[2] * px[2] as i32;
        if comps == 4 {
            d += a[3] * px[3] as i32;
        }
        let mut s = 0;
        for j in (0..n - 1).rev() {
            if d >= thresh[j] {
                s = j + 1;
                break;
            }
        }
        let e = &colors[s];
        let mut err = 0i32;
        for c in 0..comps {
            let dc = e[c] as i32 - px[c] as i32;
            err += dc * dc;
        }
        total += err as u64;
        if total > best_so_far {
            break;
        }
    }
    total
}
