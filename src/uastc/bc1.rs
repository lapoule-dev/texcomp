// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2019-2026 Binomial LLC; modifications Copyright (c) lapoule.dev
//
// Ported from Basis Universal v2.50 (tag `v2_50`, commit
// 9bebe16726b3a61c8c213eeee3b7cffb462ef34e), `transcoder/basisu_transcoder.cpp`
// and `transcoder/basisu_dds_transcoder.inl`, Copyright (C) 2019-2026 Binomial
// LLC, licensed under the Apache License, Version 2.0 (see
// LICENSE-APACHE-UPSTREAM and NOTICE). Modified by lapoule.dev: translated
// from C++ to Rust; only what the UASTC encoder's BC1 hints use (the BC1
// encoder without its high-quality passes, the two hint transcodes and the
// BC1 decoder); the single-colour tables built once on first use. Every
// floating-point expression keeps upstream's types and operation order. Each
// ported item says so at its head.

//! BC1, as far as the UASTC encoder's BC1 hints need it.

// The loops keep upstream's indexing, which keeps the port reviewable line
// for line against it.
#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

use super::tables::{COMPS, ENDPOINT_RANGES, PLANES, UNQUANT, WEIGHT_BITS};
use super::unpack::Unpacked;

/// A BC1 block: low colour, high colour (RGB565, little-endian), selectors.
pub(crate) type Bc1 = [u8; 8];

fn low(b: &Bc1) -> u32 {
    b[0] as u32 | ((b[1] as u32) << 8)
}

fn high(b: &Bc1) -> u32 {
    b[2] as u32 | ((b[3] as u32) << 8)
}

fn set_low(b: &mut Bc1, c: u32) {
    b[0] = c as u8;
    b[1] = (c >> 8) as u8;
}

fn set_high(b: &mut Bc1, c: u32) {
    b[2] = c as u8;
    b[3] = (c >> 8) as u8;
}

#[derive(Clone, Copy, Default)]
struct MatchEntry {
    hi: u8,
    lo: u8,
}

// Ported from Basis Universal v2.50, `prepare_bc1_single_color_table` (for
// `g_bc1_match5_equals_1` and `g_bc1_match6_equals_1`). Apache-2.0,
// (C) Binomial LLC. Modified: built once on first use.
fn match_tables() -> &'static ([MatchEntry; 256], [MatchEntry; 256]) {
    static TABLES: OnceLock<([MatchEntry; 256], [MatchEntry; 256])> = OnceLock::new();
    TABLES.get_or_init(|| {
        let build = |bits: u32| {
            let size = 1i32 << bits;
            let expand = |v: i32| {
                if bits == 5 {
                    (v << 3) | (v >> 2)
                } else {
                    (v << 2) | (v >> 4)
                }
            };
            let mut t = [MatchEntry::default(); 256];
            for (i, e) in t.iter_mut().enumerate() {
                let mut lowest = 256;
                for lo in 0..size {
                    for hi in 0..size {
                        let (lo_e, hi_e) = (expand(lo), expand(hi));
                        let mut err = ((hi_e * 2 + lo_e) / 3 - i as i32).abs();
                        err += ((hi_e - lo_e).abs() * 3) / 100;
                        if err < lowest {
                            *e = MatchEntry {
                                hi: hi as u8,
                                lo: lo as u8,
                            };
                            lowest = err;
                        }
                    }
                }
            }
            t
        };
        (build(5), build(6))
    })
}

// Ported from Basis Universal v2.50, `encode_bc1_solid_block`.
// Apache-2.0, (C) Binomial LLC. Modified.
fn solid_block(r: usize, g: usize, b: usize) -> Bc1 {
    let (m5, m6) = match_tables();
    let mut mask = 0xAAu8;
    let mut max16 = ((m5[r].hi as u32) << 11) | ((m6[g].hi as u32) << 5) | m5[b].hi as u32;
    let mut min16 = ((m5[r].lo as u32) << 11) | ((m6[g].lo as u32) << 5) | m5[b].lo as u32;
    if min16 == max16 {
        mask = 0;
        if min16 > 0 {
            min16 -= 1;
        } else {
            max16 = 1;
            min16 = 0;
            mask = 0x55;
        }
    }
    if max16 < min16 {
        std::mem::swap(&mut max16, &mut min16);
        mask ^= 0x55;
    }
    let mut out = [0u8; 8];
    set_low(&mut out, max16);
    set_high(&mut out, min16);
    out[4..].fill(mask);
    out
}

// Ported from Basis Universal v2.50, `bc1_find_sels`. Apache-2.0,
// (C) Binomial LLC. Modified.
fn find_sels(px: &[[u8; 4]; 16], l: [u32; 3], h: [u32; 3]) -> [u8; 16] {
    let e5 = |v: u32| (v << 3) | (v >> 2);
    let e6 = |v: u32| (v << 2) | (v >> 4);
    let mut br = [e5(l[0]), 0, 0, e5(h[0])];
    let mut bg = [e6(l[1]), 0, 0, e6(h[1])];
    let mut bb = [e5(l[2]), 0, 0, e5(h[2])];
    for c in [&mut br, &mut bg, &mut bb] {
        c[1] = (c[0] * 2 + c[3]) / 3;
        c[2] = (c[3] * 2 + c[0]) / 3;
    }
    let (mut ar, mut ag, mut ab) = (
        br[3] as i32 - br[0] as i32,
        bg[3] as i32 - bg[0] as i32,
        bb[3] as i32 - bb[0] as i32,
    );
    let dots: [i32; 4] =
        std::array::from_fn(|i| br[i] as i32 * ar + bg[i] as i32 * ag + bb[i] as i32 * ab);
    let (t0, t1, t2) = (dots[0] + dots[1], dots[1] + dots[2], dots[2] + dots[3]);
    ar *= 2;
    ag *= 2;
    ab *= 2;
    const SELS: [u8; 4] = [3, 2, 1, 0];
    std::array::from_fn(|i| {
        let d = px[i][0] as i32 * ar + px[i][1] as i32 * ag + px[i][2] as i32 * ab;
        SELS[(d <= t0) as usize + (d < t1) as usize + (d < t2) as usize]
    })
}

// Ported from Basis Universal v2.50, `compute_least_squares_endpoints_rgb`
// (the transcoder's BC1 version, in f32). Apache-2.0, (C) Binomial LLC.
// Modified.
fn least_squares(px: &[[u8; 4]; 16], sels: &[u8; 16]) -> Option<([f32; 3], [f32; 3])> {
    const WEIGHT_VALS: [u32; 4] = [0x000009, 0x010204, 0x040201, 0x090000];
    let (mut q00, mut t) = ([0u32; 3], [0u32; 3]);
    let mut accum = 0u32;
    for i in 0..16 {
        let sel = sels[i] as u32;
        for c in 0..3 {
            t[c] += px[i][c] as u32;
            q00[c] += sel * px[i][c] as u32;
        }
        accum += WEIGHT_VALS[sel as usize];
    }
    let q00f = q00.map(|v| v as f32);
    let q10: [f32; 3] = std::array::from_fn(|c| t[c] as f32 * 3.0f32 - q00f[c]);
    let z00 = ((accum >> 16) & 0xff) as f32;
    let z10 = ((accum >> 8) & 0xff) as f32;
    let z11 = (accum & 0xff) as f32;
    let z01 = z10;
    let mut det = z00 * z11 - z01 * z10;
    if det.abs() < 1e-8f32 {
        return None;
    }
    det = 3.0f32 / det;
    let (iz00, iz01, iz10, iz11) = (z11 * det, -z01 * det, -z10 * det, z00 * det);
    let mut xl: [f32; 3] = std::array::from_fn(|c| iz00 * q00f[c] + iz01 * q10[c]);
    let mut xh: [f32; 3] = std::array::from_fn(|c| iz10 * q00f[c] + iz11 * q10[c]);
    for c in 0..3 {
        if xl[c] < 0.0 || xh[c] > 255.0 {
            let lo = px.iter().map(|p| p[c]).min().unwrap_or(0);
            let hi = px.iter().map(|p| p[c]).max().unwrap_or(0);
            if lo == hi {
                xl[c] = lo as f32;
                xh[c] = hi as f32;
            }
        }
    }
    Some((xl, xh))
}

fn to5(v: u32) -> u32 {
    let v = v * 31 + 128;
    ((v + (v >> 8)) >> 8) & 0xff
}

fn to6(v: u32) -> u32 {
    let v = v * 63 + 128;
    ((v + (v >> 8)) >> 8) & 0xff
}

// Ported from Basis Universal v2.50, `encode_bc1` (flags 0 or
// `cEncodeBC1UseSelectors`, one least-squares pass). Apache-2.0,
// (C) Binomial LLC. Modified: `selectors` stands for the block's selector
// bytes upstream reads in the use-selectors mode.
pub(crate) fn encode(px: &[[u8; 4]; 16], selectors: Option<[u8; 4]>) -> Bc1 {
    let (m5, m6) = match_tables();
    let mut avg = [-1i32, 0, 0];
    let mut sels;
    let (mut l, mut h);
    if let Some(s) = selectors {
        const TRAN: [u8; 4] = [0, 3, 1, 2];
        let bits = u32::from_le_bytes(s);
        sels = std::array::from_fn(|i| TRAN[((bits >> (i * 2)) & 3) as usize]);
    } else {
        let f = px[0];
        if px.iter().all(|p| p[..3] == f[..3]) {
            return solid_block(f[0] as usize, f[1] as usize, f[2] as usize);
        }
        let mut total = [f[0] as i32, f[1] as i32, f[2] as i32];
        let mut max = total;
        let mut min = total;
        for p in &px[1..] {
            for c in 0..3 {
                let v = p[c] as i32;
                max[c] = max[c].max(v);
                min[c] = min[c].min(v);
                total[c] += v;
            }
        }
        avg = total.map(|t| (t + 8) >> 4);
        let mut icov = [0i32; 6];
        for p in px {
            let (r, g, b) = (
                p[0] as i32 - avg[0],
                p[1] as i32 - avg[1],
                p[2] as i32 - avg[2],
            );
            icov[0] += r * r;
            icov[1] += r * g;
            icov[2] += r * b;
            icov[3] += g * g;
            icov[4] += g * b;
            icov[5] += b * b;
        }
        let cov = icov.map(|v| v as f32 * (1.0f32 / 255.0f32));
        let (mut xr, mut xg, mut xb) = (
            (max[0] - min[0]) as f32,
            (max[1] - min[1]) as f32,
            (max[2] - min[2]) as f32,
        );
        for _ in 0..4 {
            let r = xr * cov[0] + xg * cov[1] + xb * cov[2];
            let g = xr * cov[1] + xg * cov[3] + xb * cov[4];
            let b = xr * cov[2] + xg * cov[4] + xb * cov[5];
            xr = r;
            xg = g;
            xb = b;
        }
        let k = xr.abs().max(xg.abs()).max(xb.abs());
        let (mut sr, mut sg, mut sb) = (306i32, 601i32, 117i32);
        if k >= 2.0 {
            let m = 1024.0f32 / k;
            sr = (xr * m) as i32;
            sg = (xg * m) as i32;
            sb = (xb * m) as i32;
        }
        let (mut low_dot, mut high_dot, mut low_c, mut high_c) = (i32::MAX, i32::MIN, 0, 0);
        for (i, p) in px.iter().enumerate() {
            let dot = p[0] as i32 * sr + p[1] as i32 * sg + p[2] as i32 * sb;
            if dot < low_dot {
                low_dot = dot;
                low_c = i;
            }
            if dot > high_dot {
                high_dot = dot;
                high_c = i;
            }
        }
        l = [
            to5(px[low_c][0] as u32),
            to6(px[low_c][1] as u32),
            to5(px[low_c][2] as u32),
        ];
        h = [
            to5(px[high_c][0] as u32),
            to6(px[high_c][1] as u32),
            to5(px[high_c][2] as u32),
        ];
        sels = find_sels(px, l, h);
    }

    // One least-squares pass.
    if let Some((xl, xh)) = least_squares(px, &sels) {
        let q = |x: f32, s: f32, max: i32| ((x * s + 0.5f32) as i32).clamp(0, max) as u32;
        let (s5, s6) = (31.0f32 / 255.0f32, 63.0f32 / 255.0f32);
        l = [q(xl[0], s5, 31), q(xl[1], s6, 63), q(xl[2], s5, 31)];
        h = [q(xh[0], s5, 31), q(xh[1], s6, 63), q(xh[2], s5, 31)];
    } else {
        if avg[0] < 0 {
            let mut total = [0i32; 3];
            for p in px {
                for c in 0..3 {
                    total[c] += p[c] as i32;
                }
            }
            avg = total.map(|t| (t + 8) >> 4);
        }
        let a = avg.map(|v| v as usize);
        l = [m5[a[0]].hi as u32, m6[a[1]].hi as u32, m5[a[2]].hi as u32];
        h = [m5[a[0]].lo as u32, m6[a[1]].lo as u32, m5[a[2]].lo as u32];
    }
    sels = find_sels(px, l, h);

    let mut lc = l[2] | (l[1] << 5) | (l[0] << 11);
    let mut hc = h[2] | (h[1] << 5) | (h[0] << 11);
    let mut out = [0u8; 8];
    if lc == hc {
        let mut mask = 0u8;
        if hc > 0 {
            hc -= 1;
        } else {
            hc = 0;
            lc = 1;
            mask = 0x55;
        }
        set_low(&mut out, lc);
        set_high(&mut out, hc);
        out[4..].fill(mask);
    } else {
        let mut invert = 0u8;
        if lc < hc {
            std::mem::swap(&mut lc, &mut hc);
            invert = 0x55;
        }
        set_low(&mut out, lc);
        set_high(&mut out, hc);
        const TRANS: [u32; 4] = [0, 2, 3, 1];
        let mut packed = 0u32;
        for i in 0..16 {
            packed |= TRANS[sels[i] as usize] << (i * 2);
        }
        for (k, b) in out[4..].iter_mut().enumerate() {
            *b = (packed >> (8 * k)) as u8 ^ invert;
        }
    }
    out
}

// Ported from Basis Universal v2.50, `s_uastc_to_bc1_weights`.
// Apache-2.0, (C) Binomial LLC. Modified.
const UASTC_TO_BC1: [&[u8]; 6] = [
    &[],
    &[0, 1],
    &[0, 2, 3, 1],
    &[0, 0, 2, 2, 3, 3, 1, 1],
    &[0, 0, 0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1, 1, 1],
    &[
        0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1,
        1, 1,
    ],
];

fn first_plane_selectors(u: &Unpacked, invert: bool) -> u32 {
    let tran = UASTC_TO_BC1[WEIGHT_BITS[u.mode] as usize];
    let shift = PLANES[u.mode] as usize - 1;
    let mut sels = 0u32;
    for i in (0..16).rev() {
        let mut s = tran[u.weights[i << shift] as usize] as u32;
        if invert {
            s ^= 1;
        }
        sels = (sels << 2) | s;
    }
    sels
}

// Ported from Basis Universal v2.50, `dxt1_block::pack_color` (scaled,
// bias 127). Apache-2.0, (C) Binomial LLC. Modified.
fn pack565(r: u32, g: u32, b: u32) -> u32 {
    let (r, g, b) = (
        (r * 31 + 127) / 255,
        (g * 63 + 127) / 255,
        (b * 31 + 127) / 255,
    );
    b.min(31) | (g.min(63) << 5) | (r.min(31) << 11)
}

// Ported from Basis Universal v2.50, `transcode_uastc_to_bc1_hint0`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// BC1 from the first subset's endpoints and the first plane's weights.
pub(crate) fn hint0(u: &Unpacked) -> Bc1 {
    let unq = &UNQUANT[ENDPOINT_RANGES[u.mode] as usize];
    let e = |i: usize| unq[u.endpoints[i] as usize] as u32;
    let mut b = [0u8; 8];
    if COMPS[u.mode] == 2 {
        set_low(&mut b, pack565(e(0), e(0), e(0)));
        set_high(&mut b, pack565(e(1), e(1), e(1)));
    } else {
        set_low(&mut b, pack565(e(0), e(2), e(4)));
        set_high(&mut b, pack565(e(1), e(3), e(5)));
    }
    if low(&b) == high(&b) {
        let (mut lc, mut hc) = (low(&b), high(&b));
        let mut mask = 0u8;
        if hc > 0 {
            hc -= 1;
        } else {
            hc = 0;
            lc = 1;
            mask = 0x55;
        }
        set_low(&mut b, lc);
        set_high(&mut b, hc);
        b[4..].fill(mask);
    } else {
        let mut invert = false;
        if low(&b) < high(&b) {
            b.swap(0, 2);
            b.swap(1, 3);
            invert = true;
        }
        b[4..].copy_from_slice(&first_plane_selectors(u, invert).to_le_bytes());
    }
    b
}

// Ported from Basis Universal v2.50, `transcode_uastc_to_bc1_hint1` (not
// high quality). Apache-2.0, (C) Binomial LLC. Modified.
/// BC1 from the first plane's weights, endpoints by least squares.
pub(crate) fn hint1(u: &Unpacked, decoded: &[[u8; 4]; 16]) -> Bc1 {
    encode(decoded, Some(first_plane_selectors(u, false).to_le_bytes()))
}

// Ported from Basis Universal v2.50, `bcu::unpack_bc1` (BC1 mode, alpha
// set). Apache-2.0, (C) Binomial LLC. Modified.
pub(crate) fn decode(b: &Bc1) -> [[u8; 4]; 16] {
    let unpack = |c: u32| {
        let (r, g, b) = ((c >> 11) & 31, (c >> 5) & 63, c & 31);
        [
            (r << 3) | (r >> 2),
            (g << 2) | (g >> 4),
            (b << 3) | (b >> 2),
        ]
    };
    let (l, h) = (low(b), high(b));
    let (c0, c1) = (unpack(l), unpack(h));
    let mut c = [[0u8; 4]; 4];
    c[0] = [c0[0] as u8, c0[1] as u8, c0[2] as u8, 255];
    c[1] = [c1[0] as u8, c1[1] as u8, c1[2] as u8, 255];
    if l > h {
        c[2] = std::array::from_fn(|i| {
            if i < 3 {
                ((c0[i] * 2 + c1[i]) / 3) as u8
            } else {
                255
            }
        });
        c[3] = std::array::from_fn(|i| {
            if i < 3 {
                ((c1[i] * 2 + c0[i]) / 3) as u8
            } else {
                255
            }
        });
    } else {
        c[2] = std::array::from_fn(|i| {
            if i < 3 {
                ((c0[i] + c1[i]) / 2) as u8
            } else {
                255
            }
        });
        c[3] = [0, 0, 0, 0];
    }
    std::array::from_fn(|i| {
        let (x, y) = (i % 4, i / 4);
        c[((b[4 + y] >> (x * 2)) & 3) as usize]
    })
}
