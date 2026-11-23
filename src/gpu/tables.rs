// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! The WGSL constants each kernel is compiled with: its [`Quality`], the
//! format's tables, and for ASTC the configurations it tries. Generated here
//! from the specification's procedures (`crate::astc`) and the crate's BC7
//! tables, so the kernels hold no hand-copied table.

use std::fmt::Write as _;

use super::Quality;
use crate::astc;

fn list<T: std::fmt::Display>(v: impl IntoIterator<Item = T>, suffix: &str) -> String {
    v.into_iter()
        .map(|x| format!("{x}{suffix}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The quality constants every kernel reads: `TIER`, `K`, `ITERS`, and
/// `EW`, the error weight of R, G, B and A.
pub(crate) fn quality(q: Quality) -> String {
    let q = q.clamped();
    let [r, g, b] = q.channel_weights();
    format!(
        "const TIER: u32 = {}u;\nconst K: u32 = {}u;\nconst ITERS: u32 = {}u;\nconst EW = vec4f({r:?}, {g:?}, {b:?}, 1.0);\n",
        q.mode_tier, q.partition_candidates, q.refine_iterations,
    )
}

/// For each integer weight 0..=64, the index of `weights` with the largest
/// weight at or below it (low byte) and the smallest at or above it (second
/// byte): the two an ideal weight between v and v + 1 is bracketed by.
pub(crate) fn bracket(weights: &[u32]) -> [u32; 65] {
    std::array::from_fn(|v| {
        let v = v as u32;
        let below = (0..weights.len())
            .filter(|&k| weights[k] <= v)
            .max_by_key(|&k| weights[k])
            .unwrap_or(0);
        let above = (0..weights.len())
            .filter(|&k| weights[k] >= v)
            .min_by_key(|&k| weights[k])
            .unwrap_or(0);
        below as u32 | (above as u32) << 8
    })
}

/// One bit per texel: set where `pattern` puts the texel in `subset`.
fn mask(pattern: &[u8; 16], subset: u8) -> u32 {
    pattern
        .iter()
        .enumerate()
        .filter(|(_, &s)| s == subset)
        .fold(0, |m, (i, _)| m | 1 << i)
}

/// BC7's partition tables (Khronos Data Format 1.3, Tables 114–118): `P2`
/// the second subset's mask of each two-subset partition, `A2` its anchor;
/// `P3` the second and third subsets' masks of each three-subset partition
/// (low and high 16 bits), `A3` their anchors (low and high byte).
pub(crate) fn bc7() -> String {
    use crate::uastc::bc7::{ANCHOR2, ANCHOR3_1, ANCHOR3_2, PARTITION2, PARTITION3};
    let p2 = PARTITION2.iter().map(|p| mask(p, 1));
    let p3 = PARTITION3.iter().map(|p| mask(p, 1) | mask(p, 2) << 16);
    let a3 = ANCHOR3_1
        .iter()
        .zip(ANCHOR3_2)
        .map(|(&a, b)| a as u32 | (b as u32) << 8);
    // Table 119.
    let weights: [&[u32]; 3] = [
        &[0, 21, 43, 64],
        &[0, 9, 18, 27, 37, 46, 55, 64],
        &[0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64],
    ];
    let ws = weights.iter().flat_map(|w| bracket(w));
    format!(
        "const P2 = array<u32, 64>({});\nconst A2 = array<u32, 64>({});\nconst P3 = array<u32, 64>({});\nconst A3 = array<u32, 64>({});\nconst WSEL = array<u32, 195>({});\n",
        list(p2, "u"),
        list(ANCHOR2, "u"),
        list(p3, "u"),
        list(a3, "u"),
        list(ws, "u"),
    )
}

/// One ASTC 4×4 configuration a kernel tries: partitions, colour endpoint
/// mode, a 4×4 weight grid of one plane, and the ranges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AstcConfig {
    pub parts: u32,
    pub cem: u32,
    /// The weight range index (0..=11).
    pub weight_range: usize,
    /// The endpoint range index the decoder derives (§23.13).
    pub endpoint_range: usize,
    /// The 11-bit block mode.
    pub mode: u32,
}

/// Endpoint values per partition of a colour endpoint mode.
fn cem_values(cem: u32) -> u32 {
    match cem {
        0 => 2,
        6 => 4,
        8 => 6,
        12 => 8,
        _ => unreachable!("a mode the kernel does not write"),
    }
}

impl AstcConfig {
    /// The configuration, if it is a valid block: weights of 24 to 96 bits
    /// (§23.11) and room left for endpoints of at least 0..5.
    pub(crate) fn new(parts: u32, cem: u32, weight_range: usize) -> Option<Self> {
        let weight_bits = astc::ise_bits(16, weight_range);
        if !(24..=96).contains(&weight_bits) {
            return None;
        }
        // Block mode, partition count, then the CEM (4 bits) for one
        // partition, or the partition index (10) and the CEM field (6).
        let header = if parts == 1 { 17 } else { 29 };
        let endpoint_range =
            astc::endpoint_range(parts * cem_values(cem), 128 - header - weight_bits)?;
        Some(Self {
            parts,
            cem,
            weight_range,
            endpoint_range,
            mode: astc::block_mode(weight_range, false),
        })
    }
}

// Weight range indices (Table 155 order).
const W0_2: usize = 1;
const W0_3: usize = 2;
const W0_5: usize = 4;
const W0_7: usize = 5;
const W0_11: usize = 7;
const W0_15: usize = 8;
const W0_19: usize = 9;
const W0_23: usize = 10;
const W0_31: usize = 11;

/// The configurations tried at `tier`, for opaque blocks and for blocks with
/// alpha.
pub(crate) fn astc_configs(tier: u32) -> (Vec<AstcConfig>, Vec<AstcConfig>) {
    let mut opaque = vec![(1, 8, W0_11)];
    let mut alpha = vec![(1, 12, W0_5)];
    if tier >= 1 {
        opaque.extend([
            (1, 8, W0_7),
            (1, 8, W0_15),
            (1, 8, W0_19),
            (1, 8, W0_23),
            (1, 8, W0_31),
            (1, 0, W0_15),
            (1, 0, W0_31),
            (1, 6, W0_15),
            (1, 6, W0_31),
        ]);
        alpha.extend([(1, 12, W0_7), (1, 12, W0_11)]);
    }
    if tier >= 2 {
        opaque.extend([(2, 8, W0_2), (2, 8, W0_3), (2, 8, W0_5), (2, 8, W0_7)]);
        alpha.extend([(2, 12, W0_2), (2, 12, W0_3)]);
    }
    let build = |v: Vec<(u32, u32, usize)>| {
        v.into_iter()
            .map(|(p, c, w)| AstcConfig::new(p, c, w).expect("a valid configuration"))
            .collect()
    };
    (build(opaque), build(alpha))
}

/// ASTC's tables and configurations for `tier`:
///
/// - `TRIT_ENC`, `QUINT_ENC`: the smallest packing of each trit quintuple and
///   quint triple (§23.12);
/// - `RANGE`: per range index, its bits, trits and quints (bytes 0, 1, 2);
/// - `WUNQ`: per weight range, the 0..64 weight of each stored value (32 a
///   range);
/// - `EQ`: per endpoint range used, for each 0..255 value, the stored value
///   nearest it and what that decodes to (`code | value << 8`), 256 a range;
/// - `CFG_O`, `CFG_A`: the configurations for opaque blocks and for blocks
///   with alpha, as `(parts | cem << 4, weight range, EQ offset, block mode |
///   endpoint range << 16)`;
/// - `AP2`: the distinct two-partition layouts, `seed | mask << 16`.
pub(crate) fn astc(tier: u32) -> String {
    let (opaque, alpha) = astc_configs(tier);
    let mut ranges: Vec<usize> = opaque
        .iter()
        .chain(&alpha)
        .map(|c| c.endpoint_range)
        .collect();
    ranges.sort_unstable();
    ranges.dedup();
    let mut eq = Vec::with_capacity(256 * ranges.len());
    for &r in &ranges {
        for v in 0..256u32 {
            let code = (0..astc::levels(r))
                .min_by_key(|&c| astc::unquantize_endpoint(c, r).abs_diff(v))
                .expect("a range has values");
            eq.push(code | astc::unquantize_endpoint(code, r) << 8);
        }
    }
    let cfg = |c: &AstcConfig| {
        let slot = ranges.iter().position(|&r| r == c.endpoint_range).unwrap_or(0);
        format!(
            "vec4u({}u, {}u, {}u, {}u)",
            c.parts | c.cem << 4,
            c.weight_range,
            slot * 256,
            c.mode | (c.endpoint_range as u32) << 16
        )
    };
    let mut wunq = Vec::with_capacity(12 * 32);
    for r in 0..12 {
        for v in 0..32 {
            wunq.push(if v < astc::levels(r) {
                astc::unquantize_weight_range(v, r)
            } else {
                0
            });
        }
    }
    let mut wsel = Vec::with_capacity(12 * 65);
    for r in 0..12 {
        let w: Vec<u32> = (0..astc::levels(r))
            .map(|v| astc::unquantize_weight_range(v, r))
            .collect();
        wsel.extend(bracket(&w));
    }
    let layouts = astc::two_partition_layouts();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "const TRIT_ENC = array<u32, 243>({});",
        list(astc::trit_encodings(), "u")
    );
    let _ = writeln!(
        out,
        "const QUINT_ENC = array<u32, 125>({});",
        list(astc::quint_encodings(), "u")
    );
    let _ = writeln!(
        out,
        "const RANGE = array<u32, 21>({});",
        list(astc::RANGES.iter().map(|&(b, t, q)| b | t << 8 | q << 16), "u")
    );
    let _ = writeln!(out, "const WUNQ = array<f32, 384>({});", list(wunq, ".0"));
    let _ = writeln!(out, "const WSEL = array<u32, 780>({});", list(wsel, "u"));
    let _ = writeln!(
        out,
        "const EQ = array<u32, {}>({});",
        eq.len(),
        list(eq, "u")
    );
    for (name, v) in [("O", &opaque), ("A", &alpha)] {
        let _ = writeln!(
            out,
            "const NCFG_{name}: u32 = {}u;\nconst CFG_{name} = array<vec4u, {}>({});",
            v.len(),
            v.len(),
            v.iter().map(cfg).collect::<Vec<_>>().join(", ")
        );
    }
    let _ = writeln!(
        out,
        "const NP2: u32 = {}u;\nconst AP2 = array<u32, {}>({});",
        layouts.len(),
        layouts.len(),
        list(layouts.iter().map(|&(s, m)| s | m << 16), "u")
    );
    out
}
