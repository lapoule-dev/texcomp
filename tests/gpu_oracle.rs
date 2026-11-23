// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! Every block the GPU encoders write, at every quality level, decoded by
//! two decoders that know nothing of this crate: a software one (the
//! `texture2ddecoder` crate, the oracle) and the GPU's own.
//!
//! # What this covers
//!
//! - **Valid, as predicted:** at every [`Quality::level`], every block of
//!   the six aerial tiles and of synthetic tiles built to call on every mode
//!   (hard two- and three-colour splits, gradients, decorrelated channels,
//!   alpha of every kind) decodes texel for texel the same, within one step,
//!   through the oracle and through the GPU. An invalid ASTC block decodes
//!   to the error colour on the GPU, a misread BC7 field to garbage, so the
//!   two would part.
//! - **Every mode is exercised:** the blocks written at the highest level
//!   use every BC7 mode, 0 to 7, and every ASTC configuration kind
//!   (luminance, base + scale, RGB direct, RGBA direct, one and two
//!   partitions).
//! - **Opaque stays opaque:** every texel the source draws at alpha 255
//!   decodes at exactly 255.
//! - **Levels never regress:** each tile's PSNR at every level is at least
//!   level 0's, less 0.1 dB of noise.
//! - **Bounded error on structured random blocks:** solid blocks within 1,
//!   gradients and two-colour splits within the bounds stated per format.
//! - **Determinism:** the same texels give the same bytes, call after call.
//!
//! # What it does not cover
//!
//! Formats the test machine's GPU does not sample (it skips them), and the
//! sRGB decode (the linear view decodes the stored values; see
//! `gpu_quality.rs`). A machine without a GPU skips; `TEXCOMP_REQUIRE_GPU=1`
//! makes that a failure.

#[path = "support/quality.rs"]
mod quality;

use std::collections::BTreeSet;

use quality::*;
use texcomp::gpu::{GpuEncoder, Quality};
use texcomp::BlockFormat;

/// A tile whose 4×4 blocks are each drawn by `block(bx, by, rng)`.
fn blockwise(name: &str, mut block: impl FnMut(u32, u32, &mut u64) -> [[u8; 4]; 16]) -> Tile {
    let mut rgba8 = vec![0u8; (TILE * TILE * 4) as usize];
    let mut rng = 0x1234_5678_9abc_def1u64;
    for by in 0..TILE / 4 {
        for bx in 0..TILE / 4 {
            let texels = block(bx, by, &mut rng);
            for (i, t) in texels.iter().enumerate() {
                let (x, y) = (bx * 4 + (i as u32 & 3), by * 4 + (i as u32 >> 2));
                let at = ((y * TILE + x) * 4) as usize;
                rgba8[at..at + 4].copy_from_slice(t);
            }
        }
    }
    Tile {
        name: name.into(),
        chain: chain(Image {
            width: TILE,
            height: TILE,
            rgba8,
        }),
    }
}

fn next(rng: &mut u64) -> u32 {
    *rng ^= *rng >> 12;
    *rng ^= *rng << 25;
    *rng ^= *rng >> 27;
    (rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
}

fn colour(rng: &mut u64, alpha: bool) -> [u8; 4] {
    let v = next(rng);
    let a = if alpha { (next(rng) % 256) as u8 } else { 255 };
    [v as u8, (v >> 8) as u8, (v >> 16) as u8, a]
}

/// The structured random blocks: solid, gradient, two-colour split, three-
/// colour split, decorrelated channels; opaque or with alpha.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Solid,
    Gradient,
    Split2,
    Split3,
    Decorrelated,
}

const KINDS: [Kind; 5] = [
    Kind::Solid,
    Kind::Gradient,
    Kind::Split2,
    Kind::Split3,
    Kind::Decorrelated,
];

fn kind_of(bx: u32, by: u32) -> Kind {
    KINDS[((bx + 3 * by) % 5) as usize]
}

fn structured_block(kind: Kind, rng: &mut u64, alpha: bool) -> [[u8; 4]; 16] {
    let a = colour(rng, alpha);
    let b = colour(rng, alpha);
    let c = colour(rng, alpha);
    let lerp = |x: u8, y: u8, t: f32| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    // A random straight edge through the block.
    let (nx, ny, d) = (
        (next(rng) % 7) as f32 - 3.0,
        (next(rng) % 7) as f32 - 3.0,
        (next(rng) % 5) as f32 - 2.0,
    );
    std::array::from_fn(|i| {
        let (x, y) = ((i & 3) as f32 - 1.5, (i >> 2) as f32 - 1.5);
        match kind {
            Kind::Solid => a,
            Kind::Gradient => {
                let t = ((x + y) / 6.0 + 0.5).clamp(0.0, 1.0);
                std::array::from_fn(|ch| lerp(a[ch], b[ch], t))
            }
            Kind::Split2 => {
                if nx * x + ny * y > d {
                    a
                } else {
                    b
                }
            }
            Kind::Split3 => {
                if x < -0.5 {
                    a
                } else if y < 0.0 {
                    b
                } else {
                    c
                }
            }
            Kind::Decorrelated => {
                // Red rises left to right, green top to bottom: no one line
                // through RGB fits both.
                [
                    lerp(a[0], b[0], (x + 1.5) / 3.0),
                    lerp(a[1], b[1], (y + 1.5) / 3.0),
                    a[2],
                    a[3],
                ]
            }
        }
    })
}

fn structured(alpha: bool) -> Tile {
    blockwise(
        if alpha { "structured-alpha" } else { "structured" },
        |bx, by, rng| structured_block(kind_of(bx, by), rng, alpha),
    )
}

/// Every tile these tests encode: the aerial ones, the synthetic ones, and
/// the structured random blocks, opaque and with alpha.
fn tiles() -> Vec<Tile> {
    let mut t = aerial_tiles();
    t.extend(synthetic_tiles());
    t.push(structured(false));
    t.push(structured(true));
    t
}

struct Encoded {
    blocks: Vec<Vec<u8>>,
    oracle: Vec<Vec<u8>>,
    hardware: Vec<Vec<u8>>,
}

fn encode_all(gpu: &Gpu, encoder: &GpuEncoder, tiles: &[Tile]) -> Vec<Encoded> {
    let textures = encode(gpu, encoder, tiles, true);
    tiles
        .iter()
        .zip(&textures)
        .map(|(tile, texture)| {
            let sizes: Vec<(u32, u32)> = tile.chain.iter().map(|l| (l.width, l.height)).collect();
            let blocks = read_blocks(gpu, texture, &sizes);
            let oracle = blocks
                .iter()
                .zip(&sizes)
                .map(|(b, &(w, h))| oracle_decode(encoder.format(), b, w, h))
                .collect();
            let hardware = gpu_decode(gpu, texture, &sizes);
            Encoded {
                blocks,
                oracle,
                hardware,
            }
        })
        .collect()
}

/// **Every block at every level is valid and decodes as predicted, opaque
/// stays opaque, every mode is exercised, and no level falls below level
/// 0.**
#[test]
fn every_level_decodes_through_the_oracle() {
    let Some(gpu) = gpu() else { return };
    let tiles = tiles();
    for format in formats(&gpu) {
        let mut level0: Vec<(f64, f64)> = Vec::new();
        let mut kinds_at_top = BTreeSet::new();
        for level in 0..=Quality::MAX_LEVEL {
            let encoder = GpuEncoder::with_quality(&gpu.device, format, Quality::level(level));
            let encoded = encode_all(&gpu, &encoder, &tiles);
            for (tile, e) in tiles.iter().zip(&encoded) {
                for (l, (src, (oracle, hw))) in tile
                    .chain
                    .iter()
                    .zip(e.oracle.iter().zip(&e.hardware))
                    .enumerate()
                {
                    let worst = oracle.iter().zip(hw).map(|(a, b)| a.abs_diff(*b)).max();
                    assert!(
                        worst <= Some(1),
                        "{format:?} level {level} {} mip {l}: oracle and GPU differ by {worst:?}",
                        tile.name
                    );
                    assert_opaque_blocks_stay_opaque(&src.rgba8, oracle, src.width, src.height)
                        .unwrap_or_else(|a| {
                            panic!("{format:?} level {level} {} mip {l}: an opaque block decodes at alpha {a}", tile.name)
                        });
                }
                if level == Quality::MAX_LEVEL {
                    for block in e.blocks[0].as_chunks::<16>().0.iter() {
                        kinds_at_top.insert(block_kind(format, block));
                    }
                }
            }
            // RGB PSNR and luma PSNR per tile.
            let scores: Vec<(f64, f64)> = tiles
                .iter()
                .zip(&encoded)
                .map(|(t, e)| psnr_rgb_y(&t.chain[0].rgba8, &e.oracle[0]))
                .collect();
            if level == 0 {
                level0 = scores;
            } else {
                for ((tile, s), r) in tiles.iter().zip(&scores).zip(&level0) {
                    assert!(
                        s.0 >= r.0 - 0.1,
                        "{format:?} level {level} {}: {:.2} dB, level 0 {:.2} dB",
                        tile.name,
                        s.0,
                        r.0
                    );
                }
            }
        }
        let expected: &[&str] = match format {
            BlockFormat::Bc7 => &[
                "mode 0", "mode 1", "mode 2", "mode 3", "mode 4", "mode 5", "mode 6", "mode 7",
            ],
            _ => &[
                "1P CEM0", "1P CEM6", "1P CEM8", "1P CEM12", "2P CEM8", "2P CEM12", "dual",
            ],
        };
        for e in expected {
            assert!(
                kinds_at_top.iter().any(|k| k.starts_with(e) || k.ends_with(e)),
                "{format:?}: no block of kind {e} at the highest level ({kinds_at_top:?})"
            );
        }
    }
}

/// Every texel of a block whose source texels are all opaque decodes at
/// exactly 255: an opaque tile must not turn 254. (A block with any alpha
/// below 255 interpolates alpha like the other channels.) `Err` is the alpha
/// found.
fn assert_opaque_blocks_stay_opaque(src: &[u8], got: &[u8], w: u32, h: u32) -> Result<(), u8> {
    for by in 0..h.div_ceil(4) {
        for bx in 0..w.div_ceil(4) {
            let texels = (0..16u32).filter_map(|i| {
                let (x, y) = (bx * 4 + (i & 3), by * 4 + (i >> 2));
                (x < w && y < h).then(|| ((y * w + x) * 4 + 3) as usize)
            });
            let at: Vec<usize> = texels.collect();
            if at.iter().all(|&a| src[a] == 255) {
                if let Some(&a) = at.iter().find(|&&a| got[a] != 255) {
                    return Err(got[a]);
                }
            }
        }
    }
    Ok(())
}

/// The largest error over a block's texels and channels, and its RMS.
fn block_error(src: &[u8], got: &[u8], bx: u32, by: u32) -> (u8, f64) {
    let (mut worst, mut se) = (0u8, 0.0);
    for i in 0..16u32 {
        let at = (((by * 4 + (i >> 2)) * TILE + bx * 4 + (i & 3)) * 4) as usize;
        for c in 0..4 {
            let d = src[at + c].abs_diff(got[at + c]);
            worst = worst.max(d);
            se += (d as f64) * (d as f64);
        }
    }
    (worst, (se / 64.0).sqrt())
}

/// **Structured random blocks round-trip within bounds**, at the default
/// level: a solid block within 1 step on every channel, and the RMS error
/// of each kind of block within its bound. The bounds were measured on
/// Apple M2 and set just beyond the worst block of these 4096: a looser
/// result is a regression, not noise.
#[test]
fn structured_random_blocks_round_trip_within_bounds() {
    let Some(gpu) = gpu() else { return };
    for format in formats(&gpu) {
        let encoder = GpuEncoder::new(&gpu.device, format);
        for alpha in [false, true] {
            let tile = structured(alpha);
            let e = &encode_all(&gpu, &encoder, std::slice::from_ref(&tile))[0];
            let mut worst = [0.0f64; 5];
            for by in 0..TILE / 4 {
                for bx in 0..TILE / 4 {
                    let kind = kind_of(bx, by);
                    let (max, rms) = block_error(&tile.chain[0].rgba8, &e.oracle[0], bx, by);
                    if kind == Kind::Solid {
                        assert!(max <= 1, "{format:?} alpha {alpha}: a solid block off by {max}");
                    }
                    let k = KINDS.iter().position(|&x| x == kind).unwrap_or(0);
                    worst[k] = worst[k].max(rms);
                }
            }
            eprintln!("{format:?} alpha {alpha}: worst RMS per kind {KINDS:?}: {worst:.2?}");
            let bound = bounds(format, alpha);
            for (k, (w, b)) in worst.iter().zip(bound).enumerate() {
                assert!(
                    *w <= b,
                    "{format:?} alpha {alpha} {:?}: worst RMS {w:.2} beyond {b}",
                    KINDS[k]
                );
            }
        }
    }
}

/// RMS bounds per kind (solid, gradient, split in two, split in three,
/// decorrelated).
fn bounds(format: BlockFormat, alpha: bool) -> [f64; 5] {
    match (format, alpha) {
        (BlockFormat::Bc7, false) => [1.0, 3.5, 1.5, 1.5, 1.5],
        (BlockFormat::Bc7, true) => [1.0, 4.0, 1.5, 3.5, 1.5],
        (_, false) => [1.0, 3.5, 1.0, 3.5, 1.5],
        (_, true) => [1.0, 7.0, 1.0, 8.5, 3.5],
    }
}

/// **The same texels give the same bytes**, at every level, call after
/// call.
#[test]
fn encoding_is_deterministic() {
    let Some(gpu) = gpu() else { return };
    let tiles = vec![aerial_tiles().swap_remove(5), structured(true)];
    for format in formats(&gpu) {
        for level in 0..=Quality::MAX_LEVEL {
            let encoder = GpuEncoder::with_quality(&gpu.device, format, Quality::level(level));
            let a = encode_all(&gpu, &encoder, &tiles);
            let b = encode_all(&gpu, &encoder, &tiles);
            for ((x, y), t) in a.iter().zip(&b).zip(&tiles) {
                assert!(x.blocks == y.blocks, "{format:?} level {level} {}", t.name);
            }
        }
    }
}
