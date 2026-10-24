// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! The CPU UASTC encoder against the blocks upstream's encoder wrote.
//!
//! `tests/fixtures/uastc/tiles` holds the nine 256² tiles the upstream tool
//! encoded, and `level<N>/<tile>.uastc` the blocks it wrote at effort level
//! N (see the fixtures' README for the command lines and the commit).
//!
//! # What this covers
//!
//! - Every block of the nine main tiles and the eight 64² coverage tiles
//!   (luminance–alpha, alpha noise, partition-friendly blocks), encoded at
//!   each ported level, compared byte
//!   for byte with upstream's block: the ASTC part, the transcoding hints
//!   (BC1, ETC1, EAC), everything. The count of differing blocks is printed
//!   per level and per upstream mode.
//! - The whole-image entry point against the per-block one, on a size that
//!   is not a multiple of 4 (edge texels replicated as upstream does).
//!
//! # What it does not cover
//!
//! Levels above those ported (3 and 4), and inputs other than these tiles.

use std::path::{Path, PathBuf};

use texcomp::uastc::{self, Level, UastcBlock};

const TILES: [&str; 9] = [
    "alps-0",
    "nile-1",
    "coast-2",
    "modis-3",
    "paris-night-0",
    "gradient",
    "stripes",
    "alpha-edge",
    "noise",
];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/uastc")
}

fn tile(name: &str) -> image::RgbaImage {
    // The main set; the coverage set is read in `compare`.
    image::open(fixtures().join(format!("tiles/{name}.png")))
        .expect("tile")
        .to_rgba8()
}

fn block_at(img: &image::RgbaImage, bx: u32, by: u32) -> [[u8; 4]; 16] {
    let mut b = [[0u8; 4]; 16];
    for (i, texel) in b.iter_mut().enumerate() {
        let (x, y) = (bx * 4 + i as u32 % 4, by * 4 + i as u32 / 4);
        *texel = img.get_pixel(x, y).0;
    }
    b
}

const COVERAGE_TILES: [&str; 8] = [
    "modis-3-crop",
    "alps-0-crop",
    "stripes-crop",
    "grey",
    "grey-alpha",
    "alpha-noise",
    "alpha-imagery",
    "partitions",
];

/// Encodes every main and coverage tile at `level` and counts the blocks
/// that differ from upstream's, per upstream mode.
fn compare(level: Level, n: u32) {
    let mut total = 0usize;
    let mut per_mode = [(0usize, 0usize); uastc::MODES];
    let mut first = Vec::new();
    let sets = [
        ("tiles", "", &TILES[..], 64u32),
        ("coverage-tiles", "coverage/", &COVERAGE_TILES[..], 16),
    ];
    for (dir, prefix, names, side) in sets {
        for name in names {
            let img = image::open(fixtures().join(format!("{dir}/{name}.png")))
                .expect("tile")
                .to_rgba8();
            let want = std::fs::read(fixtures().join(format!("{prefix}level{n}/{name}.uastc")))
                .expect("vectors");
            let (blocks, _) = want.as_chunks::<16>();
            for by in 0..side {
                for bx in 0..side {
                    let want = UastcBlock::from_bytes(blocks[(by * side + bx) as usize]);
                    let got = UastcBlock::encode(&block_at(&img, bx, by), level);
                    let mode = want.mode().expect("valid") as usize;
                    per_mode[mode].0 += 1;
                    total += 1;
                    if got != want {
                        per_mode[mode].1 += 1;
                        if first.len() < 8 {
                            first.push(format!(
                                "{name} block ({bx}, {by}), upstream mode {mode}, ours {:?}:\n  got  {:02x?}\n  want {:02x?}",
                                got.mode(),
                                got.to_bytes(),
                                want.to_bytes()
                            ));
                        }
                    }
                }
            }
        }
    }
    let wrong: usize = per_mode.iter().map(|m| m.1).sum();
    println!("level {n}: {wrong} of {total} blocks differ; (blocks, differing) per upstream mode: {per_mode:?}");
    assert!(
        wrong == 0,
        "level {n}: {wrong} of {total} blocks differ; first:\n{}",
        first.join("\n")
    );
}

/// **Level 0 writes upstream's blocks, byte for byte.**
#[test]
fn level_0_is_byte_identical() {
    compare(Level::Fastest, 0);
}

/// **Level 1 writes upstream's blocks, byte for byte.**
#[test]
fn level_1_is_byte_identical() {
    compare(Level::Faster, 1);
}

/// **Level 2 writes upstream's blocks, byte for byte.**
#[test]
fn level_2_is_byte_identical() {
    compare(Level::Default, 2);
}

/// **The whole-image entry point encodes each block as the per-block one
/// does**, replicating the edge texels of a size that is not a multiple of 4.
#[test]
fn image_entry_point_replicates_edges() {
    let img = tile("modis-3");
    let (w, h) = (13u32, 6u32);
    let mut rgba = Vec::new();
    for y in 0..h {
        for x in 0..w {
            rgba.extend_from_slice(&img.get_pixel(x, y).0);
        }
    }
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    let mut out = vec![0u8; (bw * bh * 16) as usize];
    uastc::encode_image(&rgba, w, h, Level::Fastest, &mut out).expect("sizes match");
    for by in 0..bh {
        for bx in 0..bw {
            let mut b = [[0u8; 4]; 16];
            for (i, t) in b.iter_mut().enumerate() {
                let x = (bx * 4 + i as u32 % 4).min(w - 1);
                let y = (by * 4 + i as u32 / 4).min(h - 1);
                *t = img.get_pixel(x, y).0;
            }
            let at = ((by * bw + bx) * 16) as usize;
            assert_eq!(
                &out[at..at + 16],
                &UastcBlock::encode(&b, Level::Fastest).to_bytes(),
                "block ({bx}, {by})"
            );
        }
    }
    assert!(uastc::encode_image(&rgba, w, h, Level::Fastest, &mut out[..16]).is_err());
}
