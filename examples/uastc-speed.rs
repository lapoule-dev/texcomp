// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! Measures the CPU UASTC encoder: milliseconds per 256² tile with its full
//! mip chain, on one thread, at each level.
//!
//! The tiles are the twenty `gpu_quality` cuts of the NASA imagery fixtures,
//! each with its mips (triangle filter) down to 1×1: 5,461 blocks per tile.
//! Run it in release:
//!
//! ```sh
//! cargo run --release --example uastc-speed
//! ```
//!
//! Under wasm32-wasip1 the fixtures directory must be mapped into the guest
//! (`--dir`), see the crate's CI.

use std::time::Instant;

use image::imageops::{self, FilterType};
use texcomp::uastc::{self, Level};

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/imagery");
    let mut chains: Vec<Vec<image::RgbaImage>> = Vec::new();
    for name in [
        "alps.jpeg",
        "nile.jpeg",
        "coast.jpeg",
        "modis.jpeg",
        "paris-night.png",
    ] {
        let full = image::open(dir.join(name)).expect("fixture").to_rgba8();
        for q in 0..4u32 {
            let tile = imageops::crop_imm(&full, (q % 2) * 256, (q / 2) * 256, 256, 256).to_image();
            let mut chain = vec![tile];
            while chain
                .last()
                .map(|l| l.width() > 1 || l.height() > 1)
                .unwrap_or(false)
            {
                let prev = chain.last().expect("level");
                let (w, h) = ((prev.width() / 2).max(1), (prev.height() / 2).max(1));
                chain.push(imageops::resize(prev, w, h, FilterType::Triangle));
            }
            chains.push(chain);
        }
    }
    let blocks: usize = chains[0]
        .iter()
        .map(|l| (l.width().div_ceil(4) * l.height().div_ceil(4)) as usize)
        .sum();
    println!("{} tiles, {blocks} blocks per tile with mips", chains.len());
    // The best of three runs: the machine may be busy.
    for (level, label) in [(Level::Fastest, 0), (Level::Faster, 1), (Level::Default, 2)] {
        let mut best = f64::MAX;
        let mut checksum = 0u64;
        for _ in 0..3 {
            let start = Instant::now();
            checksum = 0;
            for chain in &chains {
                for l in chain {
                    let mut out =
                        vec![0u8; (l.width().div_ceil(4) * l.height().div_ceil(4) * 16) as usize];
                    uastc::encode_image(l.as_raw(), l.width(), l.height(), level, &mut out)
                        .expect("sizes");
                    checksum = checksum.wrapping_add(out.iter().map(|&b| b as u64).sum::<u64>());
                }
            }
            best = best.min(start.elapsed().as_secs_f64() * 1000.0 / chains.len() as f64);
        }
        let mtexel = (blocks * 16) as f64 / (best * 1000.0);
        println!("level {label}: {best:.1} ms per tile with mips ({mtexel:.2} Mtexel/s), checksum {checksum}");
    }
}
