// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! The encoders' quality and cost, measured on real imagery and printed as
//! Markdown tables:
//!
//! ```text
//! cargo run --release --features gpu --example quality-report -- [--out DIR]
//! ```
//!
//! Every tile (six IGN aerial orthophotos, twenty NASA tiles) is encoded
//! with its full mip chain on the GPU, its blocks are read back and decoded
//! by a software decoder independent of this crate (the oracle), and the
//! decode is checked against the GPU's own decoder. Scores are PSNR over RGB
//! and over luma (BT.601) and SSIM over luma, at level 0 and over the whole
//! chain; per block kind (BC7 mode, ASTC configuration) the share of blocks
//! and their PSNR. Encode time is the compute pass's GPU time from
//! timestamps, per 256² tile with its mips. The CPU path (UASTC at upstream's
//! effort levels, transcoded) is scored the same way.
//!
//! With `--out DIR`, A/B sheets (source | decoded | difference ×8, crops at
//! 4×) are written there as PNG.

#[path = "../tests/support/quality.rs"]
mod quality;

use std::collections::BTreeMap;
use std::time::Instant;

use quality::*;
use texcomp::gpu::{GpuEncoder, Quality};
use texcomp::BlockFormat;

/// Scores of one tile at one setting.
#[derive(Default, Clone)]
struct TileScore {
    psnr_rgb: f64,
    psnr_y: f64,
    ssim_y: f64,
    /// PSNR over RGB of the whole chain (every level of 8×8 or more).
    chain_psnr_rgb: f64,
}

/// Squared error and texel count per block kind, over level 0.
type Kinds = BTreeMap<String, (u64, f64)>;

fn score_tile(tile: &Tile, decoded: &[Vec<u8>]) -> TileScore {
    let l0 = &tile.chain[0];
    let (psnr_rgb, psnr_y) = psnr_rgb_y(&l0.rgba8, &decoded[0]);
    let ssim = ssim_y(&l0.rgba8, &decoded[0], l0.width, l0.height);
    let (mut se, mut n) = (0.0, 0.0);
    for (level, got) in tile.chain.iter().zip(decoded) {
        if level.width >= 8 && level.height >= 8 {
            se += sse(&level.rgba8, got).0;
            n += (level.width * level.height * 3) as f64;
        }
    }
    TileScore {
        psnr_rgb,
        psnr_y,
        ssim_y: ssim,
        chain_psnr_rgb: psnr(se / n),
    }
}

fn add_kinds(kinds: &mut Kinds, format: BlockFormat, tile: &Tile, blocks: &[u8], decoded: &[u8]) {
    let l0 = &tile.chain[0];
    let bx = l0.width / 4;
    for (b, block) in blocks.as_chunks::<16>().0.iter().enumerate() {
        let (ox, oy) = ((b as u32 % bx) * 4, (b as u32 / bx) * 4);
        let mut se = 0.0;
        for i in 0..16u32 {
            let at = (((oy + (i >> 2)) * l0.width + ox + (i & 3)) * 4) as usize;
            se += sse(&l0.rgba8[at..at + 4], &decoded[at..at + 4]).0;
        }
        let e = kinds.entry(block_kind(format, block)).or_default();
        e.0 += 1;
        e.1 += se;
    }
}

/// Source | decoded | |difference|×8 of a 64² crop at (x, y), magnified 4×.
fn ab_sheet(path: &std::path::Path, src: &Image, got: &[u8], x0: u32, y0: u32) {
    const CROP: u32 = 64;
    const ZOOM: u32 = 4;
    let side = CROP * ZOOM;
    let mut sheet = image::RgbaImage::new(side * 3 + 16, side);
    for p in sheet.pixels_mut() {
        *p = image::Rgba([255, 255, 255, 255]);
    }
    for y in 0..side {
        for x in 0..side {
            let (sx, sy) = (x0 + x / ZOOM, y0 + y / ZOOM);
            let at = ((sy * src.width + sx) * 4) as usize;
            let s = &src.rgba8[at..at + 4];
            let g = &got[at..at + 4];
            let d = |c: usize| ((s[c] as i32 - g[c] as i32).unsigned_abs() * 8).min(255) as u8;
            sheet.put_pixel(x, y, image::Rgba([s[0], s[1], s[2], 255]));
            sheet.put_pixel(side + 8 + x, y, image::Rgba([g[0], g[1], g[2], 255]));
            sheet.put_pixel(2 * side + 16 + x, y, image::Rgba([d(0), d(1), d(2), 255]));
        }
    }
    sheet.save(path).expect("A/B sheet");
}

/// The A/B crops: the tile and the crop's top-left texel.
const AB: [(&str, u32, u32); 3] = [
    ("lyon-suburbs", 96, 96),
    ("paris-urban", 128, 0),
    ("mont-blanc", 64, 128),
];

struct Run {
    scores: Vec<TileScore>,
    kinds: Kinds,
    ms_per_tile: Option<(f64, f64)>,
    /// Largest difference, any texel, any channel, between the oracle's
    /// decode and the GPU's.
    oracle_vs_gpu: u8,
}

/// The settings measured: `level N` for [`Quality::level`], or the fields
/// written out (`tier,k,iters,luma`).
fn quality_of(setting: &str) -> Quality {
    if let Some(n) = setting.strip_prefix("level ") {
        return Quality::level(n.parse().expect("a level"));
    }
    let v: Vec<u32> = setting
        .split(',')
        .map(|x| x.trim().parse().expect("tier,k,iters,luma"))
        .collect();
    Quality {
        mode_tier: v[0],
        partition_candidates: v[1],
        refine_iterations: v[2],
        luma_weighting: v[3],
    }
}

fn encoder_for(gpu: &Gpu, format: BlockFormat, setting: &str) -> GpuEncoder {
    GpuEncoder::with_quality(&gpu.device, format, quality_of(setting))
}

fn run_gpu(
    gpu: &Gpu,
    format: BlockFormat,
    setting: &str,
    tiles: &[Tile],
    out: Option<&std::path::Path>,
) -> Run {
    let encoder = encoder_for(gpu, format, setting);
    let textures = encode(gpu, &encoder, tiles, true);
    let mut scores = Vec::new();
    let mut kinds = Kinds::new();
    let mut worst = 0u8;
    for (tile, texture) in tiles.iter().zip(&textures) {
        let sizes: Vec<(u32, u32)> = tile.chain.iter().map(|l| (l.width, l.height)).collect();
        let blocks = read_blocks(gpu, texture, &sizes);
        let decoded: Vec<Vec<u8>> = blocks
            .iter()
            .zip(&sizes)
            .map(|(b, &(w, h))| oracle_decode(format, b, w, h))
            .collect();
        let hardware = gpu_decode(gpu, texture, &sizes);
        for (a, b) in decoded.iter().zip(&hardware) {
            for (x, y) in a.iter().zip(b) {
                worst = worst.max(x.abs_diff(*y));
            }
        }
        add_kinds(&mut kinds, format, tile, &blocks[0], &decoded[0]);
        scores.push(score_tile(tile, &decoded));
        if let Some(dir) = out {
            for (name, x, y) in AB {
                if tile.name == name {
                    let path = dir.join(format!("{name}-{format:?}-{}.png", setting.replace([' ', ','], "-")).to_lowercase());
                    ab_sheet(&path, &tile.chain[0], &decoded[0], x, y);
                }
            }
        }
    }
    Run {
        scores,
        kinds,
        ms_per_tile: kernel_ms_per_tile(gpu, &encoder, tiles, 5),
        oracle_vs_gpu: worst,
    }
}

fn run_cpu(format: BlockFormat, level: texcomp::uastc::Level, tiles: &[Tile]) -> (Vec<TileScore>, f64) {
    let mut scores = Vec::new();
    let start = Instant::now();
    let mut decoded_all = Vec::new();
    for tile in tiles {
        let mut decoded = Vec::new();
        for l in &tile.chain {
            let n = (l.width.div_ceil(4) * l.height.div_ceil(4) * 16) as usize;
            let mut uastc = vec![0; n];
            texcomp::uastc::encode_image(&l.rgba8, l.width, l.height, level, &mut uastc)
                .expect("uastc");
            let mut blocks = vec![0; n];
            match format {
                BlockFormat::Bc7 => texcomp::uastc::transcode_to_bc7(&uastc, &mut blocks),
                _ => texcomp::uastc::transcode_to_astc_4x4(&uastc, &mut blocks),
            }
            .expect("transcode");
            decoded.push((blocks, l.width, l.height));
        }
        decoded_all.push(decoded);
    }
    let ms = start.elapsed().as_secs_f64() * 1e3 / tiles.len() as f64;
    for (tile, decoded) in tiles.iter().zip(decoded_all) {
        let d: Vec<Vec<u8>> = decoded
            .iter()
            .map(|(b, w, h)| oracle_decode(format, b, *w, *h))
            .collect();
        scores.push(score_tile(tile, &d));
    }
    (scores, ms)
}

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = v.collect();
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn min(v: impl Iterator<Item = f64>) -> f64 {
    v.fold(f64::INFINITY, f64::min)
}

fn summary_row(label: &str, s: &[TileScore], ms: Option<(f64, f64)>) -> String {
    format!(
        "| {label} | {:.2} | {:.2} | {:.4} | {:.2} | {:.2} | {:.4} | {} |",
        mean(s.iter().map(|t| t.psnr_rgb)),
        mean(s.iter().map(|t| t.psnr_y)),
        mean(s.iter().map(|t| t.ssim_y)),
        min(s.iter().map(|t| t.psnr_rgb)),
        mean(s.iter().map(|t| t.chain_psnr_rgb)),
        min(s.iter().map(|t| t.ssim_y)),
        ms.map_or("–".into(), |(gpu, wall)| format!("{gpu:.1} ({wall:.1})")),
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut out = None;
    let mut cpu = true;
    let mut settings: Vec<String> = (0..=Quality::MAX_LEVEL).map(|l| format!("level {l}")).collect();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = args.next().map(std::path::PathBuf::from),
            "--no-cpu" => cpu = false,
            // Settings to measure, `;`-separated: `level 2;2,6,4,50`.
            "--settings" => {
                settings = args
                    .next()
                    .expect("settings")
                    .split(';')
                    .map(str::to_owned)
                    .collect()
            }
            other => panic!("unknown argument {other}"),
        }
    }
    if let Some(dir) = &out {
        std::fs::create_dir_all(dir).expect("out dir");
    }
    let Some(gpu) = gpu() else { return };
    println!("Adapter: {:?}\n", gpu.adapter.get_info().name);
    let aerial = aerial_tiles();
    let nasa = nasa_tiles();
    let synthetic = if std::env::var_os("TEXCOMP_REPORT_SYNTHETIC").is_some() {
        synthetic_tiles()
    } else {
        Vec::new()
    };
    let header = "| Setting | PSNR RGB | PSNR Y | SSIM Y | worst PSNR RGB | chain PSNR RGB | worst SSIM Y | ms / tile: GPU (wall) |\n|---|---|---|---|---|---|---|---|";
    for format in formats(&gpu) {
        for (set_name, tiles) in [
            ("IGN aerial (6)", &aerial),
            ("NASA (20)", &nasa),
            ("synthetic (4)", &synthetic),
        ] {
            if tiles.is_empty() {
                continue;
            }
            println!("## {format:?}, {set_name}\n\n{header}");
            let mut per_tile = Vec::new();
            let mut kinds_all = Vec::new();
            for setting in &settings {
                let run = run_gpu(&gpu, format, setting, tiles, out.as_deref());
                println!("{}", summary_row(setting, &run.scores, run.ms_per_tile));
                assert!(
                    run.oracle_vs_gpu <= 1,
                    "{format:?} {setting}: oracle and GPU decodes differ by {}",
                    run.oracle_vs_gpu
                );
                per_tile.push((setting.clone(), run.scores));
                kinds_all.push((setting.clone(), run.kinds));
            }
            if cpu {
                use texcomp::uastc::Level;
                for (name, level) in [
                    ("CPU UASTC 0", Level::Fastest),
                    ("CPU UASTC 1", Level::Faster),
                    ("CPU UASTC 2", Level::Default),
                ] {
                    let (scores, ms) = run_cpu(format, level, tiles);
                    println!("{} (CPU ms / tile: {ms:.1})", summary_row(name, &scores, None));
                }
            }
            println!("\nPer tile, level 0 (PSNR RGB / PSNR Y / SSIM Y):\n");
            print!("| Tile |");
            for (s, _) in &per_tile {
                print!(" {s} |");
            }
            print!("\n|---|");
            for _ in &per_tile {
                print!("---|");
            }
            println!();
            for (i, tile) in tiles.iter().enumerate() {
                print!("| {} |", tile.name);
                for (_, scores) in &per_tile {
                    let t = &scores[i];
                    print!(" {:.2} / {:.2} / {:.4} |", t.psnr_rgb, t.psnr_y, t.ssim_y);
                }
                println!();
            }
            println!("\nBlock kinds, level 0 (share of blocks, PSNR RGB of those blocks):\n");
            for (setting, kinds) in &kinds_all {
                let total: u64 = kinds.values().map(|k| k.0).sum();
                let row: Vec<String> = kinds
                    .iter()
                    .map(|(k, (n, se))| {
                        format!(
                            "{k}: {:.1} % at {:.2} dB",
                            100.0 * *n as f64 / total as f64,
                            psnr(se / (*n as f64 * 48.0))
                        )
                    })
                    .collect();
                println!("- {setting}: {}", row.join("; "));
            }
            println!();
        }
    }
    if let Some(dir) = &out {
        println!("A/B sheets in {}", dir.display());
    }
}
