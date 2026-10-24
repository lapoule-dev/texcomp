// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! The CPU transcoders against vectors captured from upstream.
//!
//! `tests/fixtures/uastc` holds UASTC blocks the upstream encoder wrote,
//! their ASTC 4×4 transcodes and digests of their decoded texels, all
//! captured from the upstream tool at the commit `NOTICE` names (the
//! fixtures' README has the command lines):
//!
//! - the main set: nine 256² tiles (five cut from the imagery fixtures the
//!   way `gpu_quality` cuts them, four synthetic) at effort levels 0, 1, 2;
//! - the coverage set: eight 64² tiles at levels 3 and 4, made to reach the
//!   modes the main set never does (luminance–alpha, alpha, three-subset,
//!   mode 7's patterns).
//!
//! # What this covers
//!
//! - **ASTC 4×4:** every block's transcode, byte for byte (114,688 blocks).
//! - **Decode:** every block's sixteen texels, in both the linear and the
//!   sRGB decode, compared through a 32-bit FNV-1a digest of the 64 bytes
//!   (the fixtures keep digests, not texels, to stay small).
//! - All 19 UASTC modes, each reached at least once (modes 2 and 13 only a
//!   handful of times; the per-mode counts are printed).
//!
//! # What it does not cover
//!
//! The encoder (the inputs are kept in `tiles/` and `coverage-tiles/` for
//! it), BC7 and the other transcodes, and blocks no upstream encoder writes
//! (a hand-made block can hold values the encoder never emits; the unit
//! tests only check such blocks are read or refused without panicking).

use std::path::{Path, PathBuf};

use texcomp::uastc::{self, UastcBlock};

/// The main set: 256² tiles, effort levels 0–2 (the levels the encoder
/// port targets).
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

/// The coverage set: 64² tiles, effort levels 3–4, for the modes the main
/// set never reaches.
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

/// One captured tile at one level: where its files are, and its blocks.
struct Case {
    name: String,
    stem: PathBuf,
    blocks: usize,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for level in [0, 1, 2] {
        for tile in TILES {
            cases.push(Case {
                name: format!("{tile} level {level}"),
                stem: fixtures().join(format!("level{level}/{tile}")),
                blocks: 64 * 64,
            });
        }
    }
    for level in [3, 4] {
        for tile in COVERAGE_TILES {
            cases.push(Case {
                name: format!("{tile} level {level}"),
                stem: fixtures().join(format!("coverage/level{level}/{tile}")),
                blocks: 16 * 16,
            });
        }
    }
    cases
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/uastc")
}

fn read(case: &Case, ext: &str) -> Vec<u8> {
    let path = case.stem.with_extension(ext);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn blocks(bytes: &[u8]) -> impl Iterator<Item = UastcBlock> + '_ {
    let (blocks, rest) = bytes.as_chunks::<16>();
    assert!(rest.is_empty(), "whole blocks");
    blocks.iter().map(|&b| UastcBlock::from_bytes(b))
}

/// 32-bit FNV-1a, the digest the capture tool writes.
fn fnv1a(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c_9dc5u32, |h, &b| {
        (h ^ b as u32).wrapping_mul(0x0100_0193)
    })
}

/// **Every block transcodes to the same ASTC 4×4 bytes as upstream's.**
#[test]
fn astc_transcode_is_bit_exact() {
    let mut total = 0usize;
    let mut wrong = Vec::new();
    for case in cases() {
        let src = read(&case, "uastc");
        let want = read(&case, "astc");
        assert_eq!(src.len(), case.blocks * 16, "{}", case.name);
        assert_eq!(want.len(), src.len(), "{}", case.name);
        for (i, (block, want)) in blocks(&src).zip(want.as_chunks::<16>().0).enumerate() {
            total += 1;
            let got = block.to_astc_4x4();
            if got.as_ref() != Ok(want) {
                wrong.push(format!(
                    "{} block {i} (mode {:?}): got {got:02x?}, want {want:02x?}",
                    case.name,
                    block.mode()
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {total} blocks differ; first: {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(8)]
    );
}

/// **Every block decodes to the same texels as upstream's reference
/// decoder, linear and sRGB.**
#[test]
fn decode_is_bit_exact() {
    let mut total = 0usize;
    let mut wrong = Vec::new();
    for case in cases() {
        let src = read(&case, "uastc");
        let digests = read(&case, "decoded");
        assert_eq!(digests.len(), case.blocks * 8, "{}", case.name);
        for (i, (block, want)) in blocks(&src).zip(digests.as_chunks::<8>().0).enumerate() {
            total += 1;
            for (srgb, want) in [(false, &want[..4]), (true, &want[4..])] {
                let want = u32::from_le_bytes(want.try_into().expect("4 bytes"));
                let texels = block.decode_rgba8(srgb);
                let got = texels.as_ref().map(|t| fnv1a(t.as_flattened()));
                if got != Ok(want) {
                    wrong.push(format!(
                        "{} block {i} (mode {:?}, srgb {srgb}): {got:08x?} vs {want:08x}",
                        case.name,
                        block.mode()
                    ));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} decodes differ; first: {:#?}",
        wrong.len(),
        total * 2,
        &wrong[..wrong.len().min(8)]
    );
}

/// **The slice transcode agrees with the per-block one**, and the vectors
/// reach the modes they should.
#[test]
fn slice_transcode_and_mode_coverage() {
    let mut seen = [0usize; uastc::MODES];
    for case in cases() {
        let src = read(&case, "uastc");
        let want = read(&case, "astc");
        let mut got = vec![0u8; src.len()];
        uastc::transcode_to_astc_4x4(&src, &mut got).expect("valid blocks");
        assert!(got == want, "{}", case.name);
        for block in blocks(&src) {
            seen[block.mode().expect("valid mode") as usize] += 1;
        }
    }
    println!("blocks per UASTC mode: {seen:?}");
    for (mode, &n) in seen.iter().enumerate() {
        assert!(n > 0, "mode {mode} never reached: {seen:?}");
    }
}

/// Rewrites `tiles/*.png`, the capture's inputs: the 256² tiles cut from the
/// imagery fixtures the way `gpu_quality` cuts them, and its synthetic tiles.
/// Run by hand (`cargo test --test uastc_reference -- --ignored`) before a
/// new capture; the committed PNGs are what upstream encoded.
#[test]
#[ignore = "writes the capture's input tiles"]
fn cut_reference_tiles() {
    const TILE: u32 = 256;
    let imagery = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/imagery");
    let out = fixtures().join("tiles");
    std::fs::create_dir_all(&out).expect("tiles dir");
    let save = |name: &str, rgba8: Vec<u8>| {
        let img = image::RgbaImage::from_raw(TILE, TILE, rgba8).expect("256²");
        img.save(out.join(format!("{name}.png"))).expect("png");
    };
    for (name, q) in [
        ("alps", 0u32),
        ("nile", 1),
        ("coast", 2),
        ("modis", 3),
        ("paris-night", 0),
    ] {
        let file = if name == "paris-night" {
            "paris-night.png".to_string()
        } else {
            format!("{name}.jpeg")
        };
        let bytes = std::fs::read(imagery.join(&file)).expect("fixture");
        let full = image::load_from_memory(&bytes).expect("decodes").to_rgba8();
        assert_eq!(full.dimensions(), (512, 512), "{file}");
        let (ox, oy) = ((q % 2) * TILE, (q / 2) * TILE);
        let mut rgba8 = Vec::with_capacity((TILE * TILE * 4) as usize);
        for y in 0..TILE {
            let at = (((oy + y) * 512 + ox) * 4) as usize;
            rgba8.extend_from_slice(&full.as_raw()[at..at + (TILE * 4) as usize]);
        }
        save(&format!("{name}-{q}"), rgba8);
    }
    let hash = |x: u32, y: u32| {
        let mut h = x.wrapping_mul(0x9e37_79b1) ^ y.wrapping_mul(0x85eb_ca77);
        h ^= h >> 15;
        h = h.wrapping_mul(0xc2b2_ae3d);
        h ^ (h >> 13)
    };
    let synthetic = |f: &dyn Fn(u32, u32) -> [u8; 4]| {
        let mut rgba8 = Vec::with_capacity((TILE * TILE * 4) as usize);
        for y in 0..TILE {
            for x in 0..TILE {
                rgba8.extend_from_slice(&f(x, y));
            }
        }
        rgba8
    };
    save(
        "gradient",
        synthetic(&|x, y| [x as u8, y as u8, (255 - (x + y) / 2) as u8, 255]),
    );
    save(
        "stripes",
        synthetic(&|x, y| {
            if ((x + 2 * y) / 7) % 2 == 0 {
                [230, 40, 30, 255]
            } else {
                [20, 90, 210, 255]
            }
        }),
    );
    save(
        "alpha-edge",
        synthetic(&|x, y| {
            let a = (255i32 - (x as i32 - 116) * 255 / 24).clamp(0, 255) as u8;
            [(60 + y / 4) as u8, (120 + x / 8) as u8, 70, a]
        }),
    );
    save(
        "noise",
        synthetic(&|x, y| {
            let h = hash(x, y);
            [h as u8, (h >> 8) as u8, (h >> 16) as u8, 255]
        }),
    );

    // The coverage set: 64² crops and synthetic tiles the levels 0–2 encoder
    // never meets (luminance with alpha, alpha noise, grey), encoded at
    // levels 3 and 4, where every mode is tried.
    const SMALL: u32 = 64;
    let out = fixtures().join("coverage-tiles");
    std::fs::create_dir_all(&out).expect("coverage dir");
    let save_small = |name: &str, f: &dyn Fn(u32, u32) -> [u8; 4]| {
        let mut rgba8 = Vec::with_capacity((SMALL * SMALL * 4) as usize);
        for y in 0..SMALL {
            for x in 0..SMALL {
                rgba8.extend_from_slice(&f(x, y));
            }
        }
        let img = image::RgbaImage::from_raw(SMALL, SMALL, rgba8).expect("64²");
        img.save(out.join(format!("{name}.png"))).expect("png");
    };
    let tile = |name: &str| {
        image::open(fixtures().join(format!("tiles/{name}.png")))
            .expect("tile")
            .to_rgba8()
    };
    for (name, ox, oy) in [("modis-3", 96, 96), ("alps-0", 32, 160), ("stripes", 0, 0)] {
        let img = tile(name);
        save_small(&format!("{name}-crop"), &|x, y| {
            img.get_pixel(ox + x, oy + y).0
        });
    }
    let alps = tile("alps-0");
    save_small("grey", &|x, y| {
        let [r, g, b, _] = alps.get_pixel(64 + x, 64 + y).0;
        let l = ((r as u32 * 77 + g as u32 * 150 + b as u32 * 29) >> 8) as u8;
        [l, l, l, 255]
    });
    save_small("grey-alpha", &|x, y| {
        let [r, ..] = alps.get_pixel(128 + x, 32 + y).0;
        let a = (hash(x / 3, y / 5) as u8) | 0x0f;
        [r, r, r, a]
    });
    save_small("alpha-noise", &|x, y| {
        let h = hash(x + 1000, y);
        [h as u8, (h >> 8) as u8, (h >> 16) as u8, (h >> 24) as u8]
    });
    let modis = tile("modis-3");
    save_small("alpha-imagery", &|x, y| {
        let [r, g, b, _] = modis.get_pixel(x, 128 + y).0;
        [r, g, b, ((x * 4 + y * 3) % 256) as u8]
    });

    // Blocks only the rarer modes represent exactly, so the encoder picks
    // them: three flat colours on a three-subset pattern from levels the
    // three-subset mode stores exactly (mode 3); two two-colour subsets on
    // the patterns only mode 7 has (mode 7); RGBA on four exact
    // interpolation steps with full-precision endpoints (mode 14).
    // ASTC partition seeds from the upstream UASTC tables
    // (`g_astc_bc7_common_partitions3`, `g_bc7_3_astc2_common_partitions`;
    // Basis Universal v2.50, Apache-2.0, (C) Binomial LLC).
    const SEEDS3: [u32; 11] = [260, 74, 32, 156, 183, 15, 745, 0, 335, 902, 254];
    const SEEDS2_MODE7: [u32; 19] = [
        36, 48, 61, 137, 161, 183, 226, 281, 302, 307, 479, 495, 593, 594, 605, 799, 812, 988, 993,
    ];
    const LEVELS7: [u8; 12] = [0, 23, 46, 69, 92, 116, 139, 163, 186, 209, 232, 255];
    const LEVELS12: [u8; 40] = [
        0, 6, 13, 19, 26, 32, 39, 45, 52, 58, 65, 71, 78, 84, 91, 97, 104, 110, 117, 123, 132, 138,
        145, 151, 158, 164, 171, 177, 184, 190, 197, 203, 210, 216, 223, 229, 236, 242, 249, 255,
    ];
    let pick = |h: u32, levels: &[u8]| -> [u8; 4] {
        let l = |s: u32| levels[(h >> s) as usize % levels.len()];
        [l(0), l(6), l(12), 255]
    };
    save_small("partitions", &|x, y| {
        let (bx, by, t) = (x / 4, y / 4, (y % 4) * 4 + x % 4);
        let h = hash(bx + 500, by);
        match (bx + by) % 3 {
            0 => {
                let seed = SEEDS3[h as usize % 11];
                let s = astc_partition(seed, x % 4, y % 4, 3);
                pick(hash(bx * 3 + s, by + 900), &LEVELS7)
            }
            1 => {
                let seed = SEEDS2_MODE7[h as usize % 19];
                let s = astc_partition(seed, x % 4, y % 4, 2);
                let end = hash(t + 77, h) & 1;
                pick(hash(bx * 4 + s * 2 + end, by + 1300), &LEVELS12)
            }
            _ => {
                let lo = hash(bx, by + 2000);
                let hi = hash(bx, by + 3000);
                let w = [0u32, 21, 43, 64][(hash(t, h) & 3) as usize];
                let c = |s: u32| {
                    let (l, h) = ((lo >> s) & 255, (hi >> s) & 255);
                    let (l, h) = ((l << 8) | l, (h << 8) | h);
                    (((l * (64 - w) + h * w + 32) >> 6) >> 8) as u8
                };
                [c(0), c(8), c(16), c(24)]
            }
        }
    });
}

/// ASTC's partition function for a 4×4 block (Khronos Data Format 1.3,
/// §23.21), written from the specification.
fn astc_partition(seed: u32, x: u32, y: u32, count: u32) -> u32 {
    let (x, y) = (x << 1, y << 1);
    let seed = seed + (count - 1) * 1024;
    let mut p = seed;
    p ^= p >> 15;
    p = p.wrapping_sub(p << 17);
    p = p.wrapping_add(p << 7);
    p = p.wrapping_add(p << 4);
    p ^= p >> 5;
    p = p.wrapping_add(p << 16);
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    let r = p;
    let sq = |s: u32| ((r >> s) & 0xf) * ((r >> s) & 0xf);
    let (sh1, sh2) = if seed & 1 != 0 {
        (
            if seed & 2 != 0 { 4 } else { 5 },
            if count == 3 { 6 } else { 5 },
        )
    } else {
        (
            if count == 3 { 6 } else { 5 },
            if seed & 2 != 0 { 4 } else { 5 },
        )
    };
    let a = ((sq(0) >> sh1) * x + (sq(4) >> sh2) * y + (r >> 14)) & 0x3f;
    let b = ((sq(8) >> sh1) * x + (sq(12) >> sh2) * y + (r >> 10)) & 0x3f;
    let c = if count < 3 {
        0
    } else {
        ((sq(16) >> sh1) * x + (sq(20) >> sh2) * y + (r >> 6)) & 0x3f
    };
    if a >= b && a >= c {
        0
    } else if b >= c {
        1
    } else {
        2
    }
}
