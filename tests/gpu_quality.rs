// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! GPU in the loop: encode real imagery with the compute encoder, decode it
//! back **with the GPU's own block decoder**, and hold the result to a quality
//! gate.
//!
//! # What this covers
//!
//! - Every block format the test machine's adapter samples and the encoder
//!   writes (on Apple silicon: BC7 and ASTC 4×4; elsewhere whatever the
//!   adapter offers).
//! - Twenty 256² tiles cut from five NASA tiles (`fixtures/imagery`) and four
//!   synthetic ones (a gradient, hard stripes, an alpha edge, noise), each with
//!   its full mip chain down to 1×1.
//! - Per tile and per level: PSNR over RGB, ΔE2000 (mean, p99, max), PSNR over
//!   alpha for the tile that has alpha, and a check that no texel the source
//!   draws decodes to black or transparent.
//!
//! # What it does not cover
//!
//! The blocks are read back through `textureLoad` on the linear (`Unorm`)
//! variant of each format, which decodes exactly. The sRGB variant holds the
//! same blocks; `the_srgb_variant_decodes_the_same_blocks` checks it decodes
//! close to the linear one, not the sampler's filtering of it. Formats the
//! test machine lacks (BC7 on an iPhone, ASTC on a desktop NVIDIA) are not
//! exercised here: which device gets which format is the application's
//! choice (tuile's `select`), and its capability matrix covers that choice, not
//! these pixels.
//!
//! A machine without a usable adapter skips; `TEXCOMP_REQUIRE_GPU=1` turns the
//! skip into a failure.

use std::fmt::Write as _;
use std::time::Instant;

use palette::color_difference::Ciede2000;
use palette::{IntoColor, Lab, Srgb};
use texcomp::gpu::{feature, runs_on, texture_format, GpuEncoder, Level, GPU_FORMATS};
use texcomp::BlockFormat;

/// One RGBA8 level.
struct DecodedTexture {
    width: u32,
    height: u32,
    rgba8: Vec<u8>,
}

/// The adapter samples `format`, filtered, behind its feature.
fn samples(adapter: &wgpu::Adapter, format: BlockFormat) -> bool {
    adapter.features().contains(feature(format))
        && adapter
            .get_texture_format_features(texture_format(format, true))
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
}

fn runs_kernels(adapter: &wgpu::Adapter) -> bool {
    runs_on(
        adapter.get_downlevel_capabilities().flags,
        &adapter.limits(),
    )
}

/// The gate, per format. Measured on Apple M2 (Metal) on 21 Oct. 2026 and
/// set about 1 dB / 20 % beyond the worst tile, so it catches a regression
/// without being a wish (the PR has the table).
///
/// - `real` holds the twenty NASA tiles at every level of 8×8 texels or more.
///   The levels below that are one to four blocks, drawn only where the whole
///   tile covers a handful of pixels, and their statistics over at most 64
///   texels say little: they are held to the hole check only.
/// - `synthetic` holds the gradient, stripes and alpha-edge tiles at level 0.
///   Their small levels are steep 2-D colour ramps no imagery has, which no
///   one-line-per-block format fits; they are reported, not gated.
/// - The noise tile is incompressible by construction (white noise has no
///   structure for 8 bits per texel to keep): it is held to the hole check
///   only.
struct Gate {
    real: Bound,
    synthetic: Bound,
    /// PSNR over alpha, on the alpha-edge tile at level 0.
    psnr_alpha: f64,
}

struct Bound {
    psnr: f64,
    de_mean: f64,
    de_p99: f64,
}

fn gate(format: BlockFormat) -> Gate {
    match format {
        // Measured worst: real 40.30 dB (paris-night#1, level 5), ΔE mean
        // 2.088 (modis#3, level 5), p99 5.353 (nile#0, level 5); synthetic
        // 48.13 dB, ΔE mean 0.363, p99 1.181; alpha 62.44 dB.
        BlockFormat::Bc7 => Gate {
            real: Bound {
                psnr: 39.0,
                de_mean: 2.5,
                de_p99: 6.5,
            },
            synthetic: Bound {
                psnr: 45.0,
                de_mean: 0.5,
                de_p99: 1.5,
            },
            psnr_alpha: 55.0,
        },
        // Measured worst: real 36.41 dB and p99 9.397 (paris-night#1, level
        // 0: bright points on black, which one partition per block fits
        // worst), ΔE mean 2.429 (modis#3, level 5); synthetic 51.14 dB, ΔE
        // mean 0.258, p99 1.135; alpha 56.65 dB.
        BlockFormat::Astc4x4 => Gate {
            real: Bound {
                psnr: 35.0,
                de_mean: 3.0,
                de_p99: 11.5,
            },
            synthetic: Bound {
                psnr: 48.0,
                de_mean: 0.4,
                de_p99: 1.5,
            },
            psnr_alpha: 50.0,
        },
        _ => unreachable!("a format the encoder doesn't write"),
    }
}

const TILE: u32 = 256;

struct Gpu {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Option<Gpu> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                // The adapter's own limits, not bucketed ones: the gate
                // requests them all.
                apply_limit_buckets: false,
            })
            .await
        {
            Ok(a) => a,
            Err(e) => {
                assert!(
                    std::env::var_os("TEXCOMP_REQUIRE_GPU").is_none(),
                    "TEXCOMP_REQUIRE_GPU is set and there is no adapter: {e}"
                );
                eprintln!("skipping: no GPU adapter ({e})");
                return None;
            }
        };
        let features = GPU_FORMATS
            .into_iter()
            .filter(|f| samples(&adapter, *f))
            .fold(wgpu::Features::empty(), |acc, f| acc | feature(f));
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("texcomp quality"),
                required_features: features,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("a device with the adapter's own compression features");
        Some(Gpu {
            adapter,
            device,
            queue,
        })
    })
}

/// The formats this adapter can run the encoder for.
fn formats(gpu: &Gpu) -> Vec<BlockFormat> {
    if !runs_kernels(&gpu.adapter) {
        return Vec::new();
    }
    GPU_FORMATS
        .into_iter()
        .filter(|f| samples(&gpu.adapter, *f))
        .collect()
}

// --- tiles ------------------------------------------------------------------

struct Tile {
    name: String,
    chain: Vec<DecodedTexture>,
}

/// The level and its mips down to 1×1, each halving filtered with a triangle
/// (tent) filter, as tuile's decode does.
fn chain(base: DecodedTexture) -> Vec<DecodedTexture> {
    use image::{imageops, ImageBuffer, Rgba};
    let mut levels = vec![base];
    loop {
        let prev = levels.last().expect("level 0");
        if prev.width <= 1 && prev.height <= 1 {
            return levels;
        }
        let view = ImageBuffer::<Rgba<u8>, _>::from_raw(prev.width, prev.height, &prev.rgba8[..])
            .expect("tight RGBA8");
        let (width, height) = ((prev.width / 2).max(1), (prev.height / 2).max(1));
        let next = imageops::resize(&view, width, height, imageops::FilterType::Triangle);
        levels.push(DecodedTexture {
            width,
            height,
            rgba8: next.into_raw(),
        });
    }
}

fn fixture_tiles() -> Vec<Tile> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/imagery");
    let mut tiles = Vec::new();
    for name in [
        "alps.jpeg",
        "nile.jpeg",
        "coast.jpeg",
        "modis.jpeg",
        "paris-night.png",
    ] {
        let bytes = std::fs::read(dir.join(name)).expect("fixture");
        let img = image::load_from_memory(&bytes).expect("decodes").to_rgba8();
        let full = DecodedTexture {
            width: img.width(),
            height: img.height(),
            rgba8: img.into_raw(),
        };
        assert_eq!((full.width, full.height), (512, 512), "{name}");
        for q in 0..4u32 {
            let (ox, oy) = ((q % 2) * TILE, (q / 2) * TILE);
            let mut rgba8 = Vec::with_capacity((TILE * TILE * 4) as usize);
            for y in 0..TILE {
                let at = (((oy + y) * full.width + ox) * 4) as usize;
                rgba8.extend_from_slice(&full.rgba8[at..at + (TILE * 4) as usize]);
            }
            tiles.push(Tile {
                name: format!("{}#{q}", name.split('.').next().unwrap_or(name)),
                chain: chain(DecodedTexture {
                    width: TILE,
                    height: TILE,
                    rgba8,
                }),
            });
        }
    }
    tiles
}

fn synthetic(name: &str, f: impl Fn(u32, u32) -> [u8; 4]) -> Tile {
    let mut rgba8 = Vec::with_capacity((TILE * TILE * 4) as usize);
    for y in 0..TILE {
        for x in 0..TILE {
            rgba8.extend_from_slice(&f(x, y));
        }
    }
    Tile {
        name: name.into(),
        chain: chain(DecodedTexture {
            width: TILE,
            height: TILE,
            rgba8,
        }),
    }
}

fn synthetic_tiles() -> Vec<Tile> {
    // A cheap deterministic hash for the noise tile.
    let hash = |x: u32, y: u32| {
        let mut h = x.wrapping_mul(0x9e37_79b1) ^ y.wrapping_mul(0x85eb_ca77);
        h ^= h >> 15;
        h = h.wrapping_mul(0xc2b2_ae3d);
        h ^ (h >> 13)
    };
    vec![
        synthetic("gradient", |x, y| {
            [x as u8, y as u8, (255 - (x + y) / 2) as u8, 255]
        }),
        synthetic("stripes", |x, y| {
            // Hard diagonal edges of saturated colour: partition territory.
            if ((x + 2 * y) / 7) % 2 == 0 {
                [230, 40, 30, 255]
            } else {
                [20, 90, 210, 255]
            }
        }),
        synthetic("alpha-edge", |x, y| {
            // A drape's edge: imagery fading out over 24 texels, then gone.
            let a = (255i32 - (x as i32 - 116) * 255 / 24).clamp(0, 255) as u8;
            [(60 + y / 4) as u8, (120 + x / 8) as u8, 70, a]
        }),
        synthetic("noise", |x, y| {
            let h = hash(x, y);
            [h as u8, (h >> 8) as u8, (h >> 16) as u8, 255]
        }),
    ]
}

// --- decode back ------------------------------------------------------------

const DECODE: &str = r#"
struct P { level: u32, width: u32, height: u32, pad: u32 }
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.width || id.y >= p.height) { return; }
    out[id.y * p.width + id.x] = pack4x8unorm(textureLoad(tex, vec2i(id.xy), i32(p.level)));
}
"#;

/// Every level of `texture`, decoded by the GPU, as tightly packed RGBA8.
fn decode(gpu: &Gpu, texture: &wgpu::Texture, sizes: &[(u32, u32)]) -> Vec<Vec<u8>> {
    use wgpu::util::DeviceExt;
    let device = &gpu.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("decode"),
        source: wgpu::ShaderSource::Wgsl(DECODE.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("decode"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut cmd = device.create_command_encoder(&Default::default());
    let mut readbacks = Vec::new();
    for (level, &(w, h)) in sizes.iter().enumerate() {
        let bytes = (w * h * 4) as u64;
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[level as u32, w, h, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = cmd.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        cmd.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
        readbacks.push(read);
    }
    gpu.queue.submit([cmd.finish()]);
    readbacks
        .iter()
        .map(|read| {
            let slice = read.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
            let data = slice.get_mapped_range().expect("mapped").to_vec();
            read.unmap();
            data
        })
        .collect()
}

fn encode(gpu: &Gpu, encoder: &GpuEncoder, tiles: &[Tile], srgb: bool) -> Vec<wgpu::Texture> {
    let levels: Vec<Vec<Level<'_>>> = tiles
        .iter()
        .map(|t| {
            t.chain
                .iter()
                .map(|l| Level {
                    width: l.width,
                    height: l.height,
                    rgba8: &l.rgba8,
                })
                .collect()
        })
        .collect();
    let refs: Vec<&[Level<'_>]> = levels.iter().map(Vec::as_slice).collect();
    let mut cmd = gpu.device.create_command_encoder(&Default::default());
    let out = encoder.encode(&gpu.device, &gpu.queue, &mut cmd, &refs, srgb);
    gpu.queue.submit([cmd.finish()]);
    out.into_iter()
        .map(|r| r.expect("a valid 256² chain encodes"))
        .collect()
}

// --- metrics ----------------------------------------------------------------

struct Score {
    psnr: f64,
    psnr_alpha: f64,
    de_mean: f64,
    de_p99: f64,
    de_max: f64,
    /// Texels the source draws that came back black or transparent.
    holes: usize,
}

fn psnr(mse: f64) -> f64 {
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

fn lab(p: &[u8]) -> Lab {
    Srgb::new(p[0], p[1], p[2])
        .into_format::<f32>()
        .into_linear()
        .into_color()
}

fn score(src: &[u8], got: &[u8]) -> Score {
    let n = src.len() / 4;
    let (mut se, mut se_a) = (0.0f64, 0.0f64);
    let mut de: Vec<f32> = Vec::with_capacity(n);
    let mut holes = 0;
    for (s, g) in src.as_chunks::<4>().0.iter().zip(got.as_chunks::<4>().0) {
        for c in 0..3 {
            let d = s[c] as f64 - g[c] as f64;
            se += d * d;
        }
        let d = s[3] as f64 - g[3] as f64;
        se_a += d * d;
        // ΔE only where the texel is drawn: a transparent texel has no colour.
        if s[3] > 0 {
            de.push(lab(s).difference(lab(g)));
        }
        let drawn = s[3] == 255 && s[..3].iter().any(|&v| v > 16);
        let black = g[..3].iter().all(|&v| v == 0) || g[3] < 128;
        if drawn && black {
            holes += 1;
        }
    }
    de.sort_by(f32::total_cmp);
    let mean = de.iter().map(|&v| v as f64).sum::<f64>() / de.len().max(1) as f64;
    let at = |q: f64| {
        de.get(((de.len() as f64 - 1.0) * q).round() as usize)
            .map_or(0.0, |&v| v as f64)
    };
    Score {
        psnr: psnr(se / (3 * n) as f64),
        psnr_alpha: psnr(se_a / n as f64),
        de_mean: mean,
        de_p99: at(0.99),
        de_max: at(1.0),
        holes,
    }
}

/// Source | decoded | |difference|×8, side by side, for a human to open.
fn save_sheet(name: &str, src: &[u8], got: &[u8], w: u32, h: u32) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test-renders");
    std::fs::create_dir_all(&dir).expect("render dir");
    let mut sheet = image::RgbaImage::new(w * 3, h);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let s = &src[i..i + 4];
            let g = &got[i..i + 4];
            sheet.put_pixel(x, y, image::Rgba([s[0], s[1], s[2], 255]));
            sheet.put_pixel(w + x, y, image::Rgba([g[0], g[1], g[2], 255]));
            let d = |c: usize| ((s[c] as i32 - g[c] as i32).unsigned_abs() * 8).min(255) as u8;
            sheet.put_pixel(2 * w + x, y, image::Rgba([d(0), d(1), d(2), 255]));
        }
    }
    sheet.save(dir.join(format!("{name}.png"))).expect("sheet");
}

// --- tests ------------------------------------------------------------------

/// **Every tile, every level, every format the adapter has: within the
/// gate, and never a hole.**
#[test]
fn encoded_imagery_decodes_within_the_gate() {
    let Some(gpu) = gpu() else { return };
    let formats = formats(&gpu);
    if formats.is_empty() {
        eprintln!("skipping: the adapter runs no block format the encoder writes");
        return;
    }
    let mut tiles = fixture_tiles();
    tiles.extend(synthetic_tiles());
    let mut report = String::new();
    let mut failures = Vec::new();
    for format in formats {
        let g = gate(format);
        let encoder = GpuEncoder::new(&gpu.device, format);
        let textures = encode(&gpu, &encoder, &tiles, false);
        let _ = writeln!(
            report,
            "\n{format:?}\n{:<16} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}",
            "tile", "level", "PSNR", "PSNR-A", "ΔE mean", "ΔE p99", "ΔE max", "holes"
        );
        let (mut worst_psnr, mut worst_mean, mut worst_p99) = (f64::INFINITY, 0.0f64, 0.0f64);
        for (tile, texture) in tiles.iter().zip(&textures) {
            let sizes: Vec<_> = tile.chain.iter().map(|l| (l.width, l.height)).collect();
            let decoded = decode(&gpu, texture, &sizes);
            for (level, (src, got)) in tile.chain.iter().zip(&decoded).enumerate() {
                let s = score(&src.rgba8, got);
                let _ = writeln!(
                    report,
                    "{:<16} {:>5} {:>8.2} {:>8.2} {:>8.3} {:>8.3} {:>8.3} {:>6}",
                    tile.name, level, s.psnr, s.psnr_alpha, s.de_mean, s.de_p99, s.de_max, s.holes
                );
                if level == 0 {
                    save_sheet(
                        &format!("{format:?}-{}", tile.name),
                        &src.rgba8,
                        got,
                        src.width,
                        src.height,
                    );
                }
                let at = format!("{format:?} {} level {level}", tile.name);
                if s.holes > 0 {
                    failures.push(format!("{at}: {} texels decode to black", s.holes));
                }
                let real = tile.name.contains('#');
                let bound = if real && src.width >= 8 && src.height >= 8 {
                    Some(&g.real)
                } else if !real && tile.name != "noise" && level == 0 {
                    Some(&g.synthetic)
                } else {
                    None
                };
                if let Some(b) = bound {
                    worst_psnr = worst_psnr.min(s.psnr);
                    worst_mean = worst_mean.max(s.de_mean);
                    worst_p99 = worst_p99.max(s.de_p99);
                    if s.psnr < b.psnr {
                        failures.push(format!("{at}: PSNR {:.2} < {}", s.psnr, b.psnr));
                    }
                    if s.de_mean > b.de_mean {
                        failures.push(format!("{at}: ΔE mean {:.3} > {}", s.de_mean, b.de_mean));
                    }
                    if s.de_p99 > b.de_p99 {
                        failures.push(format!("{at}: ΔE p99 {:.3} > {}", s.de_p99, b.de_p99));
                    }
                }
                if tile.name == "alpha-edge" && level == 0 && s.psnr_alpha < g.psnr_alpha {
                    failures.push(format!(
                        "{at}: alpha PSNR {:.2} < {}",
                        s.psnr_alpha, g.psnr_alpha
                    ));
                }
            }
        }
        let _ = writeln!(
            report,
            "{format:?} worst gated: PSNR {worst_psnr:.2} dB, ΔE mean {worst_mean:.3}, ΔE p99 {worst_p99:.3}"
        );
    }
    println!("{report}");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test-renders");
    std::fs::create_dir_all(&dir).expect("render dir");
    std::fs::write(dir.join("quality.txt"), &report).expect("report");
    assert!(failures.is_empty(), "{}\n{report}", failures.join("\n"));
}

/// **The sRGB variant holds the same blocks**: the format the viewer samples
/// imagery as decodes, through the hardware's sRGB path, to nearly what the
/// linear variant does (within the sRGB round trip of an 8-bit value).
#[test]
fn the_srgb_variant_decodes_the_same_blocks() {
    let Some(gpu) = gpu() else { return };
    let tiles: Vec<Tile> = fixture_tiles().into_iter().take(2).collect();
    for format in formats(&gpu) {
        let encoder = GpuEncoder::new(&gpu.device, format);
        let linear = encode(&gpu, &encoder, &tiles, false);
        let srgb = encode(&gpu, &encoder, &tiles, true);
        for ((tile, a), b) in tiles.iter().zip(&linear).zip(&srgb) {
            assert_eq!(b.format(), texture_format(format, true));
            let sizes = [(TILE, TILE)];
            let a = &decode(&gpu, a, &sizes)[0];
            let b = &decode(&gpu, b, &sizes)[0];
            // `b` is linear light; back to sRGB-encoded bytes.
            let to_srgb = |v: u8| {
                let l = v as f64 / 255.0;
                let s = if l <= 0.003_130_8 {
                    l * 12.92
                } else {
                    1.055 * l.powf(1.0 / 2.4) - 0.055
                };
                (s * 255.0).round() as i32
            };
            let worst = a
                .as_chunks::<4>()
                .0
                .iter()
                .zip(b.as_chunks::<4>().0)
                .flat_map(|(p, q)| (0..3).map(move |c| (p[c] as i32 - to_srgb(q[c])).abs()))
                .max()
                .unwrap_or(0);
            // Linear light in 8 bits loses the darkest sRGB steps: a round
            // trip through it is good to a few codes at the bottom of the range.
            assert!(
                worst <= 13,
                "{format:?} {}: sRGB decode off by {worst}",
                tile.name
            );
        }
    }
}

/// **Throughput**, reported, not gated: the twenty fixture tiles with their
/// chains, encoded in one call, timed to the GPU's completion.
#[test]
fn encode_time_is_reported() {
    let Some(gpu) = gpu() else { return };
    let tiles = fixture_tiles();
    for format in formats(&gpu) {
        let encoder = GpuEncoder::new(&gpu.device, format);
        // Warm the pipeline up.
        let _ = encode(&gpu, &encoder, &tiles[..1], true);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let runs = 5;
        let start = Instant::now();
        for _ in 0..runs {
            let _ = encode(&gpu, &encoder, &tiles, true);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
        }
        let per_tile = start.elapsed().as_secs_f64() * 1e3 / (runs * tiles.len()) as f64;
        println!(
            "{format:?}: {per_tile:.3} ms per 256² tile with its mips (wall clock, upload + encode + copy), {}",
            gpu.adapter.get_info().name
        );
    }
}

/// **Every live adapter that runs the kernels can be given the formats it
/// samples**: for every adapter this machine enumerates, on every backend, a
/// device requested with those formats' features succeeds.
#[test]
fn every_live_adapter_can_request_its_formats() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    for adapter in adapters {
        let info = adapter.get_info();
        let formats: Vec<_> = GPU_FORMATS
            .into_iter()
            .filter(|f| samples(&adapter, *f))
            .collect();
        println!(
            "{:?} {}: runs the kernels: {}, samples {formats:?}",
            info.backend,
            info.name,
            runs_kernels(&adapter)
        );
        let features = formats
            .iter()
            .fold(wgpu::Features::empty(), |acc, f| acc | feature(*f));
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("formats"),
            required_features: features,
            ..Default::default()
        }))
        .unwrap_or_else(|e| panic!("{:?} {}: {e}", info.backend, info.name));
    }
}
