// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! What the quality tests and `examples/quality-report.rs` share: a device,
//! the imagery tiles, an encode whose blocks are read back, two independent
//! decodes of those blocks (the GPU's own decoder and a software one, the
//! oracle), and the scores: PSNR over RGB and over luma, SSIM over luma.

#![allow(dead_code)]

use texcomp::gpu::{feature, runs_on, texture_format, GpuEncoder, Level, GPU_FORMATS};
use texcomp::BlockFormat;

/// One RGBA8 image, rows top to bottom.
#[derive(Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// A tile and its mips down to 1×1.
pub struct Tile {
    pub name: String,
    pub chain: Vec<Image>,
}

pub const TILE: u32 = 256;

pub struct Gpu {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// The adapter samples `format`, filtered, behind its feature.
pub fn samples(adapter: &wgpu::Adapter, format: BlockFormat) -> bool {
    adapter.features().contains(feature(format))
        && adapter
            .get_texture_format_features(texture_format(format, true))
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
}

/// A device with every block format the adapter samples, and timestamps
/// where it has them. `None` without an adapter, unless
/// `TEXCOMP_REQUIRE_GPU` is set, which makes that a failure.
pub fn gpu() -> Option<Gpu> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
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
        let mut features = GPU_FORMATS
            .into_iter()
            .filter(|f| samples(&adapter, *f))
            .fold(wgpu::Features::empty(), |acc, f| acc | feature(f));
        features |= adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
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

/// The formats this adapter samples and runs the encoder for.
pub fn formats(gpu: &Gpu) -> Vec<BlockFormat> {
    if !runs_on(
        gpu.adapter.get_downlevel_capabilities().flags,
        &gpu.adapter.limits(),
    ) {
        return Vec::new();
    }
    GPU_FORMATS
        .into_iter()
        .filter(|f| samples(&gpu.adapter, *f))
        .collect()
}

// --- tiles ------------------------------------------------------------------

/// The level and its mips down to 1×1, each halving filtered with a triangle
/// (tent) filter, as tuile's decode does.
pub fn chain(base: Image) -> Vec<Image> {
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
        levels.push(Image {
            width,
            height,
            rgba8: next.into_raw(),
        });
    }
}

fn load(path: &std::path::Path) -> Image {
    let bytes = std::fs::read(path).expect("fixture");
    let img = image::load_from_memory(&bytes)
        .expect("decodes")
        .to_rgba8();
    Image {
        width: img.width(),
        height: img.height(),
        rgba8: img.into_raw(),
    }
}

fn fixtures() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The six IGN aerial orthophoto tiles (`fixtures/aerial`), 256² each.
pub fn aerial_tiles() -> Vec<Tile> {
    [
        "paris-urban",
        "beauce-fields",
        "vosges-forest",
        "mont-blanc",
        "brittany-coast",
        "lyon-suburbs",
    ]
    .into_iter()
    .map(|name| {
        let img = load(&fixtures().join(format!("aerial/{name}.jpeg")));
        assert_eq!((img.width, img.height), (TILE, TILE), "{name}");
        Tile {
            name: name.to_owned(),
            chain: chain(img),
        }
    })
    .collect()
}

/// The twenty 256² tiles cut from the five NASA tiles (`fixtures/imagery`).
pub fn nasa_tiles() -> Vec<Tile> {
    let mut tiles = Vec::new();
    for name in [
        "alps.jpeg",
        "nile.jpeg",
        "coast.jpeg",
        "modis.jpeg",
        "paris-night.png",
    ] {
        let full = load(&fixtures().join("imagery").join(name));
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
                chain: chain(Image {
                    width: TILE,
                    height: TILE,
                    rgba8,
                }),
            });
        }
    }
    tiles
}

pub fn synthetic(name: &str, f: impl Fn(u32, u32) -> [u8; 4]) -> Tile {
    let mut rgba8 = Vec::with_capacity((TILE * TILE * 4) as usize);
    for y in 0..TILE {
        for x in 0..TILE {
            rgba8.extend_from_slice(&f(x, y));
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

/// A cheap deterministic hash.
pub fn hash(x: u32, y: u32) -> u32 {
    let mut h = x.wrapping_mul(0x9e37_79b1) ^ y.wrapping_mul(0x85eb_ca77);
    h ^= h >> 15;
    h = h.wrapping_mul(0xc2b2_ae3d);
    h ^ (h >> 13)
}

/// A gradient, hard stripes, an alpha edge and noise.
pub fn synthetic_tiles() -> Vec<Tile> {
    vec![
        synthetic("gradient", |x, y| {
            [x as u8, y as u8, (255 - (x + y) / 2) as u8, 255]
        }),
        synthetic("stripes", |x, y| {
            if ((x + 2 * y) / 7) % 2 == 0 {
                [230, 40, 30, 255]
            } else {
                [20, 90, 210, 255]
            }
        }),
        synthetic("alpha-edge", |x, y| {
            let a = (255i32 - (x as i32 - 116) * 255 / 24).clamp(0, 255) as u8;
            [(60 + y / 4) as u8, (120 + x / 8) as u8, 70, a]
        }),
        synthetic("noise", |x, y| {
            let h = hash(x, y);
            [h as u8, (h >> 8) as u8, (h >> 16) as u8, 255]
        }),
    ]
}

// --- encode and read back ---------------------------------------------------

pub fn levels_of(tile: &Tile) -> Vec<Level<'_>> {
    tile.chain
        .iter()
        .map(|l| Level {
            width: l.width,
            height: l.height,
            rgba8: &l.rgba8,
        })
        .collect()
}

/// Every tile encoded in one call, through `encode_into`, into textures
/// that can also be copied from (the blocks are read back); the textures, in
/// order.
pub fn encode(gpu: &Gpu, encoder: &GpuEncoder, tiles: &[Tile], srgb: bool) -> Vec<wgpu::Texture> {
    use texcomp::gpu::{Destination, SourceLevel, TileInto};
    let textures: Vec<wgpu::Texture> = tiles
        .iter()
        .map(|t| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("texcomp quality"),
                size: wgpu::Extent3d {
                    width: t.chain[0].width,
                    height: t.chain[0].height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: t.chain.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: texture_format(encoder.format(), srgb),
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[match texture_format(encoder.format(), false) {
                    f if srgb => f,
                    _ => texture_format(encoder.format(), true),
                }],
            })
        })
        .collect();
    let levels: Vec<Vec<SourceLevel<'_>>> = tiles
        .iter()
        .map(|t| levels_of(t).into_iter().map(SourceLevel::from).collect())
        .collect();
    let destinations: Vec<Vec<Destination<'_>>> = tiles
        .iter()
        .zip(&textures)
        .map(|(t, texture)| {
            (0..t.chain.len() as u32)
                .map(|mip_level| Destination {
                    texture,
                    mip_level,
                    origin: wgpu::Origin3d::ZERO,
                })
                .collect()
        })
        .collect();
    let into: Vec<TileInto<'_>> = levels
        .iter()
        .zip(&destinations)
        .map(|(l, d)| TileInto {
            levels: l,
            destinations: d,
        })
        .collect();
    let mut cmd = gpu.device.create_command_encoder(&Default::default());
    let out = encoder.encode_into(&gpu.device, &gpu.queue, &mut cmd, &into);
    gpu.queue.submit([cmd.finish()]);
    for r in out {
        r.expect("a valid chain encodes");
    }
    textures
}

/// GPU time of the encode pass for `tiles`, from timestamps, and wall
/// time of the call to the GPU's completion (upload included), each in
/// milliseconds per tile: the medians of `runs` calls after one warm-up.
/// `None` without timestamps.
pub fn kernel_ms_per_tile(
    gpu: &Gpu,
    encoder: &GpuEncoder,
    tiles: &[Tile],
    runs: usize,
) -> Option<(f64, f64)> {
    if !gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        return None;
    }
    let device = &gpu.device;
    let set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let levels: Vec<Vec<Level<'_>>> = tiles.iter().map(levels_of).collect();
    let refs: Vec<&[Level<'_>]> = levels.iter().map(Vec::as_slice).collect();
    let mut scratch = texcomp::gpu::EncoderScratch::new();
    let mut times = Vec::new();
    let mut walls = Vec::new();
    for run in 0..=runs {
        let start = std::time::Instant::now();
        let mut cmd = device.create_command_encoder(&Default::default());
        let out = encoder.encode_with(
            &mut scratch,
            device,
            &gpu.queue,
            &mut cmd,
            &refs,
            true,
            Some(wgpu::ComputePassTimestampWrites {
                query_set: &set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        );
        assert!(out.iter().all(Result::is_ok));
        cmd.resolve_query_set(&set, 0..2, &resolve, 0);
        cmd.copy_buffer_to_buffer(&resolve, 0, &read, 0, 16);
        gpu.queue.submit([cmd.finish()]);
        let slice = read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let ticks: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        read.unmap();
        if run > 0 {
            walls.push(start.elapsed().as_secs_f64() * 1e3 / tiles.len() as f64);
            let ns = ticks[1].saturating_sub(ticks[0]) as f64 * gpu.queue.get_timestamp_period() as f64;
            times.push(ns / 1e6 / tiles.len() as f64);
        }
    }
    times.sort_by(f64::total_cmp);
    walls.sort_by(f64::total_cmp);
    Some((times[times.len() / 2], walls[walls.len() / 2]))
}

fn read_buffer(gpu: &Gpu, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = slice.get_mapped_range().expect("mapped").to_vec();
    buffer.unmap();
    data
}

/// The blocks of every level of `texture`, row by row, 16 bytes each.
pub fn read_blocks(gpu: &Gpu, texture: &wgpu::Texture, sizes: &[(u32, u32)]) -> Vec<Vec<u8>> {
    let mut cmd = gpu.device.create_command_encoder(&Default::default());
    let mut reads = Vec::new();
    for (level, &(w, h)) in sizes.iter().enumerate() {
        let (bx, by) = (w.div_ceil(4), h.div_ceil(4));
        let row = (bx * 16).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * by) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        cmd.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(by),
                },
            },
            wgpu::Extent3d {
                width: bx * 4,
                height: by * 4,
                depth_or_array_layers: 1,
            },
        );
        reads.push((buffer, row, bx, by));
    }
    gpu.queue.submit([cmd.finish()]);
    reads
        .into_iter()
        .map(|(buffer, row, bx, by)| {
            let data = read_buffer(gpu, &buffer);
            let mut out = Vec::with_capacity((bx * by * 16) as usize);
            for y in 0..by {
                let at = (y * row) as usize;
                out.extend_from_slice(&data[at..at + (bx * 16) as usize]);
            }
            out
        })
        .collect()
}

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

/// Every level of `texture` decoded by the GPU's own block decoder through
/// `textureLoad` on the format's linear view, as tightly packed RGBA8.
pub fn gpu_decode(gpu: &Gpu, texture: &wgpu::Texture, sizes: &[(u32, u32)]) -> Vec<Vec<u8>> {
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
    // The linear view decodes the stored values exactly.
    let linear = match texture.format() {
        wgpu::TextureFormat::Bc7RgbaUnormSrgb => wgpu::TextureFormat::Bc7RgbaUnorm,
        wgpu::TextureFormat::Astc { block, .. } => wgpu::TextureFormat::Astc {
            block,
            channel: wgpu::AstcChannel::Unorm,
        },
        f => f,
    };
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        format: Some(linear),
        ..Default::default()
    });
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
    readbacks.iter().map(|read| read_buffer(gpu, read)).collect()
}

/// One block decoded by the software oracle, 16 texels RGBA8, row-major.
pub fn oracle_block(format: BlockFormat, block: &[u8]) -> [[u8; 4]; 16] {
    let mut out = [0u32; 16];
    match format {
        BlockFormat::Bc7 => texture2ddecoder::decode_bc7_block(block, &mut out),
        BlockFormat::Astc4x4 => texture2ddecoder::decode_astc_block(block, 4, 4, &mut out),
        _ => unreachable!("a format the encoder doesn't write"),
    }
    // The oracle writes 0xAARRGGBB.
    out.map(|p| {
        let [b, g, r, a] = p.to_le_bytes();
        [r, g, b, a]
    })
}

/// A `width`×`height` level of blocks decoded by the software oracle, as
/// tightly packed RGBA8.
pub fn oracle_decode(format: BlockFormat, blocks: &[u8], width: u32, height: u32) -> Vec<u8> {
    let bx = width.div_ceil(4);
    let mut out = vec![0u8; (width * height * 4) as usize];
    for (b, block) in blocks.as_chunks::<16>().0.iter().enumerate() {
        let texels = oracle_block(format, block);
        let (ox, oy) = ((b as u32 % bx) * 4, (b as u32 / bx) * 4);
        for (i, t) in texels.iter().enumerate() {
            let (x, y) = (ox + (i as u32 & 3), oy + (i as u32 >> 2));
            if x < width && y < height {
                let at = ((y * width + x) * 4) as usize;
                out[at..at + 4].copy_from_slice(t);
            }
        }
    }
    out
}

// --- scores -----------------------------------------------------------------

pub fn psnr(mse: f64) -> f64 {
    if mse <= 0.0 {
        99.0
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

/// BT.601 luma of an RGB texel.
pub fn luma(p: &[u8]) -> f64 {
    0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64
}

/// Squared error summed over R, G and B, and over luma.
pub fn sse(src: &[u8], got: &[u8]) -> (f64, f64) {
    let (mut rgb, mut y) = (0.0, 0.0);
    for (s, g) in src.as_chunks::<4>().0.iter().zip(got.as_chunks::<4>().0.iter()) {
        for c in 0..3 {
            let d = s[c] as f64 - g[c] as f64;
            rgb += d * d;
        }
        let d = luma(s) - luma(g);
        y += d * d;
    }
    (rgb, y)
}

/// PSNR over RGB and over luma.
pub fn psnr_rgb_y(src: &[u8], got: &[u8]) -> (f64, f64) {
    let n = (src.len() / 4) as f64;
    let (rgb, y) = sse(src, got);
    (psnr(rgb / (3.0 * n)), psnr(y / n))
}

/// SSIM over luma (Wang et al. 2004): an 11×11 Gaussian window of σ 1.5,
/// K1 0.01, K2 0.03, mean over every window inside the image.
pub fn ssim_y(src: &[u8], got: &[u8], width: u32, height: u32) -> f64 {
    let (w, h) = (width as usize, height as usize);
    if w < 11 || h < 11 {
        return f64::NAN;
    }
    let a: Vec<f64> = src.as_chunks::<4>().0.iter().map(|p| luma(p)).collect();
    let b: Vec<f64> = got.as_chunks::<4>().0.iter().map(|p| luma(p)).collect();
    let mut g = [0.0f64; 11];
    for (i, v) in g.iter_mut().enumerate() {
        let x = i as f64 - 5.0;
        *v = (-x * x / (2.0 * 1.5 * 1.5)).exp();
    }
    let s: f64 = g.iter().sum();
    g.iter_mut().for_each(|v| *v /= s);
    let (c1, c2) = ((0.01f64 * 255.0).powi(2), (0.03f64 * 255.0).powi(2));
    let mut total = 0.0;
    let mut count = 0usize;
    for y in 0..=h - 11 {
        for x in 0..=w - 11 {
            let (mut ma, mut mb, mut aa, mut bb, mut ab) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for j in 0..11 {
                for i in 0..11 {
                    let k = g[i] * g[j];
                    let (va, vb) = (a[(y + j) * w + x + i], b[(y + j) * w + x + i]);
                    ma += k * va;
                    mb += k * vb;
                    aa += k * va * va;
                    bb += k * vb * vb;
                    ab += k * va * vb;
                }
            }
            let (sa, sb, sab) = (aa - ma * ma, bb - mb * mb, ab - ma * mb);
            total += ((2.0 * ma * mb + c1) * (2.0 * sab + c2))
                / ((ma * ma + mb * mb + c1) * (sa + sb + c2));
            count += 1;
        }
    }
    total / count as f64
}

// --- modes ------------------------------------------------------------------

/// What a block is: its BC7 mode, or its ASTC partition count, colour
/// endpoint mode, weight range and planes.
pub fn block_kind(format: BlockFormat, block: &[u8]) -> String {
    match format {
        BlockFormat::Bc7 => match block[0].trailing_zeros() {
            m @ 0..=7 => format!("mode {m}"),
            _ => "invalid".into(),
        },
        BlockFormat::Astc4x4 => astc_kind(block),
        _ => unreachable!(),
    }
}

fn astc_kind(block: &[u8]) -> String {
    let bits = u128::from_le_bytes(block.try_into().expect("16 bytes"));
    let field = |at: u32, n: u32| ((bits >> at) & ((1u128 << n) - 1)) as u32;
    let mode = field(0, 11);
    if mode & 0x1ff == 0x1fc {
        return "void-extent".into();
    }
    if mode & 3 == 0 && mode & 0xc == 0 {
        return "reserved".into();
    }
    // The weight range (Table 146): R0 R1 R2 and P, for a mode whose two low
    // bits are not 00 (the only layout the encoder writes).
    let (r, p) = if mode & 3 != 0 {
        ((mode >> 4) & 1 | (mode & 3) << 1, (mode >> 9) & 1)
    } else {
        ((mode >> 4) & 1 | ((mode >> 2) & 3) << 1, (mode >> 9) & 1)
    };
    let max = match (p, r) {
        (0, 2) => 1,
        (0, 3) => 2,
        (0, 4) => 3,
        (0, 5) => 4,
        (0, 6) => 5,
        (0, 7) => 7,
        (1, 2) => 9,
        (1, 3) => 11,
        (1, 4) => 15,
        (1, 5) => 19,
        (1, 6) => 23,
        (1, 7) => 31,
        _ => 0,
    };
    let dual = (mode >> 10) & 1 == 1 && mode & 3 != 0;
    let parts = field(11, 2) + 1;
    let cem = if parts == 1 {
        field(13, 4)
    } else if field(23, 2) == 0 {
        field(25, 4)
    } else {
        99
    };
    format!(
        "{parts}P CEM{cem} w0..{max}{}",
        if dual { " dual" } else { "" }
    )
}
