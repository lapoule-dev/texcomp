// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! GPU in the loop: what a caller needs to spread encodes over frames.
//!
//! - An [`EncoderScratch`] keeps the encoder's buffers from one call to the
//!   next, and a call through it encodes what a fresh
//!   [`GpuEncoder::encode`].
//! - Timestamps the caller asks for bracket the encode pass.
//!
//! # What this does not cover
//!
//! The tiles are compared with a fresh encode texel for texel, as the GPU
//! decodes them at every level; their quality is `gpu_quality`'s. Timestamps
//! are checked to be written and ordered, not to be accurate: that is the
//! driver's. A machine without a usable adapter
//! skips (`TEXCOMP_REQUIRE_GPU=1` turns the skip into a failure), and an
//! adapter without timestamp queries skips the timestamp test.

use texcomp::gpu::{
    feature, runs_on, texture_format, EncoderScratch, GpuEncoder, Level, GPU_FORMATS,
};
use texcomp::BlockFormat;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: BlockFormat,
    timestamps: bool,
}

/// A device with the first block format the adapter samples, and timestamp
/// queries when it has them.
fn gpu() -> Option<Gpu> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
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
        if !runs_on(
            adapter.get_downlevel_capabilities().flags,
            &adapter.limits(),
        ) {
            eprintln!("skipping: the adapter does not run the kernels");
            return None;
        }
        let Some(format) = GPU_FORMATS
            .into_iter()
            .find(|f| adapter.features().contains(feature(*f)))
        else {
            eprintln!("skipping: the adapter samples no format the encoder writes");
            return None;
        };
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut features = feature(format);
        if timestamps {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("texcomp pacing"),
                required_features: features,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("a device with the adapter's own features");
        Some(Gpu {
            device,
            queue,
            format,
            timestamps,
        })
    })
}

/// A `side`² tile with its chain down to 1×1, a pattern seeded by `seed`.
fn tile(side: u32, seed: u32) -> Vec<Vec<u8>> {
    let mut levels = Vec::new();
    let mut s = side;
    loop {
        let mut px = Vec::with_capacity((s * s * 4) as usize);
        for y in 0..s {
            for x in 0..s {
                let h = (x.wrapping_mul(31) ^ y.wrapping_mul(17)).wrapping_add(seed * 97);
                px.extend_from_slice(&[
                    (x * 255 / s) as u8,
                    (y * 255 / s) as u8,
                    (h % 251) as u8,
                    255,
                ]);
            }
        }
        levels.push(px);
        if s == 1 {
            return levels;
        }
        s /= 2;
    }
}

fn as_levels(chain: &[Vec<u8>], side: u32) -> Vec<Level<'_>> {
    chain
        .iter()
        .enumerate()
        .map(|(l, px)| Level {
            width: (side >> l).max(1),
            height: (side >> l).max(1),
            rgba8: px,
        })
        .collect()
}

fn chain_levels(chains: &[Vec<Vec<u8>>], side: u32) -> Vec<Vec<Level<'_>>> {
    chains.iter().map(|c| as_levels(c, side)).collect()
}

fn slices<'a, 'b>(levels: &'a [Vec<Level<'b>>]) -> Vec<&'a [Level<'b>]> {
    levels.iter().map(Vec::as_slice).collect()
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

/// Every level of `texture`, decoded by the GPU's block decoder. The texture
/// is the linear variant, which decodes exactly, so equal blocks decode equal.
fn decoded(gpu: &Gpu, texture: &wgpu::Texture) -> Vec<Vec<u8>> {
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
    let mut reads = Vec::new();
    for level in 0..texture.mip_level_count() {
        let w = (texture.width() >> level).max(1);
        let h = (texture.height() >> level).max(1);
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
            contents: &[level, w, h, 0u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
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
        reads.push(read);
    }
    gpu.queue.submit([cmd.finish()]);
    reads
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

fn fresh(gpu: &Gpu, encoder: &GpuEncoder, tiles: &[&[Level<'_>]]) -> Vec<wgpu::Texture> {
    let mut cmd = gpu.device.create_command_encoder(&Default::default());
    let out = encoder.encode(&gpu.device, &gpu.queue, &mut cmd, tiles, false);
    gpu.queue.submit([cmd.finish()]);
    out.into_iter().map(|r| r.expect("encodes")).collect()
}

fn through(
    gpu: &Gpu,
    encoder: &GpuEncoder,
    scratch: &mut EncoderScratch,
    tiles: &[&[Level<'_>]],
) -> Vec<wgpu::Texture> {
    let mut cmd = gpu.device.create_command_encoder(&Default::default());
    let out = encoder.encode_with(
        scratch,
        &gpu.device,
        &gpu.queue,
        &mut cmd,
        tiles,
        false,
        None,
    );
    gpu.queue.submit([cmd.finish()]);
    out.into_iter().map(|r| r.expect("encodes")).collect()
}

/// **A scratch sized for the work holds its buffers**: call after call, it
/// neither grows nor allocates again.
#[test]
fn a_sized_scratch_is_reused_across_calls() {
    let Some(gpu) = gpu() else { return };
    let encoder = GpuEncoder::new(&gpu.device, gpu.format);
    let chains: Vec<_> = (0..4).map(|s| tile(256, s)).collect();
    let levels: Vec<_> = chains.iter().map(|c| as_levels(c, 256)).collect();
    let tiles: Vec<&[Level<'_>]> = levels.iter().map(Vec::as_slice).collect();

    let mut scratch = EncoderScratch::with_capacity(&gpu.device, &encoder, 4, 256, 256);
    let held = scratch.bytes();
    assert!(
        held > 0,
        "a scratch with a capacity holds its buffers up front"
    );
    for _ in 0..3 {
        through(&gpu, &encoder, &mut scratch, &tiles);
        assert_eq!(scratch.bytes(), held, "no call outgrows its capacity");
    }
    // Fewer tiles fit in the same buffers.
    through(&gpu, &encoder, &mut scratch, &tiles[..1]);
    assert_eq!(scratch.bytes(), held);
}

/// **Through a scratch, what a fresh encode gives**: after the
/// scratch has served other tiles, after it has grown, at every level.
#[test]
fn a_scratch_encodes_what_a_fresh_encode_does() {
    let Some(gpu) = gpu() else { return };
    let encoder = GpuEncoder::new(&gpu.device, gpu.format);
    let first: Vec<_> = (0..2).map(|s| tile(256, s)).collect();
    let second: Vec<_> = (10..14).map(|s| tile(256, s)).collect();
    let small: Vec<_> = (20..23).map(|s| tile(64, s)).collect();
    let (first, second, small) = (
        chain_levels(&first, 256),
        chain_levels(&second, 256),
        chain_levels(&small, 64),
    );

    // Room for one tile: the second call has to grow it.
    let mut scratch = EncoderScratch::with_capacity(&gpu.device, &encoder, 1, 256, 256);
    let before = scratch.bytes();
    for set in [slices(&first), slices(&second), slices(&small)] {
        let got = through(&gpu, &encoder, &mut scratch, &set);
        let want = fresh(&gpu, &encoder, &set);
        assert_eq!(got.len(), want.len());
        for (t, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.format(), texture_format(gpu.format, false));
            assert_eq!(decoded(&gpu, g), decoded(&gpu, w), "tile {t}");
        }
    }
    assert!(scratch.bytes() > before, "four tiles outgrow room for one");

    // An empty scratch grows on first use.
    let mut empty = EncoderScratch::new();
    assert_eq!(empty.bytes(), 0);
    let set = slices(&second);
    let got = through(&gpu, &encoder, &mut empty, &set);
    assert!(empty.bytes() > 0);
    assert_eq!(
        decoded(&gpu, &got[3]),
        decoded(&gpu, &fresh(&gpu, &encoder, &set)[3])
    );
}

/// **The caller's timestamps bracket the encode pass**: both are written,
/// the end after the beginning.
#[test]
fn timestamps_bracket_the_encode_pass() {
    let Some(gpu) = gpu() else { return };
    if !gpu.timestamps {
        eprintln!("skipping: the adapter has no timestamp queries");
        return;
    }
    let encoder = GpuEncoder::new(&gpu.device, gpu.format);
    let chains: Vec<_> = (0..4).map(|s| tile(256, s)).collect();
    let levels: Vec<_> = chains.iter().map(|c| as_levels(c, 256)).collect();
    let tiles: Vec<&[Level<'_>]> = levels.iter().map(Vec::as_slice).collect();

    let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("encode pass"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut scratch = EncoderScratch::new();
    let mut cmd = gpu.device.create_command_encoder(&Default::default());
    let out = encoder.encode_with(
        &mut scratch,
        &gpu.device,
        &gpu.queue,
        &mut cmd,
        &tiles,
        true,
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &queries,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }),
    );
    assert!(out.iter().all(Result::is_ok));
    cmd.resolve_query_set(&queries, 0..2, &resolve, 0);
    cmd.copy_buffer_to_buffer(&resolve, 0, &read, 0, 16);
    gpu.queue.submit([cmd.finish()]);
    let slice = read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let stamps: Vec<u64> = slice
        .get_mapped_range()
        .expect("mapped")
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect();
    read.unmap();
    let (begin, end) = (stamps[0], stamps[1]);
    let ns = (end.saturating_sub(begin)) as f64 * gpu.queue.get_timestamp_period() as f64;
    println!(
        "{:?}: encode pass {:.3} ms for 4 tiles of 256² with mips",
        gpu.format,
        ns / 1e6
    );
    assert!(begin > 0, "the beginning timestamp is written");
    assert!(
        end > begin,
        "the pass ends after it begins ({begin} → {end})"
    );
}
