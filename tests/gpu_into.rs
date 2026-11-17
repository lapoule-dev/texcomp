// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! GPU in the loop: [`GpuEncoder::encode_into`], the encoder writing into a
//! caller's texture from texels already on the GPU.
//!
//! - Texels read from memory, from a buffer with padded rows and from a layer
//!   of an RGBA8 texture array encode to the same blocks, and those decode to
//!   what [`GpuEncoder::encode`] gives for the same texels.
//! - Encoding into layer N of a block-compressed array, at an origin, writes
//!   that region and nothing else: the other layers and the rest of the layer
//!   keep their bytes.
//! - Real imagery taken from a texture layer and written into an array layer
//!   holds `gpu_quality`'s gate for real imagery, decoded by the GPU.
//! - A tile with a bad source or destination gets its error, records nothing
//!   (no validation error on the device) and leaves the other tiles of the
//!   call encoded.
//!
//! # What this does not cover
//!
//! The quality gate itself, over twenty tiles and both texture variants, is
//! `gpu_quality`'s; here one tile is held to it. Blocks are compared as bytes
//! read back from the destination, which needs `COPY_SRC` on it. A machine
//! without a usable adapter skips (`TEXCOMP_REQUIRE_GPU=1` turns the skip into
//! a failure).

use texcomp::gpu::{
    feature, runs_on, texture_format, Destination, EncodeError, GpuEncoder, Level, Source,
    SourceLevel, TileInto, GPU_FORMATS,
};
use texcomp::BlockFormat;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    formats: Vec<BlockFormat>,
}

/// A device with every block format the adapter has that the encoder writes.
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
        let formats: Vec<BlockFormat> = GPU_FORMATS
            .into_iter()
            .filter(|f| adapter.features().contains(feature(*f)))
            .collect();
        if formats.is_empty() {
            eprintln!("skipping: the adapter samples no format the encoder writes");
            return None;
        }
        let features = formats
            .iter()
            .fold(wgpu::Features::empty(), |acc, f| acc | feature(*f));
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("texcomp encode_into"),
                required_features: features,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("a device with the adapter's own features");
        Some(Gpu {
            device,
            queue,
            formats,
        })
    })
}

impl Gpu {
    /// Submits `cmd`, waits for it, and fails on any validation error the
    /// recording or the submission raised.
    fn run(&self, record: impl FnOnce(&mut wgpu::CommandEncoder)) {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut cmd = self.device.create_command_encoder(&Default::default());
        record(&mut cmd);
        self.queue.submit([cmd.finish()]);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("validation error: {e}");
        }
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = slice.get_mapped_range().expect("mapped").to_vec();
        buffer.unmap();
        data
    }
}

/// A `side`² chain down to 1×1, a pattern seeded by `seed`; odd seeds have
/// varying alpha, which the kernels encode differently.
fn chain(side: u32, seed: u32) -> Vec<Vec<u8>> {
    let mut levels = Vec::new();
    let mut s = side;
    loop {
        let mut px = Vec::with_capacity((s * s * 4) as usize);
        for y in 0..s {
            for x in 0..s {
                let h = (x.wrapping_mul(31) ^ y.wrapping_mul(17)).wrapping_add(seed * 97);
                let a = if seed % 2 == 1 {
                    ((x + y) * 255 / (2 * s)) as u8
                } else {
                    255
                };
                px.extend_from_slice(&[
                    (x * 255 / s) as u8,
                    (y * 255 / s) as u8,
                    (h % 251) as u8,
                    a,
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

fn side_of(side: u32, level: usize) -> u32 {
    (side >> level).max(1)
}

/// A block-compressed texture the tests can write and read back.
fn block_texture(
    gpu: &Gpu,
    format: BlockFormat,
    side: u32,
    mips: u32,
    layers: u32,
) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("destination"),
        size: wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: layers,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: texture_format(format, false),
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// An RGBA8 array of `layers` layers holding `chain` in layer `layer` at
/// `origin` of every mip level, zeros elsewhere.
fn rgba_texture(
    gpu: &Gpu,
    format: wgpu::TextureFormat,
    side: u32,
    layers: u32,
    chain: &[Vec<u8>],
    chain_side: u32,
    at: (u32, u32, u32),
) -> wgpu::Texture {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("source"),
        size: wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: layers,
        },
        mip_level_count: chain.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (l, px) in chain.iter().enumerate() {
        let s = side_of(chain_side, l);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: l as u32,
                origin: wgpu::Origin3d {
                    x: at.0 >> l,
                    y: at.1 >> l,
                    z: at.2,
                },
                aspect: wgpu::TextureAspect::All,
            },
            px,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(s * 4),
                rows_per_image: Some(s),
            },
            wgpu::Extent3d {
                width: s,
                height: s,
                depth_or_array_layers: 1,
            },
        );
    }
    texture
}

/// The blocks of one mip level and layer, rows of blocks packed tightly.
fn read_blocks(gpu: &Gpu, texture: &wgpu::Texture, mip: u32, layer: u32) -> Vec<u8> {
    let bx = side_of(texture.width(), mip as usize).div_ceil(4);
    let by = side_of(texture.height(), mip as usize).div_ceil(4);
    let row = bx * 16;
    let pitch = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (pitch * by) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    gpu.run(|cmd| {
        cmd.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: Some(by),
                },
            },
            wgpu::Extent3d {
                width: bx * 4,
                height: by * 4,
                depth_or_array_layers: 1,
            },
        );
    });
    let padded = gpu.read(&buffer);
    padded
        .chunks(pitch as usize)
        .flat_map(|r| r[..row as usize].to_vec())
        .collect()
}

/// Every level and layer of `texture`, as `read_blocks` gives them.
fn all_blocks(gpu: &Gpu, texture: &wgpu::Texture) -> Vec<Vec<Vec<u8>>> {
    (0..texture.depth_or_array_layers())
        .map(|z| {
            (0..texture.mip_level_count())
                .map(|m| read_blocks(gpu, texture, m, z))
                .collect()
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

/// Every mip level of layer `layer` of `texture` (a linear block format,
/// which decodes exactly), decoded by the GPU, `sizes[l]` texels from the
/// corner.
fn decoded(gpu: &Gpu, texture: &wgpu::Texture, layer: u32, sizes: &[(u32, u32)]) -> Vec<Vec<u8>> {
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
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: layer,
        array_layer_count: Some(1),
        ..Default::default()
    });
    let mut reads = Vec::new();
    gpu.run(|cmd| {
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
                contents: &[level as u32, w, h, 0u32]
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
    });
    reads.iter().map(|r| gpu.read(r)).collect()
}

/// Destinations for a chain at `at` (x, y of level 0, layer) of `texture`.
fn destinations(
    texture: &wgpu::Texture,
    levels: usize,
    at: (u32, u32, u32),
) -> Vec<Destination<'_>> {
    (0..levels)
        .map(|l| Destination {
            texture,
            mip_level: l as u32,
            origin: wgpu::Origin3d {
                x: at.0 >> l,
                y: at.1 >> l,
                z: at.2,
            },
        })
        .collect()
}

/// **Memory, a padded buffer and a texture layer encode to the same
/// blocks**, which decode to what `encode` gives for the same texels.
#[test]
fn every_source_encodes_the_same_blocks() {
    let Some(gpu) = gpu() else { return };
    const SIDE: u32 = 64;
    for &format in &gpu.formats {
        let encoder = GpuEncoder::new(&gpu.device, format);
        for seed in [2, 3] {
            let px = chain(SIDE, seed);
            let n = px.len();

            // From memory.
            let cpu_levels: Vec<SourceLevel<'_>> = px
                .iter()
                .enumerate()
                .map(|(l, p)| SourceLevel {
                    width: side_of(SIDE, l),
                    height: side_of(SIDE, l),
                    source: Source::Cpu(p),
                })
                .collect();

            // From a buffer: each level at an odd multiple of 4, with 12
            // bytes of padding after every row.
            let mut bytes = vec![0xa5u8; 20];
            let mut placed = Vec::new();
            for (l, p) in px.iter().enumerate() {
                let s = side_of(SIDE, l) as usize;
                let pitch = s * 4 + 12;
                placed.push((bytes.len() as u64, pitch as u32));
                for row in p.chunks(s * 4) {
                    bytes.extend_from_slice(row);
                    bytes.extend_from_slice(&[0x5a; 12]);
                }
            }
            let buffer = wgpu::util::DeviceExt::create_buffer_init(
                &gpu.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("source texels"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::COPY_SRC,
                },
            );
            let buffer_levels: Vec<SourceLevel<'_>> = placed
                .iter()
                .enumerate()
                .map(|(l, &(offset, bytes_per_row))| SourceLevel {
                    width: side_of(SIDE, l),
                    height: side_of(SIDE, l),
                    source: Source::Buffer {
                        buffer: &buffer,
                        offset,
                        bytes_per_row,
                    },
                })
                .collect();

            // From layer 1 of an sRGB array, at (64, 0) of a 128² layer.
            let rgba = rgba_texture(
                &gpu,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                2 * SIDE,
                3,
                &px,
                SIDE,
                (SIDE, 0, 1),
            );
            let texture_levels: Vec<SourceLevel<'_>> = (0..n)
                .map(|l| SourceLevel {
                    width: side_of(SIDE, l),
                    height: side_of(SIDE, l),
                    source: Source::Texture {
                        texture: &rgba,
                        mip_level: l as u32,
                        origin: wgpu::Origin3d {
                            x: SIDE >> l,
                            y: 0,
                            z: 1,
                        },
                    },
                })
                .collect();

            let out: Vec<wgpu::Texture> = (0..3)
                .map(|_| block_texture(&gpu, format, SIDE, n as u32, 1))
                .collect();
            let dsts: Vec<Vec<Destination<'_>>> =
                out.iter().map(|t| destinations(t, n, (0, 0, 0))).collect();
            let tiles = [
                TileInto {
                    levels: &cpu_levels,
                    destinations: &dsts[0],
                },
                TileInto {
                    levels: &buffer_levels,
                    destinations: &dsts[1],
                },
                TileInto {
                    levels: &texture_levels,
                    destinations: &dsts[2],
                },
            ];
            gpu.run(|cmd| {
                let results = encoder.encode_into(&gpu.device, &gpu.queue, cmd, &tiles);
                assert!(results.iter().all(Result::is_ok), "{results:?}");
            });
            let blocks: Vec<_> = out.iter().map(|t| all_blocks(&gpu, t)).collect();
            assert_eq!(blocks[0], blocks[1], "{format:?} #{seed}: buffer vs memory");
            assert_eq!(
                blocks[0], blocks[2],
                "{format:?} #{seed}: texture vs memory"
            );
            assert!(
                blocks[0].iter().flatten().flatten().any(|&b| b != 0),
                "blocks were written"
            );

            // `encode`'s texture has no COPY_SRC: compare the decoded texels.
            let level_refs: Vec<Level<'_>> = px
                .iter()
                .enumerate()
                .map(|(l, p)| Level {
                    width: side_of(SIDE, l),
                    height: side_of(SIDE, l),
                    rgba8: p,
                })
                .collect();
            let mut fresh = None;
            gpu.run(|cmd| {
                let r = encoder.encode(&gpu.device, &gpu.queue, cmd, &[&level_refs], false);
                fresh = Some(r.into_iter().next().expect("one").expect("encodes"));
            });
            let sizes: Vec<(u32, u32)> = (0..n)
                .map(|l| (side_of(SIDE, l), side_of(SIDE, l)))
                .collect();
            assert_eq!(
                decoded(&gpu, &fresh.expect("texture"), 0, &sizes),
                decoded(&gpu, &out[0], 0, &sizes),
                "{format:?} #{seed}: encode_into vs encode"
            );
        }
    }
}

/// **Layer N of an array, at an origin, is all that is written**: the other
/// layers, and the rest of that layer, keep their bytes.
#[test]
fn only_the_destination_region_is_written() {
    let Some(gpu) = gpu() else { return };
    const SIDE: u32 = 32;
    const ARRAY: u32 = 64;
    for &format in &gpu.formats {
        let encoder = GpuEncoder::new(&gpu.device, format);
        let mips = 7; // 64² down to 1×1
        let array = block_texture(&gpu, format, ARRAY, mips, 4);
        // Every layer and level filled with a known pattern of bytes.
        for z in 0..4 {
            for m in 0..mips {
                let b = side_of(ARRAY, m as usize).div_ceil(4);
                let fill: Vec<u8> = (0..b * b * 16)
                    .map(|i| (i as u8).wrapping_mul(7) ^ (z * 16 + m) as u8)
                    .collect();
                gpu.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &array,
                        mip_level: m,
                        origin: wgpu::Origin3d { x: 0, y: 0, z },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &fill,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(b * 16),
                        rows_per_image: Some(b),
                    },
                    wgpu::Extent3d {
                        width: b * 4,
                        height: b * 4,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        let before = all_blocks(&gpu, &array);

        // A 32² chain (32 down to 4: four levels) at (32, 32) of layer 2, and
        // a 32² level 0 alone at (0, 32) of layer 1, in one call.
        let px = chain(SIDE, 4);
        let solo = chain(SIDE, 5);
        // Down to 4×4 only: below that, a level at (32 >> m, 32 >> m) is no
        // longer on a block boundary of its mip.
        let levels: Vec<SourceLevel<'_>> = px[..4]
            .iter()
            .enumerate()
            .map(|(l, p)| SourceLevel {
                width: side_of(SIDE, l),
                height: side_of(SIDE, l),
                source: Source::Cpu(p),
            })
            .collect();
        let solo_level = [SourceLevel {
            width: SIDE,
            height: SIDE,
            source: Source::Cpu(&solo[0]),
        }];
        let to = destinations(&array, levels.len(), (32, 32, 2));
        let solo_to = destinations(&array, 1, (0, 32, 1));
        gpu.run(|cmd| {
            let results = encoder.encode_into(
                &gpu.device,
                &gpu.queue,
                cmd,
                &[
                    TileInto {
                        levels: &levels,
                        destinations: &to,
                    },
                    TileInto {
                        levels: &solo_level,
                        destinations: &solo_to,
                    },
                ],
            );
            assert_eq!(results, [Ok(()), Ok(())]);
        });
        let after = all_blocks(&gpu, &array);

        // What the same texels give alone, in a texture of their own.
        let own = |levels: &[SourceLevel<'_>]| {
            let t = block_texture(&gpu, format, SIDE, levels.len() as u32, 1);
            let d = destinations(&t, levels.len(), (0, 0, 0));
            gpu.run(|cmd| {
                let r = encoder.encode_into(
                    &gpu.device,
                    &gpu.queue,
                    cmd,
                    &[TileInto {
                        levels,
                        destinations: &d,
                    }],
                );
                assert_eq!(r, [Ok(())]);
            });
            all_blocks(&gpu, &t).remove(0)
        };
        let chain_blocks = own(&levels);
        let solo_blocks = own(&solo_level);

        for z in 0..4u32 {
            for m in 0..mips {
                let b = side_of(ARRAY, m as usize).div_ceil(4) as usize;
                let mut want = before[z as usize][m as usize].clone();
                // (layer, level of the tile, origin in texels, its blocks)
                let written: Option<(&Vec<u8>, u32, u32)> = match (z, m) {
                    (2, m) if (m as usize) < chain_blocks.len() => {
                        Some((&chain_blocks[m as usize], 32 >> m, 32 >> m))
                    }
                    (1, 0) => Some((&solo_blocks[0], 0, 32)),
                    _ => None,
                };
                if let Some((blocks, x, y)) = written {
                    let tb = side_of(SIDE, m as usize).div_ceil(4) as usize;
                    let (bx, by) = ((x / 4) as usize, (y / 4) as usize);
                    for row in 0..tb {
                        let at = ((by + row) * b + bx) * 16;
                        want[at..at + tb * 16]
                            .copy_from_slice(&blocks[row * tb * 16..(row + 1) * tb * 16]);
                    }
                }
                assert_eq!(
                    after[z as usize][m as usize], want,
                    "{format:?}: layer {z}, level {m}"
                );
            }
        }
    }
}

/// **Real imagery from a texture layer into an array layer holds the gate**
/// `gpu_quality` sets for real imagery, decoded by the GPU.
#[test]
fn imagery_from_a_texture_layer_holds_the_gate() {
    let Some(gpu) = gpu() else { return };
    const TILE: u32 = 256;
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/imagery/alps.jpeg");
    let img = image::load_from_memory(&std::fs::read(path).expect("fixture"))
        .expect("decodes")
        .to_rgba8();
    let crop = image::imageops::crop_imm(&img, 128, 128, TILE, TILE).to_image();
    let mut px = vec![crop.clone().into_raw()];
    let mut prev = crop;
    while prev.width() > 1 {
        let s = prev.width() / 2;
        prev = image::imageops::resize(&prev, s, s, image::imageops::FilterType::Triangle);
        px.push(prev.clone().into_raw());
    }
    let n = px.len();
    let rgba = rgba_texture(
        &gpu,
        wgpu::TextureFormat::Rgba8Unorm,
        TILE,
        2,
        &px,
        TILE,
        (0, 0, 1),
    );
    let levels: Vec<SourceLevel<'_>> = (0..n)
        .map(|l| SourceLevel {
            width: side_of(TILE, l),
            height: side_of(TILE, l),
            source: Source::Texture {
                texture: &rgba,
                mip_level: l as u32,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
            },
        })
        .collect();
    for &format in &gpu.formats {
        // Measured worst over the twenty real tiles, minus about 1 dB
        // (`gpu_quality`'s `gate`).
        let gate = match format {
            BlockFormat::Bc7 => 39.0,
            BlockFormat::Astc4x4 => 35.0,
            _ => unreachable!("a format the encoder doesn't write"),
        };
        let encoder = GpuEncoder::new(&gpu.device, format);
        let array = block_texture(&gpu, format, TILE, n as u32, 3);
        let to = destinations(&array, n, (0, 0, 2));
        gpu.run(|cmd| {
            let r = encoder.encode_into(
                &gpu.device,
                &gpu.queue,
                cmd,
                &[TileInto {
                    levels: &levels,
                    destinations: &to,
                }],
            );
            assert_eq!(r, [Ok(())]);
        });
        let sizes: Vec<(u32, u32)> = (0..n)
            .map(|l| (side_of(TILE, l), side_of(TILE, l)))
            .collect();
        let got = decoded(&gpu, &array, 2, &sizes);
        for (l, (src, got)) in px.iter().zip(&got).enumerate() {
            if side_of(TILE, l) < 8 {
                continue;
            }
            let se: f64 = src
                .chunks_exact(4)
                .zip(got.chunks_exact(4))
                .map(|(s, g)| {
                    (0..3)
                        .map(|c| (s[c] as f64 - g[c] as f64).powi(2))
                        .sum::<f64>()
                })
                .sum();
            let mse = se / (src.len() / 4 * 3) as f64;
            let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
            eprintln!("{format:?} level {l}: {psnr:.2} dB");
            assert!(psnr >= gate, "{format:?} level {l}: {psnr:.2} dB < {gate}");
        }
    }
}

/// **A bad source or destination is refused before the GPU sees it**: its
/// tile gets the error and records nothing, and the valid tile of the same
/// call is encoded.
#[test]
fn bad_sources_and_destinations_are_refused() {
    let Some(gpu) = gpu() else { return };
    let format = gpu.formats[0];
    let encoder = GpuEncoder::new(&gpu.device, format);
    let texture = |format: wgpu::TextureFormat,
                   usage: wgpu::TextureUsages,
                   side: u32,
                   mips: u32,
                   layers: u32| {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: layers,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let copy = wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
    let block = texture(texture_format(format, true), copy, 16, 1, 2);
    let block_no_dst = texture(
        texture_format(format, false),
        wgpu::TextureUsages::COPY_SRC,
        16,
        1,
        1,
    );
    // The other block format, on a device that may not have it: a plain
    // RGBA8 texture stands in as a format this encoder doesn't write.
    let not_ours = texture(wgpu::TextureFormat::Rgba8Unorm, copy, 16, 1, 1);
    let rgba = texture(wgpu::TextureFormat::Rgba8Unorm, copy, 16, 1, 2);
    let rgba_no_src = texture(
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureUsages::COPY_DST,
        16,
        1,
        1,
    );
    let bgra = texture(wgpu::TextureFormat::Bgra8Unorm, copy, 16, 1, 1);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16 * 16 * 4,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buffer_no_src = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16 * 16 * 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let px = vec![200u8; 16 * 16 * 4];
    let cpu = SourceLevel {
        width: 16,
        height: 16,
        source: Source::Cpu(&px),
    };
    let from_texture = |texture, z| SourceLevel {
        width: 16,
        height: 16,
        source: Source::Texture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z },
        },
    };
    let from_buffer = |buffer, offset, bytes_per_row| SourceLevel {
        width: 16,
        height: 16,
        source: Source::Buffer {
            buffer,
            offset,
            bytes_per_row,
        },
    };
    let at = |texture, mip_level, x, y, z| Destination {
        texture,
        mip_level,
        origin: wgpu::Origin3d { x, y, z },
    };
    let ok_to = [at(&block, 0, 0, 0, 1)];
    let cases: Vec<(Vec<SourceLevel<'_>>, Vec<Destination<'_>>, EncodeError)> = vec![
        (
            vec![cpu],
            vec![],
            EncodeError::Destinations {
                levels: 1,
                destinations: 0,
            },
        ),
        (
            vec![cpu],
            vec![at(&not_ours, 0, 0, 0, 0)],
            EncodeError::DestinationFormat {
                level: 0,
                got: wgpu::TextureFormat::Rgba8Unorm,
            },
        ),
        (
            vec![cpu],
            vec![at(&block_no_dst, 0, 0, 0, 0)],
            EncodeError::DestinationUsage { level: 0 },
        ),
        (
            vec![cpu],
            vec![at(&block, 1, 0, 0, 0)],
            EncodeError::DestinationRegion { level: 0 },
        ),
        (
            vec![cpu],
            vec![at(&block, 0, 0, 0, 2)],
            EncodeError::DestinationRegion { level: 0 },
        ),
        (
            vec![cpu],
            vec![at(&block, 0, 4, 0, 0)],
            EncodeError::DestinationRegion { level: 0 },
        ),
        (
            vec![cpu],
            vec![at(&block, 0, 2, 0, 0)],
            EncodeError::DestinationRegion { level: 0 },
        ),
        (
            vec![from_texture(&bgra, 0)],
            ok_to.to_vec(),
            EncodeError::SourceFormat {
                level: 0,
                got: wgpu::TextureFormat::Bgra8Unorm,
            },
        ),
        (
            vec![from_texture(&rgba_no_src, 0)],
            ok_to.to_vec(),
            EncodeError::SourceUsage { level: 0 },
        ),
        (
            vec![from_texture(&rgba, 2)],
            ok_to.to_vec(),
            EncodeError::SourceRegion { level: 0 },
        ),
        (
            vec![from_buffer(&buffer_no_src, 0, 64)],
            ok_to.to_vec(),
            EncodeError::SourceUsage { level: 0 },
        ),
        (
            vec![from_buffer(&buffer, 4, 64)],
            ok_to.to_vec(),
            EncodeError::SourceRegion { level: 0 },
        ),
        (
            vec![from_buffer(&buffer, 2, 64)],
            ok_to.to_vec(),
            EncodeError::SourceRegion { level: 0 },
        ),
        (
            vec![from_buffer(&buffer, 0, 60)],
            ok_to.to_vec(),
            EncodeError::SourceRegion { level: 0 },
        ),
        (
            vec![from_buffer(&buffer, 0, 66)],
            ok_to.to_vec(),
            EncodeError::SourceRegion { level: 0 },
        ),
        (vec![], vec![], EncodeError::Empty),
    ];
    let good = [from_buffer(&buffer, 0, 64)];
    let good_to = [at(&block, 0, 0, 0, 0)];
    for (levels, to, want) in &cases {
        let before = all_blocks(&gpu, &block);
        gpu.run(|cmd| {
            let r = encoder.encode_into(
                &gpu.device,
                &gpu.queue,
                cmd,
                &[
                    TileInto {
                        levels,
                        destinations: to,
                    },
                    TileInto {
                        levels: &good,
                        destinations: &good_to,
                    },
                ],
            );
            assert_eq!(r[0].as_ref().err(), Some(want));
            assert_eq!(r[1], Ok(()));
        });
        let after = all_blocks(&gpu, &block);
        assert_eq!(
            after[1], before[1],
            "{want:?}: layer 1, its destination, is untouched"
        );
    }
}
