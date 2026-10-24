// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! The compute encoder: RGBA8 levels in, block-compressed wgpu textures out,
//! in one compute pass per call.
//!
//! The kernels are WGSL (`common.wgsl` plus one file per format), written
//! from the Khronos Data Format Specification 1.3:
//!
//! - **BC7**: modes 6 (one subset, 4-bit indices) and 1 (two subsets out of
//!   64 partitions, 3-bit indices). An opaque block keeps the better of the
//!   two; a block with alpha takes mode 6.
//! - **ASTC 4×4 LDR**: one partition, CEM 8 with a weight range of 0..11 for
//!   opaque blocks, CEM 12 with 0..5 for blocks with alpha, endpoints stored
//!   as plain bytes.
//!
//! [`runs_on`] says whether a device can run them; [`GPU_FORMATS`] lists
//! what they write; [`texture_format`] and [`feature`] name the wgpu
//! format and the device feature it needs.

use std::borrow::Cow;

use crate::BlockFormat;

/// The formats the kernels write.
pub const GPU_FORMATS: [BlockFormat; 2] = [BlockFormat::Bc7, BlockFormat::Astc4x4];

/// The device feature a texture of `format` needs.
pub fn feature(format: BlockFormat) -> wgpu::Features {
    match format {
        BlockFormat::Bc7 => wgpu::Features::TEXTURE_COMPRESSION_BC,
        BlockFormat::Astc4x4 => wgpu::Features::TEXTURE_COMPRESSION_ASTC,
    }
}

/// The wgpu format of `format`. `srgb` is what imagery is sampled as; the
/// linear variant holds the same blocks and decodes them exactly.
pub fn texture_format(format: BlockFormat, srgb: bool) -> wgpu::TextureFormat {
    match (format, srgb) {
        (BlockFormat::Bc7, true) => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
        (BlockFormat::Bc7, false) => wgpu::TextureFormat::Bc7RgbaUnorm,
        (BlockFormat::Astc4x4, srgb) => wgpu::TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: if srgb {
                wgpu::AstcChannel::UnormSrgb
            } else {
                wgpu::AstcChannel::Unorm
            },
        },
    }
}

/// A device with these capabilities runs the kernels: compute shaders, two
/// storage buffers in the compute stage, an 8×8 workgroup, enough workgroups
/// per dimension for a 4096² level, and a storage binding large enough for a
/// batch of tiles.
///
/// It says nothing about formats: a device that runs the kernels still needs
/// [`feature`] of the format to sample it.
pub fn runs_on(downlevel: wgpu::DownlevelFlags, limits: &wgpu::Limits) -> bool {
    downlevel.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        && limits.max_storage_buffers_per_shader_stage >= 2
        && limits.max_compute_workgroup_size_x >= WORKGROUP
        && limits.max_compute_workgroup_size_y >= WORKGROUP
        && limits.max_compute_invocations_per_workgroup >= WORKGROUP * WORKGROUP
        && limits.max_compute_workgroups_per_dimension >= 1024
        && limits.max_storage_buffer_binding_size >= MIN_STORAGE_BINDING
}

/// Side of the encoder's workgroup, in blocks.
const WORKGROUP: u32 = 8;

/// The smallest storage binding the encoder accepts from a device: room for
/// a batch of several 256² tiles with their mips. Every WebGPU device has
/// 128 MiB.
const MIN_STORAGE_BINDING: u64 = 16 << 20;

/// Rows of blocks in the output are padded to this, the alignment of a
/// buffer-to-texture copy.
const ROW_ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// One mip level of a tile, as tightly packed RGBA8.
#[derive(Clone, Copy, Debug)]
pub struct Level<'a> {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// `width * height * 4` bytes, rows top to bottom.
    pub rgba8: &'a [u8],
}

/// Why a tile was not encoded. Every one of these is a tile to upload
/// uncompressed instead, never a tile to drop.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EncodeError {
    /// A tile with no level.
    #[error("a tile with no level")]
    Empty,
    /// Level 0 of a block-compressed texture must be whole blocks.
    #[error("{width}×{height}: a block-compressed texture is a multiple of 4 on both sides")]
    NotBlockAligned {
        /// Level 0 width.
        width: u32,
        /// Level 0 height.
        height: u32,
    },
    /// A level that is not the halving of the one above it.
    #[error("level {level} is {got:?}, expected {want:?}")]
    NotAMipChain {
        /// Level index.
        level: usize,
        /// Its size.
        got: (u32, u32),
        /// The size a mip chain has there.
        want: (u32, u32),
    },
    /// A level whose bytes don't match its size.
    #[error("level {level} holds {got} bytes, {want} expected")]
    LevelBytes {
        /// Level index.
        level: usize,
        /// Bytes given.
        got: usize,
        /// `width * height * 4`.
        want: usize,
    },
    /// One tile larger than the device's storage binding.
    #[error("a tile of {bytes} bytes exceeds the device's storage binding of {limit}")]
    TooLarge {
        /// The tile's input or output bytes.
        bytes: u64,
        /// `max_storage_buffer_binding_size`.
        limit: u64,
    },
}

/// The GPU's block encoder for one format.
///
/// Built once per device. [`GpuEncoder::encode`] records into the caller's
/// command encoder, so the encode, the copy into the texture and the draw
/// that first samples it can share a submission.
pub struct GpuEncoder {
    format: BlockFormat,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform_align: u64,
    max_binding: u64,
}

/// The job uniform, `Job` in `common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Job {
    src: u32,
    width: u32,
    height: u32,
    dst: u32,
    row_words: u32,
    blocks_x: u32,
    blocks_y: u32,
    _pad: u32,
}

/// Where one level lands in the output buffer.
struct Placed {
    tile: usize,
    level: u32,
    job: Job,
    /// Byte offset of the level's blocks in the output.
    offset: u64,
    row_bytes: u32,
}

impl GpuEncoder {
    /// Compiles the encoder for `format`.
    pub fn new(device: &wgpu::Device, format: BlockFormat) -> Self {
        let source = shader_source(format);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tuile block encoder"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tuile block encoder"),
            entries: &[
                storage(0, true),
                storage(1, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Job>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tuile block encoder"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tuile block encoder"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let limits = device.limits();
        Self {
            format,
            pipeline,
            layout,
            uniform_align: limits.min_uniform_buffer_offset_alignment.max(32) as u64,
            max_binding: limits.max_storage_buffer_binding_size,
        }
    }

    /// The format this encoder writes.
    pub fn format(&self) -> BlockFormat {
        self.format
    }

    /// Encodes each tile's mip chain into a new texture, recording one
    /// compute pass and the copies into `cmd`.
    ///
    /// The texels are uploaded with `queue.write_buffer`, which lands before
    /// the next submission, so `cmd` must be submitted on that same queue.
    /// `srgb` picks the sRGB view of the format, which is what imagery is
    /// sampled as.
    ///
    /// The result has one entry per tile, in order. A tile that cannot be
    /// encoded gets its [`EncodeError`] and costs nothing: the caller uploads
    /// it uncompressed.
    ///
    /// Every call creates its buffers; [`GpuEncoder::encode_with`] keeps them
    /// in an [`EncoderScratch`] instead.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cmd: &mut wgpu::CommandEncoder,
        tiles: &[&[Level<'_>]],
        srgb: bool,
    ) -> Vec<Result<wgpu::Texture, EncodeError>> {
        let mut scratch = EncoderScratch::new();
        self.encode_with(&mut scratch, device, queue, cmd, tiles, srgb, None)
    }

    /// [`GpuEncoder::encode`], for a caller that spreads encodes over frames:
    /// the buffers live in `scratch` from one call to the next, and the
    /// compute pass writes `timestamp_writes` when given.
    ///
    /// `scratch` grows when a call needs more than it holds and never
    /// shrinks; [`EncoderScratch::with_capacity`] sizes it up front. Its
    /// buffers are rewritten by every call, through `queue.write_buffer`, which
    /// lands at the next submission: **submit `cmd` before the next call with
    /// the same scratch**, or that call's texels replace these before they are
    /// encoded. One call per frame, submitted with the frame, is the intended
    /// use.
    ///
    /// `timestamp_writes` needs [`wgpu::Features::TIMESTAMP_QUERY`] on the
    /// device. No pass is recorded when no tile is valid, and the query set is
    /// then left as it was.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_with(
        &self,
        scratch: &mut EncoderScratch,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cmd: &mut wgpu::CommandEncoder,
        tiles: &[&[Level<'_>]],
        srgb: bool,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Vec<Result<wgpu::Texture, EncodeError>> {
        let mut results: Vec<Result<wgpu::Texture, EncodeError>> = Vec::with_capacity(tiles.len());
        // Tiles go into batches whose input and output each fit one storage
        // binding; every batch is dispatched in the same compute pass.
        let mut batches: Vec<Batch<'_>> = Vec::new();
        let mut current = Batch::default();
        for (t, levels) in tiles.iter().enumerate() {
            if let Err(e) = validate(levels) {
                results.push(Err(e));
                continue;
            }
            let (src_bytes, dst_bytes) = tile_bytes(levels);
            let largest = src_bytes.max(dst_bytes);
            if largest > self.max_binding {
                results.push(Err(EncodeError::TooLarge {
                    bytes: largest,
                    limit: self.max_binding,
                }));
                continue;
            }
            if !current.placed.is_empty()
                && (current.src_bytes + src_bytes > self.max_binding
                    || current.dst_bytes + dst_bytes > self.max_binding)
            {
                batches.push(std::mem::take(&mut current));
            }
            current.add(t, levels);
            let (width, height) = (levels[0].width, levels[0].height);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tuile imagery (block-compressed)"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: levels.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: texture_format(self.format, srgb),
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            results.push(Ok(texture));
        }
        if !current.placed.is_empty() {
            batches.push(current);
        }
        if batches.is_empty() {
            return results;
        }

        let needs: Vec<Need> = batches
            .iter()
            .map(|b| Need {
                src: b.src_bytes,
                dst: b.dst_bytes,
                jobs: b.placed.len() as u64 * self.uniform_align,
            })
            .collect();
        scratch.fit(device, self, &needs);
        for (batch, slot) in batches.iter().zip(&scratch.slots) {
            self.upload(queue, batch, slot);
        }
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tuile block encode"),
                timestamp_writes,
            });
            pass.set_pipeline(&self.pipeline);
            for (batch, slot) in batches.iter().zip(&scratch.slots) {
                for (j, placed) in batch.placed.iter().enumerate() {
                    let offset = (j as u64 * self.uniform_align) as u32;
                    pass.set_bind_group(0, &slot.bind_group, &[offset]);
                    pass.dispatch_workgroups(
                        placed.job.blocks_x.div_ceil(WORKGROUP),
                        placed.job.blocks_y.div_ceil(WORKGROUP),
                        1,
                    );
                }
            }
        }
        for (batch, slot) in batches.iter().zip(&scratch.slots) {
            for placed in &batch.placed {
                let Ok(texture) = &results[placed.tile] else {
                    continue;
                };
                cmd.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &slot.output,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: placed.offset,
                            bytes_per_row: Some(placed.row_bytes),
                            rows_per_image: Some(placed.job.blocks_y),
                        },
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: placed.level,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    // A level smaller than a block is copied at its
                    // physical (block-rounded) size.
                    wgpu::Extent3d {
                        width: placed.job.blocks_x * BlockFormat::BLOCK_DIM,
                        height: placed.job.blocks_y * BlockFormat::BLOCK_DIM,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        results
    }

    /// Writes one batch's texels and jobs into its slot.
    fn upload(&self, queue: &wgpu::Queue, batch: &Batch<'_>, slot: &Slot) {
        let mut at = 0u64;
        for level in &batch.levels {
            queue.write_buffer(&slot.input, at, level);
            at += level.len() as u64;
        }
        let stride = self.uniform_align as usize;
        let mut jobs = vec![0u8; stride * batch.placed.len()];
        for (j, placed) in batch.placed.iter().enumerate() {
            jobs[j * stride..j * stride + std::mem::size_of::<Job>()]
                .copy_from_slice(bytemuck::bytes_of(&placed.job));
        }
        queue.write_buffer(&slot.jobs, 0, &jobs);
    }
}

/// The encoder's buffers, kept from one [`GpuEncoder::encode_with`] call to
/// the next so that encoding a tile allocates nothing but its texture.
///
/// It holds one input, one output and one job buffer per storage binding a
/// call fills (one for a few dozen 256² tiles), grows when a call needs more,
/// and never shrinks. [`EncoderScratch::bytes`] is the GPU memory it holds.
/// Built for one encoder, it serves any encoder of the same device.
#[derive(Default)]
pub struct EncoderScratch {
    slots: Vec<Slot>,
}

impl EncoderScratch {
    /// An empty scratch: the first call allocates what it needs.
    pub fn new() -> Self {
        Self::default()
    }

    /// A scratch that holds, up front, what one call encoding `tiles` tiles of
    /// `width`×`height` with their full mip chains needs: such calls allocate
    /// no buffer. A tile larger than the device's storage binding gets no
    /// room, as [`GpuEncoder::encode_with`] refuses it.
    pub fn with_capacity(
        device: &wgpu::Device,
        encoder: &GpuEncoder,
        tiles: u32,
        width: u32,
        height: u32,
    ) -> Self {
        let mut scratch = Self::new();
        let (src, dst, levels) = chain_io(width, height);
        let largest = src.max(dst);
        if tiles == 0 || largest > encoder.max_binding {
            return scratch;
        }
        // The same packing as `encode_with`'s batches.
        let per_slot = (encoder.max_binding / largest).min(tiles as u64);
        let mut left = tiles as u64;
        let mut needs = Vec::new();
        while left > 0 {
            let n = left.min(per_slot);
            needs.push(Need {
                src: n * src,
                dst: n * dst,
                jobs: n * levels * encoder.uniform_align,
            });
            left -= n;
        }
        scratch.fit(device, encoder, &needs);
        scratch
    }

    /// The GPU bytes the scratch holds.
    pub fn bytes(&self) -> u64 {
        self.slots
            .iter()
            .map(|s| s.input.size() + s.output.size() + s.jobs.size())
            .sum()
    }

    /// Makes slot `i` hold `needs[i]`, for `encoder`'s bind group layout.
    fn fit(&mut self, device: &wgpu::Device, encoder: &GpuEncoder, needs: &[Need]) {
        for (i, need) in needs.iter().enumerate() {
            match self.slots.get_mut(i) {
                Some(slot) if slot.holds(need) => {
                    if slot.layout != encoder.layout {
                        slot.bind_group = Slot::bind(
                            device,
                            &encoder.layout,
                            &slot.input,
                            &slot.output,
                            &slot.jobs,
                        );
                        slot.layout = encoder.layout.clone();
                    }
                }
                Some(slot) => {
                    // Grown to the larger of what it held and what is needed,
                    // so sizes that alternate settle.
                    let grown = Need {
                        src: need.src.max(slot.input.size()),
                        dst: need.dst.max(slot.output.size()),
                        jobs: need.jobs.max(slot.jobs.size()),
                    };
                    *slot = Slot::new(device, encoder, &grown);
                }
                None => self.slots.push(Slot::new(device, encoder, need)),
            }
        }
    }
}

/// What one batch needs, in bytes.
struct Need {
    src: u64,
    dst: u64,
    jobs: u64,
}

/// One batch's buffers and the bind group over them.
struct Slot {
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    jobs: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// The layout `bind_group` was made for.
    layout: wgpu::BindGroupLayout,
}

impl Slot {
    fn new(device: &wgpu::Device, encoder: &GpuEncoder, need: &Need) -> Self {
        let input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tuile block encoder texels"),
            size: need.src.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tuile block encoder blocks"),
            size: need.dst.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let jobs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tuile block encoder jobs"),
            size: need.jobs.max(encoder.uniform_align),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = Self::bind(device, &encoder.layout, &input, &output, &jobs);
        Self {
            input,
            output,
            jobs,
            bind_group,
            layout: encoder.layout.clone(),
        }
    }

    fn holds(&self, need: &Need) -> bool {
        self.input.size() >= need.src
            && self.output.size() >= need.dst
            && self.jobs.size() >= need.jobs
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        jobs: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tuile block encoder"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: jobs,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<Job>() as u64),
                    }),
                },
            ],
        })
    }
}

/// Tiles packed into one input and one output buffer.
#[derive(Default)]
struct Batch<'a> {
    placed: Vec<Placed>,
    levels: Vec<&'a [u8]>,
    src_bytes: u64,
    dst_bytes: u64,
}

impl<'a> Batch<'a> {
    fn add(&mut self, tile: usize, levels: &[Level<'a>]) {
        for (l, level) in levels.iter().enumerate() {
            let (blocks_x, blocks_y, row_bytes) = level_layout(level.width, level.height);
            self.placed.push(Placed {
                tile,
                level: l as u32,
                job: Job {
                    src: (self.src_bytes / 4) as u32,
                    width: level.width,
                    height: level.height,
                    dst: (self.dst_bytes / 4) as u32,
                    row_words: row_bytes / 4,
                    blocks_x,
                    blocks_y,
                    _pad: 0,
                },
                offset: self.dst_bytes,
                row_bytes,
            });
            self.levels.push(level.rgba8);
            self.src_bytes += level.rgba8.len() as u64;
            self.dst_bytes += row_bytes as u64 * blocks_y as u64;
        }
    }
}

/// Blocks across, blocks down, and the padded bytes of a row of blocks in
/// the output.
fn level_layout(width: u32, height: u32) -> (u32, u32, u32) {
    let blocks_x = width.div_ceil(BlockFormat::BLOCK_DIM);
    let blocks_y = height.div_ceil(BlockFormat::BLOCK_DIM);
    let row_bytes = (blocks_x * BlockFormat::BLOCK_BYTES).next_multiple_of(ROW_ALIGN);
    (blocks_x, blocks_y, row_bytes)
}

/// Input bytes, output bytes and levels of a `width`×`height` chain down to
/// 1×1, as the encoder lays them out.
fn chain_io(width: u32, height: u32) -> (u64, u64, u64) {
    let (mut w, mut h) = (width.max(1), height.max(1));
    let (mut src, mut dst, mut levels) = (0, 0, 0);
    loop {
        let (_, blocks_y, row_bytes) = level_layout(w, h);
        src += w as u64 * h as u64 * 4;
        dst += row_bytes as u64 * blocks_y as u64;
        levels += 1;
        if w == 1 && h == 1 {
            return (src, dst, levels);
        }
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
}

/// Input and output bytes of one tile's chain.
fn tile_bytes(levels: &[Level<'_>]) -> (u64, u64) {
    levels.iter().fold((0, 0), |(s, d), level| {
        let (_, blocks_y, row_bytes) = level_layout(level.width, level.height);
        (
            s + level.rgba8.len() as u64,
            d + row_bytes as u64 * blocks_y as u64,
        )
    })
}

fn validate(levels: &[Level<'_>]) -> Result<(), EncodeError> {
    let Some(first) = levels.first() else {
        return Err(EncodeError::Empty);
    };
    if first.width == 0
        || first.height == 0
        || first.width % BlockFormat::BLOCK_DIM != 0
        || first.height % BlockFormat::BLOCK_DIM != 0
    {
        return Err(EncodeError::NotBlockAligned {
            width: first.width,
            height: first.height,
        });
    }
    for (l, level) in levels.iter().enumerate() {
        let want = ((first.width >> l).max(1), (first.height >> l).max(1));
        if (level.width, level.height) != want {
            return Err(EncodeError::NotAMipChain {
                level: l,
                got: (level.width, level.height),
                want,
            });
        }
        let bytes = (level.width * level.height * 4) as usize;
        if level.rgba8.len() != bytes {
            return Err(EncodeError::LevelBytes {
                level: l,
                got: level.rgba8.len(),
                want: bytes,
            });
        }
    }
    Ok(())
}

/// The format's kernel: the shared part, the generated tables, the format.
fn shader_source(format: BlockFormat) -> String {
    let common = include_str!("common.wgsl");
    match format {
        BlockFormat::Bc7 => format!("{common}\n{}", include_str!("bc7.wgsl")),
        BlockFormat::Astc4x4 => {
            format!("{common}\n{}\n{}", astc_tables(), include_str!("astc.wgsl"))
        }
    }
}

/// `TRIT_ENC`, `Q12` and `Q6` for `astc.wgsl`, as WGSL constants.
fn astc_tables() -> String {
    let list = |v: &[u32]| {
        v.iter()
            .map(|x| format!("{x}u"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let floats = |v: &[u32]| {
        v.iter()
            .map(|x| format!("{x}.0"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let q12: Vec<u32> = (0..12)
        .map(|v| crate::astc::unquantize_weight(v, 1, 2))
        .collect();
    let q6: Vec<u32> = (0..6)
        .map(|v| crate::astc::unquantize_weight(v, 1, 1))
        .collect();
    format!(
        "const TRIT_ENC = array<u32, 243>({});\nconst Q12 = array<f32, 12>({});\nconst Q6 = array<f32, 6>({});\n",
        list(&crate::astc::trit_encodings()),
        floats(&q12),
        floats(&q6),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astc::*;

    /// **Every trit quintuple has an encoding, and it decodes back.**
    #[test]
    fn every_trit_quintuple_round_trips() {
        let table = trit_encodings();
        for (key, &t) in table.iter().enumerate() {
            assert!(t < 256, "quintuple {key} has no encoding");
            let [t0, t1, t2, t3, t4] = decode_trits(t);
            assert_eq!((t0 + 3 * t1 + 9 * t2 + 27 * t3 + 81 * t4) as usize, key);
        }
        // A partial last group: only T0 and T1 may be set.
        for (t0, &t) in table.iter().take(3).enumerate() {
            assert_eq!(t, t0 as u32, "(t0, 0, 0, 0, 0) is t0 itself");
        }
    }

    /// **The weight ranges are the spec's**, as the published unquantisation
    /// tables list them.
    #[test]
    fn weight_unquantisation_matches_the_spec() {
        let q12: Vec<u32> = (0..12).map(|v| unquantize_weight(v, 1, 2)).collect();
        assert_eq!(q12, [0, 64, 17, 47, 5, 59, 23, 41, 11, 53, 28, 36]);
        let q6: Vec<u32> = (0..6).map(|v| unquantize_weight(v, 1, 1)).collect();
        assert_eq!(q6, [0, 64, 12, 52, 25, 39]);
    }

    /// **The block modes say what the kernel assumes**: a 4×4 grid, one
    /// plane, and the intended weight range (Table 146/147).
    #[test]
    fn block_modes_decode_to_the_intended_grid_and_range() {
        // (mode, P, ρ)
        for (mode, p, rho) in [(0x251u32, 1, 0b011), (0x043, 0, 0b110)] {
            assert_ne!(mode & 3, 0, "the first five rows of Table 147");
            assert_eq!((mode >> 2) & 3, 0, "row W+4, H+2");
            assert_eq!(((mode >> 7) & 3) + 4, 4, "width");
            assert_eq!(((mode >> 5) & 3) + 2, 4, "height");
            assert_eq!((mode >> 10) & 1, 0, "single plane");
            assert_eq!((mode >> 9) & 1, p);
            let r = ((mode >> 4) & 1) | ((mode & 3) << 1);
            assert_eq!(r, rho);
        }
    }

    /// **The shaders are valid WGSL** for every format, on the validator wgpu
    /// runs, with the capabilities a compute device has.
    #[test]
    fn every_kernel_validates() {
        for format in BlockFormat::PREFERENCE {
            let source = shader_source(format);
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{format:?}: {}", e.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{format:?}: {}", e.emit_to_string(&source)));
        }
    }

    /// **A broken tile is refused before it reaches the GPU.**
    #[test]
    fn invalid_chains_are_refused() {
        let px = vec![0u8; 8 * 8 * 4];
        let half = vec![0u8; 4 * 4 * 4];
        let ok = [
            Level {
                width: 8,
                height: 8,
                rgba8: &px,
            },
            Level {
                width: 4,
                height: 4,
                rgba8: &half,
            },
        ];
        assert_eq!(validate(&ok), Ok(()));
        assert_eq!(validate(&[]), Err(EncodeError::Empty));
        let odd = vec![0u8; 6 * 6 * 4];
        assert!(matches!(
            validate(&[Level {
                width: 6,
                height: 6,
                rgba8: &odd
            }]),
            Err(EncodeError::NotBlockAligned { .. })
        ));
        assert!(matches!(
            validate(&[ok[0], ok[0]]),
            Err(EncodeError::NotAMipChain { level: 1, .. })
        ));
        assert!(matches!(
            validate(&[Level {
                width: 8,
                height: 8,
                rgba8: &half
            }]),
            Err(EncodeError::LevelBytes { .. })
        ));
    }

    /// **Each compute capability the kernels lean on is a gate**, and WebGL2
    /// (no compute) never passes.
    #[test]
    fn every_compute_limit_is_a_gate() {
        let all = wgpu::DownlevelFlags::all();
        let base = wgpu::Limits::default();
        assert!(runs_on(all, &base), "WebGPU's guaranteed limits suffice");
        assert!(runs_on(all, &wgpu::Limits::downlevel_defaults()));
        assert!(!runs_on(
            wgpu::DownlevelFlags::empty(),
            &wgpu::Limits::downlevel_webgl2_defaults()
        ));
        assert!(!runs_on(wgpu::DownlevelFlags::empty(), &base), "no compute");
        let squeezes: [fn(&mut wgpu::Limits); 6] = [
            |l| l.max_storage_buffers_per_shader_stage = 1,
            |l| l.max_compute_workgroup_size_x = 4,
            |l| l.max_compute_workgroup_size_y = 4,
            |l| l.max_compute_invocations_per_workgroup = 32,
            |l| l.max_compute_workgroups_per_dimension = 512,
            |l| l.max_storage_buffer_binding_size = 1 << 20,
        ];
        for (i, squeeze) in squeezes.iter().enumerate() {
            let mut limits = base.clone();
            squeeze(&mut limits);
            assert!(!runs_on(all, &limits), "limit {i} must gate the kernels");
        }
    }

    /// **The wgpu formats are 4×4 blocks of 16 bytes**, sRGB and linear
    /// variants of each other, behind the feature named for them.
    #[test]
    fn formats_map_to_wgpu() {
        for f in BlockFormat::PREFERENCE {
            let srgb = texture_format(f, true);
            assert!(srgb.is_compressed());
            assert!(srgb.is_srgb());
            assert_eq!(srgb.block_dimensions(), (4, 4));
            assert_eq!(srgb.block_copy_size(None), Some(BlockFormat::BLOCK_BYTES));
            assert_eq!(texture_format(f, false).add_srgb_suffix(), srgb);
            assert_eq!(srgb.required_features(), feature(f));
        }
    }
}
