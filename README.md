<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) lapoule.dev -->

# texcomp

GPU block-compressed textures, encoded on the fly from RGBA8, in pure Rust.

Imagery and other textures usually arrive as JPEG or PNG and are decoded to
RGBA8. texcomp encodes them, just before upload, to a block format the GPU
samples natively: 8 bits per texel instead of 32, so a quarter of the bytes
stay resident. Compressed textures can also be stored and served as files:
`texcomp::ktx2` reads and writes them in the Khronos KTX 2.0 container.

It is built for [tuile](https://github.com/lapoule-dev/tuile)'s viewer, and
does not depend on it.

## Formats

Both formats are 4×4 texels in 16 bytes.

| Format | Sampled natively by |
|---|---|
| **BC7** | desktop GPUs, desktop WebGPU |
| **ASTC 4×4** (LDR) | phones (iOS, Android), the Quest, mobile WebGPU and WebGL2 |

Apple silicon samples both. `BlockFormat::PREFERENCE` lists them best first;
which one a device gets is the application's choice, made from what the
device reports.

## Two paths

- **On the GPU** (feature `gpu`, wgpu 30): WGSL compute kernels write BC7 and
  ASTC 4×4 blocks from RGBA8 mip levels, and `GpuEncoder` dispatches them. One
  call records one compute pass and the copies into the caller's command
  encoder. `gpu::runs_on` says whether a device can run the kernels. BC7 uses
  all eight of its modes; ASTC one or two partitions, one or two weight
  planes, four colour endpoint modes and weight ranges up to 0..31. How hard
  they search is a `gpu::Quality`, numbers compiled into the kernel.
- **On the CPU** (default build: `thiserror`, and `ruzstd` and `miniz_oxide`
  for KTX 2.0 supercompression, all pure Rust;
  `wasm32-unknown-unknown` clean): a port of the block layer of
  [Basis Universal](https://github.com/BinomialLLC/basis_universal) v2.50, for
  devices without compute shaders, such as WebGL2. UASTC LDR 4×4 is its
  intermediate: `texcomp::uastc` encodes RGBA8 to it
  (`UastcBlock::encode`, `encode_image`), decodes it and transcodes it to
  ASTC 4×4 and BC7. Every block is byte-identical to upstream's, checked
  against vectors captured from upstream (`tests/uastc_encode.rs`,
  `tests/uastc_reference.rs`), at upstream's effort levels 0, 1 and 2.

## Quality and cost

Measured by `examples/quality-report.rs` on Apple M2 (Metal), 23 Nov. 2026:
six 256² IGN aerial orthophoto tiles (`tests/fixtures/aerial`, Licence
Ouverte 2.0) and twenty 256² tiles cut from NASA imagery
(`tests/fixtures/imagery`, public domain), each encoded with its full mip
chain. The blocks are read back and decoded by a software decoder
independent of this crate (`texture2ddecoder`), checked against the GPU's
own decode. PSNR over RGB and over luma (BT.601) and SSIM over luma are
for level 0 of each tile, averaged; "worst" is the worst tile. GPU time is
the compute pass's, from timestamps, per 256² tile with its mips (wall time
to completion is within 1 ms of it).

Aerial (IGN, 6 tiles):

| Format, `Quality::level` | PSNR RGB | PSNR Y | SSIM Y | worst PSNR RGB | GPU ms / tile |
|---|---|---|---|---|---|
| BC7, 0 (the 0.3 search) | 45.85 | 47.83 | 0.9982 | 43.19 | 4.2 |
| BC7, **1 (default)** | 45.95 | 47.91 | 0.9982 | 43.29 | 12.1 |
| BC7, 2 | 46.00 | 47.95 | 0.9983 | 43.36 | 32.0 |
| BC7, 4 | 46.10 | 48.04 | 0.9983 | 43.49 | 144.1 |
| ASTC 4×4, 0 (the 0.3 search) | 43.97 | 45.30 | 0.9974 | 40.88 | 1.4 |
| ASTC 4×4, **1 (default)** | 46.79 | 50.22 | 0.9988 | 44.29 | 26.6 |
| ASTC 4×4, 2 | 46.80 | 50.20 | 0.9988 | 44.30 | 36.0 |
| ASTC 4×4, 4 | 46.81 | 50.20 | 0.9988 | 44.31 | 100.4 |

NASA (20 tiles):

| Format, `Quality::level` | PSNR RGB | PSNR Y | SSIM Y | worst PSNR RGB | GPU ms / tile |
|---|---|---|---|---|---|
| BC7, 0 | 48.00 | 50.28 | 0.9983 | 42.23 | 4.8 |
| BC7, **1** | 48.13 | 50.33 | 0.9983 | 42.58 | 12.2 |
| BC7, 2 | 48.39 | 50.54 | 0.9983 | 43.84 | 28.3 |
| BC7, 4 | 48.45 | 50.56 | 0.9983 | 44.00 | 113.3 |
| ASTC 4×4, 0 | 47.14 | 49.90 | 0.9982 | 36.40 | 1.0 |
| ASTC 4×4, **1** | 49.65 | 52.49 | 0.9987 | 41.64 | 25.4 |
| ASTC 4×4, 2 | 49.74 | 52.61 | 0.9987 | 42.03 | 35.5 |
| ASTC 4×4, 4 | 49.78 | 52.64 | 0.9987 | 42.17 | 91.4 |

- **ASTC gains the most**: +2.8 dB RGB and +4.9 dB luma on aerial imagery
  at level 1, +5.2 dB on the worst tile (night lights, which two partitions
  fit). The extra weight ranges and base + scale endpoints carry most of it;
  two partitions and two planes the hard blocks.
- **BC7 was already near what its block allows** with modes 6 and 1: the
  other six modes add 0.1 to 0.5 dB on average and up to 1.6 dB on the
  worst tile, at 3 to 30 times the GPU time.
- Level 1 is the default: the bulk of the ASTC gain, and a BC7 cost a frame
  budget still absorbs. Level 2 adds three-subset BC7 modes and more ASTC
  partitioned configurations for the hardest tiles.
- For comparison, the CPU path (UASTC at upstream's effort 0, transcoded)
  scores 44.3 dB (BC7) and 45.1 dB (ASTC) on the aerial tiles, at 70 ms of
  CPU per tile.
- `tests/gpu_oracle.rs` holds every level to valid blocks (oracle and GPU
  decodes agree), opaque blocks to alpha 255, every tile to at least level
  0's PSNR, and structured random blocks to stated error bounds;
  `tests/gpu_quality.rs` holds the default to its quality gate. Without a
  GPU they skip; `TEXCOMP_REQUIRE_GPU=1` makes that a failure.

## Usage

```toml
[dependencies]
texcomp = { git = "https://github.com/lapoule-dev/texcomp", features = ["gpu"] }
```

```rust
use texcomp::gpu::{feature, runs_on, GpuEncoder, Level};
use texcomp::BlockFormat;

/// Makes one RGBA8 tile resident as blocks. `None`: upload it as RGBA8.
fn make_resident(
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    chain: &[Level<'_>],
) -> Option<wgpu::Texture> {
    let downlevel = adapter.get_downlevel_capabilities().flags;
    if !runs_on(downlevel, &device.limits()) {
        return None;
    }
    // The device must have been requested with the format's feature.
    let format = BlockFormat::PREFERENCE
        .into_iter()
        .find(|f| device.features().contains(feature(*f)))?;
    // Built once per device and format in a real application.
    let encoder = GpuEncoder::new(device, format);
    let mut cmd = device.create_command_encoder(&Default::default());
    let mut textures = encoder.encode(device, queue, &mut cmd, &[chain], true);
    queue.submit([cmd.finish()]);
    textures.pop()?.ok()
}
```

`chain` is level 0 and its mips down to 1×1, each tightly packed RGBA8. Level
0 must be a multiple of 4 on both sides. A tile that cannot be encoded comes
back as an `EncodeError`: the caller uploads it uncompressed, never drops it.

`GpuEncoder::new` compiles the kernel at `Quality::DEFAULT`;
`GpuEncoder::with_quality` takes another, such as `Quality::level(2)` or one
written field by field:

```rust
use texcomp::gpu::{GpuEncoder, Quality};
use texcomp::BlockFormat;

fn encoders(device: &wgpu::Device) -> (GpuEncoder, GpuEncoder) {
    // Phones: two partitions too, worth their cost on ASTC.
    let astc = GpuEncoder::with_quality(device, BlockFormat::Astc4x4, Quality::level(2));
    // Desktops: the default.
    let bc7 = GpuEncoder::new(device, BlockFormat::Bc7);
    assert_eq!(bc7.quality(), Quality::DEFAULT);
    (astc, bc7)
}
```

## Spreading encodes over frames

An application that encodes while it renders paces the work, a few tiles per
frame. Three things help:

- **`tile_cost(format, width, height)`** (default build): the blocks, the
  resident bytes and the RGBA8 bytes read for one tile with its mips, before
  anything runs. A 256² tile is 5,463 blocks, 87,408 bytes resident.
- **`EncoderScratch`**: `GpuEncoder::encode_with` keeps the input, output and
  job buffers in it from one call to the next, so a call allocates nothing but
  its textures. `EncoderScratch::with_capacity` sizes it for so many tiles of
  a given size; it grows when a call needs more and never shrinks. Its
  buffers are rewritten by each call, through `queue.write_buffer`, so the
  command buffer of one call must be submitted before the next call with the
  same scratch.
- **Timestamps**: `encode_with` takes an optional
  `wgpu::ComputePassTimestampWrites`, written by the encode pass, for a
  device with `TIMESTAMP_QUERY`.

```rust
use texcomp::gpu::{EncoderScratch, GpuEncoder, Level};

/// One frame's share of the encode work, timed on the GPU.
fn encode_some(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &GpuEncoder,
    scratch: &mut EncoderScratch,
    queries: &wgpu::QuerySet,
    tiles: &[&[Level<'_>]],
) -> Vec<Option<wgpu::Texture>> {
    let mut cmd = device.create_command_encoder(&Default::default());
    let out = encoder.encode_with(
        scratch,
        device,
        queue,
        &mut cmd,
        tiles,
        true,
        Some(wgpu::ComputePassTimestampWrites {
            query_set: queries,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }),
    );
    // Resolve `queries` here, then submit before the next call.
    queue.submit([cmd.finish()]);
    out.into_iter().map(Result::ok).collect()
}
```

`tests/gpu_pacing.rs` checks that a sized scratch neither grows nor
reallocates call after call, that a call through a scratch decodes texel for
texel like a fresh `encode`, and that the timestamps are written in order.
The GPU time per tile of each format and quality level is in the table of
[Quality and cost](#quality-and-cost).

## Encoding into your own texture, from the GPU

`GpuEncoder::encode` makes a new texture per tile from texels in memory.
`GpuEncoder::encode_into` (and `encode_into_with`, its scratch and timestamp
form) writes the blocks into **textures the caller keeps**, at a mip level, a
block-aligned origin and an array layer it names, such as one layer of a
block-compressed texture array. Each level's texels can come from memory or
**from the GPU**:

- `Source::Cpu(&[u8])`: tightly packed RGBA8, uploaded as `encode` does;
- `Source::Buffer { buffer, offset, bytes_per_row }`: RGBA8 in a buffer with
  `COPY_SRC`, rows any multiple of 4 bytes apart;
- `Source::Texture { texture, mip_level, origin }`: a region of an
  `Rgba8Unorm` or `Rgba8UnormSrgb` texture (`origin.z` is the array layer)
  with `COPY_SRC`, such as a layer a compute shader just wrote.

GPU texels are copied into the encoder's input by the same command encoder,
just ahead of the compute pass, and the blocks are copied into the
destinations just after it: one submission, no readback, and what the GPU
wrote earlier in that command encoder is what gets encoded. The kernels bind
no texture. The blocks are those `encode` writes for the same texels, from
any source.

Every source and destination is checked before anything is recorded: a
destination must be in the encoder's format (sRGB or linear), have
`COPY_DST`, and hold the level's blocks inside its mip level and layer. A
tile that fails gets its `EncodeError` (`DestinationFormat`,
`DestinationRegion`, `SourceUsage`, …), records nothing, and leaves its
destinations as they were. A tile of one level is level 0 alone, without mips.

```rust
use texcomp::gpu::{Destination, GpuEncoder, Source, SourceLevel, TileInto};

/// Compresses layer `from` of an RGBA8 array the GPU just drew (256², with
/// `COPY_SRC`) into layer `to` of a BC7 or ASTC array (with `COPY_DST`).
fn compress_layer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &GpuEncoder,
    cmd: &mut wgpu::CommandEncoder,
    drawn: &wgpu::Texture,
    from: u32,
    cache: &wgpu::Texture,
    to: u32,
) -> bool {
    let levels = [SourceLevel {
        width: 256,
        height: 256,
        source: Source::Texture {
            texture: drawn,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: from },
        },
    }];
    let destinations = [Destination {
        texture: cache,
        mip_level: 0,
        origin: wgpu::Origin3d { x: 0, y: 0, z: to },
    }];
    let tile = TileInto {
        levels: &levels,
        destinations: &destinations,
    };
    // Recorded after the pass that draws `drawn`, before the one that samples
    // `cache`, in the same command encoder.
    encoder.encode_into(device, queue, cmd, &[tile])[0].is_ok()
}
```

`tests/gpu_into.rs` checks that memory, a padded buffer and a texture layer
encode to the same blocks, which decode like `encode`'s; that encoding into
layer N at an origin writes that region and nothing else; that imagery taken
from a texture layer holds `gpu_quality`'s gate; and that every bad source or
destination is refused without a device error.

## KTX 2.0 files

`texcomp::ktx2` (default build, wasm-clean) reads and writes KTX 2.0 files,
written from the Khronos KTX 2.0 and Data Format 1.4 specifications. All of
what it declares is implemented, both ways:

- **formats**: UASTC (`vkFormat` UNDEFINED with a UASTC data format
  descriptor, every channel id), **BC7**, **ASTC 4×4** and **RGBA8**, linear
  or sRGB, each with its descriptor sample by sample, premultiplied alpha
  included;
- **texture types**: 1D (RGBA8), 2D, 3D, arrays, cubemaps and cubemap
  arrays, any number of mips, or `levelCount` 0 (`generate_mips`, RGBA8);
  `Texture::image` cuts one layer, face and depth slice out of a level;
- **supercompression**: none, **Zstandard** and **ZLIB**, level by level
  (`Supercompression`, pure Rust: `ruzstd` and `miniz_oxide`);
- **key/value data**: `KTXorientation` (`r`, `rd`, `rdi`: a file whose texels
  do not start at the top left is refused, since read as stored it would
  come out flipped), `KTXwriter`, and every other pair kept as it is
  (`key_values`), written sorted.

What it does not read is refused with an error that names it: ETC1S and its
BasisLZ supercompression, 3D arrays (which KTX 2.0 does not define), and
other `vkFormat`s.

```rust
use texcomp::ktx2::{self, Format, Supercompression, Texture, UastcChannels};

/// A UASTC tile served as a file, then read back and made BC7.
fn round_trip(levels: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    // Level 0 first, each its 16-byte blocks row by row.
    let mut tile = Texture::new(Format::Uastc(UastcChannels::Rgb), true, 256, 256, levels);
    tile.supercompression = Supercompression::Zstd;
    let file = ktx2::write(&tile)?; // served as ktx2::MEDIA_TYPE, image/ktx2
    let back = ktx2::read(&file)?;
    let mut bc7 = Vec::new();
    for level in &back.levels {
        let mut out = vec![0; level.len()];
        texcomp::uastc::transcode_to_bc7(level, &mut out)?;
        bc7.push(out);
    }
    Ok(bc7)
}
```

The reader validates every size and offset against the file and never
panics (`tests/ktx2.rs` and `tests/ktx2_complete.rs` truncate and corrupt
files by the thousand, supercompressed ones included). Three outside
parties check it:

- an independent reader, the `ktx2` crate, reads every file the writer
  makes, for every type, format and scheme, to the same header, levels,
  descriptor and pairs;
- KTX-Software's own files (RGBA8 2D, array with mips, 3D, UASTC with
  Zstandard; `tests/fixtures/ktx2/khronos`) read to their shapes and texels,
  and upstream Basis Universal's files transcode bit-exact;
- with `TEXCOMP_KTX` naming the Khronos `ktx` tool (KTX-Software 4.4.2),
  `ktx validate` passes every file written, `ktx extract` (libktx, the
  reference Zstandard and ZLIB) reads every image of them as `Texture::image`
  holds it, and libktx's Basis transcoder turns real UASTC blocks into the
  BC7 and ASTC blocks this crate transcodes to. `TEXCOMP_REQUIRE_KTX=1`
  makes the tool's absence a failure.

## Features

| Feature | Default | What it adds |
|---|---|---|
| `gpu` | no | the WGSL compute encoder and its dispatch, on wgpu 30 (no default wgpu features: the application picks its backends) |

Without `gpu` the crate depends on nothing GPU-related.

## Minimum Rust version

1.97.

## Status

- **Done**: the GPU encoders (BC7 in all eight modes, ASTC 4×4 with one or
  two partitions and planes), `Quality`, their quality gate and oracle, the
  pacing API (`tile_cost`, `EncoderScratch`, pass timestamps), and
  `encode_into` (caller's textures and layers, sources on the GPU); on the CPU,
  the UASTC encoder at effort levels 0–2, UASTC decode and UASTC to ASTC 4×4
  and BC7, all byte-identical to upstream; KTX 2.0 read and write, the
  whole container (every texture type, UASTC, BC7, ASTC 4×4, RGBA8,
  Zstandard and ZLIB).
- **Not done**: ETC1S/BasisLZ in KTX 2.0 (refused).

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. Copyright lapoule.dev. Third-party attributions are in [`NOTICE`](./NOTICE).

The files ported from Basis Universal (Apache-2.0, © Binomial LLC) are
offered under the same terms. They carry the upstream attribution and a notice
of modification at the head of each ported item; see `NOTICE` and
`LICENSE-APACHE-UPSTREAM`. "Basis Universal" is a trademark of Binomial LLC,
named here only to identify the project this code is ported from.
