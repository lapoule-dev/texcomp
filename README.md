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
  encoder. `gpu::runs_on` says whether a device can run the kernels.
- **On the CPU** (default build, `std` and `thiserror` only,
  `wasm32-unknown-unknown` clean): a port of the block layer of
  [Basis Universal](https://github.com/BinomialLLC/basis_universal) v2.50, for
  devices without compute shaders, such as WebGL2. UASTC LDR 4×4 is its
  intermediate: `texcomp::uastc` encodes RGBA8 to it
  (`UastcBlock::encode`, `encode_image`), decodes it and transcodes it to
  ASTC 4×4 and BC7. Every block is byte-identical to upstream's, checked
  against vectors captured from upstream (`tests/uastc_encode.rs`,
  `tests/uastc_reference.rs`), at upstream's effort levels 0, 1 and 2.

## Quality and cost

Measured by `tests/gpu_quality.rs` on Apple M2 (Metal). Twenty 256² tiles are
cut from five NASA imagery tiles (`tests/fixtures/imagery`, public domain) and
encoded with their full mip chains. The GPU's own block decoder reads them
back, and they are scored per tile and per level.

| Format | Worst PSNR | Worst ΔE2000 mean | Worst ΔE2000 p99 | Encode, per 256² tile with mips |
|---|---|---|---|---|
| BC7 | 40.3 dB | 2.09 | 5.35 | 5.1 ms |
| ASTC 4×4 | 36.4 dB (night lights) | 2.43 | 9.40 | 1.9 ms |

- Typical levels on imagery score 43–50 dB. The worst ASTC tile is night
  lights: bright points on black, which one partition per block fits worst.
- No texel anywhere decodes to a hole (black or transparent where the source
  draws).
- The test holds every tile to a gate set just beyond these worst values, so
  a regression fails it. `TEXCOMP_REQUIRE_GPU=1` turns its skip on a machine
  without a GPU into a failure.
- Encode time is wall clock (upload, encode and copy, to the GPU's
  completion) in a release build, as `encode_time_is_reported` prints it.

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
texel like a fresh `encode`, and that the timestamps are written in order. On
Apple M2 its BC7 pass (four 256² tiles with mips, first pass, no warm-up)
measured 5 to 7 ms per tile across runs.

## KTX 2.0 files

`texcomp::ktx2` (default build, wasm-clean) reads and writes a KTX 2.0 file
holding one 2D texture with its mips, written from the Khronos KTX 2.0 and
Data Format 1.4 specifications:

- **UASTC** (`vkFormat` UNDEFINED with a UASTC data format descriptor),
  **BC7** and **ASTC 4×4**, linear or sRGB (the `*_SRGB_BLOCK` `vkFormat`
  and the sRGB transfer function);
- no supercompression: a file with Zstandard or BasisLZ is refused with
  `Ktx2Error::Supercompression`, as are arrays, cubemaps, 3D textures and
  other formats;
- the writer records `KTXorientation` = `rd` and `KTXwriter`, levels
  smallest first, each aligned to 16 bytes. The reader refuses any other
  orientation, since read as stored the image would come out flipped.

```rust
use texcomp::ktx2::{self, Format, Texture, UastcChannels};

/// A UASTC tile served as a file, then read back and made BC7.
fn round_trip(levels: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let tile = Texture {
        format: Format::Uastc(UastcChannels::Rgb),
        srgb: true,
        width: 256,
        height: 256,
        levels, // level 0 first, each its 16-byte blocks row by row
    };
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
panics (`tests/ktx2.rs` truncates and corrupts files by the thousand). It
reads upstream Basis Universal's KTX 2.0 files, whose every level then
transcodes bit-exact to upstream's; the files the writer makes pass the
Khronos `ktx validate` with no warning and upstream's `-unpack`
(`tests/fixtures/ktx2`).

## Features

| Feature | Default | What it adds |
|---|---|---|
| `gpu` | no | the WGSL compute encoder and its dispatch, on wgpu 30 (no default wgpu features: the application picks its backends) |

Without `gpu` the crate depends on nothing GPU-related.

## Minimum Rust version

1.97.

## Status

- **Done**: the GPU encoders (BC7, ASTC 4×4), their quality gate, and the
  pacing API (`tile_cost`, `EncoderScratch`, pass timestamps); on the CPU,
  the UASTC encoder at effort levels 0–2, UASTC decode and UASTC to ASTC 4×4
  and BC7, all byte-identical to upstream; KTX 2.0 read and write (UASTC,
  BC7, ASTC 4×4, no supercompression).
- **Next**: Zstandard supercompression in KTX 2.0.

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. Copyright lapoule.dev. Third-party attributions are in [`NOTICE`](./NOTICE).

The files ported from Basis Universal (Apache-2.0, © Binomial LLC) are
offered under the same terms. They carry the upstream attribution and a notice
of modification at the head of each ported item; see `NOTICE` and
`LICENSE-APACHE-UPSTREAM`. "Basis Universal" is a trademark of Binomial LLC,
named here only to identify the project this code is ported from.
