<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) lapoule.dev -->

# texcomp

GPU block-compressed textures, encoded on the fly from RGBA8, in pure Rust.

Imagery and other textures usually arrive as JPEG or PNG and are decoded to
RGBA8. texcomp encodes them, just before upload, to a block format the GPU
samples natively: 8 bits per texel instead of 32, so a quarter of the bytes
stay resident. Nothing here reads or writes a compressed file.

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
- **On the CPU** (default build, nothing but `std`, `wasm32-unknown-unknown`
  clean): a port of the block layer of Basis Universal (UASTC LDR 4×4 and its
  transcoder to ASTC), for devices without compute shaders, such as WebGL2.
  *In progress, not in this release.*

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

## Features

| Feature | Default | What it adds |
|---|---|---|
| `gpu` | no | the WGSL compute encoder and its dispatch, on wgpu 30 (no default wgpu features: the application picks its backends) |

Without `gpu` the crate depends on nothing GPU-related.

## Minimum Rust version

1.97.

## Status

- **Done**: the GPU encoders (BC7, ASTC 4×4), their quality gate, and the
  pacing API (`tile_cost`, `EncoderScratch`, pass timestamps).
- **Next**: the CPU path, in steps. First UASTC unpack and UASTC to ASTC,
  bit-exact against upstream reference vectors; then the UASTC encoder.
  UASTC to BC7 is deferred: desktop WebGL2 is rare and falls back to RGBA8.

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. Copyright lapoule.dev. Third-party attributions are in [`NOTICE`](./NOTICE).
