<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) lapoule.dev -->

# Changelog

## 0.3.0

Additive: `encode` and `encode_with` keep their signatures, and write the
same blocks.

- `GpuEncoder::encode_into` and `GpuEncoder::encode_into_with` write the
  blocks into textures the caller keeps, at a mip level, a block-aligned
  origin and an array layer given per level (`Destination`, `TileInto`).
- A level's texels can be on the GPU (`SourceLevel`, `Source`): a buffer with
  any 4-byte row pitch, or a region of a layer of an `Rgba8Unorm` or
  `Rgba8UnormSrgb` texture. They are copied into the encoder's input by the
  caller's command encoder ahead of the compute pass: no readback.
- `EncodeError` gains `Destinations`, `DestinationFormat`, `DestinationUsage`,
  `DestinationRegion`, `SourceFormat`, `SourceUsage` and `SourceRegion`,
  returned by `encode_into` before anything is recorded. A `match` on
  `EncodeError` without a wildcard arm needs them.
- The kernels read a level's rows at a stride (the job's former padding
  word), which is the level's width for texels in memory.
- A test scans the crate's WGSL for a texture or sampler bound to a `let`,
  which WebKit rejects.

## 0.2.0

- CPU UASTC decode and transcode to ASTC 4×4 and BC7, and the CPU UASTC
  encoder at effort levels 0, 1 and 2, all byte-identical to upstream Basis
  Universal.
- KTX 2.0 read and write (UASTC, BC7, ASTC 4×4, with mips).

## 0.1.0

- BC7 and ASTC 4×4 compute encoders in WGSL with their wgpu dispatch, the
  GPU quality gate, and the pacing API (`tile_cost`, `EncoderScratch`, pass
  timestamps), on wgpu 30.
