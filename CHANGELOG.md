<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) lapoule.dev -->

# Changelog

## 0.4.0

Breaking: `ktx2::Texture` has new fields (build a 2D texture with
`Texture::new`), and `ktx2::Format` a new variant. `GpuEncoder::new` keeps
its signature and now encodes at `Quality::DEFAULT` (level 1), which writes
other, better blocks at a higher cost than 0.3 (`Quality::level(0)` is 0.3's
search).

- **BC7, all eight modes.** Modes 0 and 2 (three subsets), 3 and 7 (two),
  4 and 5 (rotation, separate alpha) join 1 and 6. Partitioned modes fit
  the `K` best partitions of a moment estimate; every fit iterates
  quantise, assign, least squares.
- **ASTC 4×4, more configurations.** Weight ranges 0..7 to 0..31,
  luminance (CEM 0) and RGB base + scale (CEM 6) endpoints, two weight
  planes (one channel on its own plane, the best of R, G, B and alpha), two
  partitions among every distinct layout the partition function makes, and
  the same for blocks with alpha. Each configuration's endpoint range is the
  one a decoder derives from the bits left.
- **`gpu::Quality`**, numbers compiled into the kernel: `mode_tier`,
  `partition_candidates`, `refine_iterations`, `luma_weighting`.
  `Quality::level(0..=4)` derives them from one number;
  `GpuEncoder::with_quality` takes one, `GpuEncoder::quality` says which.
- Indices and weights are assigned by projecting each texel on its segment
  and comparing the two weights that bracket it, instead of every weight.
- The kernels' ASTC and BC7 tables are generated from the specification's
  procedures and the crate's BC7 tables.
- **KTX 2.0, the whole container.** 1D, 2D, 3D, arrays, cubemaps and
  cubemap arrays; RGBA8 (`Format::Rgba8`) beside UASTC, BC7 and ASTC 4×4;
  Zstandard and ZLIB supercompression (`Supercompression`, pure Rust);
  `levelCount` 0 (`generate_mips`); premultiplied alpha; every key/value
  pair kept (`key_values`); `Texture::image` cuts one image out of a level.
  ETC1S/BasisLZ, 3D arrays and other `vkFormat`s are refused by name.
- New dependencies: `ruzstd` and `miniz_oxide` (default build, both pure
  Rust and wasm-clean).
- Tests: `gpu_oracle` decodes every block of every level through an
  independent decoder and the GPU's; `ktx2_complete` checks every texture
  type, format and scheme against the `ktx2` crate, KTX-Software's files and,
  with `TEXCOMP_KTX`, the Khronos `ktx` tool; `examples/quality-report`
  prints the quality and cost tables.

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
