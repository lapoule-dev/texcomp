<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) lapoule.dev -->

# UASTC reference vectors

Blocks written by the upstream UASTC encoder, their ASTC 4×4 and BC7 transcodes and
digests of their decoded texels, captured once from upstream Basis Universal
**v2.50** (tag `v2_50`, commit `9bebe16726b3a61c8c213eeee3b7cffb462ef34e`,
Apache-2.0, © Binomial LLC) for `tests/uastc_reference.rs`. Captured on
24 Oct. 2026 on the k8s builder (linux/amd64, g++ 12.2.0, Debian bookworm,
CMake `Release`, default options); `toolchain.txt` records it.

## Inputs

- `tiles/`: nine 256² RGBA8 PNGs. Five are cut from `../imagery` the way
  `tests/gpu_quality.rs` cuts them (quadrant `q` of the 512² tile, named
  `<tile>-<q>`), four are its synthetic tiles (gradient, stripes, alpha edge,
  noise). Encoded at effort levels 0, 1 and 2.
- `coverage-tiles/`: eight 64² RGBA8 PNGs made to reach the modes the main
  set never does (crops, grey, grey with alpha, alpha noise, imagery with an
  alpha ramp, and blocks only the three-subset, mode-7 and RGBA 2-bit modes
  represent exactly). Encoded at effort levels 3 and 4, where every mode is
  tried.

Both are written by `cargo test --test uastc_reference -- --ignored
cut_reference_tiles`; the committed PNGs are what upstream encoded.

## Outputs

`level<N>/<tile>.*` (main set) and `coverage/level<N>/<tile>.*`:

| File | Contents |
|---|---|
| `.uastc` | the UASTC blocks, 16 bytes each, row-major, as the tool wrote them |
| `.astc` | each block's ASTC 4×4 transcode (`transcode_uastc_to_astc`) |
| `.bc7` | each block's BC7 transcode (`transcode_uastc_to_bc7`) |
| `.decoded` | per block, two little-endian `u32`: the 32-bit FNV-1a digest of the 64 RGBA8 bytes `unpack_uastc` decodes, linear then sRGB |

Every `.astc` and `.bc7` file was also checked, inside the capture, byte for
byte against the tool's own `basisu -unpack -format_only 10 -ktx_only` (ASTC)
and `-format_only 6` (BC7) output. The BC7 files were added on 24 Oct. 2026
by a second run of the same capture, whose other outputs came out identical
to the committed ones, byte for byte.

Mode coverage (blocks per mode 0–18, all sets): printed by
`slice_transcode_and_mode_coverage`; every mode appears at least once.

## Command lines

For each tile and level (all of them, in order, in `commands.txt`):

```sh
basisu -uastc -uastc_level <N> -ktx2 -ktx2_no_zstandard -no_multithreading \
    -no_stats -output_file /work/run/<tile>.ktx2 -file <tile>.png
```

UASTC LDR 4×4, no RDO (no `-uastc_rdo_l`), no Zstandard, no mipmaps, one
thread; the capture asserts the tool reports `Effort Level (0-4): <N>`. The
level-0 blocks of each `.ktx2` are then read by `capture/dump.cpp`, built
against upstream's transcoder alone, which writes the four files.

## Reproducing

From this directory, on a remote builder (never needed in CI):

```sh
docker buildx build --builder stl-builder-k8s --platform linux/amd64 \
    -f capture/Dockerfile --output type=local,dest=<dir> .
```

`<dir>` then holds `level0..2/`, `coverage/`, `commands.txt`,
`toolchain.txt` and upstream's licence.

Size: about 6.5 MB of outputs and 1 MB of input PNGs.
