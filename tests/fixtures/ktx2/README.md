<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) lapoule.dev -->

# KTX 2.0 cross-check files

For `tests/ktx2.rs`: KTX 2.0 files written by upstream Basis Universal
**v2.50** (tag `v2_50`, commit `9bebe16726b3a61c8c213eeee3b7cffb462ef34e`,
Apache-2.0, © Binomial LLC), and files written by texcomp's writer that two
independent readers accepted: upstream's tool and the Khronos KTX-Software
**v4.4.2** `ktx` tool. Captured on 24 Oct. 2026 on the k8s builder
(linux/amd64); `toolchain.txt` records it.

## Contents

| Path | What |
|---|---|
| `inputs/alps-0-100x60.png` | a 100×60 crop of `../uastc/tiles/alps-0.png`, for a chain that is neither square nor in whole blocks (written by `cargo test --test ktx2 -- --ignored cut_ktx2_inputs`) |
| `upstream/<name>.ktx2` | upstream's UASTC file with its full mip chain, no Zstandard |
| `upstream/<name>.astc`, `.bc7` | every level of it as upstream's own `-unpack` transcodes it to ASTC 4×4 (format 10) and BC7 (format 6), level 0 first, concatenated |
| `ours/<name>-<format>.ktx2` | texcomp's writer: the upstream file read and written again (`uastc`), and its levels transcoded by texcomp to `bc7` and `astc`, linear and `-srgb` (written by `cargo test --test ktx2 -- --ignored write_cross_check_files`) |
| `validate.txt` | the capture's log of the checks below |
| `commands.txt` | the upstream command lines |

`<name>` is `alps-0` (256²), `alpha-imagery` (64², with alpha, from
`../uastc/coverage-tiles`) and `alps-0-100x60`; `ours/` covers the last two.

## What the capture checked

- Every file of `ours/` passes `ktx validate` with no error and no warning.
  (Upstream's own files pass too, with one warning: they carry a custom key.)
- Upstream's `basisu -info` reads every UASTC file of `ours/`, and its
  `-unpack` transcodes every level of them to the same ASTC 4×4 and BC7
  bytes as the upstream file they were rewritten from.

The last line of `validate.txt` says `ALL CHECKS PASSED`.
`the_writer_still_writes_the_files_the_tools_accepted` holds the writer to
the bytes of `ours/`, and `upstream_files_read_and_transcode_bit_exact`
checks that texcomp reads upstream's files and transcodes every level to
upstream's bytes.

## Command lines

```sh
basisu -uastc -uastc_level 1 -mipmap -ktx2 -ktx2_no_zstandard \
    -no_multithreading -no_stats -output_file <name>.ktx2 -file <name>.png
```

## Reproducing

From `tests/fixtures`, on a remote builder (never needed in CI), after
writing `ours/` with the ignored test above:

```sh
docker buildx build --builder stl-builder-k8s --platform linux/amd64 \
    -f ktx2/capture/Dockerfile --output type=local,dest=<dir> .
```

`<dir>` then holds `upstream/`, `validate.txt`, `commands.txt`,
`toolchain.txt` and upstream's licence. A second run gave the same upstream
files, byte for byte.

## `khronos/`: KTX-Software's own files

For `tests/ktx2_complete.rs`: six files of the Khronos KTX-Software test set,
tag **v4.4.2**, `tests/testimages/`, fetched on 23 Nov. 2026 from
`https://media.githubusercontent.com/media/KhronosGroup/KTX-Software/v4.4.2/tests/testimages/<name>`.
Licence: **Apache-2.0**, © 2015-2022 The Khronos Group Inc. (KTX-Software's
`REUSE.toml`, which covers every `.ktx2` of the repository).

| File | What it is | What the test checks |
|---|---|---|
| `rgba-reference-u.ktx2` | R8G8B8A8_SRGB, 128², one level | shape, texels, rewrite |
| `cyan_rgba_reference_u.ktx2` | R8G8B8A8_SRGB, 64², one level | shape, texels, rewrite |
| `arraytex_7_mipmap_reference_u.ktx2` | R8G8B8A8_SRGB, 16², 7 layers, 5 levels | array layout, texels, rewrite |
| `3dtex_7_reference_u.ktx2` | R8G8B8A8_SRGB, 16²×7, one level | 3D layout, texels, rewrite |
| `color_grid_uastc_zstd.ktx2` | UASTC RGB, 1024², Zstandard | Zstandard as libktx writes it; the blocks decode as libktx decodes them |
| `rgba-mipmap-reference-basis.ktx2` | ETC1S, BasisLZ | refused with `Supercompression(1)` |

The texels are pinned by the FNV-1a of level 0 as `ktx extract --raw` (libktx
4.4.2) writes it, image after image; for the UASTC file that is its decode
to RGB8, which this crate's decode must match. `print_khronos_hashes`
(ignored; needs `TEXCOMP_KTX`) prints them again.

## The Khronos tool, live

With `TEXCOMP_KTX` naming the `ktx` tool of KTX-Software 4.4.2 (the macOS
package's `ktx` binary and `libktx.4.dylib`, unpacked without installing;
a wrapper setting `DYLD_LIBRARY_PATH` will do), `tests/ktx2_complete.rs`
also checks every file the writer makes for every texture type, format and
supercompression scheme with `ktx validate` (no error, no warning but the
custom-key one the files with pairs of their own carry), compares every
image `ktx extract --raw` reads out of them with `Texture::image`, and
compares libktx's transcodes of real UASTC blocks to BC7 and ASTC with this
crate's. `TEXCOMP_REQUIRE_KTX=1` makes the tool's absence a failure.
