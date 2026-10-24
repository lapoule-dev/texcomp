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
