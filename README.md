<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) lapoule.dev -->

# texcomp

GPU block-compressed textures, encoded on the fly from RGBA8, in pure Rust.

- **On the GPU** (feature `gpu`, wgpu 29): WGSL compute kernels that write
  BC7 and ASTC 4×4 from RGBA8 levels, and the host-side dispatch around them.
- **On the CPU** (default, dependency-light, `wasm32-unknown-unknown` clean):
  a port of the block layer of Basis Universal (UASTC LDR 4×4 and its
  transcoders to ASTC, BC7, ETC2), for devices that have no compute shaders or
  no format the GPU kernels write. *In progress.*

Built for [tuile](https://github.com/lapoule-dev/tuile)'s viewer, where imagery
arrives as JPEG/PNG and is made resident in the GPU's native block format just
before upload. Nothing is stored compressed.

## License

Dual-licensed under MIT or Apache-2.0, at your option. Ported code keeps its
upstream Apache-2.0 attribution; see `NOTICE`.
