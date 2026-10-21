// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! # texcomp
//!
//! GPU block-compressed textures, encoded on the fly from RGBA8.
//!
//! Imagery and other textures arrive as JPEG or PNG and are decoded to RGBA8.
//! Just before upload they are encoded to a block format the GPU samples
//! natively, which keeps a quarter of the bytes resident. Nothing here reads
//! or writes a compressed file.
//!
//! - [`gpu`] (feature `gpu`): WGSL compute kernels that write [`BlockFormat`]
//!   blocks from RGBA8 levels, and the host-side dispatch around them on
//!   wgpu 29.
//! - The CPU encoder, for devices without compute shaders or without a format
//!   the kernels write, lands in this crate next. It needs nothing but `std`
//!   and builds for `wasm32-unknown-unknown`.
//!
//! Which encoder a device uses is the application's choice, made from what
//! the device reports; [`gpu::runs_on`] says whether the kernels can run.

#[cfg(feature = "gpu")]
mod astc;
#[cfg(feature = "gpu")]
pub mod gpu;

/// A block-compressed format a texture can be made resident in.
///
/// Every variant is 4×4 texels in 16 bytes: 8 bits per texel, a quarter of
/// RGBA8. [`BlockFormat::PREFERENCE`] lists them best first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BlockFormat {
    /// BC7 (BPTC): desktop GPUs, Apple silicon under macOS, desktop WebGPU.
    Bc7,
    /// ASTC LDR with 4×4 blocks: iOS, Android and the Quest, mobile WebGPU.
    Astc4x4,
}

impl BlockFormat {
    /// The order to try formats in on a device that samples several: the
    /// best quality at 8 bits per texel first.
    pub const PREFERENCE: [BlockFormat; 2] = [BlockFormat::Bc7, BlockFormat::Astc4x4];

    /// Side of a block, in texels.
    pub const BLOCK_DIM: u32 = 4;

    /// Bytes per block.
    pub const BLOCK_BYTES: u32 = 16;

    /// The bytes one level of `width`×`height` texels takes, in whole blocks.
    pub fn level_bytes(width: u32, height: u32) -> u64 {
        let blocks_x = width.div_ceil(Self::BLOCK_DIM) as u64;
        let blocks_y = height.div_ceil(Self::BLOCK_DIM) as u64;
        blocks_x * blocks_y * Self::BLOCK_BYTES as u64
    }

    /// The bytes a `width`×`height` texture takes with its full mip chain
    /// down to 1×1, every level in whole blocks.
    pub fn chain_bytes(width: u32, height: u32) -> u64 {
        let (mut w, mut h, mut total) = (width.max(1), height.max(1), 0);
        loop {
            total += Self::level_bytes(w, h);
            if w == 1 && h == 1 {
                return total;
            }
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A quarter of RGBA8, in whole blocks.**
    #[test]
    fn sizes_round_up_to_whole_blocks() {
        assert_eq!(BlockFormat::level_bytes(256, 256), 65_536);
        assert_eq!(BlockFormat::level_bytes(2, 2), 16);
        assert_eq!(BlockFormat::level_bytes(1, 1), 16);
        assert_eq!(BlockFormat::level_bytes(6, 4), 32);
        // 256² with its chain: 64×64 blocks, then 32×32, …, 1, and the 2×2 and
        // 1×1 levels at one block each.
        assert_eq!(BlockFormat::chain_bytes(256, 256), 87_408);
        assert_eq!(BlockFormat::chain_bytes(4, 4), 48);
    }
}
