// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

// The crate doc is the README, so GitHub and rustdoc show one text and its
// example compiles as a doctest. The example needs the `gpu` feature; without
// it the doc is a pointer to the README.
#![cfg_attr(feature = "gpu", doc = include_str!("../README.md"))]
#![cfg_attr(
    not(feature = "gpu"),
    doc = "GPU block-compressed textures, encoded on the fly from RGBA8. \
           Build the docs with `--features gpu` for the full README."
)]

#[cfg(feature = "gpu")]
mod astc;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod ktx2;
pub mod uastc;

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

/// What one tile costs to make resident: its blocks and bytes with the full
/// mip chain down to 1×1, and the RGBA8 bytes the encoder reads to make them.
///
/// A pure function of the format and the size, so a scheduler can spread
/// encodes over frames before any of them runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileCost {
    /// Blocks written, every level in whole blocks.
    pub blocks: u64,
    /// Bytes resident once encoded: `blocks` times the block size.
    pub bytes: u64,
    /// Bytes of the RGBA8 chain the encoder reads, 4 per texel.
    pub rgba8_bytes: u64,
}

/// The cost of a `width`×`height` tile in `format`, mips included.
pub fn tile_cost(format: BlockFormat, width: u32, height: u32) -> TileCost {
    let (block_dim, block_bytes) = match format {
        BlockFormat::Bc7 | BlockFormat::Astc4x4 => {
            (BlockFormat::BLOCK_DIM, BlockFormat::BLOCK_BYTES)
        }
    };
    let (mut w, mut h) = (width.max(1), height.max(1));
    let mut cost = TileCost::default();
    loop {
        let blocks = w.div_ceil(block_dim) as u64 * h.div_ceil(block_dim) as u64;
        cost.blocks += blocks;
        cost.bytes += blocks * block_bytes as u64;
        cost.rgba8_bytes += w as u64 * h as u64 * 4;
        if w == 1 && h == 1 {
            return cost;
        }
        w = (w / 2).max(1);
        h = (h / 2).max(1);
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

    /// **A tile's cost counts every level**, in whole blocks, and its RGBA8
    /// source texel by texel.
    #[test]
    fn tile_cost_counts_the_whole_chain() {
        for format in BlockFormat::PREFERENCE {
            // 64², 32², …, 1 blocks, then the 2×2 and 1×1 levels at one each.
            let c = tile_cost(format, 256, 256);
            assert_eq!(c.blocks, 4096 + 1024 + 256 + 64 + 16 + 4 + 1 + 1 + 1);
            assert_eq!(c.bytes, 87_408);
            assert_eq!(c.bytes, BlockFormat::chain_bytes(256, 256));
            let texels = 65_536 + 16_384 + 4096 + 1024 + 256 + 64 + 16 + 4 + 1;
            assert_eq!(c.rgba8_bytes, 4 * texels);
            // Not square: 8×4, 4×2, 2×1, 1×1 are 2 + 1 + 1 + 1 blocks.
            let c = tile_cost(format, 8, 4);
            assert_eq!((c.blocks, c.bytes), (5, 80));
            assert_eq!(c.rgba8_bytes, 4 * (32 + 8 + 2 + 1));
        }
    }
}
