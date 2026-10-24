// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! UASTC LDR 4×4 on the CPU: the reference decode and the transcodes to
//! ASTC 4×4 and BC7.
//!
//! UASTC is a 4×4, 16-byte block format whose every block stands for an
//! ASTC 4×4 block (plus hints that make other transcodes cheap). It is the
//! intermediate of the CPU path: a block encoded once transcodes losslessly
//! to ASTC 4×4, and to BC7 within a small loss. This module reads such
//! blocks; the encoder lands next.
//!
//! - [`UastcBlock::to_astc_4x4`] (and [`transcode_to_astc_4x4`] for a run of
//!   blocks): the ASTC 4×4 block that decodes to exactly the texels the
//!   UASTC block stands for.
//! - [`UastcBlock::to_bc7`] (and [`transcode_to_bc7`]): the BC7 block, for
//!   GPUs without ASTC.
//! - [`UastcBlock::decode_rgba8`]: those texels, as ASTC's LDR decoder
//!   computes them, with the linear or the sRGB endpoint expansion.
//!
//! The code is a port of parts of an upstream transcoder (Apache-2.0; see
//! `NOTICE`), checked bit for bit against vectors captured from it
//! (`tests/uastc_reference.rs`).

mod astc;
mod bc7;
mod tables;
mod unpack;

/// Number of UASTC block modes, the solid-colour mode included.
pub const MODES: usize = tables::MODES;

/// Bytes per block, UASTC and ASTC 4×4 alike.
pub const BLOCK_BYTES: usize = 16;

/// Why a block cannot be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum InvalidBlock {
    /// The block's mode code is the one UASTC reserves.
    #[error("the block uses the reserved UASTC mode code")]
    ReservedMode,
    /// The block's partition pattern index is out of its mode's table.
    #[error("the block's partition pattern index is out of range")]
    Pattern,
}

/// Why a run of blocks cannot be transcoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TranscodeError {
    /// The input is not a whole number of blocks, or the output's length
    /// differs from the input's.
    #[error("{src} bytes in, {dst} bytes out: not the same whole number of blocks")]
    Length {
        /// The input's length in bytes.
        src: usize,
        /// The output's length in bytes.
        dst: usize,
    },
    /// One block of the run is invalid.
    #[error("block {index}: {error}")]
    Block {
        /// The block's index in the run.
        index: usize,
        /// Why it is invalid.
        error: InvalidBlock,
    },
}

/// One UASTC LDR 4×4 block, as stored: 16 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UastcBlock([u8; BLOCK_BYTES]);

impl UastcBlock {
    /// The block whose bytes are `bytes`.
    pub const fn from_bytes(bytes: [u8; BLOCK_BYTES]) -> Self {
        Self(bytes)
    }

    /// The block's bytes.
    pub const fn to_bytes(self) -> [u8; BLOCK_BYTES] {
        self.0
    }

    /// The block's mode, `0..MODES` (8 is a solid colour).
    pub fn mode(&self) -> Result<u8, InvalidBlock> {
        unpack::mode(&self.0).map(|m| m as u8)
    }

    /// The ASTC 4×4 block that decodes to the same texels.
    pub fn to_astc_4x4(&self) -> Result<[u8; BLOCK_BYTES], InvalidBlock> {
        Ok(astc::pack(&unpack::unpack(&self.0, true)?))
    }

    /// The BC7 block it transcodes to: lossless for some modes, otherwise
    /// the endpoints are requantised to BC7's precisions (about 0.75–1.5 dB
    /// below the ASTC transcode).
    pub fn to_bc7(&self) -> Result<[u8; BLOCK_BYTES], InvalidBlock> {
        Ok(bc7::pack(&unpack::unpack(&self.0, false)?))
    }

    /// The block's 16 texels, row-major, as ASTC's LDR decoder computes them.
    ///
    /// `srgb` selects the decode of an `*_SRGB` texture format, which expands
    /// the endpoints differently before interpolating (the texels are still
    /// the stored, gamma-encoded values; nothing is converted to linear).
    pub fn decode_rgba8(&self, srgb: bool) -> Result<[[u8; 4]; 16], InvalidBlock> {
        Ok(unpack::decode(&unpack::unpack(&self.0, false)?, srgb))
    }
}

/// Transcodes a run of UASTC blocks into ASTC 4×4 blocks of the same layout.
///
/// `dst` must be as long as `src`, a whole number of 16-byte blocks. When a
/// block is invalid, the blocks before it are written and the rest of `dst`
/// is left as it was.
pub fn transcode_to_astc_4x4(src: &[u8], dst: &mut [u8]) -> Result<(), TranscodeError> {
    transcode_run(src, dst, UastcBlock::to_astc_4x4)
}

fn transcode_run(
    src: &[u8],
    dst: &mut [u8],
    one: impl Fn(&UastcBlock) -> Result<[u8; BLOCK_BYTES], InvalidBlock>,
) -> Result<(), TranscodeError> {
    let (blocks, rest) = src.as_chunks::<BLOCK_BYTES>();
    let (out, out_rest) = dst.as_chunks_mut::<BLOCK_BYTES>();
    if !rest.is_empty() || !out_rest.is_empty() || out.len() != blocks.len() {
        return Err(TranscodeError::Length {
            src: src.len(),
            dst: dst.len(),
        });
    }
    for (index, (s, d)) in blocks.iter().zip(out).enumerate() {
        *d = one(&UastcBlock::from_bytes(*s))
            .map_err(|error| TranscodeError::Block { index, error })?;
    }
    Ok(())
}

/// Transcodes a run of UASTC blocks into BC7 blocks of the same layout, on
/// the same terms as [`transcode_to_astc_4x4`].
pub fn transcode_to_bc7(src: &[u8], dst: &mut [u8]) -> Result<(), TranscodeError> {
    transcode_run(src, dst, UastcBlock::to_bc7)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A solid block: mode 8's code (`0x17`, 5 bits), then RGBA.
    fn solid(rgba: [u8; 4]) -> UastcBlock {
        let v = 0x17u128 | ((u32::from_le_bytes(rgba) as u128) << 5);
        UastcBlock::from_bytes(v.to_le_bytes())
    }

    /// **A solid block decodes to its colour and transcodes to ASTC's
    /// void-extent block of that colour** (Khronos Data Format 1.3, §23.13:
    /// `0xDFC` in the low 12 bits, the extent all ones, then RGBA as 16-bit
    /// UNORM).
    #[test]
    fn solid_blocks() {
        let b = solid([10, 20, 30, 40]);
        assert_eq!(b.mode(), Ok(8));
        assert_eq!(b.decode_rgba8(false), Ok([[10, 20, 30, 40]; 16]));
        assert_eq!(b.decode_rgba8(true), Ok([[10, 20, 30, 40]; 16]));
        let astc = b.to_astc_4x4().expect("valid");
        assert_eq!(u16::from_le_bytes([astc[0], astc[1]]) & 0xfff, 0xdfc);
        assert_eq!(&astc[2..8], &[0xff; 6]);
        assert_eq!(&astc[8..], &[10, 10, 20, 20, 30, 30, 40, 40]);
    }

    /// **The reserved mode code and out-of-range patterns are errors, not
    /// panics**, and a run names the block.
    #[test]
    fn invalid_blocks() {
        // Code 0x45 (7 bits) is the reserved 20th code.
        let mut bytes = [0u8; 16];
        bytes[0] = 0x45;
        let reserved = UastcBlock::from_bytes(bytes);
        assert_eq!(reserved.mode(), Err(InvalidBlock::ReservedMode));
        assert_eq!(reserved.to_astc_4x4(), Err(InvalidBlock::ReservedMode));
        assert_eq!(
            reserved.decode_rgba8(false),
            Err(InvalidBlock::ReservedMode)
        );

        // Mode 3 (code 0x3, 5 bits), 15 hint bits, then a 4-bit pattern
        // index of 15 (there are 11).
        let v: u128 = 0x3 | (15u128 << (5 + 15));
        let bad = UastcBlock::from_bytes(v.to_le_bytes());
        assert_eq!(bad.mode(), Ok(3));
        assert_eq!(bad.to_astc_4x4(), Err(InvalidBlock::Pattern));
        assert_eq!(bad.to_bc7(), Err(InvalidBlock::Pattern));
        assert_eq!(bad.decode_rgba8(true), Err(InvalidBlock::Pattern));

        let mut run = solid([1, 2, 3, 4]).to_bytes().to_vec();
        run.extend_from_slice(&bad.to_bytes());
        let mut out = vec![0; run.len()];
        assert_eq!(
            transcode_to_astc_4x4(&run, &mut out),
            Err(TranscodeError::Block {
                index: 1,
                error: InvalidBlock::Pattern
            })
        );
        assert_eq!(
            transcode_to_astc_4x4(&run[..20], &mut out[..20]),
            Err(TranscodeError::Length { src: 20, dst: 20 })
        );
        assert_eq!(
            transcode_to_astc_4x4(&run, &mut out[..16]),
            Err(TranscodeError::Length { src: 32, dst: 16 })
        );
    }

    /// **Arbitrary bytes either read or are refused, never panic**, over a
    /// deterministic sweep that reaches every mode code.
    #[test]
    fn arbitrary_bytes_never_panic() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut modes = [0usize; MODES];
        for _ in 0..50_000 {
            let mut bytes = [0u8; 16];
            for b in &mut bytes {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *b = state as u8;
            }
            let block = UastcBlock::from_bytes(bytes);
            if let Ok(mode) = block.mode() {
                modes[mode as usize] += 1;
            }
            let _ = block.to_astc_4x4();
            let _ = block.to_bc7();
            let _ = block.decode_rgba8(false);
            let _ = block.decode_rgba8(true);
        }
        assert!(modes.iter().all(|&n| n > 0), "{modes:?}");
    }
}
