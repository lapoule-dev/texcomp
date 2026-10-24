// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! KTX 2.0 files holding one 2D texture of compressed blocks with its mips.
//!
//! The Khronos container lets pre-compressed imagery be stored and served as
//! files (`image/ktx2`), for instance as the tiles of an archive: a tile cut
//! out of it opens in any KTX 2.0 tool.
//!
//! Written from the Khronos *KTX File Format Specification*, version 2.0
//! (<https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html>), and the
//! *Khronos Data Format Specification* 1.4 for the descriptor
//! (<https://registry.khronos.org/DataFormat/specs/1.4/dataformat.1.4.html>).
//! Section numbers in the comments are those documents'.
//!
//! What is supported, both ways:
//!
//! - one 2D texture (no array, cubemap or depth), level 0 and any number of
//!   its mips (normally the full chain down to 1×1);
//! - [`Format::Uastc`]: `vkFormat` `VK_FORMAT_UNDEFINED` with a Basic Data
//!   Format Descriptor of colour model `KHR_DF_MODEL_UASTC`;
//! - [`Format::Bc7`] (`VK_FORMAT_BC7_{UNORM,SRGB}_BLOCK`) and
//!   [`Format::Astc4x4`] (`VK_FORMAT_ASTC_4x4_{UNORM,SRGB}_BLOCK`), the
//!   formats this crate encodes and transcodes to;
//! - the sRGB transfer function, recorded in `vkFormat` and in the
//!   descriptor;
//! - supercompression scheme 0 (none) only. A supercompressed file is
//!   refused with [`Ktx2Error::Supercompression`].
//!
//! [`read`] validates every size and offset against the file and never
//! panics, whatever the bytes.

use crate::BlockFormat;

/// The media type of a KTX 2.0 file (KTX 2.0 §7).
pub const MEDIA_TYPE: &str = "image/ktx2";

/// The block format of a [`Texture`]'s levels. Every format is 4×4 texels in
/// 16 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    /// UASTC LDR 4×4, to be transcoded before it is sampled
    /// ([`crate::uastc`]). The channels say what the blocks hold, to help
    /// pick a transcode target.
    Uastc(UastcChannels),
    /// BC7.
    Bc7,
    /// ASTC LDR 4×4.
    Astc4x4,
}

/// What a UASTC texture's blocks hold: the channel id of its descriptor's
/// sample (KTX 2.0 §3.10.3, Khronos Data Format 1.4 §5.6.14).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UastcChannels {
    /// Opaque colour (`KHR_DF_CHANNEL_UASTC_RGB`).
    Rgb,
    /// Colour and alpha (`KHR_DF_CHANNEL_UASTC_RGBA`).
    Rgba,
    /// One channel, R replicated in G and B (`KHR_DF_CHANNEL_UASTC_RRR`).
    Rrr,
    /// Two channels, R replicated in G and B and G in alpha
    /// (`KHR_DF_CHANNEL_UASTC_RRRG`).
    Rrrg,
    /// Two channels, R and G (`KHR_DF_CHANNEL_UASTC_RG`).
    Rg,
}

/// One 2D texture of 4×4 blocks with its mips, as a KTX 2.0 file holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Texture {
    /// The blocks' format.
    pub format: Format,
    /// Whether the texels are sRGB-encoded: the `*_SRGB_BLOCK` `vkFormat`
    /// and the sRGB transfer function in the descriptor.
    pub srgb: bool,
    /// Width of level 0, in texels.
    pub width: u32,
    /// Height of level 0, in texels.
    pub height: u32,
    /// The levels, level 0 first, each its blocks row by row (16 bytes per
    /// block, every level in whole blocks). Level `p` is
    /// `max(1, width >> p)` × `max(1, height >> p)` texels.
    pub levels: Vec<Vec<u8>>,
}

/// Why a file cannot be read, or a [`Texture`] written.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Ktx2Error {
    /// The file does not start with the KTX 2.0 identifier (§3.1).
    #[error("not a KTX 2.0 file: the identifier differs")]
    Identifier,
    /// A part of the file lies, at least partly, beyond its end.
    #[error(
        "{what} ({length} bytes at offset {offset}) lies beyond the file's end ({file} bytes)"
    )]
    OutOfBounds {
        /// The part of the file.
        what: &'static str,
        /// Its offset from the start of the file.
        offset: u64,
        /// Its length.
        length: u64,
        /// The file's length.
        file: usize,
    },
    /// A supercompression scheme other than 0 (none). Zstandard (2) is not
    /// supported yet.
    #[error("supercompression scheme {0} is not supported, only 0 (none)")]
    Supercompression(u32),
    /// A `vkFormat` other than the five this module reads.
    #[error("vkFormat {0} is not supported (UNDEFINED with UASTC, BC7 or ASTC 4x4 only)")]
    VkFormat(u32),
    /// A valid kind of texture this module does not read.
    #[error("not supported: {0}")]
    Unsupported(&'static str),
    /// The header or the index breaks a rule of the specification, or a
    /// texture to write has no level or an impossible size.
    #[error("invalid header: {0}")]
    Header(&'static str),
    /// The data format descriptor is invalid or does not match the format.
    #[error("invalid data format descriptor: {0}")]
    Dfd(&'static str),
    /// The key/value data is invalid.
    #[error("invalid key/value data: {0}")]
    KeyValue(&'static str),
    /// The file says its texels start other than at the top left
    /// (`KTXorientation`, §5.2): read as stored, the image would be flipped.
    #[error("orientation {0:?} is not supported, only \"rd\"")]
    Orientation(String),
    /// A level's entry in the level index, or a level to write, is invalid.
    #[error("level {level}: {reason}")]
    Level {
        /// The level, 0 being the largest.
        level: usize,
        /// What is wrong with it.
        reason: &'static str,
    },
}

impl Format {
    /// The `vkFormat` of a texture in this format (KTX 2.0 §3.2).
    pub const fn vk_format(self, srgb: bool) -> u32 {
        match (self, srgb) {
            (Format::Uastc(_), _) => VK_FORMAT_UNDEFINED,
            (Format::Bc7, false) => VK_FORMAT_BC7_UNORM_BLOCK,
            (Format::Bc7, true) => VK_FORMAT_BC7_SRGB_BLOCK,
            (Format::Astc4x4, false) => VK_FORMAT_ASTC_4X4_UNORM_BLOCK,
            (Format::Astc4x4, true) => VK_FORMAT_ASTC_4X4_SRGB_BLOCK,
        }
    }

    /// The block format the GPU samples, or `None` for UASTC, which is
    /// transcoded first.
    pub const fn block_format(self) -> Option<BlockFormat> {
        match self {
            Format::Uastc(_) => None,
            Format::Bc7 => Some(BlockFormat::Bc7),
            Format::Astc4x4 => Some(BlockFormat::Astc4x4),
        }
    }

    /// The descriptor's colour model (Khronos Data Format 1.4 §5.6).
    const fn color_model(self) -> u8 {
        match self {
            Format::Uastc(_) => KHR_DF_MODEL_UASTC,
            Format::Bc7 => KHR_DF_MODEL_BC7,
            Format::Astc4x4 => KHR_DF_MODEL_ASTC,
        }
    }

    /// The channel id of the descriptor's one sample: the UASTC channels
    /// (§5.6.14), `KHR_DF_CHANNEL_BC7_COLOR` (§5.6.7) or
    /// `KHR_DF_CHANNEL_ASTC_DATA` (§5.6.10), all 0.
    const fn channel_id(self) -> u8 {
        match self {
            Format::Uastc(c) => c.channel_id(),
            Format::Bc7 | Format::Astc4x4 => 0,
        }
    }
}

impl From<BlockFormat> for Format {
    fn from(f: BlockFormat) -> Self {
        match f {
            BlockFormat::Bc7 => Format::Bc7,
            BlockFormat::Astc4x4 => Format::Astc4x4,
        }
    }
}

impl UastcChannels {
    /// The channel id (Khronos Data Format 1.4, table 73).
    pub const fn channel_id(self) -> u8 {
        match self {
            UastcChannels::Rgb => 0,
            UastcChannels::Rgba => 3,
            UastcChannels::Rrr => 4,
            UastcChannels::Rrrg => 5,
            UastcChannels::Rg => 6,
        }
    }

    /// The channels of a channel id, if it is one of table 73's.
    pub const fn from_channel_id(id: u8) -> Option<Self> {
        Some(match id {
            0 => UastcChannels::Rgb,
            3 => UastcChannels::Rgba,
            4 => UastcChannels::Rrr,
            5 => UastcChannels::Rrrg,
            6 => UastcChannels::Rg,
            _ => return None,
        })
    }
}

impl Texture {
    /// The number of levels of the full chain of a `width`×`height`
    /// texture, down to 1×1: `floor(log2(max(width, height))) + 1` (§3.7).
    pub const fn full_chain(width: u32, height: u32) -> usize {
        let side = if width > height { width } else { height };
        if side == 0 {
            0
        } else {
            (u32::BITS - side.leading_zeros()) as usize
        }
    }

    /// The bytes level `level` takes, in whole blocks, or `None` when that
    /// overflows.
    pub fn level_bytes(&self, level: usize) -> Option<u64> {
        level_bytes(self.width, self.height, level)
    }

    /// Checks what [`write()`] checks: a size, between 1 and the full chain's
    /// number of levels, each the length of its blocks.
    pub fn validate(&self) -> Result<(), Ktx2Error> {
        if self.width == 0 || self.height == 0 {
            return Err(Ktx2Error::Header(
                "pixelWidth and pixelHeight must not be 0",
            ));
        }
        if self.levels.is_empty() {
            return Err(Ktx2Error::Header(
                "a block-compressed texture needs at least one level",
            ));
        }
        if self.levels.len() > Self::full_chain(self.width, self.height) {
            return Err(Ktx2Error::Header(
                "more levels than the texture's size allows",
            ));
        }
        for (level, bytes) in self.levels.iter().enumerate() {
            if self.level_bytes(level) != Some(bytes.len() as u64) {
                return Err(Ktx2Error::Level {
                    level,
                    reason: "its length is not its size in whole 4x4 blocks",
                });
            }
        }
        Ok(())
    }
}

// vkFormat values (Vulkan's `VkFormat`; KTX 2.0 §3.2).
const VK_FORMAT_UNDEFINED: u32 = 0;
const VK_FORMAT_BC7_UNORM_BLOCK: u32 = 145;
const VK_FORMAT_BC7_SRGB_BLOCK: u32 = 146;
const VK_FORMAT_ASTC_4X4_UNORM_BLOCK: u32 = 157;
const VK_FORMAT_ASTC_4X4_SRGB_BLOCK: u32 = 158;

// Khronos Data Format 1.4: colour models (§5.6), primaries (§5.7) and
// transfer functions (§5.8).
const KHR_DF_MODEL_BC7: u8 = 134;
const KHR_DF_MODEL_ASTC: u8 = 162;
const KHR_DF_MODEL_ETC1S: u8 = 163;
const KHR_DF_MODEL_UASTC: u8 = 166;
const KHR_DF_PRIMARIES_BT709: u8 = 1;
const KHR_DF_TRANSFER_LINEAR: u8 = 1;
const KHR_DF_TRANSFER_SRGB: u8 = 2;
/// The `F` qualifier of a sample (§5.12): float data, which UASTC is not.
const KHR_DF_SAMPLE_DATATYPE_FLOAT: u8 = 0x80;
/// The Basic Data Format Descriptor Block's version (§5.3).
const KHR_DF_VERSIONNUMBER: u32 = 2;

/// The identifier every KTX 2.0 file starts with (§3.1).
const IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];
/// The identifier, the header (§3.2–3.8) and the index (§3.9), before the
/// level index (§3.9.7).
const HEADER_BYTES: usize = 80;
/// One level index entry: byteOffset, byteLength, uncompressedByteLength.
const LEVEL_ENTRY_BYTES: usize = 24;
/// The basic descriptor block with one sample: 24 bytes plus 16 per sample
/// (Khronos Data Format 1.4 §5.4).
const BASIC_BLOCK_BYTES: usize = 24 + 16;
/// The DFD: dfdTotalSize, then the basic block (§3.10).
const DFD_BYTES: usize = 4 + BASIC_BLOCK_BYTES;
/// A level's alignment without supercompression: lcm(texel block size,
/// 4), the texel block being 16 bytes in every format here (§3.13.2).
const LEVEL_ALIGN: u64 = 16;
/// The orientation written and accepted: texels from the top left, S to the
/// right and T downwards (§5.2).
const ORIENTATION: &[u8] = b"rd";
const KEY_ORIENTATION: &[u8] = b"KTXorientation";
const KEY_WRITER: &[u8] = b"KTXwriter";
const WRITER: &str = concat!("texcomp ", env!("CARGO_PKG_VERSION"));

/// The bytes of level `level` of a `width`×`height` texture in whole 4×4
/// blocks of 16 bytes (§2, the `levelImages` structure), `None` on
/// overflow.
fn level_bytes(width: u32, height: u32, level: usize) -> Option<u64> {
    let shift = u32::try_from(level).ok().filter(|&s| s < u32::BITS)?;
    let w = (width >> shift).max(1);
    let h = (height >> shift).max(1);
    (w.div_ceil(4) as u64)
        .checked_mul(h.div_ceil(4) as u64)?
        .checked_mul(16)
}

/// Writes `texture` as a KTX 2.0 file.
///
/// The file holds, in this order (§2): the header and index; the level
/// index; a data format descriptor with one basic block; the key/value data
/// `KTXorientation` = `rd` and `KTXwriter`; and the levels, smallest first,
/// each aligned to 16 bytes. Nothing is supercompressed.
///
/// Fails, writing nothing, when [`Texture::validate`] does.
pub fn write(texture: &Texture) -> Result<Vec<u8>, Ktx2Error> {
    texture.validate()?;
    let n = texture.levels.len();
    let dfd_offset = HEADER_BYTES + LEVEL_ENTRY_BYTES * n;
    let kvd_offset = dfd_offset + DFD_BYTES;
    let kvd = key_value_data(&[
        (KEY_ORIENTATION, &[ORIENTATION, b"\0"].concat()),
        (KEY_WRITER, &[WRITER.as_bytes(), b"\0"].concat()),
    ]);
    let data_offset = (kvd_offset + kvd.len()).next_multiple_of(LEVEL_ALIGN as usize);
    let data_bytes: usize = texture.levels.iter().map(Vec::len).sum();

    let mut f = Vec::with_capacity(data_offset + data_bytes);
    // §3.1–3.8.
    f.extend_from_slice(&IDENTIFIER);
    for v in [
        texture.format.vk_format(texture.srgb),
        1, // typeSize: 1 for VK_FORMAT_UNDEFINED and *_BLOCK formats (§3.3).
        texture.width,
        texture.height,
        0, // pixelDepth: a 2D texture (§3.4, §4.1).
        0, // layerCount: not an array (§3.5).
        1, // faceCount: not a cubemap (§3.6).
        n as u32,
        0, // supercompressionScheme: none (§3.8).
    ] {
        f.extend_from_slice(&v.to_le_bytes());
    }
    // §3.9: the index. No supercompression global data, so its offset and
    // length are 0 (§3.9.5, §3.9.6).
    for v in [dfd_offset, DFD_BYTES, kvd_offset, kvd.len()] {
        f.extend_from_slice(&(v as u32).to_le_bytes());
    }
    f.extend_from_slice(&[0; 16]);
    // §3.9.7: the level index, level 0 first, while the levels themselves
    // are stored smallest first (§3.13). Every level is a whole number of
    // 16-byte blocks, so each one after the first is aligned too.
    let mut offset = data_offset as u64;
    let mut offsets = vec![0; n];
    for p in (0..n).rev() {
        offsets[p] = offset;
        offset += texture.levels[p].len() as u64;
    }
    for (p, level) in texture.levels.iter().enumerate() {
        let len = level.len() as u64;
        // Without supercompression, byteLength = uncompressedByteLength.
        for v in [offsets[p], len, len] {
            f.extend_from_slice(&v.to_le_bytes());
        }
    }
    // §3.10.
    f.extend_from_slice(&dfd(texture.format, texture.srgb));
    // §3.11.
    f.extend_from_slice(&kvd);
    // §3.13: mipPadding, then the levels.
    f.resize(data_offset, 0);
    for level in texture.levels.iter().rev() {
        f.extend_from_slice(level);
    }
    Ok(f)
}

/// The data format descriptor: `dfdTotalSize` and one Basic Data Format
/// Descriptor Block with one sample (KTX 2.0 §3.10, Khronos Data Format 1.4
/// §5 and the 4×4 examples of §5.6.7, §5.6.10 and §5.6.14).
fn dfd(format: Format, srgb: bool) -> [u8; DFD_BYTES] {
    let mut d = [0u8; DFD_BYTES];
    d[0..4].copy_from_slice(&(DFD_BYTES as u32).to_le_bytes());
    let b = &mut d[4..];
    // Word 0: vendorId 0 (Khronos) and descriptorType 0 (basic), §5.1–5.2.
    // Word 1: versionNumber, then descriptorBlockSize (§5.3–5.4).
    b[4..8]
        .copy_from_slice(&(KHR_DF_VERSIONNUMBER | (BASIC_BLOCK_BYTES as u32) << 16).to_le_bytes());
    // Word 2: colorModel, colorPrimaries, transferFunction, flags (§5.5–5.9).
    // Flags 0: alpha not premultiplied. The sRGB transfer is required with
    // an *_SRGB vkFormat (KTX 2.0 §3.10.1), and recommended for sRGB
    // texels with VK_FORMAT_UNDEFINED.
    let transfer = if srgb {
        KHR_DF_TRANSFER_SRGB
    } else {
        KHR_DF_TRANSFER_LINEAR
    };
    b[8..12].copy_from_slice(&[format.color_model(), KHR_DF_PRIMARIES_BT709, transfer, 0]);
    // Word 3: texelBlockDimension0..3, each the size minus one: 4×4×1×1.
    b[12..16].copy_from_slice(&[3, 3, 0, 0]);
    // Words 4–5: bytesPlane0..7. One plane of 16-byte blocks.
    b[16] = 16;
    // The sample (§5.12–5.17): bitOffset 0 (16 bits), bitLength 128 stored
    // as 127, then channel id with no qualifier; samplePosition 0;
    // sampleLower 0 and sampleUpper UINT32_MAX.
    b[24..28].copy_from_slice(&[0, 0, 127, format.channel_id()]);
    b[36..40].copy_from_slice(&u32::MAX.to_le_bytes());
    d
}

/// Key/value data (§3.11): each pair is its length, the key, a NUL, the
/// value, then padding to 4 bytes. The pairs must be sorted by key.
fn key_value_data(pairs: &[(&[u8], &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for (key, value) in pairs {
        let len = key.len() + 1 + value.len();
        out.extend_from_slice(&(len as u32).to_le_bytes());
        out.extend_from_slice(key);
        out.push(0);
        out.extend_from_slice(value);
        out.resize(out.len().next_multiple_of(4), 0);
    }
    out
}

/// Reads a KTX 2.0 file holding one 2D texture in a [`Format`].
///
/// Checks the identifier, the header against the texture types read here,
/// the descriptor against `vkFormat`, the key/value data's structure and
/// order, and that every level lies inside the file, aligned, smallest
/// first, after the metadata, with exactly its size in blocks. Key/value
/// pairs other than `KTXorientation` are skipped; an orientation other than
/// `rd` is refused, since read as stored the image would come out flipped.
///
/// Never panics: any bytes give a texture or an error.
pub fn read(bytes: &[u8]) -> Result<Texture, Ktx2Error> {
    if bytes.get(..IDENTIFIER.len()) != Some(&IDENTIFIER[..]) {
        return Err(Ktx2Error::Identifier);
    }
    let file = Reader(bytes);
    let header = file.section("the header and index", 0, HEADER_BYTES as u64)?;
    let word = |i: usize| u32_le(header, 12 + 4 * i);
    let [vk_format, type_size, width, height, depth, layers, faces, level_count, scheme] =
        [0, 1, 2, 3, 4, 5, 6, 7, 8].map(word);
    // §3.8: only "none". Checked first, so a supercompressed file says so.
    if scheme != 0 {
        return Err(Ktx2Error::Supercompression(scheme));
    }
    let (format_kind, srgb_from_vk) = match vk_format {
        VK_FORMAT_UNDEFINED => (None, None),
        VK_FORMAT_BC7_UNORM_BLOCK => (Some(Format::Bc7), Some(false)),
        VK_FORMAT_BC7_SRGB_BLOCK => (Some(Format::Bc7), Some(true)),
        VK_FORMAT_ASTC_4X4_UNORM_BLOCK => (Some(Format::Astc4x4), Some(false)),
        VK_FORMAT_ASTC_4X4_SRGB_BLOCK => (Some(Format::Astc4x4), Some(true)),
        other => return Err(Ktx2Error::VkFormat(other)),
    };
    // §3.3.
    if type_size != 1 {
        return Err(Ktx2Error::Header("typeSize must be 1 for these formats"));
    }
    // §3.4–3.6 and the texture types of §4.1.
    if width == 0 || height == 0 {
        return Err(Ktx2Error::Header(
            "pixelWidth and pixelHeight must not be 0",
        ));
    }
    if depth != 0 {
        return Err(Ktx2Error::Unsupported("3D textures"));
    }
    if layers != 0 {
        return Err(Ktx2Error::Unsupported("array textures"));
    }
    match faces {
        1 => {}
        6 => return Err(Ktx2Error::Unsupported("cubemaps")),
        _ => return Err(Ktx2Error::Header("faceCount must be 1 or 6")),
    }
    // §3.7: 0 is not allowed for block-compressed formats, and the chain
    // stops at 1×1. This also bounds the level index to 32 entries.
    if level_count == 0 {
        return Err(Ktx2Error::Header(
            "levelCount 0 is not allowed for block-compressed formats",
        ));
    }
    let n = level_count as usize;
    if n > Texture::full_chain(width, height) {
        return Err(Ktx2Error::Header(
            "levelCount exceeds the levels the texture's size allows",
        ));
    }

    // §3.9: the index.
    let dfd_offset = u32_le(header, 48) as u64;
    let dfd_length = u32_le(header, 52) as u64;
    let kvd_offset = u32_le(header, 56) as u64;
    let kvd_length = u32_le(header, 60) as u64;
    let sgd_length = u64_le(header, 72);
    // §3.9.6: no scheme here has global data.
    if sgd_length != 0 {
        return Err(Ktx2Error::Header(
            "sgdByteLength must be 0 without supercompression",
        ));
    }
    let level_index = file.section(
        "the level index",
        HEADER_BYTES as u64,
        (LEVEL_ENTRY_BYTES * n) as u64,
    )?;

    // §3.10.
    let dfd = file.section("the data format descriptor", dfd_offset, dfd_length)?;
    if kvd_length != 0 && kvd_offset.checked_sub(dfd_offset) != Some(dfd_length) {
        return Err(Ktx2Error::Dfd(
            "dfdTotalSize must equal kvdByteOffset - dfdByteOffset",
        ));
    }
    let (format, srgb) = read_dfd(dfd, format_kind, srgb_from_vk)?;

    // §3.11.
    if kvd_length != 0 {
        let kvd = file.section("the key/value data", kvd_offset, kvd_length)?;
        read_key_values(kvd)?;
    }
    let metadata_end = [
        HEADER_BYTES as u64 + level_index.len() as u64,
        dfd_offset + dfd_length,
        if kvd_length != 0 {
            kvd_offset + kvd_length
        } else {
            0
        },
    ]
    .into_iter()
    .max()
    .unwrap_or(0);

    // §3.9.7 and §3.13: the levels.
    let mut levels = Vec::with_capacity(n);
    let mut next_start = None::<u64>;
    for p in 0..n {
        let e = &level_index[LEVEL_ENTRY_BYTES * p..LEVEL_ENTRY_BYTES * (p + 1)];
        let (offset, length, uncompressed) = (u64_le(e, 0), u64_le(e, 8), u64_le(e, 16));
        let bad = |reason| Err(Ktx2Error::Level { level: p, reason });
        if length != uncompressed {
            return bad("byteLength must equal uncompressedByteLength without supercompression");
        }
        if level_bytes(width, height, p) != Some(length) {
            return bad("byteLength is not the level's size in whole 4x4 blocks");
        }
        if !offset.is_multiple_of(LEVEL_ALIGN) {
            return bad("byteOffset is not aligned to 16 bytes (mipPadding)");
        }
        if offset < metadata_end {
            return bad("the level overlaps the header, index or metadata");
        }
        let data = file.section("a level's data", offset, length)?;
        // Smallest first: a level ends at or before the larger one's start.
        if let Some(start) = next_start {
            if offset + length > start {
                return bad("the level overlaps or follows the larger level");
            }
        }
        next_start = Some(offset);
        levels.push(data.to_vec());
    }
    Ok(Texture {
        format,
        srgb,
        width,
        height,
        levels,
    })
}

/// Reads the descriptor's basic block and checks it against `vkFormat`:
/// `kind` is the format `vkFormat` names (`None`: UNDEFINED) and `srgb` its
/// transfer (`None`: the descriptor says).
fn read_dfd(
    dfd: &[u8],
    kind: Option<Format>,
    srgb: Option<bool>,
) -> Result<(Format, bool), Ktx2Error> {
    // §3.10.4: dfdTotalSize, which must equal dfdByteLength.
    if dfd.len() < DFD_BYTES {
        return Err(Ktx2Error::Dfd(
            "too short for a basic block with one sample",
        ));
    }
    if u32_le(dfd, 0) as usize != dfd.len() {
        return Err(Ktx2Error::Dfd("dfdTotalSize must equal dfdByteLength"));
    }
    let b = &dfd[4..];
    // Khronos Data Format 1.4 §5.1–5.4. The first block must be the basic
    // one; any block after it is not needed here.
    let w0 = u32_le(b, 0);
    if w0 & 0x1_ffff != 0 || w0 >> 17 != 0 {
        return Err(Ktx2Error::Dfd(
            "the first block is not a Khronos basic descriptor block",
        ));
    }
    let w1 = u32_le(b, 4);
    if w1 & 0xffff != KHR_DF_VERSIONNUMBER {
        return Err(Ktx2Error::Dfd("the basic block's versionNumber is not 2"));
    }
    let block_size = (w1 >> 16) as usize;
    if block_size < BASIC_BLOCK_BYTES
        || !(block_size - 24).is_multiple_of(16)
        || block_size > b.len()
    {
        return Err(Ktx2Error::Dfd(
            "descriptorBlockSize is not 24 + 16 per sample within the DFD",
        ));
    }
    let samples = (block_size - 24) / 16;
    // §5.5–5.8.
    let (model, transfer, flags) = (b[8], b[10], b[11]);
    let _ = flags;
    if model == KHR_DF_MODEL_ETC1S {
        return Err(Ktx2Error::Unsupported("ETC1S (BasisLZ) textures"));
    }
    // §5.10–5.11: 4×4 blocks of 16 bytes in one plane.
    if b[12..16] != [3, 3, 0, 0] {
        return Err(Ktx2Error::Dfd("texelBlockDimension is not 4x4"));
    }
    if b[16..24] != [16, 0, 0, 0, 0, 0, 0, 0] {
        return Err(Ktx2Error::Dfd("bytesPlane is not one plane of 16 bytes"));
    }
    // §5.12–5.15: the first sample covers the 128 bits of the block.
    let (bit_offset, bit_length, channel) = (u16::from_le_bytes([b[24], b[25]]), b[26], b[27]);
    if bit_offset != 0 || bit_length != 127 {
        return Err(Ktx2Error::Dfd(
            "the sample does not cover the 128-bit block",
        ));
    }
    match kind {
        None => {
            // KTX 2.0 §3.10.3: UNDEFINED with the UASTC model, one sample
            // whose channel id says the channels, no float qualifier.
            if model != KHR_DF_MODEL_UASTC {
                return Err(Ktx2Error::Dfd(
                    "vkFormat UNDEFINED with a colour model other than UASTC",
                ));
            }
            if samples != 1 {
                return Err(Ktx2Error::Dfd("a UASTC descriptor has one sample"));
            }
            if channel & KHR_DF_SAMPLE_DATATYPE_FLOAT != 0 {
                return Err(Ktx2Error::Dfd(
                    "UASTC is LDR: the float qualifier must be clear",
                ));
            }
            let channels = UastcChannels::from_channel_id(channel & 0x0f)
                .ok_or(Ktx2Error::Dfd("not a UASTC channel id"))?;
            // KTX 2.0 §3.10.1 leaves the transfer free for UNDEFINED; only
            // the two a UASTC texture can be sampled as are read.
            let srgb = match transfer {
                KHR_DF_TRANSFER_SRGB => true,
                KHR_DF_TRANSFER_LINEAR => false,
                _ => return Err(Ktx2Error::Dfd("transfer function neither linear nor sRGB")),
            };
            Ok((Format::Uastc(channels), srgb))
        }
        Some(format) => {
            // KTX 2.0 §3.10.1: the model and the sizes match the format,
            // and an *_SRGB format has the sRGB transfer.
            if model != format.color_model() {
                return Err(Ktx2Error::Dfd("the colour model does not match vkFormat"));
            }
            let srgb = srgb.unwrap_or(false);
            if srgb && transfer != KHR_DF_TRANSFER_SRGB {
                return Err(Ktx2Error::Dfd(
                    "an *_SRGB vkFormat needs the sRGB transfer function",
                ));
            }
            Ok((format, srgb))
        }
    }
}

/// Walks the key/value data (§3.11): every pair within it, at least a
/// one-byte key and its NUL, UTF-8 keys sorted and unique, the padding
/// accounted for to the byte. Checks `KTXorientation` (§5.2).
fn read_key_values(kvd: &[u8]) -> Result<(), Ktx2Error> {
    let mut at = 0usize;
    let mut previous: Option<&[u8]> = None;
    while at < kvd.len() {
        let len = kvd
            .get(at..at + 4)
            .map(|b| u32_le(b, 0) as usize)
            .ok_or(Ktx2Error::KeyValue(
                "a pair's length runs past kvdByteLength",
            ))?;
        if len < 2 {
            return Err(Ktx2Error::KeyValue(
                "keyAndValueByteLength must be at least 2",
            ));
        }
        let pair = (at + 4)
            .checked_add(len)
            .and_then(|end| kvd.get(at + 4..end))
            .ok_or(Ktx2Error::KeyValue("a pair runs past kvdByteLength"))?;
        let nul = pair
            .iter()
            .position(|&b| b == 0)
            .ok_or(Ktx2Error::KeyValue("a key has no NUL terminator"))?;
        let (key, value) = (&pair[..nul], &pair[nul + 1..]);
        if key.is_empty() || std::str::from_utf8(key).is_err() {
            return Err(Ktx2Error::KeyValue("a key is empty or not UTF-8"));
        }
        // UTF-8's byte order is the code points' order.
        if previous.is_some_and(|p| p >= key) {
            return Err(Ktx2Error::KeyValue("keys are not sorted, or one repeats"));
        }
        previous = Some(key);
        if key == KEY_ORIENTATION {
            // The value is a string; readers must not rely on its NUL.
            let v = value.strip_suffix(b"\0").unwrap_or(value);
            if v != ORIENTATION {
                return Err(Ktx2Error::Orientation(
                    String::from_utf8_lossy(v).into_owned(),
                ));
            }
        }
        // valuePadding (§3.11.3).
        at = at + 4 + len.next_multiple_of(4);
    }
    if at != kvd.len() {
        return Err(Ktx2Error::KeyValue(
            "kvdByteLength is not the pairs and their padding",
        ));
    }
    Ok(())
}

/// The file, read through bounds-checked sections.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    /// `length` bytes at `offset`, or why not.
    fn section(&self, what: &'static str, offset: u64, length: u64) -> Result<&'a [u8], Ktx2Error> {
        let out = || Ktx2Error::OutOfBounds {
            what,
            offset,
            length,
            file: self.0.len(),
        };
        let start = usize::try_from(offset).map_err(|_| out())?;
        let end = offset
            .checked_add(length)
            .and_then(|e| usize::try_from(e).ok())
            .ok_or_else(out)?;
        self.0.get(start..end).ok_or_else(out)
    }
}

/// A little-endian `u32` at `at`; the caller has checked the bounds.
fn u32_le(b: &[u8], at: usize) -> u32 {
    let mut w = [0; 4];
    w.copy_from_slice(&b[at..at + 4]);
    u32::from_le_bytes(w)
}

/// A little-endian `u64` at `at`; the caller has checked the bounds.
fn u64_le(b: &[u8], at: usize) -> u64 {
    let mut w = [0; 8];
    w.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(w)
}
