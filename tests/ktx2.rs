// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! KTX 2.0 read and write, against the specification and upstream's files.
//!
//! # What this covers
//!
//! - **Round trip:** every format (UASTC with each channel id, BC7 and ASTC
//!   4×4, linear and sRGB), at sizes square and not, whole blocks and not,
//!   down to 1×1, comes back from `read(write(t))` equal, and rewriting is
//!   byte-stable.
//! - **Layout:** a written file's bytes are checked field by field against
//!   the specification, without going through `read`.
//! - **Upstream's UASTC blocks:** the 27 main-set tiles of
//!   `tests/fixtures/uastc`, wrapped in a KTX 2.0 file, read back and
//!   transcoded, give upstream's ASTC 4×4 and BC7 bytes.
//! - **Upstream's files:** KTX 2.0 files with full mip chains written by the
//!   upstream encoder (`tests/fixtures/ktx2/upstream`) are read, and every
//!   level transcodes to the bytes upstream's own `-unpack` wrote.
//! - **Independent readers accept ours:** the files in
//!   `tests/fixtures/ktx2/ours` were checked in the capture by the Khronos
//!   `ktx validate` and by upstream's `-unpack` (see that directory's
//!   README); `the_writer_still_writes_the_files_the_tools_accepted` holds
//!   the writer to those bytes.
//! - **Hostile input:** every truncation and many corruptions of valid files
//!   are read without a panic, and the classic ones give their error.
//!
//! # What it does not cover
//!
//! Supercompressed files (refused), arrays, cubemaps and 3D textures
//! (refused), and other `vkFormat`s.

use std::path::{Path, PathBuf};

use texcomp::ktx2::{self, Format, Ktx2Error, Texture, UastcChannels};
use texcomp::uastc;

const IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

/// A deterministic byte stream (xorshift64*), so the tests need no crate.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// The bytes of a level of `w`×`h` texels in whole 4×4 blocks.
fn level_len(w: u32, h: u32) -> usize {
    (w.div_ceil(4) * h.div_ceil(4) * 16) as usize
}

/// A texture of random blocks with its full chain down to 1×1.
fn texture(format: Format, srgb: bool, width: u32, height: u32, seed: u64) -> Texture {
    let mut rng = Rng(seed | 1);
    let (mut w, mut h, mut levels) = (width, height, Vec::new());
    loop {
        levels.push(rng.bytes(level_len(w, h)));
        if w == 1 && h == 1 {
            break;
        }
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    Texture {
        format,
        srgb,
        width,
        height,
        levels,
    }
}

const FORMATS: [(Format, bool); 9] = [
    (Format::Uastc(UastcChannels::Rgb), true),
    (Format::Uastc(UastcChannels::Rgba), false),
    (Format::Uastc(UastcChannels::Rrr), false),
    (Format::Uastc(UastcChannels::Rrrg), true),
    (Format::Uastc(UastcChannels::Rg), false),
    (Format::Bc7, false),
    (Format::Bc7, true),
    (Format::Astc4x4, false),
    (Format::Astc4x4, true),
];

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// **Every format comes back as written**, at every size, and writing what
/// was read gives the same bytes.
#[test]
fn round_trip_every_format() {
    for (i, (format, srgb)) in FORMATS.into_iter().enumerate() {
        for (w, h) in [(256, 256), (8, 4), (5, 3), (100, 60), (1, 1), (1, 17)] {
            let t = texture(
                format,
                srgb,
                w,
                h,
                i as u64 * 1000 + w as u64 * 7 + h as u64,
            );
            let file = ktx2::write(&t).expect("writes");
            let back = ktx2::read(&file).unwrap_or_else(|e| panic!("{format:?} {w}×{h}: {e}"));
            assert_eq!(back, t, "{format:?} srgb {srgb} {w}×{h}");
            assert_eq!(ktx2::write(&back).expect("writes"), file, "byte-stable");
        }
    }
}

/// **A chain that stops before 1×1 is a valid file too** (§3.7).
#[test]
fn round_trip_a_partial_chain() {
    let mut t = texture(Format::Bc7, true, 64, 32, 9);
    t.levels.truncate(3);
    let file = ktx2::write(&t).expect("writes");
    assert_eq!(ktx2::read(&file).expect("reads"), t);
}

/// **The bytes are the specification's**, field by field, read here without
/// the module's reader.
#[test]
fn layout_follows_the_specification() {
    for (format, srgb) in FORMATS {
        let t = texture(format, srgb, 100, 60, 3);
        let f = ktx2::write(&t).expect("writes");
        let n = t.levels.len();
        assert_eq!(n, 7, "100×60 down to 1×1");

        // §3.1, §3.2–3.8: identifier and header.
        assert_eq!(f[..12], IDENTIFIER);
        let vk_format = match (format, srgb) {
            (Format::Uastc(_), _) => 0,
            (Format::Bc7, false) => 145,
            (Format::Bc7, true) => 146,
            (Format::Astc4x4, false) => 157,
            (Format::Astc4x4, true) => 158,
            _ => unreachable!(),
        };
        assert_eq!(u32_at(&f, 12), vk_format, "vkFormat");
        assert_eq!(u32_at(&f, 16), 1, "typeSize");
        assert_eq!((u32_at(&f, 20), u32_at(&f, 24)), (100, 60));
        assert_eq!(u32_at(&f, 28), 0, "pixelDepth");
        assert_eq!(u32_at(&f, 32), 0, "layerCount");
        assert_eq!(u32_at(&f, 36), 1, "faceCount");
        assert_eq!(u32_at(&f, 40), n as u32, "levelCount");
        assert_eq!(u32_at(&f, 44), 0, "supercompressionScheme");

        // §3.9: index. The DFD follows the level index.
        let (dfd_off, dfd_len) = (u32_at(&f, 48) as usize, u32_at(&f, 52) as usize);
        let (kvd_off, kvd_len) = (u32_at(&f, 56) as usize, u32_at(&f, 60) as usize);
        assert_eq!((u64_at(&f, 64), u64_at(&f, 72)), (0, 0), "no SGD");
        assert_eq!(dfd_off, 80 + 24 * n);

        // §3.10: one basic descriptor block with one sample, 44 bytes.
        assert_eq!(dfd_len, 44);
        assert_eq!(u32_at(&f, dfd_off), 44, "dfdTotalSize");
        assert_eq!(
            kvd_off - dfd_off,
            44,
            "dfdTotalSize = kvdByteOffset - dfdByteOffset"
        );
        let d = &f[dfd_off + 4..dfd_off + 44];
        assert_eq!(u32_at(d, 0), 0, "vendorId and descriptorType");
        assert_eq!(u32_at(d, 4), 2 | (40 << 16), "versionNumber 2, size 40");
        let (model, channel) = match format {
            Format::Uastc(c) => (
                166,
                match c {
                    UastcChannels::Rgb => 0,
                    UastcChannels::Rgba => 3,
                    UastcChannels::Rrr => 4,
                    UastcChannels::Rrrg => 5,
                    UastcChannels::Rg => 6,
                },
            ),
            Format::Bc7 => (134, 0),
            Format::Astc4x4 => (162, 0),
            _ => unreachable!(),
        };
        let transfer = if srgb { 2 } else { 1 };
        assert_eq!(
            d[8..12],
            [model, 1, transfer, 0],
            "model, BT.709, transfer, flags"
        );
        assert_eq!(d[12..16], [3, 3, 0, 0], "texelBlockDimension 4×4");
        assert_eq!(d[16..24], [16, 0, 0, 0, 0, 0, 0, 0], "bytesPlane");
        assert_eq!(
            d[24..28],
            [0, 0, 127, channel],
            "bitOffset, bitLength, channel"
        );
        assert_eq!(u32_at(d, 28), 0, "samplePosition");
        assert_eq!((u32_at(d, 32), u32_at(d, 36)), (0, u32::MAX));

        // §3.11, §5.2, §5.5: sorted keys, NUL-terminated, 4-byte padded.
        let kvd = &f[kvd_off..kvd_off + kvd_len];
        let mut entries = Vec::new();
        let mut at = 0;
        while at < kvd.len() {
            let len = u32_at(kvd, at) as usize;
            entries.push(&kvd[at + 4..at + 4 + len]);
            at += 4 + len.div_ceil(4) * 4;
        }
        assert_eq!(
            at, kvd_len,
            "kvdByteLength is the entries and their padding"
        );
        assert_eq!(entries[0], b"KTXorientation\0rd\0");
        let writer = concat!("KTXwriter\0texcomp ", env!("CARGO_PKG_VERSION"), "\0");
        assert_eq!(entries[1], writer.as_bytes());
        assert_eq!(entries.len(), 2);

        // §3.9.7, §3.13: levels smallest first, aligned to lcm(16, 4), the
        // largest ending the file; byteLength = uncompressedByteLength.
        let mut end = f.len();
        for (p, level) in t.levels.iter().enumerate() {
            let at = 80 + 24 * p;
            let (off, len, ulen) = (u64_at(&f, at), u64_at(&f, at + 8), u64_at(&f, at + 16));
            assert_eq!(len, level.len() as u64, "level {p} byteLength");
            assert_eq!(ulen, len, "level {p} uncompressedByteLength");
            assert_eq!(off % 16, 0, "level {p} aligned");
            assert_eq!(
                off + len,
                end as u64,
                "level {p} right before level {}",
                p as i64 - 1
            );
            assert_eq!(&f[off as usize..(off + len) as usize], &level[..]);
            end = off as usize;
        }
        // Between the key/value data and the smallest level, only mipPadding.
        assert!(end - (kvd_off + kvd_len) < 16);
        assert!(f[kvd_off + kvd_len..end].iter().all(|&b| b == 0));
    }
}

/// **A texture the writer cannot express is an error, not a file.**
#[test]
fn the_writer_refuses_invalid_textures() {
    let good = texture(Format::Bc7, false, 8, 8, 1);
    assert!(ktx2::write(&good).is_ok());

    let mut t = good.clone();
    t.levels.clear();
    assert!(matches!(ktx2::write(&t), Err(Ktx2Error::Header(_))));

    let mut t = good.clone();
    t.levels.push(vec![0; 16]);
    assert!(
        matches!(ktx2::write(&t), Err(Ktx2Error::Header(_))),
        "5 levels for 8×8"
    );

    let mut t = good.clone();
    t.levels[1].push(0);
    assert!(matches!(
        ktx2::write(&t),
        Err(Ktx2Error::Level { level: 1, .. })
    ));

    let mut t = good.clone();
    t.width = 0;
    assert!(matches!(ktx2::write(&t), Err(Ktx2Error::Header(_))));
}

fn uastc_fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/uastc")
}

fn ktx2_fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ktx2")
}

fn slurp(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn transcode(
    t: &Texture,
    to: fn(&[u8], &mut [u8]) -> Result<(), uastc::TranscodeError>,
) -> Vec<Vec<u8>> {
    t.levels
        .iter()
        .map(|level| {
            let mut out = vec![0; level.len()];
            to(level, &mut out).expect("valid blocks");
            out
        })
        .collect()
}

/// **Upstream's UASTC blocks survive the container**: wrapped, read back
/// and transcoded, they give upstream's ASTC 4×4 and BC7 bytes.
#[test]
fn upstream_blocks_through_the_container_transcode_bit_exact() {
    let names = [
        "alps-0",
        "nile-1",
        "coast-2",
        "modis-3",
        "paris-night-0",
        "gradient",
        "stripes",
        "alpha-edge",
        "noise",
    ];
    for level in 0..3 {
        for name in names {
            let stem = uastc_fixtures().join(format!("level{level}/{name}"));
            let blocks = slurp(&stem.with_extension("uastc"));
            let t = Texture {
                format: Format::Uastc(UastcChannels::Rgba),
                srgb: true,
                width: 256,
                height: 256,
                levels: vec![blocks],
            };
            let back = ktx2::read(&ktx2::write(&t).expect("writes")).expect("reads");
            assert_eq!(back, t);
            let astc = transcode(&back, uastc::transcode_to_astc_4x4);
            assert_eq!(
                astc[0],
                slurp(&stem.with_extension("astc")),
                "{name} level {level} ASTC"
            );
            let bc7 = transcode(&back, uastc::transcode_to_bc7);
            assert_eq!(
                bc7[0],
                slurp(&stem.with_extension("bc7")),
                "{name} level {level} BC7"
            );
        }
    }
}

/// The upstream files: name, size.
const UPSTREAM: [(&str, u32, u32); 3] = [
    ("alps-0", 256, 256),
    ("alpha-imagery", 64, 64),
    ("alps-0-100x60", 100, 60),
];

/// **Upstream's KTX 2.0 files read**, with their full chains, and every
/// level transcodes to the bytes upstream's own `-unpack` wrote.
#[test]
fn upstream_files_read_and_transcode_bit_exact() {
    for (name, w, h) in UPSTREAM {
        let dir = ktx2_fixtures().join("upstream");
        let t = ktx2::read(&slurp(&dir.join(format!("{name}.ktx2"))))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            matches!(t.format, Format::Uastc(_)),
            "{name}: {:?}",
            t.format
        );
        assert_eq!((t.width, t.height), (w, h), "{name}");
        let full = 32 - w.max(h).leading_zeros() as usize;
        assert_eq!(t.levels.len(), full, "{name}: the full chain");
        for (ext, to) in [
            (
                "astc",
                uastc::transcode_to_astc_4x4 as fn(&[u8], &mut [u8]) -> _,
            ),
            ("bc7", uastc::transcode_to_bc7),
        ] {
            let got = transcode(&t, to).concat();
            assert!(
                got == slurp(&dir.join(format!("{name}.{ext}"))),
                "{name}: {ext} levels differ"
            );
        }
    }
}

/// The files of `ours/` for one upstream input: the UASTC texture rewritten,
/// and its BC7 and ASTC transcodes, linear and sRGB.
fn ours(name: &str) -> Vec<(String, Texture)> {
    let up = ktx2::read(&slurp(
        &ktx2_fixtures().join(format!("upstream/{name}.ktx2")),
    ))
    .expect("reads");
    let mut out = vec![(format!("{name}-uastc"), up.clone())];
    for (fmt, to, tag) in [
        (
            Format::Bc7,
            uastc::transcode_to_bc7 as fn(&[u8], &mut [u8]) -> _,
            "bc7",
        ),
        (Format::Astc4x4, uastc::transcode_to_astc_4x4, "astc"),
    ] {
        for srgb in [false, true] {
            let t = Texture {
                format: fmt,
                srgb,
                levels: transcode(&up, to),
                ..up.clone()
            };
            let suffix = if srgb { "-srgb" } else { "" };
            out.push((format!("{name}-{tag}{suffix}"), t));
        }
    }
    out
}

/// **The writer writes, byte for byte, the files `ktx validate` and
/// upstream's `-unpack` accepted** in the capture.
#[test]
fn the_writer_still_writes_the_files_the_tools_accepted() {
    for (name, _, _) in &UPSTREAM[1..] {
        for (file, t) in ours(name) {
            let path = ktx2_fixtures().join(format!("ours/{file}.ktx2"));
            assert!(ktx2::write(&t).expect("writes") == slurp(&path), "{file}");
        }
    }
}

/// Writes `tests/fixtures/ktx2/ours`, for the capture to check. Run by hand.
#[test]
#[ignore = "writes the files the capture checks; see tests/fixtures/ktx2/README.md"]
fn write_cross_check_files() {
    let dir = ktx2_fixtures().join("ours");
    std::fs::create_dir_all(&dir).expect("dir");
    for (name, _, _) in &UPSTREAM[1..] {
        for (file, t) in ours(name) {
            std::fs::write(
                dir.join(format!("{file}.ktx2")),
                ktx2::write(&t).expect("writes"),
            )
            .expect("writes");
        }
    }
}

/// Cuts the non-square input upstream encodes. Run by hand.
#[test]
#[ignore = "writes an input of the capture; see tests/fixtures/ktx2/README.md"]
fn cut_ktx2_inputs() {
    let src = image::open(uastc_fixtures().join("tiles/alps-0.png"))
        .expect("png")
        .to_rgba8();
    let crop = image::imageops::crop_imm(&src, 37, 101, 100, 60).to_image();
    let dir = ktx2_fixtures().join("inputs");
    std::fs::create_dir_all(&dir).expect("dir");
    crop.save(dir.join("alps-0-100x60.png")).expect("png");
}

/// Offsets of the fields the corruption cases poke.
const VK_FORMAT: usize = 12;
const TYPE_SIZE: usize = 16;
const PIXEL_WIDTH: usize = 20;
const PIXEL_DEPTH: usize = 28;
const LAYER_COUNT: usize = 32;
const FACE_COUNT: usize = 36;
const LEVEL_COUNT: usize = 40;
const SCHEME: usize = 44;
const DFD_OFFSET: usize = 48;
const DFD_LENGTH: usize = 52;
const KVD_LENGTH: usize = 60;
const SGD_LENGTH: usize = 72;
const LEVEL0: usize = 80;

fn poke32(f: &mut [u8], at: usize, v: u32) {
    f[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn poke64(f: &mut [u8], at: usize, v: u64) {
    f[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn valid() -> Vec<u8> {
    ktx2::write(&texture(Format::Uastc(UastcChannels::Rgb), true, 20, 12, 5)).expect("writes")
}

/// **The classic corruptions give their error.**
#[test]
fn corrupt_files_give_their_error() {
    let f = valid();
    let dfd = u32_at(&f, DFD_OFFSET) as usize;
    let kvd = u32_at(&f, 56) as usize;
    let case = |edit: &dyn Fn(&mut Vec<u8>)| {
        let mut g = f.clone();
        edit(&mut g);
        ktx2::read(&g)
    };
    assert_eq!(case(&|g| g[5] = b'1'), Err(Ktx2Error::Identifier));
    assert_eq!(case(&|g| g.truncate(11)), Err(Ktx2Error::Identifier));
    assert_eq!(
        case(&|g| poke32(g, SCHEME, 2)),
        Err(Ktx2Error::Supercompression(2))
    );
    assert_eq!(
        case(&|g| poke32(g, SCHEME, 1)),
        Err(Ktx2Error::Supercompression(1))
    );
    assert_eq!(
        case(&|g| poke32(g, VK_FORMAT, 37)),
        Err(Ktx2Error::VkFormat(37))
    );
    assert!(matches!(
        case(&|g| poke32(g, TYPE_SIZE, 4)),
        Err(Ktx2Error::Header(_))
    ));
    assert!(matches!(
        case(&|g| poke32(g, PIXEL_WIDTH, 0)),
        Err(Ktx2Error::Header(_))
    ));
    assert!(matches!(
        case(&|g| poke32(g, PIXEL_DEPTH, 4)),
        Err(Ktx2Error::Unsupported(_))
    ));
    assert!(matches!(
        case(&|g| poke32(g, LAYER_COUNT, 2)),
        Err(Ktx2Error::Unsupported(_))
    ));
    assert!(matches!(
        case(&|g| poke32(g, FACE_COUNT, 6)),
        Err(Ktx2Error::Unsupported(_))
    ));
    assert!(matches!(
        case(&|g| poke32(g, FACE_COUNT, 2)),
        Err(Ktx2Error::Header(_))
    ));
    assert!(matches!(
        case(&|g| poke32(g, LEVEL_COUNT, 0)),
        Err(Ktx2Error::Header(_))
    ));
    assert!(
        matches!(
            case(&|g| poke32(g, LEVEL_COUNT, 6)),
            Err(Ktx2Error::Header(_))
        ),
        "20×12 has 5"
    );
    assert!(matches!(
        case(&|g| poke32(g, LEVEL_COUNT, u32::MAX)),
        Err(Ktx2Error::Header(_))
    ));
    assert!(matches!(
        case(&|g| poke64(g, SGD_LENGTH, 8)),
        Err(Ktx2Error::Header(_))
    ));
    // A huge level 0 whose level index still claims the old sizes.
    assert!(matches!(
        case(&|g| poke32(g, PIXEL_WIDTH, u32::MAX)),
        Err(Ktx2Error::Header(_) | Ktx2Error::Level { .. })
    ));
    // DFD.
    assert!(matches!(
        case(&|g| poke32(g, DFD_OFFSET, u32::MAX)),
        Err(Ktx2Error::OutOfBounds { .. })
    ));
    assert!(matches!(
        case(&|g| poke32(g, DFD_LENGTH, 40)),
        Err(Ktx2Error::Dfd(_))
    ));
    assert!(
        matches!(case(&|g| poke32(g, dfd, 48)), Err(Ktx2Error::Dfd(_))),
        "dfdTotalSize"
    );
    assert!(
        matches!(
            case(&|g| g[dfd + 4 + 8] = 163),
            Err(Ktx2Error::Unsupported(_))
        ),
        "ETC1S"
    );
    assert!(
        matches!(case(&|g| g[dfd + 4 + 8] = 134), Err(Ktx2Error::Dfd(_))),
        "BC7 model on UNDEFINED"
    );
    assert!(
        matches!(case(&|g| g[dfd + 4 + 10] = 8), Err(Ktx2Error::Dfd(_))),
        "transfer"
    );
    assert!(
        matches!(case(&|g| g[dfd + 4 + 12] = 5), Err(Ktx2Error::Dfd(_))),
        "6×4 blocks"
    );
    assert!(
        matches!(case(&|g| g[dfd + 4 + 16] = 8), Err(Ktx2Error::Dfd(_))),
        "bytesPlane0"
    );
    assert!(
        matches!(case(&|g| g[dfd + 4 + 27] = 9), Err(Ktx2Error::Dfd(_))),
        "channel"
    );
    assert!(
        matches!(case(&|g| g[dfd + 4 + 27] = 0x80), Err(Ktx2Error::Dfd(_))),
        "float"
    );
    assert!(
        matches!(case(&|g| g[dfd + 4 + 6] = 41), Err(Ktx2Error::Dfd(_))),
        "block size"
    );
    assert!(
        matches!(case(&|g| g[dfd + 4 + 4] = 1), Err(Ktx2Error::Dfd(_))),
        "version"
    );
    // Key/value data.
    assert!(matches!(
        case(&|g| poke32(g, kvd, u32::MAX)),
        Err(Ktx2Error::KeyValue(_))
    ));
    assert!(matches!(
        case(&|g| poke32(g, kvd, 1)),
        Err(Ktx2Error::KeyValue(_))
    ));
    assert!(matches!(
        case(&|g| {
            let v = u32_at(g, KVD_LENGTH) - 4;
            poke32(g, KVD_LENGTH, v)
        }),
        Err(Ktx2Error::KeyValue(_))
    ));
    assert!(
        matches!(
            case(&|g| {
                g[kvd + 4 + 14] = b'x';
                g[kvd + 4 + 17] = b'x';
            }),
            Err(Ktx2Error::KeyValue(_))
        ),
        "key without NUL"
    );
    assert!(
        matches!(
            case(&|g| g[kvd + 4 + 3] = b'z'),
            Err(Ktx2Error::KeyValue(_))
        ),
        "keys out of order"
    );
    assert_eq!(
        case(&|g| g[kvd + 4 + 16] = b'u'),
        Err(Ktx2Error::Orientation("ru".into()))
    );
    // Level index.
    assert!(matches!(
        case(&|g| poke64(g, LEVEL0, u64::MAX)),
        Err(Ktx2Error::Level { level: 0, .. } | Ktx2Error::OutOfBounds { .. })
    ));
    assert!(
        matches!(
            case(&|g| {
                let v = u64_at(g, LEVEL0) - 4;
                poke64(g, LEVEL0, v)
            }),
            Err(Ktx2Error::Level { level: 0, .. })
        ),
        "misaligned"
    );
    assert!(
        matches!(
            case(&|g| {
                let v = u64_at(g, LEVEL0) - 16;
                poke64(g, LEVEL0, v)
            }),
            Err(Ktx2Error::Level { level: 1, .. })
        ),
        "overlaps level 1"
    );
    assert!(
        matches!(
            case(&|g| poke64(g, LEVEL0 + 8, 16)),
            Err(Ktx2Error::Level { level: 0, .. })
        ),
        "byteLength"
    );
    assert!(
        matches!(
            case(&|g| poke64(g, LEVEL0 + 16, 0)),
            Err(Ktx2Error::Level { level: 0, .. })
        ),
        "uncompressed"
    );
    assert!(
        matches!(
            case(&|g| poke64(g, LEVEL0 + 24 * 4, 0)),
            Err(Ktx2Error::Level { level: 4, .. })
        ),
        "into the header"
    );
    assert!(matches!(
        case(&|g| {
            g.pop();
        }),
        Err(Ktx2Error::OutOfBounds { .. })
    ));
}

/// **Every truncation of a valid file is an error**, and none panics.
#[test]
fn every_truncation_is_an_error() {
    for f in [
        valid(),
        ktx2::write(&texture(Format::Astc4x4, true, 9, 33, 2)).expect("writes"),
    ] {
        for len in 0..f.len() {
            assert!(ktx2::read(&f[..len]).is_err(), "{len} of {} bytes", f.len());
        }
    }
}

/// **Random corruptions never panic**: every byte of the metadata replaced
/// in turn by a few values, then many files with several random bytes
/// changed, read whatever they hold.
#[test]
fn corruptions_never_panic() {
    let files = [
        valid(),
        ktx2::write(&texture(Format::Bc7, false, 16, 16, 4)).expect("writes"),
    ];
    for f in &files {
        let meta = u32_at(f, 56) as usize + u32_at(f, KVD_LENGTH) as usize;
        for at in 0..meta {
            for v in [0, 1, 3, 0x7f, 0x80, 0xff, f[at] ^ 0x10] {
                let mut g = f.clone();
                g[at] = v;
                let _ = ktx2::read(&g);
            }
        }
        let mut rng = Rng(0x5eed);
        for _ in 0..20_000 {
            let mut g = f.clone();
            for _ in 0..1 + rng.below(8) {
                // Mostly in the metadata, where the offsets are.
                let at = if rng.below(4) == 0 {
                    rng.below(g.len())
                } else {
                    rng.below(meta)
                };
                g[at] = rng.next() as u8;
            }
            let _ = ktx2::read(&g);
        }
    }
    // And bytes that only start like a KTX 2.0 file.
    let mut rng = Rng(77);
    for len in [12, 79, 80, 104, 200, 1000] {
        for _ in 0..500 {
            let mut g = rng.bytes(len);
            g[..12].copy_from_slice(&IDENTIFIER[..12.min(len)]);
            assert!(ktx2::read(&g).is_err());
        }
    }
}
