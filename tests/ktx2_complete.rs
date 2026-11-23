// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! KTX 2.0 as a whole: every texture type, every format, every
//! supercompression scheme the module declares, against two independent
//! readers and the Khronos files.
//!
//! # What this covers
//!
//! - **Round trip:** 1D, 2D, 3D, arrays, cubemaps and cubemap arrays, in
//!   UASTC, BC7, ASTC 4×4 and RGBA8, linear and sRGB, stored plain, with
//!   Zstandard and with ZLIB, with key/value pairs, premultiplied alpha and
//!   `levelCount` 0, come back from `read(write(t))` equal.
//! - **An independent reader** (the `ktx2` crate) reads every one of those
//!   files to the same header, levels, descriptor and key/value pairs.
//! - **The Khronos tool** (`ktx` of KTX-Software 4.4.2), when
//!   `TEXCOMP_KTX` names it: `ktx validate` passes every file with no error
//!   and no warning, and `ktx extract --raw` (libktx, the reference
//!   Zstandard and ZLIB) gives every image's bytes as `Texture::image`
//!   holds them; for UASTC, its Basis transcoder gives the BC7 and ASTC
//!   blocks this crate transcodes to. `TEXCOMP_REQUIRE_KTX=1` turns the
//!   skip without the tool into a failure.
//! - **KTX-Software's own files** (`tests/fixtures/ktx2/khronos`): RGBA8 2D,
//!   array with mips, 3D, UASTC with Zstandard, read to their documented
//!   shapes, their texels checked, and written back to files the readers
//!   accept; BasisLZ refused.
//! - **Hostile supercompressed input:** truncations and corruptions of
//!   Zstandard and ZLIB files are errors, never panics.
//!
//! # What it does not cover
//!
//! ETC1S/BasisLZ (refused), `vkFormat`s other than those of `Format`
//! (refused), and the levels of the Zstandard encoder (the pure-Rust one
//! writes its fastest level).

use std::path::{Path, PathBuf};
use std::process::Command;

use texcomp::ktx2::{self, Format, Ktx2Error, Supercompression, Texture, UastcChannels};

/// A deterministic byte stream (xorshift64*).
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

    /// Bytes with runs, so that they compress.
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let v = self.next() as u8;
            let run = 1 + self.below(6);
            for _ in 0..run.min(n - out.len()) {
                out.push(v);
            }
        }
        out
    }
}

/// A texture of the given shape, its levels random, `levels` of them.
#[allow(clippy::too_many_arguments)]
fn shaped(
    format: Format,
    srgb: bool,
    (width, height, depth): (u32, u32, u32),
    layers: u32,
    faces: u32,
    levels: usize,
    supercompression: Supercompression,
    seed: u64,
) -> Texture {
    let mut t = Texture::new(format, srgb, width, height, Vec::new());
    t.depth = depth;
    t.layers = layers;
    t.faces = faces;
    t.supercompression = supercompression;
    let mut rng = Rng(seed | 1);
    for level in 0..levels {
        let n = t.level_bytes(level).expect("a size") as usize;
        let bytes = if matches!(format, Format::Uastc(_)) {
            // Valid UASTC blocks, which the Khronos tool transcodes to
            // validate: random texels, encoded.
            (0..n / 16)
                .flat_map(|_| {
                    let px = rng.bytes(64);
                    let texels: [[u8; 4]; 16] =
                        std::array::from_fn(|i| px[4 * i..4 * i + 4].try_into().expect("4"));
                    texcomp::uastc::UastcBlock::encode(&texels, texcomp::uastc::Level::Fastest)
                        .to_bytes()
                })
                .collect()
        } else {
            rng.bytes(n)
        };
        t.levels.push(bytes);
    }
    t
}

/// Every type of §4.1 in the formats that can be it: (name, format, size,
/// layers, faces, levels).
fn shapes() -> Vec<(&'static str, Format, (u32, u32, u32), u32, u32, usize)> {
    let uastc = Format::Uastc(UastcChannels::Rgba);
    vec![
        ("1d", Format::Rgba8, (37, 0, 0), 0, 1, 6),
        ("1d-array", Format::Rgba8, (16, 0, 0), 3, 1, 5),
        ("2d-rgba8", Format::Rgba8, (33, 17, 0), 0, 1, 6),
        ("2d-bc7", Format::Bc7, (64, 20, 0), 0, 1, 7),
        ("2d-astc", Format::Astc4x4, (13, 9, 0), 0, 1, 4),
        ("2d-uastc", uastc, (32, 32, 0), 0, 1, 6),
        ("3d-rgba8", Format::Rgba8, (8, 4, 6), 0, 1, 4),
        ("3d-bc7", Format::Bc7, (16, 8, 5), 0, 1, 5),
        ("array-astc", Format::Astc4x4, (20, 12, 0), 4, 1, 5),
        ("array-rgba8", Format::Rgba8, (8, 8, 0), 2, 1, 4),
        ("cube-bc7", Format::Bc7, (16, 16, 0), 0, 6, 5),
        ("cube-rgba8", Format::Rgba8, (4, 4, 0), 0, 6, 3),
        ("cube-array-uastc", uastc, (8, 8, 0), 3, 6, 4),
    ]
}

const SCHEMES: [Supercompression; 3] = [
    Supercompression::None,
    Supercompression::Zstd,
    Supercompression::Zlib,
];

/// Every shape, sRGB and linear, every scheme, some with extras.
fn all_textures() -> Vec<(String, Texture)> {
    let mut out = Vec::new();
    for (i, (name, format, size, layers, faces, levels)) in shapes().into_iter().enumerate() {
        for (j, scheme) in SCHEMES.into_iter().enumerate() {
            for srgb in [false, true] {
                let mut t = shaped(
                    format,
                    srgb,
                    size,
                    layers,
                    faces,
                    levels,
                    scheme,
                    (i * 10 + j * 2 + srgb as usize) as u64,
                );
                if srgb && j == 1 {
                    t.premultiplied_alpha = true;
                    t.key_values = vec![
                        ("KTXswizzle".into(), b"rgba\0".to_vec()),
                        ("tile/source".into(), vec![1, 2, 3, 0, 255]),
                    ];
                }
                out.push((format!("{name}-{scheme:?}-{}", if srgb { "srgb" } else { "linear" }), t));
            }
        }
    }
    // levelCount 0: level 0, the mips to be generated.
    let mut t = shaped(
        Format::Rgba8,
        true,
        (24, 10, 0),
        0,
        1,
        1,
        Supercompression::Zstd,
        99,
    );
    t.generate_mips = true;
    out.push(("generate-mips".into(), t));
    out
}

/// **Every type, format and scheme comes back as written**, and rewriting
/// is byte-stable.
#[test]
fn every_type_format_and_scheme_round_trips() {
    for (name, t) in all_textures() {
        let file = ktx2::write(&t).unwrap_or_else(|e| panic!("{name}: {e}"));
        let back = ktx2::read(&file).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(back, t, "{name}");
        assert_eq!(ktx2::write(&back).expect("writes"), file, "{name}: byte-stable");
    }
}

/// **Supercompression shrinks what compresses**, and a level is stored as
/// its scheme's stream (checked by the independent reader's view of it).
#[test]
fn supercompressed_levels_are_their_schemes_streams() {
    let mut t = shaped(
        Format::Bc7,
        true,
        (64, 64, 0),
        0,
        1,
        7,
        Supercompression::None,
        3,
    );
    t.levels[0] = vec![7; t.levels[0].len()];
    let plain = ktx2::write(&t).expect("writes");
    for (scheme, magic) in [
        (Supercompression::Zstd, &[0x28, 0xb5, 0x2f, 0xfd][..]),
        (Supercompression::Zlib, &[0x78][..]),
    ] {
        t.supercompression = scheme;
        let file = ktx2::write(&t).expect("writes");
        assert!(file.len() < plain.len() / 2, "{scheme:?}: {} vs {}", file.len(), plain.len());
        let reader = ::ktx2::Reader::new(&file[..]).expect("the ktx2 crate reads it");
        for (level, l) in reader.levels().enumerate() {
            assert!(l.data.starts_with(magic), "{scheme:?} level {level}");
            assert_eq!(l.uncompressed_byte_length, t.levels[level].len() as u64);
        }
    }
}

/// **An independent reader** (the `ktx2` crate) sees what was written: the
/// header, the levels (plain ones byte for byte), the descriptor and the
/// key/value pairs.
#[test]
fn an_independent_reader_reads_every_file() {
    for (name, t) in all_textures() {
        let file = ktx2::write(&t).expect("writes");
        let r = ::ktx2::Reader::new(&file[..]).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let h = r.header();
        let vk = h.format.map_or(0, |f| f.value());
        assert_eq!(vk, t.format.vk_format(t.srgb), "{name}");
        assert_eq!(
            (h.pixel_width, h.pixel_height, h.pixel_depth),
            (t.width, t.height, t.depth),
            "{name}"
        );
        assert_eq!((h.layer_count, h.face_count), (t.layers, t.faces), "{name}");
        let count = if t.generate_mips { 0 } else { t.levels.len() as u32 };
        assert_eq!(h.level_count, count, "{name}");
        let scheme = h.supercompression_scheme.map_or(0, |s| s.value());
        assert_eq!(
            scheme,
            match t.supercompression {
                Supercompression::None => 0,
                Supercompression::Zstd => 2,
                Supercompression::Zlib => 3,
                _ => unreachable!(),
            },
            "{name}"
        );
        assert_eq!(r.levels().len(), t.levels.len(), "{name}");
        for (level, l) in r.levels().enumerate() {
            assert_eq!(l.uncompressed_byte_length, t.levels[level].len() as u64, "{name}");
            if t.supercompression == Supercompression::None {
                assert!(l.data == &t.levels[level][..], "{name} level {level}");
            }
        }
        let dfd = r.basic_dfd().expect("a basic descriptor block");
        let model = dfd.color_model.map_or(0, |m| m.value());
        let expected_model = match t.format {
            Format::Uastc(_) => 166,
            Format::Bc7 => 134,
            Format::Astc4x4 => 162,
            Format::Rgba8 => 1,
            _ => unreachable!(),
        };
        assert_eq!(model, expected_model, "{name}");
        let transfer = dfd.transfer_function.map_or(0, |f| f.value());
        assert_eq!(transfer, if t.srgb { 2 } else { 1 }, "{name}");
        assert_eq!(r.is_alpha_premultiplied(), Some(t.premultiplied_alpha), "{name}");
        let samples = &dfd.sample_information;
        if t.format == Format::Rgba8 {
            assert_eq!(dfd.bytes_planes[0], 4, "{name}");
            let channels: Vec<(u16, u8, u8)> = samples
                .iter()
                .map(|s| (s.bit_offset, s.bit_length.get(), s.channel_type))
                .collect();
            assert_eq!(channels, [(0, 8, 0), (8, 8, 1), (16, 8, 2), (24, 8, 15)], "{name}");
            let alpha_linear = samples[3]
                .channel_type_qualifiers
                .contains(::ktx2::dfd::ChannelTypeQualifiers::LINEAR);
            assert_eq!(alpha_linear, t.srgb, "{name}: sRGB alpha is linear");
        } else {
            assert_eq!(dfd.bytes_planes[0], 16, "{name}");
            assert_eq!(samples.len(), 1, "{name}");
            assert_eq!(samples[0].bit_length.get(), 128, "{name}");
        }
        let kv: Vec<(String, Vec<u8>)> = r
            .key_value_data()
            .map(|(k, v)| (k.to_owned(), v.to_vec()))
            .collect();
        let orientation = match (t.height, t.depth) {
            (0, _) => &b"r\0"[..],
            (_, 0) => b"rd\0",
            _ => b"rdi\0",
        };
        assert!(kv.contains(&("KTXorientation".into(), orientation.to_vec())), "{name}");
        assert!(r.writer().is_some_and(|w| w.starts_with("texcomp ")), "{name}");
        for pair in &t.key_values {
            assert!(kv.contains(pair), "{name}: {pair:?}");
        }
        assert!(
            kv.windows(2).all(|w| w[0].0.as_bytes() < w[1].0.as_bytes()),
            "{name}: sorted"
        );
    }
}

/// **What the writer cannot write it refuses**, writing nothing.
#[test]
fn the_writer_refuses_what_ktx2_does_not_define() {
    let ok = || shaped(Format::Bc7, false, (16, 16, 0), 0, 1, 1, Supercompression::None, 1);
    let refuse = |edit: &dyn Fn(&mut Texture)| {
        let mut t = ok();
        edit(&mut t);
        ktx2::write(&t)
    };
    assert!(ktx2::write(&ok()).is_ok());
    assert!(matches!(refuse(&|t| t.faces = 2), Err(Ktx2Error::Header(_))));
    assert!(matches!(
        refuse(&|t| {
            t.faces = 6;
            t.width = 32;
        }),
        Err(Ktx2Error::Header(_))
    ), "non-square cubemap");
    assert!(matches!(
        refuse(&|t| {
            t.depth = 4;
            t.layers = 2;
        }),
        Err(Ktx2Error::Unsupported(_))
    ), "3D array");
    assert!(matches!(refuse(&|t| t.height = 0), Err(Ktx2Error::Header(_))), "1D BC7");
    assert!(matches!(refuse(&|t| t.generate_mips = true), Err(Ktx2Error::Header(_))));
    assert!(matches!(refuse(&|t| t.layers = 2), Err(Ktx2Error::Level { level: 0, .. })));
    for key in ["", "a\0b", "KTXwriter", "KTXorientation"] {
        assert!(
            matches!(
                refuse(&|t| t.key_values = vec![(key.into(), vec![])]),
                Err(Ktx2Error::KeyValue(_))
            ),
            "{key:?}"
        );
    }
    assert!(matches!(
        refuse(&|t| t.key_values = vec![("k".into(), vec![]), ("k".into(), vec![1])]),
        Err(Ktx2Error::KeyValue(_))
    ));
}

/// **`Texture::image` cuts the images in the file's order**: layer, then
/// face, then depth slice (§3.13).
#[test]
fn images_are_cut_in_the_files_order() {
    let mut t = shaped(
        Format::Rgba8,
        false,
        (2, 2, 0),
        2,
        6,
        1,
        Supercompression::None,
        1,
    );
    t.levels[0] = (0..2 * 6).flat_map(|i| [i as u8; 16]).collect();
    for layer in 0..2 {
        for face in 0..6 {
            let image = t.image(0, layer, face, 0).expect("an image");
            assert_eq!(image, [(layer * 6 + face) as u8; 16]);
        }
    }
    assert_eq!(t.image(0, 2, 0, 0), None);
    assert_eq!(t.image(0, 0, 6, 0), None);
    assert_eq!(t.image(0, 0, 0, 1), None);
    let mut v = shaped(
        Format::Bc7,
        false,
        (4, 4, 3),
        0,
        1,
        2,
        Supercompression::None,
        2,
    );
    v.levels[0] = (0..3).flat_map(|z| [z as u8; 16]).collect();
    for z in 0..3 {
        assert_eq!(v.image(0, 0, 0, z).expect("a slice"), [z as u8; 16]);
    }
    // Level 1 of a depth of 3 is one slice.
    assert_eq!(v.level_size(1), Some((2, 2, 1)));
    assert!(v.image(1, 0, 0, 1).is_none());
}

// --- the Khronos tool -------------------------------------------------------

/// The `ktx` tool, if `TEXCOMP_KTX` names it.
fn ktx_tool() -> Option<PathBuf> {
    match std::env::var_os("TEXCOMP_KTX") {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            assert!(
                std::env::var_os("TEXCOMP_REQUIRE_KTX").is_none(),
                "TEXCOMP_REQUIRE_KTX is set and TEXCOMP_KTX does not name the ktx tool"
            );
            eprintln!("skipping: TEXCOMP_KTX does not name the Khronos ktx tool");
            None
        }
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("texcomp-ktx2-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// `ktx validate` on `path`: its output, and whether it passed without an
/// error or a warning, but the one a custom key gives (7010: the files with
/// pairs of their own carry one on purpose).
fn validate(tool: &Path, path: &Path) -> (bool, String) {
    let out = Command::new(tool)
        .arg("validate")
        .arg(path)
        .output()
        .expect("runs ktx");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let clean = text
        .lines()
        .filter(|l| l.contains("error") || (l.contains("warning-") && !l.contains("warning-7010")))
        .count()
        == 0;
    (out.status.success() && clean, text)
}

/// `ktx extract --raw` of one image of `t` (written at `path`), with an
/// optional transcode. The tool takes a layer, face or depth option only for
/// a texture that has them.
#[allow(clippy::too_many_arguments)]
fn extract(
    tool: &Path,
    path: &Path,
    t: &Texture,
    level: usize,
    layer: u32,
    face: u32,
    slice: u32,
    transcode: Option<&str>,
) -> Vec<u8> {
    let out = path.with_extension(format!("{level}-{layer}-{face}-{slice}.raw"));
    let mut cmd = Command::new(tool);
    cmd.arg("extract").arg("--raw");
    if let Some(t) = transcode {
        cmd.args(["--transcode", t]);
    }
    cmd.args(["--level", &level.to_string()]);
    if t.layers > 0 {
        cmd.args(["--layer", &layer.to_string()]);
    }
    if t.faces == 6 {
        cmd.args(["--face", &face.to_string()]);
    }
    if t.depth > 0 {
        cmd.args(["--depth", &slice.to_string()]);
    }
    let status = cmd.arg(path).arg(&out).output().expect("runs ktx");
    assert!(
        status.status.success(),
        "ktx extract {}: {}",
        path.display(),
        String::from_utf8_lossy(&status.stderr)
    );
    std::fs::read(&out).expect("extracted")
}

/// **`ktx validate` accepts every file the writer makes, with no warning,
/// and libktx reads every image of it as `Texture::image` holds it.**
#[test]
fn the_khronos_tool_accepts_and_reads_every_file() {
    let Some(tool) = ktx_tool() else { return };
    let dir = scratch("tool");
    for (name, t) in all_textures() {
        let path = dir.join(format!("{name}.ktx2"));
        std::fs::write(&path, ktx2::write(&t).expect("writes")).expect("writes");
        let (ok, text) = validate(&tool, &path);
        assert!(ok, "{name}: {text}");
        // Random bytes are no valid UASTC: libktx would decode them, not
        // extract them (`libktx_transcodes_uastc_as_this_crate_does` takes
        // real blocks).
        if matches!(t.format, Format::Uastc(_)) {
            continue;
        }
        // Every image of the first and last levels, decompressed by libktx.
        for level in [0, t.levels.len() - 1] {
            let (_, _, d) = t.level_size(level).expect("a size");
            for layer in 0..t.layers.max(1) {
                for face in 0..t.faces {
                    for slice in 0..d {
                        let raw = extract(&tool, &path, &t, level, layer, face, slice, None);
                        let ours = t.image(level, layer, face, slice).expect("an image");
                        assert!(raw == ours, "{name}: level {level} layer {layer} face {face} slice {slice}");
                    }
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// **UASTC through the Basis transcoder libktx carries gives this crate's
/// BC7 and ASTC blocks**, on real UASTC blocks.
#[test]
fn libktx_transcodes_uastc_as_this_crate_does() {
    let Some(tool) = ktx_tool() else { return };
    let dir = scratch("transcode");
    // Real UASTC: an imagery tile encoded by this crate's encoder.
    let img = image::open(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/aerial/lyon-suburbs.jpeg"),
    )
    .expect("jpeg")
    .to_rgba8();
    let mut uastc = vec![0u8; 256 * 256];
    texcomp::uastc::encode_image(img.as_raw(), 256, 256, texcomp::uastc::Level::Fastest, &mut uastc)
        .expect("encodes");
    for scheme in SCHEMES {
        let mut t = Texture::new(Format::Uastc(UastcChannels::Rgb), true, 256, 256, vec![uastc.clone()]);
        t.supercompression = scheme;
        let path = dir.join(format!("uastc-{scheme:?}.ktx2"));
        std::fs::write(&path, ktx2::write(&t).expect("writes")).expect("writes");
        let (ok, text) = validate(&tool, &path);
        assert!(ok, "{scheme:?}: {text}");
        for (target, ours) in [
            ("bc7", texcomp::uastc::transcode_to_bc7 as fn(&[u8], &mut [u8]) -> _),
            ("astc", texcomp::uastc::transcode_to_astc_4x4),
        ] {
            let theirs = extract(&tool, &path, &t, 0, 0, 0, 0, Some(target));
            let mut mine = vec![0u8; uastc.len()];
            ours(&uastc, &mut mine).expect("transcodes");
            assert!(theirs == mine, "{scheme:?} {target}: libktx's transcode differs");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// --- KTX-Software's files ---------------------------------------------------

fn khronos(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/ktx2/khronos")
            .join(name),
    )
    .expect("fixture")
}

/// Level 0 of an RGB UASTC texture decoded by this crate, as RGB8 rows.
fn uastc_rgb8(t: &Texture) -> Vec<u8> {
    let (w, h) = (t.width as usize, t.height as usize);
    let bx = w.div_ceil(4);
    let mut out = vec![0u8; w * h * 3];
    for (b, block) in t.levels[0].chunks_exact(16).enumerate() {
        let texels = texcomp::uastc::UastcBlock::from_bytes(block.try_into().expect("16 bytes"))
            .decode_rgba8(false)
            .expect("a valid block");
        for (i, p) in texels.iter().enumerate() {
            let (x, y) = ((b % bx) * 4 + (i & 3), (b / bx) * 4 + (i >> 2));
            if x < w && y < h {
                out[(y * w + x) * 3..][..3].copy_from_slice(&p[..3]);
            }
        }
    }
    out
}

/// FNV-1a, to pin a level's bytes in a line.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    })
}

/// **KTX-Software's files read to their documented shapes**, their texels
/// as `ktx extract` gives them (pinned by hash in the fixtures' README), and
/// rewritten they read back the same.
#[test]
fn khronos_files_read_and_rewrite() {
    // (file, format, sRGB, size, layers, faces, levels, scheme, FNV-1a of
    // level 0 as `ktx extract --raw --all` concatenates it).
    let cases: [(&str, Format, bool, (u32, u32, u32), u32, u32, usize, Supercompression, u64); 5] = [
        ("rgba-reference-u.ktx2", Format::Rgba8, true, (128, 128, 0), 0, 1, 1, Supercompression::None, 0x53e6_afeb_39bf_9b16),
        ("cyan_rgba_reference_u.ktx2", Format::Rgba8, true, (64, 64, 0), 0, 1, 1, Supercompression::None, 0x4fa9_b333_b957_a325),
        ("arraytex_7_mipmap_reference_u.ktx2", Format::Rgba8, true, (16, 16, 0), 7, 1, 5, Supercompression::None, 0x8bf2_6dae_66a8_1f25),
        ("3dtex_7_reference_u.ktx2", Format::Rgba8, true, (16, 16, 7), 0, 1, 1, Supercompression::None, 0x8bf2_6dae_66a8_1f25),
        ("color_grid_uastc_zstd.ktx2", Format::Uastc(UastcChannels::Rgb), true, (1024, 1024, 0), 0, 1, 1, Supercompression::Zstd, 0xc5ba_5f19_7eed_2fe8),
    ];
    for (name, format, srgb, (w, h, d), layers, faces, levels, scheme, hash) in cases {
        let t = ktx2::read(&khronos(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((t.format, t.srgb), (format, srgb), "{name}");
        assert_eq!((t.width, t.height, t.depth), (w, h, d), "{name}");
        assert_eq!((t.layers, t.faces, t.levels.len()), (layers, faces, levels), "{name}");
        assert_eq!(t.supercompression, scheme, "{name}");
        // `ktx extract --raw` decodes UASTC (to RGB8 for RGB channels):
        // compared with this crate's decode of the blocks read.
        let level0 = match t.format {
            Format::Uastc(_) => uastc_rgb8(&t),
            _ => t.levels[0].clone(),
        };
        assert_eq!(fnv(&level0), hash, "{name}: level 0 differs from ktx extract's");
        let again = ktx2::read(&ktx2::write(&t).expect("writes")).expect("reads");
        assert_eq!(again, t, "{name}");
    }
    assert_eq!(
        ktx2::read(&khronos("rgba-mipmap-reference-basis.ktx2")),
        Err(Ktx2Error::Supercompression(1)),
        "BasisLZ"
    );
}

/// Prints the FNV-1a of level 0 of each Khronos fixture as `ktx extract
/// --raw --all` writes it, for the README. Run by hand with the tool.
#[test]
#[ignore = "prints the hashes khronos_files_read_and_rewrite pins"]
fn print_khronos_hashes() {
    let Some(tool) = ktx_tool() else { return };
    let dir = scratch("hashes");
    for name in [
        "rgba-reference-u.ktx2",
        "cyan_rgba_reference_u.ktx2",
        "arraytex_7_mipmap_reference_u.ktx2",
        "3dtex_7_reference_u.ktx2",
        "color_grid_uastc_zstd.ktx2",
    ] {
        let path = dir.join(name);
        std::fs::write(&path, khronos(name)).expect("copies");
        let t = ktx2::read(&khronos(name)).expect("reads");
        let (_, _, d) = t.level_size(0).expect("size");
        let mut level0 = Vec::new();
        for layer in 0..t.layers.max(1) {
            for face in 0..t.faces {
                for slice in 0..d {
                    level0.extend(extract(&tool, &path, &t, 0, layer, face, slice, None));
                }
            }
        }
        println!("{name}: {:#018x}", fnv(&level0));
    }
}

// --- hostile input ----------------------------------------------------------

/// **Truncated and corrupted supercompressed files are errors, never
/// panics**, and a level whose stream decompresses to another size is
/// refused.
#[test]
fn supercompressed_corruptions_never_panic() {
    for scheme in [Supercompression::Zstd, Supercompression::Zlib] {
        let t = shaped(Format::Astc4x4, true, (32, 16, 0), 2, 1, 6, scheme, 11);
        let f = ktx2::write(&t).expect("writes");
        for len in 0..f.len() {
            assert!(ktx2::read(&f[..len]).is_err(), "{scheme:?}: {len} of {}", f.len());
        }
        let mut rng = Rng(0xfeed);
        for _ in 0..5_000 {
            let mut g = f.clone();
            for _ in 0..1 + rng.below(6) {
                let at = rng.below(g.len());
                g[at] = rng.next() as u8;
            }
            let _ = ktx2::read(&g);
        }
        // Level 0's uncompressedByteLength one block short.
        let mut g = f.clone();
        let at = 80 + 16;
        let v = u64::from_le_bytes(g[at..at + 8].try_into().unwrap()) - 16;
        g[at..at + 8].copy_from_slice(&v.to_le_bytes());
        assert!(matches!(ktx2::read(&g), Err(Ktx2Error::Level { level: 0, .. })));
    }
}
