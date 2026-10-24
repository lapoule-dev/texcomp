// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2019-2026 Binomial LLC; modifications Copyright (c) lapoule.dev
//
// Ported from Basis Universal v2.50 (tag `v2_50`, commit
// 9bebe16726b3a61c8c213eeee3b7cffb462ef34e), `transcoder/basisu_transcoder.cpp`,
// Copyright (C) 2019-2026 Binomial LLC, licensed under the Apache License,
// Version 2.0 (see LICENSE-APACHE-UPSTREAM and NOTICE). Modified by
// lapoule.dev: translated from C++ to Rust, writing into byte arrays instead
// of 32-bit words.

//! Writing the ASTC 4×4 block a UASTC block stands for.

use super::tables::{
    ASTC_BLOCK_MODE, BISE_RANGES, ENDPOINT_RANGES, QUINT_ENCODE, SOLID, TRIT_ENCODE,
};
use super::unpack::Unpacked;

// Ported from Basis Universal v2.50, `astc_set_bits`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// ORs the low `count` bits of `value` in at bit `pos`, a byte at a time.
/// As upstream, bits of `value` above `count` that share the last byte are
/// written too; every caller passes values that fit.
fn set_bits(buf: &mut [u8], pos: &mut usize, mut value: u32, mut count: u32) {
    while count > 0 {
        let n = count.min(8 - (*pos & 7) as u32);
        buf[*pos >> 3] |= (value << (*pos & 7)) as u8;
        *pos += n as usize;
        count -= n;
        value >>= n;
    }
}

// Ported from Basis Universal v2.50, `astc_set_bits_1_to_9`.
// Apache-2.0, (C) Binomial LLC. Modified.
fn set_bits_1_to_9(buf: &mut [u8], pos: &mut usize, code: u32, size: u32) {
    if size > 0 {
        let shift = (*pos & 7) as u32;
        let val = code << shift;
        let at = *pos >> 3;
        buf[at] |= val as u8;
        if size > 8 - shift {
            buf[at + 1] |= (val >> 8) as u8;
        }
        *pos += size as usize;
    }
}

fn extract_bits(bits: u32, low: u32, high: u32) -> u32 {
    (bits >> low) & ((1 << (high - low + 1)) - 1)
}

// Ported from Basis Universal v2.50, `astc_encode_trits`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// Five values of a trit range with `n` low bits each.
fn encode_trits(buf: &mut [u8], values: &[u8; 5], pos: &mut usize, n: u32) {
    let mut trits = 0;
    let mut bits = [0u32; 5];
    let mask = (1 << n) - 1;
    for (i, &v) in values.iter().enumerate() {
        trits += (v as u32 >> n) * [1, 3, 9, 27, 81][i];
        bits[i] = v as u32 & mask;
    }
    let t = TRIT_ENCODE[trits as usize] as u32;
    set_bits(
        buf,
        pos,
        bits[0] | (extract_bits(t, 0, 1) << n) | (bits[1] << (2 + n)),
        n * 2 + 2,
    );
    set_bits(
        buf,
        pos,
        extract_bits(t, 2, 3)
            | (bits[2] << 2)
            | (extract_bits(t, 4, 4) << (2 + n))
            | (bits[3] << (3 + n))
            | (extract_bits(t, 5, 6) << (3 + n * 2))
            | (bits[4] << (5 + n * 2))
            | (extract_bits(t, 7, 7) << (5 + n * 3)),
        n * 3 + 6,
    );
}

// Ported from Basis Universal v2.50, `astc_encode_quints`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// Three values of a quint range with `n` low bits each.
fn encode_quints(buf: &mut [u8], values: &[u8; 5], pos: &mut usize, n: u32) {
    let mut quints = 0;
    let mut bits = [0u32; 3];
    let mask = (1 << n) - 1;
    for i in 0..3 {
        quints += (values[i] as u32 >> n) * [1, 5, 25][i];
        bits[i] = values[i] as u32 & mask;
    }
    let t = QUINT_ENCODE[quints as usize] as u32;
    set_bits(
        buf,
        pos,
        bits[0]
            | (extract_bits(t, 0, 2) << n)
            | (bits[1] << (3 + n))
            | (extract_bits(t, 3, 4) << (3 + n * 2))
            | (bits[2] << (5 + n * 2))
            | (extract_bits(t, 5, 6) << (5 + n * 3)),
        7 + n * 3,
    );
}

// Ported from Basis Universal v2.50, `astc_pack_bise`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// ORs `values`, integer-sequence encoded in `range`, into `dst` at bit
/// `pos`.
fn pack_bise(dst: &mut [u8; 16], values: &[u8], mut pos: usize, range: usize) {
    // Upstream packs into five 32-bit words and keeps the first four.
    let mut temp = [0u8; 20];
    let [bits, trits, quints] = BISE_RANGES[range];
    let n = bits as u32;
    let group = if trits != 0 {
        5
    } else if quints != 0 {
        3
    } else {
        0
    };
    if group != 0 {
        for chunk in values.chunks(group) {
            let mut vals = [0u8; 5];
            vals[..chunk.len()].copy_from_slice(chunk);
            if group == 5 {
                encode_trits(&mut temp, &vals, &mut pos, n);
            } else {
                encode_quints(&mut temp, &vals, &mut pos, n);
            }
        }
    } else {
        for &v in values {
            set_bits_1_to_9(&mut temp, &mut pos, v as u32, n);
        }
    }
    for (d, t) in dst.iter_mut().zip(temp) {
        *d |= t;
    }
}

// Ported from Basis Universal v2.50, `pack_astc_solid_block`.
// Apache-2.0, (C) Binomial LLC. Modified.
/// An ASTC void-extent block of one LDR colour.
fn solid_block(rgba: [u8; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&[0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
    let mut pos = 64;
    for c in rgba {
        let c = c as u32;
        set_bits(&mut out, &mut pos, c | (c << 8), 16);
    }
    out
}

// Ported from Basis Universal v2.50, `pack_astc_block` and
// `transcode_uastc_to_astc`. Apache-2.0, (C) Binomial LLC. Modified.
/// The ASTC 4×4 block of a block read with blue contraction undone.
pub(crate) fn pack(u: &Unpacked) -> [u8; 16] {
    if u.mode == SOLID {
        return solid_block(u.solid);
    }
    let mut out = [0u8; 16];
    let mode = ASTC_BLOCK_MODE[u.mode];
    out[0] = mode as u8;
    out[1] = (mode >> 8) as u8;

    let total_weights = if u.dual_plane { 32 } else { 16 };
    let mut pos = 11; // block mode bits
    let bits_per_weight = BISE_RANGES[u.weight_range as usize][0] as usize;

    set_bits_1_to_9(&mut out, &mut pos, u.subsets as u32 - 1, 2);
    if u.subsets == 1 {
        set_bits_1_to_9(&mut out, &mut pos, u.cem as u32, 4);
    } else {
        set_bits(&mut out, &mut pos, u.seed as u32, 10);
        // All subsets share the CEM: two zero bits, then the CEM.
        set_bits_1_to_9(&mut out, &mut pos, ((u.cem as u32) << 2) & 63, 6);
    }

    if u.dual_plane {
        let mut ccs_pos = 128 - total_weights * bits_per_weight - 2;
        set_bits_1_to_9(&mut out, &mut ccs_pos, u.ccs as u32, 2);
    }

    let pairs = (1 + (u.cem as usize >> 2)) * u.subsets;
    pack_bise(
        &mut out,
        &u.endpoints[..pairs * 2],
        pos,
        ENDPOINT_RANGES[u.mode] as usize,
    );

    // The weights, from the block's top bit down, each bit-reversed.
    let n = bits_per_weight;
    for (i, &w) in u.weights.iter().take(total_weights).enumerate() {
        let ofs = 128 - n - i * n;
        let rev = (w as u32).reverse_bits() >> (32 - n);
        let rev = rev << (ofs & 7);
        let at = ofs >> 3;
        out[at] |= rev as u8;
        if at + 1 < 16 {
            out[at + 1] |= (rev >> 8) as u8;
        }
    }
    out
}
