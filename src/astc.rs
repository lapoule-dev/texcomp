// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! The parts of ASTC's decoding the encoders invert, written from the Khronos
//! Data Format Specification 1.3: §23.12 (integer sequence encoding), §23.17
//! (weight unquantisation), §23.18 (endpoint unquantisation, through the
//! crate's table), §23.21 (the partition function) and Tables 146–147 (block
//! modes).

/// The five trits that the packed byte `t` decodes to (§23.12).
pub(crate) fn decode_trits(t: u32) -> [u32; 5] {
    let bit = |v: u32, i: u32| (v >> i) & 1;
    let bits = |v: u32, hi: u32, lo: u32| (v >> lo) & ((1 << (hi - lo + 1)) - 1);
    let (c, t4, t3);
    if bits(t, 4, 2) == 0b111 {
        c = (bits(t, 7, 5) << 2) | bits(t, 1, 0);
        t4 = 2;
        t3 = 2;
    } else {
        c = bits(t, 4, 0);
        if bits(t, 6, 5) == 0b11 {
            t4 = 2;
            t3 = bit(t, 7);
        } else {
            t4 = bit(t, 7);
            t3 = bits(t, 6, 5);
        }
    }
    let (t2, t1, t0);
    if bits(c, 1, 0) == 0b11 {
        t2 = 2;
        t1 = bit(c, 4);
        t0 = (bit(c, 3) << 1) | (bit(c, 2) & !bit(c, 3) & 1);
    } else if bits(c, 3, 2) == 0b11 {
        t2 = 2;
        t1 = 2;
        t0 = bits(c, 1, 0);
    } else {
        t2 = bit(c, 4);
        t1 = bits(c, 3, 2);
        t0 = (bit(c, 1) << 1) | (bit(c, 0) & !bit(c, 1) & 1);
    }
    [t0, t1, t2, t3, t4]
}

/// For each of the 243 trit quintuples `t0 + 3·t1 + 9·t2 + 27·t3 + 81·t4`,
/// the smallest byte that decodes to it.
///
/// Smallest matters: a block's last group of weights is partial, its
/// unused trits are 0, and the bits past the stream's end are not
/// stored. The smallest encoding of `(t0, 0, 0, 0, 0)` is `t0` itself,
/// whose high bits are 0 and survive the truncation.
pub(crate) fn trit_encodings() -> Vec<u32> {
    let mut table = vec![u32::MAX; 243];
    for t in 0..256u32 {
        let [t0, t1, t2, t3, t4] = decode_trits(t);
        let key = (t0 + 3 * t1 + 9 * t2 + 27 * t3 + 81 * t4) as usize;
        if table[key] == u32::MAX {
            table[key] = t;
        }
    }
    table
}

/// The three quints that the packed 7 bits `q` decode to (§23.12).
pub(crate) fn decode_quints(q: u32) -> [u32; 3] {
    let bit = |v: u32, i: u32| (v >> i) & 1;
    let bits = |v: u32, hi: u32, lo: u32| (v >> lo) & ((1 << (hi - lo + 1)) - 1);
    if bits(q, 2, 1) == 0b11 && bits(q, 6, 5) == 0 {
        let q0 = 4;
        let q1 = 4;
        let q2 = (bit(q, 0) << 2) | ((bit(q, 4) & (bit(q, 0) ^ 1)) << 1) | (bit(q, 3) & (bit(q, 0) ^ 1));
        return [q0, q1, q2];
    }
    let (c, q2);
    if bits(q, 2, 1) == 0b11 {
        q2 = 4;
        c = (bits(q, 4, 3) << 3) | ((bits(q, 6, 5) ^ 0b11) << 1) | bit(q, 0);
    } else {
        q2 = bits(q, 6, 5);
        c = bits(q, 4, 0);
    }
    let (q1, q0);
    if bits(c, 2, 0) == 0b101 {
        q1 = 4;
        q0 = bits(c, 4, 3);
    } else {
        q1 = bits(c, 4, 3);
        q0 = bits(c, 2, 0);
    }
    [q0, q1, q2]
}

/// For each of the 125 quint triples `q0 + 5·q1 + 25·q2`, the smallest
/// 7-bit value that decodes to it (smallest, for the reason
/// [`trit_encodings`] gives).
pub(crate) fn quint_encodings() -> Vec<u32> {
    let mut table = vec![u32::MAX; 125];
    for q in 0..128u32 {
        let [q0, q1, q2] = decode_quints(q);
        let key = (q0 + 5 * q1 + 25 * q2) as usize;
        if table[key] == u32::MAX {
            table[key] = q;
        }
    }
    table
}

/// ASTC's 21 integer-sequence ranges, by index (§23.12, Table 155 order):
/// `(bits, trits, quints)`; the range is `0 .. (1 + 2·trits + 4·quints)·2^bits`.
pub(crate) const RANGES: [(u32, u32, u32); 21] = [
    (1, 0, 0),
    (0, 1, 0),
    (2, 0, 0),
    (0, 0, 1),
    (1, 1, 0),
    (3, 0, 0),
    (1, 0, 1),
    (2, 1, 0),
    (4, 0, 0),
    (2, 0, 1),
    (3, 1, 0),
    (5, 0, 0),
    (3, 0, 1),
    (4, 1, 0),
    (6, 0, 0),
    (4, 0, 1),
    (5, 1, 0),
    (7, 0, 0),
    (5, 0, 1),
    (6, 1, 0),
    (8, 0, 0),
];

/// The number of values in range `r`.
pub(crate) const fn levels(r: usize) -> u32 {
    let (bits, trits, quints) = RANGES[r];
    (1 + 2 * trits + 4 * quints) << bits
}

/// The bits `count` values of range `r` take in an integer sequence
/// (§23.12).
pub(crate) const fn ise_bits(count: u32, r: usize) -> u32 {
    let (bits, trits, quints) = RANGES[r];
    count * bits + (count * 8 * trits).div_ceil(5) + (count * 7 * quints).div_ceil(3)
}

/// The 0..=64 weight a stored value of weight range `r` (0..=11) decodes to
/// (§23.17, Tables 167 and 168).
pub(crate) fn unquantize_weight_range(value: u32, r: usize) -> u32 {
    let (n, trits, quints) = RANGES[r];
    if trits == 0 && quints == 0 {
        // Bit replication to six bits.
        let v = match n {
            1 => value * 63,
            2 => (value << 4) | (value << 2) | value,
            3 => (value << 3) | value,
            4 => (value << 2) | (value >> 2),
            5 => (value << 1) | (value >> 4),
            _ => unreachable!("weight ranges have 1 to 5 bits"),
        };
        return if v > 32 { v + 1 } else { v };
    }
    if n == 0 {
        return value * if trits == 1 { 32 } else { 16 };
    }
    if trits == 1 {
        unquantize_weight(value, 1, n)
    } else {
        let t = value >> n;
        let m = value & ((1 << n) - 1);
        let a7 = if m & 1 == 1 { 0x7f } else { 0 };
        let (b7, c) = match n {
            1 => (0, 28),
            2 => (((m >> 1) & 1) * 0b100_0010, 13),
            _ => unreachable!("quint weight ranges have 0 to 2 bits"),
        };
        let mut unq = (t * c + b7) ^ a7;
        unq = (a7 & 0x20) | (unq >> 2);
        if unq > 32 {
            unq + 1
        } else {
            unq
        }
    }
}

/// The 0..64 weight a stored value decodes to, for a range of
/// `trits` (1) trit and `n` (1 to 3) bits (§23.17, Tables 167 and 168).
pub(crate) fn unquantize_weight(value: u32, trits: u32, n: u32) -> u32 {
    assert_eq!(trits, 1, "trit ranges");
    let t = value >> n;
    let m = value & ((1 << n) - 1);
    let a = m & 1;
    let a7 = if a == 1 { 0x7f } else { 0 };
    let (b7, c) = match n {
        1 => (0, 50),
        2 => (((m >> 1) & 1) * 0b100_0101, 23),
        3 => (((m << 4) | (m >> 1)) & 0b110_0011, 11),
        _ => unreachable!("ranges 0..5, 0..11 and 0..23 only"),
    };
    let mut unq = t * c + b7;
    unq ^= a7;
    unq = (a7 & 0x20) | (unq >> 2);
    if unq > 32 {
        unq += 1;
    }
    unq
}

/// The 0..=255 value a stored colour endpoint of range `r` decodes to
/// (§23.18), from the crate's table of them.
pub(crate) fn unquantize_endpoint(value: u32, r: usize) -> u32 {
    crate::uastc::tables::UNQUANT[r][value as usize] as u32
}

/// The colour endpoint range a decoder derives for `values` endpoint values
/// in `bits` bits: the largest whose sequence fits (§23.13), or `None` below
/// 0..5, where the block is an error block.
pub(crate) fn endpoint_range(values: u32, bits: u32) -> Option<usize> {
    (4..21).rev().find(|&r| ise_bits(values, r) <= bits)
}

/// The 11-bit block mode of a 4×4 weight grid of weight range `r` (a range
/// index, 0..=11), one or two planes (Tables 146 and 147, the row
/// `W = B + 4, H = A + 2` with B = 0, A = 2).
pub(crate) fn block_mode(r: usize, dual_plane: bool) -> u32 {
    // The weight ranges in index order run 0..1, 0..2, 0..3, 0..4, 0..5,
    // 0..7 (P = 0, ρ = 2..=7), then 0..9, 0..11, 0..15, 0..19, 0..23, 0..31
    // (P = 1, ρ = 2..=7).
    let (rho, p) = match r {
        0..=5 => (r as u32 + 2, 0),
        6..=11 => (r as u32 - 4, 1),
        _ => unreachable!("weight ranges are 0..=11"),
    };
    let (r0, r1, r2) = (rho & 1, (rho >> 1) & 1, (rho >> 2) & 1);
    r1 | (r2 << 1) | (r0 << 4) | (2 << 5) | (p << 9) | ((dual_plane as u32) << 10)
}

/// ASTC's partition function for a 4×4 block (§23.21): the partition of
/// texel (x, y) among `count` partitions, for the 10-bit `seed`.
pub(crate) fn partition_of(seed: u32, x: u32, y: u32, count: u32) -> u32 {
    let (x, y) = (x << 1, y << 1); // small block
    let seed = seed + (count - 1) * 1024;
    let mut p = seed;
    p ^= p >> 15;
    p = p.wrapping_sub(p << 17);
    p = p.wrapping_add(p << 7);
    p = p.wrapping_add(p << 4);
    p ^= p >> 5;
    p = p.wrapping_add(p << 16);
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    let r = p;
    let mut s = [0u32; 12];
    for (i, shift) in [0, 4, 8, 12, 16, 20, 24, 28, 18, 22, 26].iter().enumerate() {
        s[i] = (r >> shift) & 0xf;
    }
    s[11] = r.rotate_right(30) & 0xf;
    for v in &mut s {
        *v *= *v;
    }
    let (sh1, sh2) = if seed & 1 != 0 {
        (
            if seed & 2 != 0 { 4 } else { 5 },
            if count == 3 { 6 } else { 5 },
        )
    } else {
        (
            if count == 3 { 6 } else { 5 },
            if seed & 2 != 0 { 4 } else { 5 },
        )
    };
    let sh3 = if seed & 0x10 != 0 { sh1 } else { sh2 };
    for (i, v) in s.iter_mut().enumerate() {
        *v >>= match i {
            0 | 2 | 4 | 6 => sh1,
            1 | 3 | 5 | 7 => sh2,
            _ => sh3,
        };
    }
    let a = (s[0] * x + s[1] * y + (r >> 14)) & 0x3f;
    let b = (s[2] * x + s[3] * y + (r >> 10)) & 0x3f;
    let c = if count < 3 {
        0
    } else {
        (s[4] * x + s[5] * y + (r >> 6)) & 0x3f
    };
    let d = if count < 4 {
        0
    } else {
        (s[6] * x + s[7] * y + (r >> 2)) & 0x3f
    };
    if a >= b && a >= c && a >= d {
        0
    } else if b >= c && b >= d {
        1
    } else if c >= d {
        2
    } else {
        3
    }
}

/// The distinct two-partition layouts of a 4×4 block: for each, the first
/// seed that makes it and its mask (bit `i` set: texel `i` in partition 1).
/// Layouts that leave a partition empty are left out, and a layout and its
/// complement (the same split with the partitions swapped) count once.
pub(crate) fn two_partition_layouts() -> Vec<(u32, u32)> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for seed in 0..1024 {
        let mut mask = 0u32;
        for i in 0..16 {
            if partition_of(seed, i & 3, i >> 2, 2) == 1 {
                mask |= 1 << i;
            }
        }
        if mask == 0 || mask == 0xffff {
            continue;
        }
        let canonical = mask.min(!mask & 0xffff);
        if seen.insert(canonical) {
            out.push((seed, mask));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every packing decodes back to its group**, to the same group as
    /// upstream's table packs it to (which, for (4, 4, 4), picks another of
    /// its encodings than the smallest), and the trits' are upstream's.
    #[test]
    fn trit_and_quint_packings_round_trip() {
        let trits = trit_encodings();
        let quints = quint_encodings();
        for (key, &q) in quints.iter().enumerate() {
            let [q0, q1, q2] = decode_quints(q);
            assert_eq!((q0 + 5 * q1 + 25 * q2) as usize, key);
        }
        let up_t: Vec<u32> = crate::uastc::tables::TRIT_ENCODE.iter().map(|&v| v as u32).collect();
        let up_q: Vec<u32> = crate::uastc::tables::QUINT_ENCODE.iter().map(|&v| v as u32).collect();
        assert_eq!(trits, up_t);
        for (key, (&ours, &theirs)) in quints.iter().zip(&up_q).enumerate() {
            assert_eq!(decode_quints(ours), decode_quints(theirs), "{key}");
            assert!(ours <= theirs, "{key}: the smallest");
        }
    }

    /// **Sequence lengths and the endpoint range rule** (§23.12–23.13), on
    /// the configurations the kernels use.
    #[test]
    fn sequence_lengths_and_endpoint_ranges() {
        assert_eq!(ise_bits(16, 7), 58, "0..11");
        assert_eq!(ise_bits(16, 4), 42, "0..5");
        assert_eq!(ise_bits(16, 9), 70, "0..19, quints");
        assert_eq!(ise_bits(32, 2), 64, "0..3, two planes");
        assert_eq!(levels(20), 256);
        assert_eq!(levels(12), 40);
        // One partition, CEM 8, weights 0..11: 53 bits leave 0..255.
        assert_eq!(endpoint_range(6, 128 - 17 - 58), Some(20));
        // Two partitions, CEM 8, weights 0..3: 67 bits, 12 values: 0..39.
        assert_eq!(endpoint_range(12, 128 - 29 - 32), Some(12));
        // Too few bits for 0..5.
        assert_eq!(endpoint_range(12, 20), None);
    }

    /// **Every distinct two-partition layout is listed once**, with the
    /// first seed that makes it.
    #[test]
    fn two_partition_layouts_are_distinct_and_complete() {
        let layouts = two_partition_layouts();
        let mut seen = std::collections::HashSet::new();
        for &(seed, mask) in &layouts {
            assert!(seed < 1024 && mask != 0 && mask != 0xffff);
            assert!(seen.insert(mask.min(!mask & 0xffff)), "{seed}: a repeat");
            for i in 0..16 {
                assert_eq!(partition_of(seed, i & 3, i >> 2, 2), (mask >> i) & 1);
            }
        }
        for seed in 0..1024 {
            let mask = (0..16).fold(0, |m, i| m | partition_of(seed, i & 3, i >> 2, 2) << i);
            if mask != 0 && mask != 0xffff {
                assert!(seen.contains(&mask.min(!mask & 0xffff)), "seed {seed} missing");
            }
        }
        assert!(layouts.len() > 100, "{}", layouts.len());
    }

    /// Writes `n` bits of `v` at `pos`.
    fn put(block: &mut u128, pos: u32, n: u32, v: u32) {
        *block |= ((v as u128) & ((1u128 << n) - 1)) << pos;
    }

    /// The integer sequence of `values` in range `r`, as bits from 0.
    fn ise(values: &[u32], r: usize) -> u128 {
        let (n, trits, quints) = RANGES[r];
        let (mut out, mut pos) = (0u128, 0u32);
        let mut emit = |v: u32, bits: u32, out: &mut u128| {
            put(out, pos, bits, v);
            pos += bits;
        };
        if trits == 1 {
            let table = trit_encodings();
            for group in values.chunks(5) {
                let mut t = [0; 5];
                let mut m = [0; 5];
                for (k, &v) in group.iter().enumerate() {
                    t[k] = v >> n;
                    m[k] = v & ((1 << n) - 1);
                }
                let p = table[(t[0] + 3 * t[1] + 9 * t[2] + 27 * t[3] + 81 * t[4]) as usize];
                for (k, (shift, len)) in [(0, 2), (2, 2), (4, 1), (5, 2), (7, 1)].into_iter().enumerate() {
                    emit(m[k], n, &mut out);
                    emit(p >> shift, len, &mut out);
                }
            }
        } else if quints == 1 {
            let table = quint_encodings();
            for group in values.chunks(3) {
                let mut q = [0; 3];
                let mut m = [0; 3];
                for (k, &v) in group.iter().enumerate() {
                    q[k] = v >> n;
                    m[k] = v & ((1 << n) - 1);
                }
                let p = table[(q[0] + 5 * q[1] + 25 * q[2]) as usize];
                for (k, (shift, len)) in [(0, 3), (3, 2), (5, 2)].into_iter().enumerate() {
                    emit(m[k], n, &mut out);
                    emit(p >> shift, len, &mut out);
                }
            }
        } else {
            for &v in values {
                emit(v, n, &mut out);
            }
        }
        // Bits past the sequence's length are not part of it.
        let len = ise_bits(values.len() as u32, r);
        if len < 128 {
            out &= (1u128 << len) - 1;
        }
        out
    }

    /// A one-partition luminance block (CEM 0) from 0 to 255 whose 16
    /// weights of range `r` all store `k`, as the decoder sees it.
    fn grey_ramp_block(r: usize, k: u32) -> [u8; 16] {
        let mut block = 0u128;
        put(&mut block, 0, 11, block_mode(r, false));
        put(&mut block, 13, 4, 0);
        let weight_bits = ise_bits(16, r);
        let er = endpoint_range(2, 128 - 17 - weight_bits).expect("room for endpoints");
        let top = (0..levels(er))
            .find(|&c| unquantize_endpoint(c, er) == 255)
            .expect("255 is in every range");
        let zero = (0..levels(er))
            .find(|&c| unquantize_endpoint(c, er) == 0)
            .expect("0 is in every range");
        block |= ise(&[zero, top], er) << 17;
        // Weights fill the block from bit 127 down, bit-reversed.
        let weights = ise(&[k; 16], r);
        block |= weights.reverse_bits();
        block.to_le_bytes()
    }

    /// **The weight unquantisation, the block modes and the integer
    /// sequence encoding are a decoder's**: a grey ramp block at every
    /// weight range and every stored weight decodes, through an
    /// independent decoder, to the value its unquantised weight gives.
    #[test]
    fn weight_ranges_decode_as_unquantised() {
        for r in 0..12 {
            if !(24..=96).contains(&ise_bits(16, r)) {
                continue;
            }
            for k in 0..levels(r) {
                let w = unquantize_weight_range(k, r);
                let mut out = [0u32; 16];
                texture2ddecoder::decode_astc_block(&grey_ramp_block(r, k), 4, 4, &mut out);
                // The decoder's LDR interpolation of 0 and 255 at w.
                let c16 = (65535 * w + 32) >> 6;
                let expected = ((c16 * 255 + 32768) / 65536) as u8;
                for p in out {
                    let [b, g, rr, a] = p.to_le_bytes();
                    assert_eq!((rr, g, b, a), (expected, expected, expected, 255), "range {r} value {k} weight {w}");
                }
            }
        }
    }
}
