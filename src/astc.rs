// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! The parts of ASTC's decoding the encoders invert, written from the Khronos
//! Data Format Specification 1.3, §23.12 (integer sequence encoding) and
//! §23.17 (weight unquantisation).

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

/// The 0..64 weight a stored value decodes to, for a range of
/// `trits` (0 or 1) trits and `n` bits (§23.17, Tables 167 and 168).
/// Only the trit ranges with one or two bits are used here.
pub(crate) fn unquantize_weight(value: u32, trits: u32, n: u32) -> u32 {
    assert_eq!(trits, 1, "only trit ranges are needed");
    let t = value >> n;
    let m = value & ((1 << n) - 1);
    let a = m & 1;
    let a7 = if a == 1 { 0x7f } else { 0 };
    let (b7, c) = match n {
        1 => (0, 50),
        2 => (((m >> 1) & 1) * 0b100_0101, 23),
        _ => unreachable!("ranges 0..5 and 0..11 only"),
    };
    let mut unq = t * c + b7;
    unq ^= a7;
    unq = (a7 & 0x20) | (unq >> 2);
    if unq > 32 {
        unq += 1;
    }
    unq
}
