// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

// ASTC LDR, 4×4 blocks, in two fixed configurations (Khronos Data Format
// Specification 1.3, §23):
//
// - an opaque block: one partition, CEM 8 (RGB direct), a 4×4 weight grid of
//   range 0..11 (a trit and two bits each: 58 bits), endpoints of range
//   0..255 (48 bits);
// - a block with alpha: one partition, CEM 12 (RGBA direct), a 4×4 weight
//   grid of range 0..5 (a trit and one bit each: 42 bits), endpoints of range
//   0..255 (64 bits).
//
// Endpoints of 0..255 are stored as plain bytes. The weights go through the
// spec's integer sequence encoding; `TRIT_ENC`, `Q12` and `Q6` are generated
// from the spec's decoding procedure by the Rust side (`encoder.rs`) and
// prefixed to this file.
//
// Block mode (Table 147, 2D, bits [3:2] = 00): D = 0, W field 0 (width 4),
// H field 2 (height 4), and the weight range ρ / P:
//   range 0..11: ρ = 011, P = 1 → 0x251
//   range 0..5:  ρ = 110, P = 0 → 0x043

const MODE_Q12: u32 = 0x251u;
const MODE_Q6: u32 = 0x043u;

// Writes `n` bits of `value` at position `pos` of the *weight* stream, which
// fills the block from bit 127 downwards, bit-reversed. Bits at or past
// `limit` are left unwritten: the decoder reads them as 0.
fn put_weight(pos: u32, n: u32, value: u32, limit: u32) {
    for (var j = 0u; j < n; j++) {
        let s = pos + j;
        if (s >= limit) {
            return;
        }
        if (((value >> j) & 1u) != 0u) {
            let b = 127u - s;
            bits[b >> 5u] = bits[b >> 5u] | (1u << (b & 31u));
        }
    }
}

// The 16 weights (indices in `best_idx`, each a value of range 3·2ⁿ) as
// trit blocks of five (Table 155).
fn put_weights(n: u32) {
    let limit = (16u * (8u + 5u * n) + 4u) / 5u;
    let mask = (1u << n) - 1u;
    for (var b = 0u; b < 4u; b++) {
        var t = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
        var m = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
        for (var k = 0u; k < 5u; k++) {
            let i = 5u * b + k;
            if (i < 16u) {
                t[k] = best_idx[i] >> n;
                m[k] = best_idx[i] & mask;
            }
        }
        let packed = TRIT_ENC[t[0] + 3u * t[1] + 9u * t[2] + 27u * t[3] + 81u * t[4]];
        var pos = b * (8u + 5u * n);
        put_weight(pos, n, m[0], limit);
        pos += n;
        put_weight(pos, 2u, packed, limit);
        pos += 2u;
        put_weight(pos, n, m[1], limit);
        pos += n;
        put_weight(pos, 2u, packed >> 2u, limit);
        pos += 2u;
        put_weight(pos, n, m[2], limit);
        pos += n;
        put_weight(pos, 1u, packed >> 4u, limit);
        pos += 1u;
        put_weight(pos, n, m[3], limit);
        pos += n;
        put_weight(pos, 2u, packed >> 5u, limit);
        pos += 2u;
        put_weight(pos, n, m[4], limit);
        pos += n;
        put_weight(pos, 1u, packed >> 7u, limit);
    }
}

// Endpoints of range 0..255, ordered so the decoder never applies blue
// contraction: CEM 8 and 12 swap and contract when the second endpoint's RGB
// sum is below the first's.
fn quantize8(s: Segment, alpha: bool) -> Segment {
    var e0 = clamp(round(s.e0), vec4f(0.0), vec4f(255.0));
    var e1 = clamp(round(s.e1), vec4f(0.0), vec4f(255.0));
    if (!alpha) {
        e0.a = 255.0;
        e1.a = 255.0;
    }
    if (e1.r + e1.g + e1.b < e0.r + e0.g + e0.b) {
        let t = e0;
        e0 = e1;
        e1 = t;
    }
    return Segment(e0, e1);
}

var<private> ends: Segment;

fn fit(alpha: bool) {
    var seg = pca_segment(ALL, select(vec4f(1.0, 1.0, 1.0, 0.0), vec4f(1.0), alpha));
    var best = 3.4e38;
    for (var it = 0; it < 3; it++) {
        let q = quantize8(seg, alpha);
        let err = assign(q.e0, q.e1, ALL);
        if (err < best) {
            best = err;
            ends = q;
            keep_indices(ALL);
        }
        seg = refine(ALL, seg);
    }
}

fn encode() {
    let alpha = !opaque();
    if (alpha) {
        for (var k = 0u; k < 6u; k++) {
            wt[k] = Q6[k];
        }
        wn = 6u;
    } else {
        for (var k = 0u; k < 12u; k++) {
            wt[k] = Q12[k];
        }
        wn = 12u;
    }
    fit(alpha);
    let e0 = vec4u(ends.e0);
    let e1 = vec4u(ends.e1);
    // Block mode, one partition (bits 11..12 = 0), the CEM, the endpoints
    // from bit 17: v0..v5 (CEM 8) or v0..v7 (CEM 12), interleaved
    // (e0.r, e1.r, e0.g, e1.g, …).
    put(0u, 11u, select(MODE_Q12, MODE_Q6, alpha));
    put(13u, 4u, select(8u, 12u, alpha));
    let channels = select(3u, 4u, alpha);
    for (var c = 0u; c < channels; c++) {
        put(17u + 16u * c, 8u, e0[c]);
        put(25u + 16u * c, 8u, e1[c]);
    }
    put_weights(select(2u, 1u, alpha));
}
