// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

// BC7, two of its eight modes (Khronos Data Format Specification 1.3, §21):
//
// - mode 6: one subset, RGBA endpoints of 7 bits plus a p-bit each, 4-bit
//   indices. Smooth blocks, and every block with alpha.
// - mode 1: two subsets out of 64 partitions, RGB endpoints of 6 bits plus a
//   p-bit shared per subset, 3-bit indices. Blocks across an edge.
//
// An opaque block is fitted both ways and keeps the lower error; a block with
// alpha takes mode 6. The partition for mode 1 is picked by an estimate (the
// variance each subset leaves off its principal axis), then fitted fully.

// Table 119: the interpolation factors of 4-bit and 3-bit indices.
const W4 = array<f32, 16>(0.0, 4.0, 9.0, 13.0, 17.0, 21.0, 26.0, 30.0, 34.0, 38.0, 43.0, 47.0, 51.0, 55.0, 60.0, 64.0);
const W3 = array<f32, 8>(0.0, 9.0, 18.0, 27.0, 37.0, 46.0, 55.0, 64.0);

// Table 114: the 2-subset partitions; bit i is texel i's subset.
const PARTITIONS = array<u32, 64>(
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80,
    0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
    0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce,
    0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
    0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a,
    0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c,
    0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
);

// Table 118: the anchor texel of the second subset.
const ANCHOR2 = array<u32, 64>(
    15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u,
    15u, 2u, 8u, 2u, 2u, 8u, 8u, 15u, 2u, 8u, 2u, 2u, 8u, 8u, 2u, 2u,
    15u, 15u, 6u, 8u, 2u, 8u, 15u, 15u, 2u, 8u, 2u, 2u, 2u, 15u, 15u, 6u,
    6u, 2u, 6u, 8u, 15u, 15u, 2u, 2u, 15u, 15u, 15u, 15u, 15u, 2u, 2u, 15u,
);

const RGBA = vec4f(1.0, 1.0, 1.0, 1.0);
const RGB = vec4f(1.0, 1.0, 1.0, 0.0);

// --- mode 6 -----------------------------------------------------------------

// The 7-bit codes and p-bit nearest an endpoint, and what they decode to.
struct Q6 {
    code: vec4u,
    p: u32,
    value: vec4f,
}

// The p-bit is shared by the four channels. An opaque block weighs alpha so
// that its alpha stays exactly 255: an opaque tile must not turn 254.
fn quantize6(e: vec4f, weights: vec4f) -> Q6 {
    var best = Q6(vec4u(0u), 0u, vec4f(0.0));
    var best_err = 3.4e38;
    for (var p = 0u; p < 2u; p++) {
        let pf = f32(p);
        let q = clamp(round((e - pf) / 2.0), vec4f(0.0), vec4f(127.0));
        let v = 2.0 * q + pf;
        let d = e - v;
        let err = dot(d * d, weights);
        if (err < best_err) {
            best_err = err;
            best = Q6(vec4u(q), p, v);
        }
    }
    return best;
}

var<private> m6_lo: Q6;
var<private> m6_hi: Q6;

fn fit6(weights: vec4f) -> f32 {
    wt = array<f32, 16>(W4[0], W4[1], W4[2], W4[3], W4[4], W4[5], W4[6], W4[7], W4[8], W4[9], W4[10], W4[11], W4[12], W4[13], W4[14], W4[15]);
    wn = 16u;
    var seg = pca_segment(ALL, RGBA);
    var best = 3.4e38;
    for (var it = 0; it < 3; it++) {
        let lo = quantize6(seg.e0, weights);
        let hi = quantize6(seg.e1, weights);
        let err = assign(lo.value, hi.value, ALL);
        if (err < best) {
            best = err;
            m6_lo = lo;
            m6_hi = hi;
            keep_indices(ALL);
        }
        seg = refine(ALL, seg);
    }
    return best;
}

fn pack6() {
    var lo = m6_lo;
    var hi = m6_hi;
    // Texel 0 is the anchor: its index's top bit is implied 0.
    if (best_idx[0] >= 8u) {
        let t = lo;
        lo = hi;
        hi = t;
        for (var i = 0u; i < 16u; i++) {
            best_idx[i] = 15u - best_idx[i];
        }
    }
    put(0u, 7u, 64u);
    for (var c = 0u; c < 4u; c++) {
        put(7u + 14u * c, 7u, lo.code[c]);
        put(14u + 14u * c, 7u, hi.code[c]);
    }
    put(63u, 1u, lo.p);
    put(64u, 1u, hi.p);
    put(65u, 3u, best_idx[0]);
    for (var i = 1u; i < 16u; i++) {
        put(68u + 4u * (i - 1u), 4u, best_idx[i]);
    }
}

// --- mode 1 -----------------------------------------------------------------

// A 6-bit code with p-bit decodes to (code << 1 | p) widened from 7 to 8 bits.
fn expand1(q: vec3f, p: f32) -> vec3f {
    let v7 = 2.0 * q + p;
    return 2.0 * v7 + floor(v7 / 64.0);
}

struct Q1 {
    lo: vec3u,
    hi: vec3u,
    p: u32,
    lo_value: vec4f,
    hi_value: vec4f,
}

fn nearest1(e: vec3f, p: f32) -> vec3f {
    // Two candidates per channel bracket the ideal code; take the nearer.
    let guess = clamp(floor((e * 127.0 / 255.0 - p) / 2.0), vec3f(0.0), vec3f(63.0));
    let up = min(guess + 1.0, vec3f(63.0));
    let dg = abs(expand1(guess, p) - e);
    let du = abs(expand1(up, p) - e);
    return select(guess, up, du < dg);
}

// One subset's endpoints share a p-bit: both values tried.
fn quantize1(s: Segment) -> Q1 {
    var best: Q1;
    var best_err = 3.4e38;
    for (var p = 0u; p < 2u; p++) {
        let pf = f32(p);
        let lo = nearest1(s.e0.rgb, pf);
        let hi = nearest1(s.e1.rgb, pf);
        let lv = expand1(lo, pf);
        let hv = expand1(hi, pf);
        let d0 = lv - s.e0.rgb;
        let d1 = hv - s.e1.rgb;
        let err = dot(d0, d0) + dot(d1, d1);
        if (err < best_err) {
            best_err = err;
            best = Q1(vec3u(lo), vec3u(hi), p, vec4f(lv, 255.0), vec4f(hv, 255.0));
        }
    }
    return best;
}

var<private> m1_q: array<Q1, 2>;

fn fit1_subset(members: u32, s: u32) -> f32 {
    var seg = pca_segment(members, RGB);
    var best = 3.4e38;
    for (var it = 0; it < 3; it++) {
        let q = quantize1(seg);
        let err = assign(q.lo_value, q.hi_value, members);
        if (err < best) {
            best = err;
            m1_q[s] = q;
            keep_indices(members);
        }
        seg = refine(members, seg);
    }
    return best;
}

// Squared error a subset leaves when its texels are put on one line and
// quantised to 8 steps along it: the variance off the principal axis, plus
// the quantisation noise of 8 levels along it.
//
// From raw moments (count, sum, sum of outer products), so a partition's two
// subsets cost one pass over the texels of one of them: the other is the
// block's total minus it.
struct Moments {
    n: f32,
    sum: vec3f,
    outer: mat3x3f,
}

fn moments(members: u32) -> Moments {
    var m = Moments(0.0, vec3f(0.0), mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0)));
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) != 0u) {
            let p = block[i].rgb;
            m.n += 1.0;
            m.sum += p;
            m.outer += mat3x3f(p * p.x, p * p.y, p * p.z);
        }
    }
    return m;
}

fn line_error(m: Moments) -> f32 {
    if (m.n < 2.0) {
        return 0.0;
    }
    let mean = m.sum / m.n;
    let cov = m.outer - mat3x3f(mean * m.sum.x, mean * m.sum.y, mean * m.sum.z);
    let trace = cov[0][0] + cov[1][1] + cov[2][2];
    var v = vec3f(1.0, 1.0, 1.0);
    var lambda = 0.0;
    for (var it = 0; it < 3; it++) {
        let next = cov * v;
        lambda = length(next);
        if (lambda < 1e-6) {
            return max(trace, 0.0);
        }
        v = next / lambda;
    }
    return max(trace - lambda, 0.0) + lambda / 49.0;
}

var<private> m1_part: u32;

fn fit1() -> f32 {
    var best_part = 0u;
    var best_est = 3.4e38;
    let all = moments(ALL);
    for (var part = 0u; part < 64u; part++) {
        let ones = moments(PARTITIONS[part]);
        let zeros = Moments(all.n - ones.n, all.sum - ones.sum, all.outer - ones.outer);
        let est = line_error(zeros) + line_error(ones);
        if (est < best_est) {
            best_est = est;
            best_part = part;
        }
    }
    m1_part = best_part;
    wt = array<f32, 16>(W3[0], W3[1], W3[2], W3[3], W3[4], W3[5], W3[6], W3[7], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    wn = 8u;
    let ones = PARTITIONS[best_part];
    return fit1_subset(ALL & ~ones, 0u) + fit1_subset(ones, 1u);
}

fn pack1() {
    let part = m1_part;
    let ones = PARTITIONS[part];
    let anchor = ANCHOR2[part];
    var q = m1_q;
    // Each subset's anchor index has its top bit implied 0.
    for (var s = 0u; s < 2u; s++) {
        let a = select(0u, anchor, s == 1u);
        if (best_idx[a] >= 4u) {
            let t = q[s].lo;
            q[s].lo = q[s].hi;
            q[s].hi = t;
            for (var i = 0u; i < 16u; i++) {
                let in_s = ((ones >> i) & 1u) == s;
                if (in_s) {
                    best_idx[i] = 7u - best_idx[i];
                }
            }
        }
    }
    put(0u, 2u, 2u);
    put(2u, 6u, part);
    for (var c = 0u; c < 3u; c++) {
        let base = 8u + 24u * c;
        put(base, 6u, q[0].lo[c]);
        put(base + 6u, 6u, q[0].hi[c]);
        put(base + 12u, 6u, q[1].lo[c]);
        put(base + 18u, 6u, q[1].hi[c]);
    }
    put(80u, 1u, q[0].p);
    put(81u, 1u, q[1].p);
    var pos = 82u;
    for (var i = 0u; i < 16u; i++) {
        let n = select(3u, 2u, i == 0u || i == anchor);
        put(pos, n, best_idx[i]);
        pos += n;
    }
}

// --- choice -----------------------------------------------------------------

// Below this mode 6 error (squared, summed over the block: an RMS of about
// one code per channel) the block is smooth enough that no partition can
// help, and the 64-partition search is skipped.
const GOOD_ENOUGH: f32 = 48.0;

var<private> idx6: array<u32, 16>;

fn encode() {
    let is_opaque = opaque();
    let e6 = fit6(select(vec4f(1.0), vec4f(1.0, 1.0, 1.0, 1.0e4), is_opaque));
    if (is_opaque && e6 > GOOD_ENOUGH) {
        idx6 = best_idx;
        let e1 = fit1();
        if (e1 < e6) {
            pack1();
            return;
        }
        best_idx = idx6;
    }
    pack6();
}
