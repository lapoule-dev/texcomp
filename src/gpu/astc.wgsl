// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

// ASTC LDR, 4×4 blocks (Khronos Data Format Specification 1.3, §23): one
// 4×4 weight grid of one plane, and per block the configuration of least
// weighted error among those the quality's `TIER` lists (`CFG_O` for opaque
// blocks, `CFG_A` for blocks with alpha; see `Quality` and `tables.rs`):
//
// - one partition: colour endpoint mode 8 (RGB direct), 0 (luminance) or 6
//   (RGB base + scale), or 12 (RGBA direct) with alpha, at several weight
//   ranges;
// - two partitions of one colour endpoint mode, 8 or 12: the `K` layouts
//   that a moment estimate ranks best among every distinct one the
//   partition function makes.
//
// The endpoint range of a configuration is the one a decoder derives from
// the bits left (§23.13), computed with the configuration. Endpoints and
// weights are written with the integer sequence encoding (§23.12), weights
// from bit 127 downwards. Every fit quantises the endpoints, assigns weights
// and refits by least squares, `ITERS` times.
//
// Generated before this file: `TRIT_ENC`, `QUINT_ENC`, `RANGE`, `WUNQ`,
// `EQ`, `NCFG_O`, `CFG_O`, `NCFG_A`, `CFG_A`, `NP2`, `AP2` and the quality
// constants.

const BIG: f32 = 3.4e38;

// --- integer sequence encoding ----------------------------------------------

// The values to write, in order.
var<private> seq: array<u32, 32>;

// Writes `n` bits of `value` at position `pos` of a sequence of `len` bits
// starting at bit `start`, or, `reversed`, filling the block from bit 127
// downwards. Bits at or past `len` are left unwritten: a decoder reads them
// as 0, and the encodings chosen are those whose truncated bits are 0.
fn seq_put(pos: u32, n: u32, value: u32, start: u32, len: u32, reversed: bool) {
    for (var j = 0u; j < n; j++) {
        let s = pos + j;
        if (s >= len) {
            return;
        }
        if (((value >> j) & 1u) != 0u) {
            let b = select(start + s, 127u - s, reversed);
            bits[b >> 5u] = bits[b >> 5u] | (1u << (b & 31u));
        }
    }
}

fn range_bits(r: u32) -> u32 {
    return RANGE[r] & 0xffu;
}

fn range_levels(r: u32) -> u32 {
    let g = RANGE[r];
    return (1u + 2u * ((g >> 8u) & 0xffu) + 4u * (g >> 16u)) << (g & 0xffu);
}

fn ise_len(count: u32, r: u32) -> u32 {
    let g = RANGE[r];
    let t = (g >> 8u) & 0xffu;
    let q = g >> 16u;
    return count * (g & 0xffu) + (count * 8u * t + 4u) / 5u + (count * 7u * q + 2u) / 3u;
}

// Writes `seq[0..count]` in range `r` at bit `start` (or reversed from bit
// 127).
fn ise_put(count: u32, r: u32, start: u32, reversed: bool) {
    let g = RANGE[r];
    let n = g & 0xffu;
    let trits = ((g >> 8u) & 0xffu) == 1u;
    let quints = (g >> 16u) == 1u;
    let len = ise_len(count, r);
    let mask = (1u << n) - 1u;
    var pos = 0u;
    if (trits) {
        for (var b = 0u; b * 5u < count; b++) {
            var t = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
            var m = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
            for (var k = 0u; k < 5u; k++) {
                let i = 5u * b + k;
                if (i < count) {
                    t[k] = seq[i] >> n;
                    m[k] = seq[i] & mask;
                }
            }
            let p = TRIT_ENC[t[0] + 3u * t[1] + 9u * t[2] + 27u * t[3] + 81u * t[4]];
            seq_put(pos, n, m[0], start, len, reversed);
            pos += n;
            seq_put(pos, 2u, p, start, len, reversed);
            pos += 2u;
            seq_put(pos, n, m[1], start, len, reversed);
            pos += n;
            seq_put(pos, 2u, p >> 2u, start, len, reversed);
            pos += 2u;
            seq_put(pos, n, m[2], start, len, reversed);
            pos += n;
            seq_put(pos, 1u, p >> 4u, start, len, reversed);
            pos += 1u;
            seq_put(pos, n, m[3], start, len, reversed);
            pos += n;
            seq_put(pos, 2u, p >> 5u, start, len, reversed);
            pos += 2u;
            seq_put(pos, n, m[4], start, len, reversed);
            pos += n;
            seq_put(pos, 1u, p >> 7u, start, len, reversed);
            pos += 1u;
        }
    } else if (quints) {
        for (var b = 0u; b * 3u < count; b++) {
            var q = array<u32, 3>(0u, 0u, 0u);
            var m = array<u32, 3>(0u, 0u, 0u);
            for (var k = 0u; k < 3u; k++) {
                let i = 3u * b + k;
                if (i < count) {
                    q[k] = seq[i] >> n;
                    m[k] = seq[i] & mask;
                }
            }
            let p = QUINT_ENC[q[0] + 5u * q[1] + 25u * q[2]];
            seq_put(pos, n, m[0], start, len, reversed);
            pos += n;
            seq_put(pos, 3u, p, start, len, reversed);
            pos += 3u;
            seq_put(pos, n, m[1], start, len, reversed);
            pos += n;
            seq_put(pos, 2u, p >> 3u, start, len, reversed);
            pos += 2u;
            seq_put(pos, n, m[2], start, len, reversed);
            pos += n;
            seq_put(pos, 2u, p >> 5u, start, len, reversed);
            pos += 2u;
        }
    } else {
        for (var i = 0u; i < count; i++) {
            seq_put(pos, n, seq[i], start, len, reversed);
            pos += n;
        }
    }
}

// --- endpoints --------------------------------------------------------------

// The stored value of the endpoint range at `base` in `EQ` nearest `v`: x
// the stored value, y what it decodes to.
fn eq(v: f32, base: u32) -> vec2u {
    let t = EQ[base + u32(clamp(round(v), 0.0, 255.0))];
    return vec2u(t & 0xffu, t >> 8u);
}

// The stored endpoint values of each partition, in the order of its colour
// endpoint mode.
var<private> pv: array<array<u32, 8>, 2>;

fn colour_weights(alpha: bool) -> vec4f {
    var w = ew;
    if (!alpha) {
        w.a = 0.0;
    }
    return w;
}

// CEM 8 (RGB direct) or, with `alpha`, 12 (RGBA direct): endpoints ordered
// so the decoder never applies blue contraction, which it does when the
// second endpoint's RGB sum is below the first's.
fn fit_direct(members: u32, s: u32, alpha: bool, base: u32) -> f32 {
    let w = colour_weights(alpha);
    let channels = select(3u, 4u, alpha);
    var seg = pca_segment(members, w);
    var best = BIG;
    for (var it = 0u; it < ITERS; it++) {
        var c0 = vec4u(0u);
        var c1 = vec4u(0u);
        var v0 = vec4f(255.0);
        var v1 = vec4f(255.0);
        for (var c = 0u; c < channels; c++) {
            let q0 = eq(seg.e0[c], base);
            let q1 = eq(seg.e1[c], base);
            c0[c] = q0.x;
            c1[c] = q1.x;
            v0[c] = f32(q0.y);
            v1[c] = f32(q1.y);
        }
        if (v1.r + v1.g + v1.b < v0.r + v0.g + v0.b) {
            let tc = c0;
            c0 = c1;
            c1 = tc;
            let tv = v0;
            v0 = v1;
            v1 = tv;
        }
        let err = assign(v0, v1, members, w);
        if (err < best) {
            best = err;
            keep_indices(members);
            for (var c = 0u; c < 4u; c++) {
                pv[s][2u * c] = c0[c];
                pv[s][2u * c + 1u] = c1[c];
            }
        }
        seg = refine(members, seg);
    }
    return best;
}

// CEM 0 (luminance direct): grey endpoints, the weighted mean of R, G, B.
fn fit_luminance(members: u32, s: u32, base: u32) -> f32 {
    let w = colour_weights(false);
    let sum = w.r + w.g + w.b;
    var lo = 255.0;
    var hi = 0.0;
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) != 0u) {
            let l = dot(block[i].rgb, w.rgb) / sum;
            lo = min(lo, l);
            hi = max(hi, l);
        }
    }
    var best = BIG;
    for (var it = 0u; it < ITERS; it++) {
        let q0 = eq(lo, base);
        let q1 = eq(hi, base);
        let e0 = vec4f(vec3f(f32(q0.y)), 255.0);
        let e1 = vec4f(vec3f(f32(q1.y)), 255.0);
        let err = assign(e0, e1, members, w);
        if (err < best) {
            best = err;
            keep_indices(members);
            pv[s][0] = q0.x;
            pv[s][1] = q1.x;
        }
        let seg = refine(members, Segment(e0, e1));
        lo = dot(seg.e0.rgb, w.rgb) / sum;
        hi = dot(seg.e1.rgb, w.rgb) / sum;
    }
    return best;
}

// CEM 6 (RGB base + scale): the second endpoint (v0, v1, v2), the first that
// scaled by v3 / 256; a line through black.
fn fit_scale(members: u32, s: u32, base: u32) -> f32 {
    let w = colour_weights(false);
    var mean = vec3f(0.0);
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) != 0u) {
            mean += block[i].rgb;
        }
    }
    if (dot(mean, mean) < 1.0) {
        return BIG;
    }
    let d = normalize(mean);
    var lo = 3.4e38;
    var hi = 0.0;
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) != 0u) {
            let t = dot(block[i].rgb, d);
            lo = min(lo, t);
            hi = max(hi, t);
        }
    }
    var best = BIG;
    for (var it = 0u; it < ITERS; it++) {
        var c1 = vec3u(0u);
        var v1 = vec3f(0.0);
        for (var c = 0u; c < 3u; c++) {
            let q = eq(d[c] * hi, base);
            c1[c] = q.x;
            v1[c] = f32(q.y);
        }
        let qs = eq(clamp(lo / max(hi, 1e-3) * 256.0, 0.0, 255.0), base);
        let v0 = floor(v1 * f32(qs.y) / 256.0);
        let e0 = vec4f(v0, 255.0);
        let e1 = vec4f(v1, 255.0);
        let err = assign(e0, e1, members, w);
        if (err < best) {
            best = err;
            keep_indices(members);
            pv[s][0] = c1.x;
            pv[s][1] = c1.y;
            pv[s][2] = c1.z;
            pv[s][3] = qs.x;
        }
        let seg = refine(members, Segment(e0, e1));
        lo = max(dot(seg.e0.rgb, d), 0.0);
        hi = max(dot(seg.e1.rgb, d), lo);
    }
    return best;
}

fn cem_values(cem: u32) -> u32 {
    switch cem {
        case 0u: { return 2u; }
        case 6u: { return 4u; }
        case 12u: { return 8u; }
        default: { return 6u; }
    }
}

// --- configurations ---------------------------------------------------------

// Writes the block of configuration `cfg` (partition layout `seed` when it
// has two) from `pv` and `best_idx`.
fn pack(cfg: vec4u, seed: u32) {
    let parts = cfg.x & 0xfu;
    let cem = cfg.x >> 4u;
    begin();
    put(0u, 11u, cfg.w & 0xffffu);
    put(11u, 2u, parts - 1u);
    var start = 17u;
    if (parts == 1u) {
        put(13u, 4u, cem);
    } else {
        put(13u, 10u, seed);
        // All partitions share the mode: the field's low two bits are 00.
        put(23u, 6u, cem << 2u);
        start = 29u;
    }
    let nv = cem_values(cem);
    for (var p = 0u; p < parts; p++) {
        for (var k = 0u; k < nv; k++) {
            seq[p * nv + k] = pv[p][k];
        }
    }
    ise_put(parts * nv, cfg.w >> 16u, start, false);
    for (var i = 0u; i < 16u; i++) {
        seq[i] = best_idx[i];
    }
    ise_put(16u, cfg.y, 0u, true);
}

fn fit_one(members: u32, s: u32, cem: u32, base: u32) -> f32 {
    switch cem {
        case 0u: { return fit_luminance(members, s, base); }
        case 6u: { return fit_scale(members, s, base); }
        case 12u: { return fit_direct(members, s, true, base); }
        default: { return fit_direct(members, s, false, base); }
    }
}

var<private> ranked: bool;

// Ranks the two-partition layouts by estimate into `top`.
fn rank_layouts() {
    if (ranked) {
        return;
    }
    ranked = true;
    top_reset();
    let all = moments(ALL);
    for (var l = 0u; l < NP2; l++) {
        let ones = moments(AP2[l] >> 16u);
        top_add(l, line_error(moments_minus(all, ones)) + line_error(ones));
    }
}

fn try_config(cfg: vec4u) {
    let parts = cfg.x & 0xfu;
    let cem = cfg.x >> 4u;
    let wr = cfg.y;
    wn = range_levels(wr);
    for (var k = 0u; k < wn; k++) {
        wt[k] = WUNQ[wr * 32u + k];
    }
    for (var v = 0u; v < 65u; v++) {
        wsel[v] = WSEL[wr * 65u + v];
    }
    if (parts == 1u) {
        let err = fit_one(ALL, 0u, cem, cfg.z);
        if (err < out_err) {
            pack(cfg, 0u);
            consider(err);
        }
        return;
    }
    rank_layouts();
    for (var c = 0u; c < K; c++) {
        let lay = AP2[top[c]];
        let ones = lay >> 16u;
        var err = fit_one(ALL & ~ones, 0u, cem, cfg.z);
        if (err >= out_err) {
            continue;
        }
        err += fit_one(ones, 1u, cem, cfg.z);
        if (err < out_err) {
            pack(cfg, lay & 0x3ffu);
            consider(err);
        }
    }
}

// Below this error (squared, summed over the block: an RMS of about one
// code per channel) a block is smooth enough that nothing else can help.
const GOOD_ENOUGH: f32 = 48.0;

fn encode() {
    ranked = false;
    if (opaque()) {
        for (var c = 0u; c < NCFG_O; c++) {
            try_config(CFG_O[c]);
            if (out_err <= GOOD_ENOUGH) {
                return;
            }
        }
    } else {
        for (var c = 0u; c < NCFG_A; c++) {
            try_config(CFG_A[c]);
            if (out_err <= GOOD_ENOUGH) {
                return;
            }
        }
    }
}
