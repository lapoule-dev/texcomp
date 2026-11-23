// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

// BC7 (Khronos Data Format Specification 1.3, §21), all eight modes:
//
// | Mode | Subsets | Partitions | Endpoints | p-bits | Indices |
// |---|---|---|---|---|---|
// | 0 | 3 | 16 | RGB 4 | one per endpoint | 3 bits |
// | 1 | 2 | 64 | RGB 6 | one per subset | 3 bits |
// | 2 | 3 | 64 | RGB 5 | none | 2 bits |
// | 3 | 2 | 64 | RGB 7 | one per endpoint | 2 bits |
// | 4 | 1 | – | RGB 5, A 6, rotation | none | 2 and 3 bits, either way |
// | 5 | 1 | – | RGB 7, A 8, rotation | none | 2 and 2 bits |
// | 6 | 1 | – | RGBA 7 | one per endpoint | 4 bits |
// | 7 | 2 | 64 | RGBA 5 | one per endpoint | 2 bits |
//
// Which modes a block tries is the quality's `TIER` (see `Quality`); a
// block with alpha tries only the modes that store it (4 to 7). Partitioned
// modes fit the `K` partitions that a moment estimate ranks best. Every fit
// quantises the endpoints, assigns indices and refits by least squares,
// `ITERS` times. The block keeps the encoding of least weighted error.
//
// Generated before this file: `P2`, `A2`, `P3`, `A3` (the partition and
// anchor tables), `WSEL` (the indices bracketing each ideal weight) and the
// quality constants.

// Table 119: the interpolation factors of 2-, 3- and 4-bit indices.
const W2 = array<f32, 4>(0.0, 21.0, 43.0, 64.0);
const W3 = array<f32, 8>(0.0, 9.0, 18.0, 27.0, 37.0, 46.0, 55.0, 64.0);
const W4 = array<f32, 16>(0.0, 4.0, 9.0, 13.0, 17.0, 21.0, 26.0, 30.0, 34.0, 38.0, 43.0, 47.0, 51.0, 55.0, 60.0, 64.0);

const BIG: f32 = 3.4e38;

fn index_weight(ibits: u32, k: u32) -> f32 {
    if (ibits == 2u) {
        return W2[k & 3u];
    }
    if (ibits == 3u) {
        return W3[k & 7u];
    }
    return W4[k & 15u];
}

fn set_weights(ibits: u32) {
    wn = 1u << ibits;
    for (var k = 0u; k < wn; k++) {
        wt[k] = index_weight(ibits, k);
    }
    let base = (ibits - 2u) * 65u;
    for (var v = 0u; v < 65u; v++) {
        wsel[v] = WSEL[base + v];
    }
}

// --- endpoints --------------------------------------------------------------

// An endpoint of `b` stored bits (code and p-bit) widens to 8 by repeating
// its top bits.
fn expand(v: u32, b: u32) -> u32 {
    return (v << (8u - b)) | (v >> (2u * b - 8u));
}

// The `n`-bit code (with p-bit `p` below it when `has_p`) whose widened
// value is nearest `e`: x = code, y = the value.
fn quant_channel(e: f32, n: u32, has_p: bool, p: u32) -> vec2u {
    let b = n + select(0u, 1u, has_p);
    let maxc = i32((1u << n) - 1u);
    let ideal = e * f32((1u << b) - 1u) / 255.0;
    var guess: i32;
    if (has_p) {
        guess = i32(floor((ideal - f32(p)) / 2.0));
    } else {
        guess = i32(floor(ideal));
    }
    var best = vec2u(0u, 0u);
    var best_d = BIG;
    for (var c = guess - 1; c <= guess + 2; c++) {
        let code = u32(clamp(c, 0, maxc));
        var v = code;
        if (has_p) {
            v = (code << 1u) | p;
        }
        let value = expand(v, b);
        let d = abs(f32(value) - e);
        if (d < best_d) {
            best_d = d;
            best = vec2u(code, value);
        }
    }
    return best;
}

struct Ep {
    code: vec4u,
    p: u32,
    value: vec4f,
}

// An endpoint quantised to `cb` bits of colour and `ab` of alpha (0: no
// alpha stored, it decodes to 255), with p-bit `p` when `has_p`.
fn quant_ep(e: vec4f, cb: u32, ab: u32, has_p: bool, p: u32) -> Ep {
    var out = Ep(vec4u(0u), p, vec4f(255.0));
    for (var c = 0u; c < 3u; c++) {
        let q = quant_channel(e[c], cb, has_p, p);
        out.code[c] = q.x;
        out.value[c] = f32(q.y);
    }
    if (ab > 0u) {
        let q = quant_channel(e.a, ab, has_p, p);
        out.code.w = q.x;
        out.value.w = f32(q.y);
    }
    return out;
}

fn ep_error(q: Ep, e: vec4f, w: vec4f) -> f32 {
    let d = q.value - e;
    return dot(d * d, w);
}

// --- subsets ----------------------------------------------------------------

// The endpoints kept per subset.
var<private> s_lo: array<Ep, 3>;
var<private> s_hi: array<Ep, 3>;

// p-bit kinds.
const P_NONE: u32 = 0u;
const P_SHARED: u32 = 1u;
const P_UNIQUE: u32 = 2u;

// Fits subset `s` (its texels `members`) with endpoints of `cb` colour and
// `ab` alpha bits and p-bits of kind `pk`, with the index weights already
// set. Returns its weighted squared error; keeps its endpoints in
// `s_lo[s]`, `s_hi[s]` and its indices in `best_idx`. Without alpha bits,
// alpha is neither fitted nor counted.
fn fit_subset(members: u32, s: u32, cb: u32, ab: u32, pk: u32) -> f32 {
    var w = ew;
    if (ab == 0u) {
        w.a = 0.0;
    }
    var seg = pca_segment(members, w);
    var best = BIG;
    for (var it = 0u; it < ITERS; it++) {
        var lo: Ep;
        var hi: Ep;
        if (pk == P_NONE) {
            lo = quant_ep(seg.e0, cb, ab, false, 0u);
            hi = quant_ep(seg.e1, cb, ab, false, 0u);
        } else {
            let l0 = quant_ep(seg.e0, cb, ab, true, 0u);
            let h0 = quant_ep(seg.e1, cb, ab, true, 0u);
            let l1 = quant_ep(seg.e0, cb, ab, true, 1u);
            let h1 = quant_ep(seg.e1, cb, ab, true, 1u);
            let el0 = ep_error(l0, seg.e0, w);
            let el1 = ep_error(l1, seg.e0, w);
            let eh0 = ep_error(h0, seg.e1, w);
            let eh1 = ep_error(h1, seg.e1, w);
            if (pk == P_SHARED) {
                if (el1 + eh1 < el0 + eh0) {
                    lo = l1;
                    hi = h1;
                } else {
                    lo = l0;
                    hi = h0;
                }
            } else {
                lo = l0;
                if (el1 < el0) {
                    lo = l1;
                }
                hi = h0;
                if (eh1 < eh0) {
                    hi = h1;
                }
            }
        }
        let err = assign(lo.value, hi.value, members, w);
        if (err < best) {
            best = err;
            s_lo[s] = lo;
            s_hi[s] = hi;
            keep_indices(members);
        }
        seg = refine(members, seg);
    }
    return best;
}

// The texels of subset `s` of partition `part` among `ns` subsets.
fn subset_mask(ns: u32, part: u32, s: u32) -> u32 {
    if (ns == 1u) {
        return ALL;
    }
    if (ns == 2u) {
        let ones = P2[part];
        return select(ALL & ~ones, ones, s == 1u);
    }
    let one = P3[part] & 0xffffu;
    let two = P3[part] >> 16u;
    if (s == 1u) {
        return one;
    }
    if (s == 2u) {
        return two;
    }
    return ALL & ~(one | two);
}

fn anchor(ns: u32, part: u32, s: u32) -> u32 {
    if (s == 0u) {
        return 0u;
    }
    if (ns == 2u) {
        return A2[part];
    }
    if (s == 1u) {
        return A3[part] & 0xffu;
    }
    return A3[part] >> 8u;
}

fn is_anchor(ns: u32, part: u32, i: u32) -> bool {
    for (var s = 0u; s < ns; s++) {
        if (anchor(ns, part, s) == i) {
            return true;
        }
    }
    return false;
}

// Writes modes 0, 1, 2, 3, 6 and 7 from the kept endpoints and indices:
// first each anchor's index gets its top bit clear, by swapping its
// subset's endpoints and reversing its indices where it is set.
fn pack_subsets(mode: u32, part: u32, ns: u32, pbits: u32, cb: u32, ab: u32, pk: u32, ib: u32) {
    let msb = 1u << (ib - 1u);
    let last = (1u << ib) - 1u;
    for (var s = 0u; s < ns; s++) {
        if (best_idx[anchor(ns, part, s)] >= msb) {
            let t = s_lo[s];
            s_lo[s] = s_hi[s];
            s_hi[s] = t;
            let m = subset_mask(ns, part, s);
            for (var i = 0u; i < 16u; i++) {
                if (((m >> i) & 1u) != 0u) {
                    best_idx[i] = last - best_idx[i];
                }
            }
        }
    }
    put(0u, mode + 1u, 1u << mode);
    var pos = mode + 1u;
    if (pbits > 0u) {
        put(pos, pbits, part);
        pos += pbits;
    }
    for (var c = 0u; c < 3u; c++) {
        for (var s = 0u; s < ns; s++) {
            put(pos, cb, s_lo[s].code[c]);
            put(pos + cb, cb, s_hi[s].code[c]);
            pos += 2u * cb;
        }
    }
    if (ab > 0u) {
        for (var s = 0u; s < ns; s++) {
            put(pos, ab, s_lo[s].code.w);
            put(pos + ab, ab, s_hi[s].code.w);
            pos += 2u * ab;
        }
    }
    if (pk == P_UNIQUE) {
        for (var s = 0u; s < ns; s++) {
            put(pos, 1u, s_lo[s].p);
            put(pos + 1u, 1u, s_hi[s].p);
            pos += 2u;
        }
    } else if (pk == P_SHARED) {
        for (var s = 0u; s < ns; s++) {
            put(pos, 1u, s_lo[s].p);
            pos += 1u;
        }
    }
    for (var i = 0u; i < 16u; i++) {
        let n = select(ib, ib - 1u, is_anchor(ns, part, i));
        put(pos, n, best_idx[i]);
        pos += n;
    }
}

// Modes 0, 1, 2, 3, 6, 7: fits every subset of partition `part` and writes
// the block if it beats the best so far.
fn try_subsets(mode: u32, part: u32) {
    var ns = 1u;
    var pbits = 0u;
    var cb = 7u;
    var ab = 0u;
    var pk = P_UNIQUE;
    var ib = 4u;
    switch mode {
        case 0u: { ns = 3u; pbits = 4u; cb = 4u; ib = 3u; }
        case 1u: { ns = 2u; pbits = 6u; cb = 6u; pk = P_SHARED; ib = 3u; }
        case 2u: { ns = 3u; pbits = 6u; cb = 5u; pk = P_NONE; ib = 2u; }
        case 3u: { ns = 2u; pbits = 6u; cb = 7u; ib = 2u; }
        case 7u: { ns = 2u; pbits = 6u; cb = 5u; ab = 5u; ib = 2u; }
        default: { ab = 7u; } // mode 6
    }
    set_weights(ib);
    var err = 0.0;
    for (var s = 0u; s < ns; s++) {
        err += fit_subset(subset_mask(ns, part, s), s, cb, ab, pk);
        if (err >= out_err) {
            return;
        }
    }
    begin();
    pack_subsets(mode, part, ns, pbits, cb, ab, pk, ib);
    consider(err);
}

// --- modes 4 and 5: rotation and separate alpha -----------------------------

// The scalar channel's kept indices and endpoints (x code, y value).
var<private> best_idx2: array<u32, 16>;
var<private> a_lo: vec2u;
var<private> a_hi: vec2u;

// Fits the scalar channel `block[i].a` with `ab`-bit endpoints and
// `ib`-bit indices. Returns its squared error, unweighted.
fn fit_scalar(ab: u32, ib: u32) -> f32 {
    let n = 1u << ib;
    var lo = 255.0;
    var hi = 0.0;
    for (var i = 0u; i < 16u; i++) {
        lo = min(lo, block[i].a);
        hi = max(hi, block[i].a);
    }
    var best = BIG;
    var ix = array<u32, 16>();
    for (var it = 0u; it < ITERS; it++) {
        let ql = quant_channel(lo, ab, false, 0u);
        let qh = quant_channel(hi, ab, false, 0u);
        var err = 0.0;
        for (var i = 0u; i < 16u; i++) {
            var be = BIG;
            for (var k = 0u; k < n; k++) {
                let w = index_weight(ib, k);
                let v = floor(((64.0 - w) * f32(ql.y) + w * f32(qh.y) + 32.0) / 64.0);
                let d = block[i].a - v;
                if (d * d < be) {
                    be = d * d;
                    ix[i] = k;
                }
            }
            err += be;
        }
        if (err < best) {
            best = err;
            a_lo = ql;
            a_hi = qh;
            best_idx2 = ix;
        }
        // Least squares on the kept indices.
        var aa = 0.0;
        var ab2 = 0.0;
        var bb = 0.0;
        var ap = 0.0;
        var bp = 0.0;
        for (var i = 0u; i < 16u; i++) {
            let b = index_weight(ib, best_idx2[i]) / 64.0;
            let a = 1.0 - b;
            aa += a * a;
            ab2 += a * b;
            bb += b * b;
            ap += a * block[i].a;
            bp += b * block[i].a;
        }
        let det = aa * bb - ab2 * ab2;
        if (abs(det) < 1e-4) {
            break;
        }
        lo = clamp((bb * ap - ab2 * bp) / det, 0.0, 255.0);
        hi = clamp((aa * bp - ab2 * ap) / det, 0.0, 255.0);
    }
    return best;
}

// Rotation `rot` swaps alpha with R (1), G (2) or B (3).
fn rotate(v: vec4f, rot: u32) -> vec4f {
    if (rot == 1u) {
        return vec4f(v.a, v.g, v.b, v.r);
    }
    if (rot == 2u) {
        return vec4f(v.r, v.a, v.b, v.g);
    }
    if (rot == 3u) {
        return vec4f(v.r, v.g, v.a, v.b);
    }
    return v;
}

// Mode 4 (`isel` 1 gives colour the 3-bit indices) or mode 5, with
// rotation `rot`: the colour fitted as one subset, the rotated channel
// alone.
fn try_rotated(mode: u32, rot: u32, isel: u32) {
    let saved = ew;
    for (var i = 0u; i < 16u; i++) {
        block[i] = rotate(block[i], rot);
    }
    ew = rotate(saved, rot);
    let cb = select(7u, 5u, mode == 4u);
    let ab = select(8u, 6u, mode == 4u);
    let cib = select(2u, 3u, mode == 4u && isel == 1u);
    let aib = select(2u, 3u, mode == 4u && isel == 0u);
    set_weights(cib);
    var err = fit_subset(ALL, 0u, cb, 0u, P_NONE);
    if (err < out_err) {
        err += ew.a * fit_scalar(ab, aib);
    }
    for (var i = 0u; i < 16u; i++) {
        block[i] = rotate(block[i], rot);
    }
    ew = saved;
    if (err >= out_err) {
        return;
    }
    // Anchors: texel 0 of each index set has its top bit clear.
    var lo = s_lo[0];
    var hi = s_hi[0];
    if (best_idx[0] >= (1u << (cib - 1u))) {
        let t = lo;
        lo = hi;
        hi = t;
        for (var i = 0u; i < 16u; i++) {
            best_idx[i] = (1u << cib) - 1u - best_idx[i];
        }
    }
    var alo = a_lo;
    var ahi = a_hi;
    if (best_idx2[0] >= (1u << (aib - 1u))) {
        let t = alo;
        alo = ahi;
        ahi = t;
        for (var i = 0u; i < 16u; i++) {
            best_idx2[i] = (1u << aib) - 1u - best_idx2[i];
        }
    }
    begin();
    put(0u, mode + 1u, 1u << mode);
    var pos = mode + 1u;
    put(pos, 2u, rot);
    pos += 2u;
    if (mode == 4u) {
        put(pos, 1u, isel);
        pos += 1u;
    }
    for (var c = 0u; c < 3u; c++) {
        put(pos, cb, lo.code[c]);
        put(pos + cb, cb, hi.code[c]);
        pos += 2u * cb;
    }
    put(pos, ab, alo.x);
    put(pos + ab, ab, ahi.x);
    pos += 2u * ab;
    // The 2-bit set first, then the other: colour first unless mode 4
    // gives colour the 3-bit set.
    let colour_first = !(mode == 4u && isel == 1u);
    for (var half = 0u; half < 2u; half++) {
        let colour = (half == 0u) == colour_first;
        let ib = select(aib, cib, colour);
        for (var i = 0u; i < 16u; i++) {
            let n = select(ib, ib - 1u, i == 0u);
            put(pos, n, select(best_idx2[i], best_idx[i], colour));
            pos += n;
        }
    }
    consider(err);
}

// --- partition candidates ---------------------------------------------------

// Ranks the two-subset partitions by estimate into `top`.
fn rank2() {
    top_reset();
    let all = moments(ALL);
    for (var part = 0u; part < 64u; part++) {
        let ones = moments(P2[part]);
        top_add(part, line_error(moments_minus(all, ones)) + line_error(ones));
    }
}

// Ranks the first `n` three-subset partitions by estimate into `top`.
fn rank3(n: u32) {
    top_reset();
    let all = moments(ALL);
    for (var part = 0u; part < n; part++) {
        let one = moments(P3[part] & 0xffffu);
        let two = moments(P3[part] >> 16u);
        let zero = moments_minus(moments_minus(all, one), two);
        top_add(part, line_error(zero) + line_error(one) + line_error(two));
    }
}

// --- choice -----------------------------------------------------------------

// Below this error (squared, summed over the block: an RMS of about one
// code per channel) a block is smooth enough that nothing else can help.
const GOOD_ENOUGH: f32 = 48.0;

fn encode() {
    let is_opaque = opaque();
    if (is_opaque) {
        // Opaque: alpha must stay exactly 255. Modes 0 to 3 store none (the
        // decoder gives 255), mode 6 is steered to it by a heavy weight, and
        // modes 4 and 5 rotate a colour channel into it.
        ew.a = 1.0e4;
    }
    try_subsets(6u, 0u);
    if (out_err <= GOOD_ENOUGH) {
        return;
    }
    if (is_opaque) {
        ew.a = 0.0;
        rank2();
        for (var c = 0u; c < K; c++) {
            try_subsets(1u, top[c]);
            if (TIER >= 1u) {
                try_subsets(3u, top[c]);
            }
        }
        if (TIER >= 1u) {
            for (var rot = 1u; rot < 4u; rot++) {
                try_rotated(5u, rot, 0u);
            }
        }
        if (TIER >= 2u) {
            rank3(16u);
            for (var c = 0u; c < K; c++) {
                try_subsets(0u, top[c]);
            }
            rank3(64u);
            for (var c = 0u; c < K; c++) {
                try_subsets(2u, top[c]);
            }
            for (var rot = 1u; rot < 4u; rot++) {
                try_rotated(4u, rot, 0u);
                try_rotated(4u, rot, 1u);
            }
        }
    } else if (TIER >= 1u) {
        for (var rot = 0u; rot < 4u; rot++) {
            try_rotated(5u, rot, 0u);
        }
        rank2();
        for (var c = 0u; c < K; c++) {
            try_subsets(7u, top[c]);
        }
        if (TIER >= 2u) {
            for (var rot = 0u; rot < 4u; rot++) {
                try_rotated(4u, rot, 0u);
                try_rotated(4u, rot, 1u);
            }
        }
    }
}
