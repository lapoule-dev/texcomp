// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

// What every block encoder shares: the job, one 4×4 block of texels, the
// 128-bit block being written, an endpoint fit by principal axis and least
// squares, the partition estimate and the best encoding kept. The generated
// constants (`TIER`, `K`, `ITERS`, `EW` and the format's tables) come first,
// then this file, then the format's own, which supplies `encode()`: it tries
// encodings, writing each into `bits` and handing its error to `consider`.
//
// Texels are the RGBA8 bytes as stored, values 0..255: the sRGB-encoded values
// when the texture is sampled as sRGB. The error every choice minimises is
// the squared distance in that space, each channel weighted by `ew`: equal
// weights make it the space PSNR is read in.

struct Job {
    // First texel of this level in `src`.
    src: u32,
    width: u32,
    height: u32,
    // First word of this level's blocks in `dst`.
    dst: u32,
    // Words between two rows of blocks (a row is padded to 256 bytes for the
    // copy into the texture).
    row_words: u32,
    blocks_x: u32,
    blocks_y: u32,
    // Texels between two rows of this level in `src`: its width for tightly
    // packed texels, more for a level copied in from a buffer or a texture
    // with padded rows.
    src_stride: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> job: Job;

// The block's texels, row-major: texel i is (i & 3, i >> 2).
var<private> block: array<vec4f, 16>;
// The block being written, least significant word first.
var<private> bits: array<u32, 4>;
// The unquantised interpolation weight (0..64) of each index the format can
// store, and how many there are. Set by the format before a fit.
var<private> wt: array<f32, 32>;
var<private> wn: u32;
// Each texel's index into `wt`, from the last `assign`, and the best kept.
var<private> idx: array<u32, 16>;
var<private> best_idx: array<u32, 16>;
// The error weight of each channel, `EW` (from the quality) unless the
// format changes it for a block.
var<private> ew: vec4f;

const ALL: u32 = 0xffffu;

fn texel(x: u32, y: u32) -> vec4f {
    // Beyond the level's edge (levels of 2×2 and 1×1) the edge is repeated:
    // those texels are never sampled, and repeating keeps them from pulling
    // the fit.
    let px = min(x, job.width - 1u);
    let py = min(y, job.height - 1u);
    let v = src[job.src + py * job.src_stride + px];
    return vec4f(
        f32(v & 255u),
        f32((v >> 8u) & 255u),
        f32((v >> 16u) & 255u),
        f32(v >> 24u),
    );
}

// Writes the low `n` bits of `value` at bit `pos` of the block (n ≤ 32).
fn put(pos: u32, n: u32, value: u32) {
    let w = pos >> 5u;
    let o = pos & 31u;
    var v = value;
    if (n < 32u) {
        v = value & ((1u << n) - 1u);
    }
    bits[w] = bits[w] | (v << o);
    if (o + n > 32u) {
        bits[w + 1u] = bits[w + 1u] | (v >> (32u - o));
    }
}

fn opaque() -> bool {
    for (var i = 0u; i < 16u; i++) {
        if (block[i].a < 255.0) {
            return false;
        }
    }
    return true;
}

// The colour index `k` decodes to between two 8-bit endpoints.
fn lerp_index(e0: vec4f, e1: vec4f, k: u32) -> vec4f {
    let w = wt[k];
    return floor(((64.0 - w) * e0 + w * e1 + 32.0) / 64.0);
}

// For each integer weight v in 0..=64, the index whose weight is the
// largest at or below v (low byte) and the smallest at or above it (second
// byte). Set by the format with `wt`.
var<private> wsel: array<u32, 65>;

// Gives each member texel the index that decodes nearest it, by the error
// weighted per channel by `w`; returns the total weighted squared error.
//
// The texel is projected on the segment (in the weighted space) to its
// ideal weight; of the indices, only the two whose weights bracket it can
// be nearest along the segment, and the decoded colour of both is compared.
fn assign(e0: vec4f, e1: vec4f, members: u32, w: vec4f) -> f32 {
    let d = e1 - e0;
    let dw = d * w;
    let dd = dot(d, dw);
    var total = 0.0;
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) == 0u) {
            continue;
        }
        var t = 0.0;
        if (dd > 1e-6) {
            t = clamp(dot(block[i] - e0, dw) / dd * 64.0, 0.0, 64.0);
        }
        let k0 = wsel[u32(floor(t))] & 0xffu;
        let k1 = (wsel[u32(ceil(t))] >> 8u) & 0xffu;
        let d0 = block[i] - lerp_index(e0, e1, k0);
        let d1 = block[i] - lerp_index(e0, e1, k1);
        let err0 = dot(d0 * d0, w);
        let err1 = dot(d1 * d1, w);
        if (err1 < err0) {
            idx[i] = k1;
            total += err1;
        } else {
            idx[i] = k0;
            total += err0;
        }
    }
    return total;
}

fn keep_indices(members: u32) {
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) != 0u) {
            best_idx[i] = idx[i];
        }
    }
}

struct Segment {
    e0: vec4f,
    e1: vec4f,
}

// The member texels' principal axis, through their mean, cut at the extreme
// projections, in the space where each channel is scaled by the square root
// of its error weight `w` (0 for a channel that is not fitted, which keeps
// the mean).
fn pca_segment(members: u32, w: vec4f) -> Segment {
    let sw = sqrt(w);
    var n = 0.0;
    var mean = vec4f(0.0);
    var lo = vec4f(255.0);
    var hi = vec4f(0.0);
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) != 0u) {
            n += 1.0;
            mean += block[i];
            lo = min(lo, block[i]);
            hi = max(hi, block[i]);
        }
    }
    mean = mean / max(n, 1.0);
    var cov = mat4x4f(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) != 0u) {
            let d = (block[i] - mean) * sw;
            cov += mat4x4f(d * d.x, d * d.y, d * d.z, d * d.w);
        }
    }
    // Power iteration, started on the bounding box's diagonal.
    var axis = (hi - lo) * sw;
    if (dot(axis, axis) < 1e-6) {
        return Segment(mean, mean);
    }
    for (var it = 0; it < 6; it++) {
        let next = cov * axis;
        let len = length(next);
        if (len < 1e-6) {
            break;
        }
        axis = next / len;
    }
    axis = normalize(axis);
    var tmin = 3.4e38;
    var tmax = -3.4e38;
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) != 0u) {
            let t = dot((block[i] - mean) * sw, axis);
            tmin = min(tmin, t);
            tmax = max(tmax, t);
        }
    }
    // Back to texel space: the axis divided by the scale, per channel.
    let fitted = select(vec4f(0.0), vec4f(1.0), w > vec4f(0.0));
    let dir = axis / max(sw, vec4f(1e-6)) * fitted;
    let e0 = clamp(mean + dir * tmin, vec4f(0.0), vec4f(255.0));
    let e1 = clamp(mean + dir * tmax, vec4f(0.0), vec4f(255.0));
    return Segment(e0, e1);
}

// Least-squares endpoints for the member texels' kept indices (`best_idx`).
// Returns the endpoints unchanged when every texel has the same weight.
fn refine(members: u32, s: Segment) -> Segment {
    var aa = 0.0;
    var ab = 0.0;
    var bb = 0.0;
    var ap = vec4f(0.0);
    var bp = vec4f(0.0);
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) != 0u) {
            let b = wt[best_idx[i]] / 64.0;
            let a = 1.0 - b;
            aa += a * a;
            ab += a * b;
            bb += b * b;
            ap += a * block[i];
            bp += b * block[i];
        }
    }
    let det = aa * bb - ab * ab;
    if (abs(det) < 1e-4) {
        return s;
    }
    let e0 = (bb * ap - ab * bp) / det;
    let e1 = (aa * bp - ab * ap) / det;
    return Segment(
        clamp(e0, vec4f(0.0), vec4f(255.0)),
        clamp(e1, vec4f(0.0), vec4f(255.0)),
    );
}

// --- partition estimates ----------------------------------------------------

// Raw moments of a set of texels' RGB, scaled by the square roots of the
// error weights: count, sum, sum of outer products. A partition's subsets
// cost one pass over all but one of them: the last is the block's total
// minus the others.
struct Moments {
    n: f32,
    sum: vec3f,
    outer: mat3x3f,
}

fn moments(members: u32) -> Moments {
    let sw = sqrt(ew.rgb);
    var m = Moments(0.0, vec3f(0.0), mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0)));
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) != 0u) {
            let p = block[i].rgb * sw;
            m.n += 1.0;
            m.sum += p;
            m.outer += mat3x3f(p * p.x, p * p.y, p * p.z);
        }
    }
    return m;
}

fn moments_minus(a: Moments, b: Moments) -> Moments {
    return Moments(a.n - b.n, a.sum - b.sum, a.outer - b.outer);
}

// Squared error a subset leaves when its texels are put on one line and
// quantised to 8 steps along it: the variance off the principal axis, plus
// the quantisation noise of 8 levels along it.
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

// The `K` best candidates by estimate, best first.
var<private> top: array<u32, 16>;
var<private> top_e: array<f32, 16>;

fn top_reset() {
    for (var i = 0u; i < 16u; i++) {
        top[i] = 0u;
        top_e[i] = 3.4e38;
    }
}

fn top_add(candidate: u32, e: f32) {
    if (e >= top_e[K - 1u]) {
        return;
    }
    var j = K - 1u;
    while (j > 0u && top_e[j - 1u] > e) {
        top_e[j] = top_e[j - 1u];
        top[j] = top[j - 1u];
        j -= 1u;
    }
    top_e[j] = e;
    top[j] = candidate;
}

// --- the best encoding so far -----------------------------------------------

var<private> out_bits: array<u32, 4>;
var<private> out_err: f32;

// Clears the block being written, for the next trial.
fn begin() {
    bits = array<u32, 4>(0u, 0u, 0u, 0u);
}

// Keeps the block just written if it beats the best so far.
fn consider(err: f32) {
    if (err < out_err) {
        out_err = err;
        out_bits = bits;
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= job.blocks_x || id.y >= job.blocks_y) {
        return;
    }
    for (var i = 0u; i < 16u; i++) {
        block[i] = texel(id.x * 4u + (i & 3u), id.y * 4u + (i >> 2u));
    }
    bits = array<u32, 4>(0u, 0u, 0u, 0u);
    out_err = 3.4e38;
    ew = EW;
    encode();
    let at = job.dst + id.y * job.row_words + id.x * 4u;
    dst[at] = out_bits[0];
    dst[at + 1u] = out_bits[1];
    dst[at + 2u] = out_bits[2];
    dst[at + 3u] = out_bits[3];
}
