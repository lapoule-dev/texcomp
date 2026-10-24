// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

// What every block encoder shares: the job, one 4×4 block of texels, the
// 128-bit block being written, and an endpoint fit by principal axis and
// least squares. The format's own file supplies `encode()` and is appended
// after this one.
//
// Texels are the RGBA8 bytes as stored, values 0..255: the sRGB-encoded values
// when the texture is sampled as sRGB. The error every choice minimises is
// the squared distance in that space, which is the space PSNR is read in.

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
    _pad: u32,
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
var<private> wt: array<f32, 16>;
var<private> wn: u32;
// Each texel's index into `wt`, from the last `assign`, and the best kept.
var<private> idx: array<u32, 16>;
var<private> best_idx: array<u32, 16>;

const ALL: u32 = 0xffffu;

fn texel(x: u32, y: u32) -> vec4f {
    // Beyond the level's edge (levels of 2×2 and 1×1) the edge is repeated:
    // those texels are never sampled, and repeating keeps them from pulling
    // the fit.
    let px = min(x, job.width - 1u);
    let py = min(y, job.height - 1u);
    let v = src[job.src + py * job.width + px];
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

// Gives each member texel the index that decodes nearest it; returns the
// total squared error.
fn assign(e0: vec4f, e1: vec4f, members: u32) -> f32 {
    var total = 0.0;
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) == 0u) {
            continue;
        }
        var best = 3.4e38;
        var bi = 0u;
        for (var k = 0u; k < wn; k++) {
            let d = block[i] - lerp_index(e0, e1, k);
            let e = dot(d, d);
            if (e < best) {
                best = e;
                bi = k;
            }
        }
        idx[i] = bi;
        total += best;
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
// projections. `channels` masks the channels that count (alpha or not).
fn pca_segment(members: u32, channels: vec4f) -> Segment {
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
            let d = (block[i] - mean) * channels;
            cov += mat4x4f(d * d.x, d * d.y, d * d.z, d * d.w);
        }
    }
    // Power iteration, started on the bounding box's diagonal.
    var axis = (hi - lo) * channels;
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
            let t = dot(block[i] - mean, axis);
            tmin = min(tmin, t);
            tmax = max(tmax, t);
        }
    }
    let keep = vec4f(1.0) - channels;
    let e0 = clamp(mean + axis * tmin, vec4f(0.0), vec4f(255.0));
    let e1 = clamp(mean + axis * tmax, vec4f(0.0), vec4f(255.0));
    // Channels that don't count keep the mean (alpha of an opaque fit).
    return Segment(e0 * channels + mean * keep, e1 * channels + mean * keep);
}

// Least-squares endpoints for the member texels' current indices. Returns
// the endpoints unchanged when every texel has the same weight.
fn refine(members: u32, s: Segment) -> Segment {
    var aa = 0.0;
    var ab = 0.0;
    var bb = 0.0;
    var ap = vec4f(0.0);
    var bp = vec4f(0.0);
    for (var i = 0u; i < 16u; i++) {
        if (((members >> i) & 1u) != 0u) {
            let b = wt[idx[i]] / 64.0;
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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= job.blocks_x || id.y >= job.blocks_y) {
        return;
    }
    for (var i = 0u; i < 16u; i++) {
        block[i] = texel(id.x * 4u + (i & 3u), id.y * 4u + (i >> 2u));
    }
    bits = array<u32, 4>(0u, 0u, 0u, 0u);
    encode();
    let at = job.dst + id.y * job.row_words + id.x * 4u;
    dst[at] = bits[0];
    dst[at + 1u] = bits[1];
    dst[at + 2u] = bits[2];
    dst[at + 3u] = bits[3];
}
