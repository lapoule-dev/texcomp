// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev
//
// Reads the level-0 UASTC blocks of an uncompressed KTX2 file the upstream
// tool wrote and writes, next to `<out>`:
//   <out>.uastc    the blocks, 16 bytes each, row-major;
//   <out>.astc     each block's ASTC 4x4 transcode (transcode_uastc_to_astc);
//   <out>.decoded  per block, two little-endian u32 FNV-1a digests of the 64
//                  RGBA8 bytes unpack_uastc decodes: linear, then sRGB.
// Built against upstream's transcoder only; see capture.sh.

#include "transcoder/basisu_transcoder.h"
#include "transcoder/basisu_transcoder_uastc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<uint8_t> slurp(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    std::vector<uint8_t> v;
    uint8_t buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
    fclose(f);
    return v;
}

static void spit(const std::string& path, const std::vector<uint8_t>& v) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f || fwrite(v.data(), 1, v.size(), f) != v.size()) { perror(path.c_str()); exit(1); }
    fclose(f);
}

static uint32_t rd32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint64_t rd64(const uint8_t* p) { return rd32(p) | ((uint64_t)rd32(p + 4) << 32); }

static uint32_t fnv1a(const uint8_t* p, size_t n) {
    uint32_t h = 0x811c9dc5u;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x01000193u; }
    return h;
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: dump <in.ktx2> <out-prefix>\n"); return 2; }
    basist::basisu_transcoder_init();
    std::vector<uint8_t> k = slurp(argv[1]);
    static const uint8_t id[12] = {0xAB, 'K', 'T', 'X', ' ', '2', '0', 0xBB, '\r', '\n', 0x1A, '\n'};
    if (k.size() < 104 || memcmp(k.data(), id, 12) != 0) { fprintf(stderr, "not KTX2\n"); return 1; }
    const uint32_t width = rd32(&k[20]), height = rd32(&k[24]), levels = rd32(&k[40]), scheme = rd32(&k[44]);
    if (scheme != 0) { fprintf(stderr, "supercompressed (scheme %u)\n", scheme); return 1; }
    if (levels != 1) { fprintf(stderr, "%u levels, want 1\n", levels); return 1; }
    const uint64_t off = rd64(&k[80]), len = rd64(&k[88]);
    const uint32_t bx = (width + 3) / 4, by = (height + 3) / 4;
    if (len != (uint64_t)bx * by * 16 || off + len > k.size()) { fprintf(stderr, "bad level 0\n"); return 1; }

    std::vector<uint8_t> uastc(k.begin() + off, k.begin() + off + len), astc(len), decoded;
    for (uint32_t i = 0; i < bx * by; i++) {
        const basist::uastc_block& blk = *reinterpret_cast<const basist::uastc_block*>(&uastc[i * 16]);
        if (!basist::transcode_uastc_to_astc(blk, &astc[i * 16])) { fprintf(stderr, "block %u: transcode failed\n", i); return 1; }
        for (int srgb = 0; srgb < 2; srgb++) {
            basist::color32 px[16];
            if (!basist::unpack_uastc(blk, px, srgb != 0)) { fprintf(stderr, "block %u: unpack failed\n", i); return 1; }
            const uint32_t h = fnv1a(reinterpret_cast<const uint8_t*>(px), 64);
            for (int b = 0; b < 4; b++) decoded.push_back((uint8_t)(h >> (8 * b)));
        }
    }
    const std::string out = argv[2];
    spit(out + ".uastc", uastc);
    spit(out + ".astc", astc);
    spit(out + ".decoded", decoded);
    printf("%s: %ux%u, %u blocks\n", argv[1], width, height, bx * by);
    return 0;
}
