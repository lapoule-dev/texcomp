# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) lapoule.dev
"""Writes every mip level of a KTX1 file (one face, no array), level 0 first,
concatenated, to a raw file."""
import struct
import sys

ktx_path, raw_path = sys.argv[1], sys.argv[2]
with open(ktx_path, "rb") as f:
    k = f.read()
assert k[:12] == b"\xabKTX 11\xbb\r\n\x1a\n", ktx_path
faces, levels, kv_len = struct.unpack_from("<III", k, 52)
assert faces == 1, ktx_path
at = 64 + kv_len
out = b""
for _ in range(max(1, levels)):
    size = struct.unpack_from("<I", k, at)[0]
    out += k[at + 4 : at + 4 + size]
    at += 4 + size + (-size % 4)
assert at == len(k), f"{ktx_path}: {len(k) - at} trailing bytes"
with open(raw_path, "wb") as f:
    f.write(out)
print(f"{raw_path}: {max(1, levels)} levels, {len(out)} bytes")
