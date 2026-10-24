# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) lapoule.dev
"""Asserts that a KTX1 file's level 0 holds exactly the bytes of a raw file."""
import struct
import sys

ktx_path, raw_path = sys.argv[1], sys.argv[2]
with open(ktx_path, "rb") as f:
    k = f.read()
assert k[:12] == b"\xabKTX 11\xbb\r\n\x1a\n", ktx_path
kv_len = struct.unpack_from("<I", k, 60)[0]
at = 64 + kv_len
size = struct.unpack_from("<I", k, at)[0]
level0 = k[at + 4 : at + 4 + size]
with open(raw_path, "rb") as f:
    raw = f.read()
assert level0 == raw, f"{ktx_path}: {len(level0)} bytes differ from {raw_path}"
print(f"{raw_path}: matches the tool's -unpack ({size} bytes)")
