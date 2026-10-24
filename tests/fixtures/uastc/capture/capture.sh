#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) lapoule.dev
#
# Runs inside the capture image (see Dockerfile). For every tile and effort
# level: encode with the upstream tool (UASTC LDR 4x4, no RDO, no Zstandard,
# no mips, one thread), dump the blocks, their ASTC 4x4 transcodes and decode
# digests, and cross-check the ASTC bytes against the upstream tool's own
# -unpack output.
#   /in/tiles           256² tiles, levels 0..2 -> /out/level<N>
#   /in/coverage-tiles  64² tiles,  levels 3..4 -> /out/coverage/level<N>
set -eu
mkdir -p /out /work

capture() { # <input dir> <output dir> <levels...>
    in=$1; out=$2; shift 2
    for level in "$@"; do
        mkdir -p "$out/level$level"
        for png in "$in"/*.png; do
            name=$(basename "$png" .png)
            rm -rf /work/run && mkdir /work/run
            cmd="basisu -uastc -uastc_level $level -ktx2 -ktx2_no_zstandard -no_multithreading -no_stats -output_file /work/run/$name.ktx2 -file $png"
            echo "$cmd" >> /out/commands.txt
            $cmd > /work/run/encode.log
            grep -q "UASTC LDR 4x4 Effort Level (0-4): $level" /work/run/encode.log
            dump "/work/run/$name.ktx2" "$out/level$level/$name"
            # The tool's own transcode (format 10 = ASTC 4x4 RGBA), as KTX1.
            (cd /work/run && basisu -unpack -format_only 10 -ktx_only "$name.ktx2" > unpack.log)
            python3 /capture/check_ktx1.py /work/run/*ASTC*.ktx "$out/level$level/$name.astc"
        done
    done
}

capture /in/tiles /out 0 1 2
capture /in/coverage-tiles /out/coverage 3 4
