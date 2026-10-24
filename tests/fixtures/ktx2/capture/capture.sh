#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) lapoule.dev
#
# Runs inside the capture image (see Dockerfile).
#  1. Upstream's KTX 2.0 files: each input encoded to UASTC with its full mip
#     chain, no Zstandard, then unpacked by the tool itself to ASTC 4x4 and
#     BC7, every level kept (-> /out/upstream).
#  2. Ours, if any (/in/ours): every file through the Khronos `ktx validate`
#     and `ktx info`; the UASTC ones also through upstream's -info and
#     -unpack, whose ASTC and BC7 levels must equal those of the upstream
#     file they were rewritten from (-> /out/validate.txt).
set -eu
mkdir -p /out/upstream /work
log=/out/validate.txt
: > "$log"
failed=0

for png in /in/tiles/alps-0.png /in/coverage-tiles/alpha-imagery.png /in/inputs/alps-0-100x60.png; do
    name=$(basename "$png" .png)
    rm -rf /work/run && mkdir /work/run
    cmd="basisu -uastc -uastc_level 1 -mipmap -ktx2 -ktx2_no_zstandard -no_multithreading -no_stats -output_file /out/upstream/$name.ktx2 -file $png"
    echo "$cmd" >> /out/commands.txt
    $cmd > /work/run/encode.log
    (cd /work/run && basisu -unpack -format_only 10 -ktx_only "/out/upstream/$name.ktx2" > unpack.log)
    python3 /capture/ktx1_levels.py /work/run/*ASTC*.ktx "/out/upstream/$name.astc"
    (cd /work/run && basisu -unpack -format_only 6 -ktx_only "/out/upstream/$name.ktx2" > unpack-bc7.log)
    python3 /capture/ktx1_levels.py /work/run/*BC7*.ktx "/out/upstream/$name.bc7"
    echo "== ktx validate upstream/$name.ktx2 (control)" >> "$log"
    ktx validate "/out/upstream/$name.ktx2" >> "$log" 2>&1 || echo "FAILED (control)" >> "$log"
done

for f in /in/ours/*.ktx2; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .ktx2)
    echo "== ktx validate ours/$name.ktx2" >> "$log"
    if ktx validate "$f" >> "$log" 2>&1; then echo "ok" >> "$log"; else echo "FAILED" >> "$log"; failed=1; fi
    echo "== ktx info ours/$name.ktx2" >> "$log"
    ktx info "$f" >> "$log" 2>&1
    case "$name" in
    *-uastc)
        src=${name%-uastc}
        echo "== basisu -info and -unpack ours/$name.ktx2, levels compared with upstream's" >> "$log"
        # Chained with &&: `set -e` does not apply inside an `if` condition.
        if (
            rm -rf /work/run && mkdir /work/run && cd /work/run \
            && basisu -info "$f" \
            && basisu -unpack -format_only 10 -ktx_only "$f" > unpack.log \
            && python3 /capture/ktx1_levels.py /work/run/*ASTC*.ktx /work/run/ours.astc \
            && cmp /work/run/ours.astc "/out/upstream/$src.astc" \
            && basisu -unpack -format_only 6 -ktx_only "$f" > unpack-bc7.log \
            && python3 /capture/ktx1_levels.py /work/run/*BC7*.ktx /work/run/ours.bc7 \
            && cmp /work/run/ours.bc7 "/out/upstream/$src.bc7"
        ) >> "$log" 2>&1; then echo "ok" >> "$log"; else echo "FAILED" >> "$log"; failed=1; fi
        ;;
    esac
done
# The outputs come back either way; the last line says how it went.
if [ "$failed" = 0 ]; then echo "ALL CHECKS PASSED" >> "$log"; else echo "SOME CHECKS FAILED" >> "$log"; fi
