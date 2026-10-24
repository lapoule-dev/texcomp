# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) lapoule.dev
#
# Captures the UASTC reference vectors from upstream Basis Universal v2.50.
# Run from tests/fixtures/uastc (see the README there):
#   docker buildx build --builder stl-builder-k8s --platform linux/amd64 \
#     -f capture/Dockerfile --output type=local,dest=<dir> .
FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git cmake make g++ python3 && rm -rf /var/lib/apt/lists/*
ARG UPSTREAM_TAG=v2_50
ARG UPSTREAM_COMMIT=9bebe16726b3a61c8c213eeee3b7cffb462ef34e
RUN git clone --depth 1 --branch ${UPSTREAM_TAG} https://github.com/BinomialLLC/basis_universal /src \
    && test "$(git -C /src rev-parse HEAD)" = "${UPSTREAM_COMMIT}"
RUN cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DBASISU_EXAMPLES=OFF \
    && cmake --build /build -j4 --target basisu \
    && install -m 755 "$(find /build /src/bin -name basisu -type f -perm -u+x | head -n 1)" /usr/local/bin/basisu \
    && (basisu -version || true)
COPY capture/dump.cpp /capture/dump.cpp
RUN g++ -O2 -std=c++17 -fno-strict-aliasing -DBASISD_SUPPORT_KTX2_ZSTD=0 -I/src \
        /capture/dump.cpp /src/transcoder/basisu_transcoder.cpp -o /usr/local/bin/dump
COPY capture/capture.sh capture/check_ktx1.py /capture/
COPY tiles /in/tiles
COPY coverage-tiles /in/coverage-tiles
RUN mkdir -p /work && sh /capture/capture.sh \
    && cp /src/LICENSE /out/UPSTREAM-LICENSE \
    && (g++ --version | head -n 1; uname -m; git -C /src rev-parse HEAD) > /out/toolchain.txt

FROM scratch
COPY --from=build /out /
