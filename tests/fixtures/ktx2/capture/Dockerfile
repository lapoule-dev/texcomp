# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) lapoule.dev
#
# Captures upstream's KTX 2.0 files and checks ours with independent readers:
# upstream Basis Universal v2.50 and the Khronos KTX-Software tools.
# Run from tests/fixtures (see ktx2/README.md):
#   docker buildx build --builder stl-builder-k8s --platform linux/amd64 \
#     -f ktx2/capture/Dockerfile --output type=local,dest=<dir> .
# The first stage's lines are those of uastc/capture/Dockerfile, so the
# builder's cache holds the upstream build.
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
# The Khronos tools, from their release (checked against its published SHA-1).
ARG KTX_VERSION=4.4.2
RUN apt-get update && apt-get install -y --no-install-recommends curl bzip2 && rm -rf /var/lib/apt/lists/* \
    && base=https://github.com/KhronosGroup/KTX-Software/releases/download/v${KTX_VERSION} \
    && file=KTX-Software-${KTX_VERSION}-Linux-x86_64.tar.bz2 \
    && curl -fsSLO "$base/$file" && curl -fsSLO "$base/$file.sha1" \
    && echo "$(cut -d' ' -f1 $file.sha1)  $file" | sha1sum -c - \
    && mkdir /opt/ktx && tar -xjf "$file" -C /opt/ktx --strip-components=1 && rm "$file" "$file.sha1" \
    && ln -s "$(find /opt/ktx -name ktx -type f -perm -u+x | head -n 1)" /usr/local/bin/ktx \
    && ktx --version
ENV LD_LIBRARY_PATH=/opt/ktx/lib
COPY ktx2/capture/capture.sh ktx2/capture/ktx1_levels.py /capture/
COPY uastc/tiles/alps-0.png /in/tiles/
COPY uastc/coverage-tiles/alpha-imagery.png /in/coverage-tiles/
COPY ktx2/inputs /in/inputs
COPY ktx2/ours /in/ours
RUN sh /capture/capture.sh \
    && cp /src/LICENSE /out/UPSTREAM-LICENSE \
    && (basisu -version 2>&1 | head -n 3; ktx --version; uname -m; git -C /src rev-parse HEAD) > /out/toolchain.txt

FROM scratch
COPY --from=build /out /
