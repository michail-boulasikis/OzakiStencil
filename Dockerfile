FROM nvidia/cuda:13.2.0-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirrors.kernel.org/g' /etc/apt/sources.list.d/ubuntu.sources || true \
    && sed -i 's/archive.ubuntu.com/mirrors.kernel.org/g' /etc/apt/sources.list || true

# RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 && \
#    echo 'Acquire::http::Timeout "10";' > /etc/apt/apt.conf.d/99timeout && \
#    echo 'Acquire::ftp::Timeout "10";' >> /etc/apt/apt.conf.d/99timeout && \
#    echo 'Acquire::Retries "3";' >> /etc/apt/apt.conf.d/99timeout

RUN apt-get update && apt-get install --fix-missing -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libgtest-dev \
    vim \
    software-properties-common \
    curl \
    wget \
    libquadmath0 \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3.13 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy scipy seaborn matplotlib pandas

# CUTLASS/CuTe is header-only: a shallow clone of a pinned release is enough.
# CMake locates it through CUTLASS_PATH (see find_path in CMakeLists.txt).
ARG CUTLASS_VERSION=v4.8.1
RUN git clone --depth 1 --branch ${CUTLASS_VERSION} https://github.com/NVIDIA/cutlass.git /opt/cutlass
ENV CUTLASS_PATH=/opt/cutlass

WORKDIR /artifact
COPY . /artifact
RUN chmod +x /artifact/entrypoint.sh /artifact/build.sh /artifact/correctness.sh /artifact/performance.sh

RUN mkdir -p /artifact/benchmark && \
    cd /artifact/benchmark && \
    git clone https://github.com/microsoft/ConvStencil.git ConvStencil && \
    cd ConvStencil && git checkout 89688a1 && \
    git apply /artifact/patch/convstencil.patch

RUN mkdir -p /artifact/benchmark && \
    cd /artifact/benchmark && \
    git clone https://github.com/neozhang307/EBISU-ICS23.git EBISU && \
    cd EBISU && git checkout dd3d844 && \
    git apply /artifact/patch/ebisu.patch

ENTRYPOINT ["/artifact/entrypoint.sh"]
