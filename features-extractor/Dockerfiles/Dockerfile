FROM nvidia/cuda:11.4.3-devel-ubuntu20.04 AS base

# Set ENV variables
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# Install essential packages, and set timezone
RUN ln -snf /usr/share/zoneinfo/Europe/Moscow /etc/localtime && \
    echo Europe/Moscow > /etc/timezone && \
    apt-get update && apt-get install -y apt-utils software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install -y \
    wget zlib1g-dev libncurses5-dev libgdbm-dev curl \
    libnss3-dev libssl-dev libreadline-dev libffi-dev \
    autoconf automake libtool build-essential unzip git \
    libx264-dev libgnutls28-dev libmp3lame-dev yasm cmake npm \
    libc6 libc6-dev libnuma1 libnuma-dev pkgconf libsqlite3-dev libbz2-dev \
    lzma liblzma-dev libbz2-dev libsm6 libxext6 libgl1 libglib2.0-0 libavdevice58 \
    python3.10 python3.10-distutils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Installing pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Set python3.10 and pip3.10 as the default python and pip versions
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 1 && \
    python --version && pip --version && pip install --upgrade pip

# Download the 11.0.10.2 version of nv-codec-headers and install. Suitable for nvidia driver 470.57.02 or newer
RUN wget https://github.com/FFmpeg/nv-codec-headers/releases/download/n11.0.10.2/nv-codec-headers-11.0.10.2.tar.gz && \
    tar -xzvf nv-codec-headers-11.0.10.2.tar.gz && \
    cd nv-codec-headers-11.0.10.2 && \
    make install && \
    cd .. && \
    rm -rf nv-codec-headers-11.0.10.2.tar.gz nv-codec-headers-11.0.10.2

# Download the n6.0 version of ffmpeg and install
ARG COMPUTE=70
RUN wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n6.0.tar.gz && \
    tar -xzvf n6.0.tar.gz && \
    cd FFmpeg-n6.0 && \
    ./configure \
        --enable-openssl \
        --enable-gpl \
        --enable-libx264 \
        --enable-libmp3lame \
        --enable-decoder=aac \
        --enable-decoder=h264 \
        --enable-decoder=h264_cuvid \
        --nvccflags="-gencode arch=compute_$COMPUTE,code=sm_$COMPUTE -O2" \
        --enable-cuda-nvcc \
        --enable-cuvid \
        --enable-nvenc \
        --enable-nonfree \
        --enable-libnpp \
        --extra-cflags=-I/usr/local/cuda-11.4/include \
        --extra-ldflags=-L/usr/local/cuda-11.4/lib64 && \
    make -j 8 && make install && \
    cd .. && \
    rm -rf n6.0.tar.gz FFmpeg-n6.0

# Install requirements
COPY requirements.txt .
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

WORKDIR /mr-feature-extraction
RUN mkdir weights/
RUN mkdir data/
RUN mkdir src/
COPY .pre-commit-config.yaml ./
COPY pyproject.toml setup.cfg ./
COPY setup.py README.md ./

# VSCode setup
FROM base AS vscode
ARG PORT=6007
ENV PORT=$PORT
ENV PASSWORD=vscode_server

# Install Visual Studio Code Server
RUN wget -qO- https://code-server.dev/install.sh | sh

EXPOSE $PORT
CMD code-server --auth password --port $PORT --host 0.0.0.0 --disable-telemetry
