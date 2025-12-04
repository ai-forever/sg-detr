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


# Install requirements
COPY Dockerfiles/requirements.txt .
COPY requirements-dev.txt .
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchtext==0.16.0 torcheval==0.0.7
RUN pip install --no-cache-dir -r requirements-dev.txt

# stand container
FROM base AS prod
WORKDIR /mr-train-repo
RUN mkdir data/
COPY src/ .
COPY configs/ .
COPY .pre-commit-config.yaml ./
COPY pyproject.toml setup.cfg ./
COPY setup.py README.md ./

# VSCode setup
FROM base AS vscode
ENV PASSWORD=vscode_server
ENV PORT=6001

# Install Visual Studio Code Server
RUN wget -qO- https://code-server.dev/install.sh | sh

EXPOSE $PORT
CMD code-server --auth password --port $PORT --host 0.0.0.0 --disable-telemetry
