# https://github.com/geyang/jaynes-starter-kit/blob/master/docker/pytorch/Dockerfile
FROM nvidia/cudagl:11.4.0-devel-ubuntu20.04
# FROM nvidia/cudagl:10.1-runtime-ubuntu18.04
# FROM ubuntu:20.04

# Install python3.8
RUN apt update && \
   DEBIAN_FRONTEND=noninteractive \
   TZ=America/Chicago \
   apt -y install \
   wget software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.8 python3.8-dev python3.8-distutils
RUN ln -s `which python3.8` /usr/local/bin/python
RUN ln -s `which python3.8` /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py

ENV PIP_NO_CACHE_DIR=1
RUN apt update
RUN yes | apt install \
    tree \
    zsh  \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev

RUN yes | apt install libcgal-qt5-dev
RUN pip install --upgrade pip
RUN pip install pytest pytest-forked lz4 pyyaml
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install datasets einops sklearn torchinfo tqdm



# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    # terminator \
    # openssh-client \
    # openssh-server \
    apt-utils \
    # dialog \
    ffmpeg \
    # nvidia-settings \
    libffi-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    # flex \
    # bison \
    gcc \
    wget \
    # module-init-tools \
    pciutils \
    xserver-xorg \
    xserver-xorg-video-fbdev \
    xauth
    # python3-tk


# install GLX-Gears (for debugging)
RUN apt-get update && apt-get install -y \
   mesa-utils \
   python3-setuptools \
   && rm -rf /var/lib/apt/lists/*


RUN pip install \
   tdqm \
   wandb \
   kornia \
   ffmpeg \
   imageio \
   imageio-ffmpeg \
   moviepy

# Dependencies from diffuser
RUN pip install \
   matplotlib==3.3.4 \
   typed-argument-parser \
   scikit-image==0.17.2 \
   scikit-video==1.1.11 \
   gitpython \
   einops \
   pillow \
   pandas \
   flax \
   jax==0.2.21 \
   ray==1.9.1 \
   crcmod \
   opencv-python

ADD requirements.txt .
RUN pip install -r requirements.txt
# RUN pip install -e .

RUN git config --global url."https://".insteadOf git://
RUN pip install git+https://github.com/JannerM/d4rl.git@13fc08c198d17ac627f528aedc70843e8f86bb71

# Always install geyang products at last
RUN pip install --ignore-installed httplib2
RUN apt-get update && apt-get install -y --no-install-recommends \
   libcurl4-openssl-dev libssl-dev
RUN pip install jaynes ml-logger waterbear params-proto functional-notations

# Workaround for a weird TensorFlow issue
RUN pip install protobuf==3.20.*



RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
# COPY vendor/Xdummy /usr/local/bin/Xdummy
# RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
# COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN mkdir -p /usr/local/etc/mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /usr/local/etc/mujoco \
    && rm mujoco.tar.gz
ENV MUJOCO_PY_MUJOCO_PATH="/usr/local/etc/mujoco/mujoco210"
ENV LD_LIBRARY_PATH="/usr/local/etc/mujoco/mujoco210/bin"
# Compile-time library location
ENV LIBRARY_PATH=$LD_LIBRARY_PATH
RUN ln -s /usr/local/etc/mujoco $HOME/.mujoco \
    && mkdir -p /mujoco_py/mujoco_py/binaries \
    && ln -s /usr/local/etc/mujoco /mujoco_py/mujoco_py/binaries/linux

