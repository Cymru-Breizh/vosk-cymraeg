FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04
LABEL maintainer="prv21fgt@bangor.ac.uk"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        make \
        automake \
        bzip2 \
        unzip \
        wget \
        libtool \
        git \
        python3 \
        zlib1g-dev \
        ca-certificates \
        gfortran \
        patch \
        sox \
        software-properties-common && \
        apt-add-repository multiverse && \
        apt-get update && \
        yes | DEBIAN_FRONTEND=noninteractive apt-get install -yqq --no-install-recommends\
            intel-mkl && \
    rm -rf /var/lib/apt/lists/*


RUN git clone https://github.com/kaldi-asr/kaldi.git /opt/kaldi
WORKDIR /opt/kaldi/tools
RUN git reset --hard 22f36e5b592b587af5ee50324271df55eff879ab
COPY openfst-1.8.3.tar.gz /opt/kaldi/tools/openfst-1.8.3.tar.gz
RUN make -j $(nproc)

WORKDIR /opt/kaldi/src
RUN ./configure --shared --use-cuda=yes
RUN make depend -j $(nproc)
RUN make -j $(nproc)
RUN find /opt/kaldi  -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \;

# Patch Kaldi recipe scripts for OpenFST 1.8 (--project_output=true -> --project_type=output).
# Required because the pinned Kaldi commit ships scripts that still use the old flag,
# but tools/Makefile pulls OpenFST 1.8.3 which rejects it. Drop this when Kaldi upstream fixes it.
RUN find /opt/kaldi/egs/wsj/s5/steps /opt/kaldi/egs/wsj/s5/utils \
    -type f -exec sed -i 's/--project_output=true/--project_type=output/g' {} +

RUN rm -rf /opt/kaldi/.git

WORKDIR /opt/kaldi/

# Install apt packages
# (only curl is required for UV, the rest are for the developer)
RUN apt update
RUN apt upgrade -y
RUN apt install -y \
    curl \
    fish \
    gawk \
    gcc \ 
    git \
    libsox-fmt-mp3 \
    locales \
    make \
    neovim \
    python2.7 \
    tar \
    wget

# Install en_US.UTF-8 (see issue #17)
RUN locale-gen en_US.UTF-8

# Force python to refer to python2
RUN ln -s python2.7 /usr/bin/python

# pocolm (Apache-2.0) for language modelling, replacing SRILM.
#
# SRILM's Research Community License limits use to non-commercial research, which
# is why published Kaldi/Vosk models carried CC-BY-NC. It was also being fetched
# from an unauthorised third-party mirror rather than SRI's download form, so the
# licence was never accepted. pocolm is permissive, so models built with it can
# ship under Apache-2.0/CC-BY; it also tunes discounting constants (and, once
# there is more than one text source, interpolation weights) on held-out dev data
# rather than using the fixed Witten-Bell discounting we had before.
#
# numpy is a genuine runtime dependency (scripts/optimize_metaparameters.py and
# scripts/internal/bfgs.py). We build `src` directly rather than running the
# top-level `make`, because pocolm's check_dependencies.sh fails the build unless
# subversion is installed -- a vestigial check inherited from Kaldi; pocolm never
# uses svn.
#
# pocolm's scripts invoke each other by bare name, relying on their python3
# shebangs plus scripts/ being on PATH -- recipes/cy/path.sh adds it. Note the
# python -> python2.7 symlink above does not affect them.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-numpy && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kaldi/tools
RUN git clone --depth 1 https://github.com/danpovey/pocolm.git pocolm && \
    make -C pocolm/src -j $(nproc)

# Setup the user profile (automatically uses host user if built with docker compose)
ARG USERNAME
ARG USER_UID
ARG USER_GID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Add sudo support.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Copy the recipe and set the owner
ARG RECIPE=cy
ADD recipes/$RECIPE /opt/kaldi/egs/$RECIPE
RUN chown -R $USER_UID:$USER_GID /opt/kaldi/egs/$RECIPE

# Change the default shell
RUN chsh -s /usr/bin/fish $USERNAME

# Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# Download the config files
RUN git clone https://gitlab.com/prvInSpace/uniplexed-commafiles ~/.config

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Fix to get rid of Git warning
RUN git config --global --add safe.directory /opt/kaldi && \
    git config --global --add safe.directory /workspaces/vosk

# Set the workdir to the mounted volume
WORKDIR /workspaces/vosk

ENTRYPOINT /usr/bin/fish

