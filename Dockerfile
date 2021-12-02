FROM ubuntu:bionic
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=$PYTHONPATH:/pywave \
    PYTHON_UNBUFFERED=1

RUN apt-get update \
&&  apt-get upgrade -y \
&&  apt-get install -y \
      build-essential \
      ffmpeg \
      libavcodec-dev \
      libavformat-dev \
      libgtk-3-dev \
      libjpeg-dev \
      libhdf5-serial-dev \
      libpng-dev \
      libswscale-dev \
      libv4l-dev \
      libx264-dev \
      libxi-dev  \
      libxmu-dev \
      libxvidcore-dev \
      pandoc \
      pkg-config \
      python3-dev \
      python3-tk \
      python-imaging-tk \
      texlive \
      texlive-latex-extra \
      texlive-xetex \
      unzip \
      wget \
&& rm -rf /var/lib/apt/lists/*

# Python packages
WORKDIR /tmp
RUN wget https://bootstrap.pypa.io/get-pip.py \
&&  python3 get-pip.py \
&&  pip install --upgrade pip
RUN pip install \
      jedi==0.17.2 \
      matplotlib \
      mock \
      nose \
      numpy \
      pandas \
      pillow \
      proj \
      pyyaml \
      scikit-image \
      scikit-learn \
      statsmodels
RUN pip install \
      ipython \
      jupyter \
      nc-time-axis \
      netCDF4 \
      shapely --no-binary shapely \
      xgboost \
      celluloid
RUN rm -rf /tmp/*
WORKDIR /root
