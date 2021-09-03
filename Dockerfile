FROM ubuntu:bionic AS base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=$PYTHONPATH:/ml-utils \
    PYTHON_UNBUFFERED=1

RUN apt-get update \
&&  apt-get upgrade -y \
&&  apt-get install -y \
      build-essential \
      cmake \
      g++-6 \
      gcc-6 \
      gfortran \
      graphviz \
      libatlas-base-dev \
      libavcodec-dev \
      libavformat-dev \
      libgeos-dev \
      libglu1-mesa  \
      libglu1-mesa-dev \
      libgtk-3-dev \
      libjpeg-dev \
      libhdf5-serial-dev \
      liblapack-dev \
      libopenblas-dev \
      libpng-dev \
      libproj-dev \
      libswscale-dev \
      libtiff-dev \
      libv4l-dev \
      libx264-dev \
      libxi-dev  \
      libxmu-dev \
      libxvidcore-dev \
      pkg-config \
      python3-dev \
      python3-tk \
      python-imaging-tk \
      unzip \
      wget \
&& rm -rf /var/lib/apt/lists/*

# Python packages
FROM base AS python
WORKDIR /tmp
RUN wget https://bootstrap.pypa.io/get-pip.py \
&&  python3 get-pip.py \
&&  pip install --upgrade pip
RUN pip install \
      beautifulsoup4 \
      cirq \
      graphviz \
      imutils \
      jedi==0.17.2 \
      matplotlib \
      mock \
      nose \
      numpy \
      opencv-contrib-python \
      pandas \
      pillow \
      progressbar2 \
      proj \
      pydotplus \
      pyproj \
      pyyaml \
      scikit-image \
      scikit-learn \
      seaborn \
      shapely \
      statsmodels \
      torch==1.8 \
      torchaudio \
      torchvision \
      qiskit
RUN pip install \
      cartopy \
      cmocean \
      fastai2 \
      ipython \
      jupyter \
      nc-time-axis \
      netCDF4 \
      pytorch-forecasting \
      shapely --no-binary shapely \
      sktime \
      sktime-dl \
      xgboost
#RUN rm -rf /tmp/*
WORKDIR /root
