FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

MAINTAINER Samir Jabari  #feel free to change that ;-)

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    freeglut3-dev \
    libcupti-dev \
    libcurl3-dev \
    libfreetype6-dev \
    libpng12-dev \
    libzmq3-dev \
    pkg-config \
    python-dev \
    python \
    unzip \
    python3 \
    python-pip \
    python3-pip \
    python3-setuptools \
    rsync \
    software-properties-common \
    inkscape \
    jed \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    pandoc \
    vim \
    libpng-dev \
    g++ \
    gfortran \
    libffi-dev \
    libhdf5-dev \
    libjpeg-dev \
    liblcms2-dev \
    libopenblas-dev \
    liblapack-dev \
    libssl-dev \
    libtiff5-dev \
    libwebp-dev \
    nano \
    libopenslide-dev \
    wget \
    zlib1g-dev \
    qt5-default \
    libvtk6-dev \
    libjasper-dev \
    libopenexr-dev \
    libgdal-dev \
    libdc1394-22-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libx264-dev \
    yasm \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libv4l-dev \
    libxine2-dev \
    libtbb-dev \
    libeigen3-dev \
    python-tk \
    python3-dev \
    python3-tk \
    ant \
    default-jdk \
    doxygen \
      && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    bc \
    cmake \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
    python-numpy \
    python-scipy \
    python-nose \
    python-h5py \
    python-skimage \
    python-matplotlib \
    python-pandas \
    python-sklearn \
    python-sympy \
    python3-numpy \
    python3-scipy \
    python3-nose \
    python3-h5py \
    python3-skimage \
    python3-matplotlib \
    python3-pandas \
    python3-sklearn \
    python3-sympy \
    && \
  apt-get clean && \
  apt-get autoremove && \
rm -rf /var/lib/apt/lists/*


RUN python -m pip install --upgrade pip
RUN pip3 install --upgrade pip

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	 python get-pip.py && \
	  rm get-pip.py
RUN python2 -m pip install ipykernel
RUN pip --no-cache-dir install  \
    Pillow \
    h5py \
    #ipykernel \
    jupyter \
    matplotlib==2.0.2 \
    seaborn \
    numpy \
    pandas \
    scipy \
    sklearn \
    bcolz \
    Cython \
    path.py \
    six \
    sphinx \
    wheel \
    zmq \
    pygments \
    Flask \
    statsmodels \
    && \
python2 -m ipykernel.kernelspec


RUN pip3 --no-cache-dir install --upgrade ipython \
    Pillow \
    h5py \
    ipykernel \
    jupyter \
    matplotlib==2.0.2 \
    seaborn \
    numpy \
    pandas \
    scipy \
    sklearn \
    bcolz \
    Cython \
    path.py \
    six \
    sphinx \
    wheel \
    zmq \
    pygments \
    Flask \
    statsmodels \
    && \
python3 -m ipykernel.kernelspec


# Install OpenSlide
RUN apt-get update && apt-get install -y openslide-tools
RUN apt-get update && apt-get install -y python3-openslide ##&&\
    #apt-get clean && \
    #rm -rf /var/lib/apt/lists/*


# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# These lines will be edited automatically by parameterized_docker_build.sh. #
# COPY _PIP_FILE_ /
# RUN pip --no-cache-dir install /_PIP_FILE_
# RUN rm -f /_PIP_FILE_

# Install TensorFlow Python2 GPU version from central repo
RUN pip --no-cache-dir install --upgrade tensorflow-gpu

# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #



# Install TensorFlow Python3 GPU version from central repo
RUN pip3 --no-cache-dir install --upgrade tensorflow-gpu
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #

# RUN ln -s /usr/bin/python3 /usr/bin/python#

# Installing Keras
RUN pip  --no-cache-dir install git+git://github.com/fchollet/keras.git@2.1.2
RUN pip3 --no-cache-dir install git+git://github.com/fchollet/keras.git@2.1.2

#Installing Pytorch
RUN python2 -m pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl &&\
    python2 -m pip install torchvision

RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl &&\
    pip3 install torchvision

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=$CUDA_HOME
ENV PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

# Install Libvips openslide
RUN apt update && apt install -y libvips libvips-dev libvips-tools libopenslide-dev

# Install OpenCV
ENV OPENCV_VERSION 3.4.0
RUN	git clone https://github.com/Itseez/opencv.git  /opt/opencv &&\
	git clone https://github.com/Itseez/opencv_contrib.git  /opt/opencv_contrib && \
	cd /opt/opencv && \
	git checkout ${OPENCV_VERSION} && \
	cd /opt/opencv_contrib && \
	git checkout ${OPENCV_VERSION}

RUN mkdir -p /opt/opencv/build && \
	cd /opt/opencv/build && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_INSTALL_PREFIX=/usr/local \
		# -D INSTALL_C_EXAMPLES=ON \
		# -D INSTALL_PYTHON_EXAMPLES=ON \
		-D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
		# -D BUILD_EXAMPLES=ON \
		.. && \
	make -j${CPUCOUNT} && \
	make install && \
	ldconfig

RUN pip install --pre jupyter-tensorboard
RUN pip3 install --pre jupyter-tensorboard


###Jupyter
# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Copy sample notebooks.
# This is my directory I set up as a workdir change this accordingly....
COPY notebooks /notebooks

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888
# Openslide Server
EXPOSE 5000

WORKDIR "/notebooks"

RUN chmod +x /run_jupyter.sh

CMD ["/run_jupyter.sh", "--allow-root"]

cleanup
RUN 	apt-get autoremove -y && \
apt-get autoclean && \
apt-get clean && \
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
