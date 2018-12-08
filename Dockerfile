# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu16.04
MAINTAINER Samir Jabari <samir.jabari@fau.de>
USER root
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python-software-properties
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.6
RUN apt-get update
RUN apt-get install curl
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6

RUN python3.6 -m pip install --upgrade pip

RUN ln -sf /usr/bin/python3.6 /usr/local/bin/python3 &&\
    ln -sf /usr/local/bin/pip /usr/local/bin/pip3

    # Install all OS dependencies for fully functional notebook server
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
        unzip \
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
        ant \
        default-jdk \
        doxygen \
        libvips \
        libvips-dev \
        libvips-tools \
          && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    bc \
    cmake \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

    RUN apt-get update && apt-get install -y \
        python3-dev \
        python3-tk \
        python3-setuptools \
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


    # libav-tools for matplotlib anim
    RUN apt-get update && \
        apt-get install -y --no-install-recommends libav-tools && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*


    RUN cd /tmp && \
        wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
        ##echo "e1045ee415162f944b6aebfe560b8fee *Miniconda3-latest-Linux-x86_64.sh" | md5sum -c - && \
        /bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b
    ENV PATH=/root/miniconda3/bin:$PATH

    RUN conda install -y \
        'python=3.6'

    RUN conda install --yes\
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        seaborn \
        numpy \
        pandas \
        scipy \
        bcolz \
        Cython \
        path.py \
        six \
        sphinx \
        wheel \
        pygments \
        Flask \
        scikit-image \
        pandas \
        tqdm \
        statsmodels \
        #pydicom \
        #tensorboardX \
        tensorflow-tensorboard \
        plotly \
        seaborn \
        ipython
        

    RUN conda install -y -c conda-forge \
        pydicom \
        tensorboardx


    RUN conda install -y \
        scikit-learn

    RUN python3.6 -m pip install --upgrade --force-reinstall --no-cache-dir --ignore-installed  \
        zmq \
        imgaug \
        kaggle \
        pretrainedmodels \
        skorch


        # Installing OpenSlide
        RUN apt-get update && apt-get install -y openslide-tools
        RUN apt-get update && apt-get install -y python3-openslide &&\
            apt-get clean && \
            rm -rf /var/lib/apt/lists/*

        # Installing OpenCV
        RUN conda install -c conda-forge opencv
           

            ENV CUDA_HOME /usr/local/cuda-9.2
            ENV LD_LIBRARY_PATH /usr/local/cuda-9.2:/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH
            ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
            ENV LIBRARY_PATH $LD_LIBRARY_PATH

            # Install facets which does not have a pip or conda package at the moment
            RUN cd /tmp && \
                git clone https://github.com/PAIR-code/facets.git && \
                cd facets && \
                jupyter nbextension install facets-dist/ --sys-prefix && \
                rm -rf facets
                


            RUN conda install -y \
                tensorboard

            # Installig Tensorflow/GPU
            RUN conda install -c aaronzs tensorflow
            

            RUN conda install -c aaronzs tensorflow-gpu
           

            # Installing Keras
            RUN conda install -c anaconda keras
            
            # Installing pytorch
            RUN conda install pytorch torchvision cuda92 -c pytorch
                
            RUN apt-get update && apt-get install libevent-dev -y

            RUN python3.6 -m pip install --upgrade pip

            RUN python3.6 -m pip install \
                setuptools \
                numpy

            RUN apt-get install -y \
                libmemcached-dev \
                zlib1g-dev

            RUN python3.6 -m pip install pylibmc

            RUN apt-get update && apt-get install build-essential

            RUN python3.6 -m pip install --user numpy==1.14.5 && \
                cd /tmp/ &&\
                git clone https://github.com/girder/large_image.git && \
                cd large_image/ && \
                python3.6 -m pip install -e .[openslide,memcached]

            RUN python3.6 -m pip install \
                scikit-build \
                cmake

            RUN cd /tmp/ &&\
                git clone https://github.com/DigitalSlideArchive/HistomicsTK.git && \
                cd HistomicsTK/ && \
                python3.6 -m pip install -e .


            RUN conda install --yes \
                'matplotlib' \
                'jupyter' \
                'numpy' \
                'scikit-image'
               

            # Import matplotlib the first time to build the font cache.
            ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
            RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot"
           

            RUN python3.6 -m pip install --upgrade --force-reinstall --no-cache-dir --ignore-installed  --pre \
                jupyter-tensorboard
                

          RUN conda install -c menpo opencv3
          RUN conda install -c glemaitre imbalanced-learn
          RUN conda install statsmodels
          RUN python3.6 -m pip install python-bioformats





RUN conda install -c soft-matter slicerator
RUN conda install -c conda-forge jpype1


RUN conda install -c conda-forge pims


###Jupyter
# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Copy sample notebooks.
#COPY notebooks /notebooks

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

  # now clean up the unwanted source to keep image size to a minimum
  RUN cd /opt && \
  	rm -rf /opt/opencv && \
  	rm -rf /opt/opencv_contrib && \
  	apt-get purge -y cmake && \
  apt-get autoremove -y --purge
