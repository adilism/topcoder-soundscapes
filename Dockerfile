FROM nvidia/cuda:10.2-runtime-ubuntu16.04

RUN apt-get update && \
    apt-get install -y curl build-essential libpng12-dev libffi-dev \
      	libboost-all-dev \
		libgflags-dev \
		libgoogle-glog-dev \
		libhdf5-serial-dev \
		libleveldb-dev \
		liblmdb-dev \
		libopencv-dev \
		libprotobuf-dev \
		libsnappy-dev \
		libsndfile1 \
		protobuf-compiler \
		python3-rtree \
		git \
        wget \
        bzip2 \
        unzip \
        ca-certificates \
		 && \
    apt-get clean && \
    rm -rf /var/tmp /tmp /var/lib/apt/lists/*

RUN curl -sSL -o installer.sh https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh && \
    /bin/bash /installer.sh -b -f && \
    rm /installer.sh

ENV PATH "/root/anaconda3/bin:$PATH"

RUN pip install tqdm==4.48.2
RUN pip install torch torchvision
RUN pip install numpy==1.18.5
RUN pip install pandas==1.0.5 --ignore-installed certifi
RUN pip install matplotlib==3.3.1
RUN pip install scikit_learn==0.23.2
RUN pip install pytorch_lightning==1.0.5 --ignore-installed PyYAML
RUN pip install timm==0.3.1
RUN pip install librosa==0.8.0
RUN pip install omegaconf==2.0.5
RUN pip install hydra-core --upgrade

WORKDIR /work

COPY . /work/

RUN chmod 777 train.sh
RUN chmod 777 test.sh