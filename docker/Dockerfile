FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
	libxext6 \
	libfontconfig1 \
	libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libavformat-dev \
    libpq-dev \
	libturbojpeg \
	software-properties-common \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD mkdir -p /workspace
RUN pip install -U catalyst --no-cache-dir

WORKDIR /workspace
