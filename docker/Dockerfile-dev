FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

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

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt --no-cache-dir

COPY requirements-dev.txt /workspace/requirements-dev.txt
RUN pip install -r /workspace/requirements-dev.txt --no-cache-dir

WORKDIR /workspace
