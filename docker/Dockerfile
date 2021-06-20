# pytorch
ARG PYTORCH_TAG
FROM pytorch/pytorch:${PYTORCH_TAG:-latest}

# apex
ARG CATALYST_APEX
# extras
ARG CATALYST_DEV
ARG CATALYST_CV
ARG CATALYST_ML
ARG CATALYST_HYDRA
ARG CATALYST_OPTUNA
ARG CATALYST_ONNX
ARG CATALYST_ONNX_GPU

#RUN apt-get update && apt-get install -y software-properties-common \
#    && rm -rf /var/lib/apt/lists/* \
#    && add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" \
#    && apt-get update && apt-get install -y \
#         build-essential \
#         libsm6 \
#         libxext6 \
#         libfontconfig1 \
#         libxrender1 \
#         libswscale-dev \
#         libtbb2 \
#         libtbb-dev \
#         libjpeg-dev \
#         libpng-dev \
#         libtiff-dev \
#         libjasper-dev \
#         libavformat-dev \
#         libpq-dev \
#         libturbojpeg \
#        git \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN if [ "$CATALYST_APEX" = true ] ; then git clone https://github.com/NVIDIA/apex && cd apex &&  python setup.py install ; else echo "Apex is not required" ; fi

CMD mkdir -p /workspace
RUN pip install -U catalyst --no-cache-dir
RUN if [ "$CATALYST_DEV" = "1" ] ; then pip install -U catalyst[dev] --no-cache-dir ; else echo "catalyst[dev] is not required" ; fi
RUN if [ "$CATALYST_CV" = "1" ] ; then pip install -U catalyst[cv] --no-cache-dir ; else echo "catalyst[cv] is not required" ; fi
RUN if [ "$CATALYST_ML" = "1" ] ; then pip install -U catalyst[ml] --no-cache-dir ; else echo "catalyst[ml] is not required" ; fi
RUN if [ "$CATALYST_HYDRA" = "1" ] ; then pip install -U catalyst[hydra] --no-cache-dir ; else echo "catalyst[hydra] is not required" ; fi
RUN if [ "$CATALYST_OPTUNA" = "1" ] ; then pip install -U catalyst[optuna] --no-cache-dir ; else echo "catalyst[optuna] is not required" ; fi
RUN if [ "$CATALYST_ONNX" = "1" ] ; then pip install -U catalyst[onnx] --no-cache-dir ; else echo "catalyst[onnx] is not required" ; fi
RUN if [ "$CATALYST_ONNX_GPU" = "1" ] ; then pip install -U catalyst[onnx-gpu] --no-cache-dir ; else echo "catalyst[onnx-gpu] is not required" ; fi

WORKDIR /workspace
