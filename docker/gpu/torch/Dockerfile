FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

# Set up locale to prevent bugs with encoding
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && \
      apt-get install -y apt-utils && \
      apt-get install -y \
            wget \
            curl \
            libsm6 \
            libxext6 \
            libxrender-dev \
            python3 \
            python-opencv \
            libmediainfo0v5 \
            python3-pip && \
      rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --upgrade pip

COPY ./matchvec /app/matchvec
COPY ./setup.py /app
COPY ./docker/gpu/torch/requirements.txt /workspace


# mmcv requires cython==0.29.12 installed
RUN pip3 install cython
RUN pip3 install -r /workspace/requirements.txt

RUN git clone --single-branch --branch dev https://github.com/ia-flash/mmdetection.git mmdetection
RUN cd /workspace/mmdetection && \
    PYTHON=python3 bash ./compile.sh && \
    pip3 install -e . && \
    cd ..

RUN mkdir /.cache && chmod -R a+rw /.cache/
WORKDIR /app

RUN pip3 install -e .

CMD [ "python3", "matchvec/app.py" ]
