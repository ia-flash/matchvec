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
            python3-pip && \
      rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --upgrade pip

COPY ./requirements.txt /workspace/requirements.txt
# mmcv requires cython==0.29.12 installed
RUN pip3 install cython
RUN pip3 install -r /workspace/requirements.txt

COPY . /workspace/

RUN git clone https://github.com/open-mmlab/mmdetection.git mmdetection
RUN cd /workspace/mmdetection && \
    git reset --hard 6050186056dcf8f50cc8115a22bfeb1fa5b56063

RUN cd /workspace/mmdetection && \
    PYTHON=python3 bash ./compile.sh && \
    pip3 install -e . && \
    cd ..

RUN mkdir /.cache && chmod -R a+rw /.cache/
WORKDIR /app

CMD [ "python3", "matchvec/app.py" ]
