FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel
RUN apt-get update &&\
    apt-get install -y python-opencv &&\
    rm -rf /var/lib/apt/lists/*

RUN pip install mmcv
WORKDIR /usr/src
RUN git clone https://github.com/open-mmlab/mmdetection.git
WORKDIR mmdetection
RUN conda install cython -y
RUN PYTHON=python3 ./compile.sh
RUN python3 setup.py install


COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python3", "matchvec/app.py" ]
