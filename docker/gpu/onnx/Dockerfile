FROM nvidia/cuda:10.0-base

ENV LANG=en_US.UTF-8

RUN apt-get update && \
      apt-get install -y apt-utils && \
      apt-get install -y \
            wget \
            curl \
            libgomp1 \
            python3 \
            python-opencv \
            protobuf-compiler \
            libprotoc-dev \
            libmediainfo0v5 \
            python3-pip && \
      rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python3", "matchvec/app.py" ]
