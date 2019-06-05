#FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel
FROM mmdetection

#RUN apt-get update &&\
#    apt-get install -y python-opencv &&\
#    rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python3", "matchvec/app.py" ]
