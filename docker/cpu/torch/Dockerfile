FROM python:3.9.5

COPY ./matchvec /app/matchvec
COPY ./setup.py /app
COPY ./docker/cpu/torch/requirements.txt /app

WORKDIR /app

RUN apt-get update && \
    apt-get install -y protobuf-compiler libprotoc-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip install --no-cache-dir -r requirements.txt

RUN  pip install -e .

CMD [ "python3", "matchvec/app.py" ]
