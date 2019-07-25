FROM python:3.7

COPY . /app
WORKDIR /app

RUN apt-get update
RUN apt-get install -y protobuf-compiler libprotoc-dev
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python3", "matchvec/app.py" ]
