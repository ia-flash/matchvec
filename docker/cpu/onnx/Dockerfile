FROM python:3.7

COPY ./matchvec /app/matchvec
COPY ./setup.py /app

WORKDIR /app

RUN apt-get update && \
    apt-get install -y protobuf-compiler libprotoc-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir Flask==1.0.2 numpy==1.16.2 pandas==0.24.2 opencv-python-headless==4.0.0.21 requests==2.21.0 Sphinx==2.0.1 sphinx-autodoc-typehints==1.6.0 flask-restplus==0.12.1 sphinx-rtd-theme==0.4.3 onnx==1.5.0 onnxruntime==0.4.0 Pillow==6.1.0 celery==4.3.0 redis==3.3.6

RUN  pip install -e .

CMD [ "python3", "matchvec/app.py" ]
