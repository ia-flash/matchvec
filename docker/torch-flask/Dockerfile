FROM mmdetection
RUN conda config --set auto_update_conda false


COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt
