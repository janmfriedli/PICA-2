FROM python:3.8.12-buster

WORKDIR /src

COPY pica2 /pica2
COPY requirements.txt /requirements.txt
COPY setup.py setup.py


RUN pip install --upgrade pip
RUN pip install .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD uvicorn pica2.api.fast:app --host 0.0.0.0 --port $PORT
