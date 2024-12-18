FROM python:3.10.6-buster
WORKDIR /memesense
COPY memesense memesense
COPY src src
#COPY setup.py setup.py
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install -r requirements.txt
CMD uvicorn src.api:app --host 0.0.0.0 --port $PORT
