FROM tensorflow/tensorflow:2.10.0
WORKDIR /memesense
COPY memesense memesense
COPY src src
#COPY setup.py setup.py
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn src.api:app --host 0.0.0.0 --port 8000
