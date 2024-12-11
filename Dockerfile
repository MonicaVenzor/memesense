FROM python:3.10.6-buster
COPY api_clon.py /api_clon.py
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn api_clon:app --host 0.0.0.0 --port $PORT
