FROM python:3.10.6-buster
COPY api.py /api.py
COPY requirements.txt /requirements.txt
RUN pip install .
CMD uvicorn api:app --host 0.0.0.0 --port 8000
