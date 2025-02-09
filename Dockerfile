FROM python:3.10.6-buster

COPY moviepicker /moviepicker
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn moviepicker.api.fast:app --host 0.0.0.0 --port $PORT
