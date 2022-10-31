FROM python:3.7.15-slim-buster
RUN apt-get update
RUN apt-get -y install gcc
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /app
COPY . /app