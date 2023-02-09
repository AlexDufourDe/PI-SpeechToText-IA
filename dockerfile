FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    python3.9\
    python3-pip

COPY src ./src
COPY requirements.txt  ./requirements.txt

RUN pip3 install -r requirements.txt

WORKDIR ./src/phase2




