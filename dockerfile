FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    python3.9\
    python3-pip

RUN apt-get install -y alsa-base\
    alsa-utils\
    libportaudio2

COPY src ./src

COPY requirements.txt  ./requirements.txt

RUN pip3 install -r requirements.txt




