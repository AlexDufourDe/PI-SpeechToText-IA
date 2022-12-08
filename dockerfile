FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    python3\
    python3-pip

COPY src ./
COPY requirements.txt  ./requirements.txt

RUN pip install -r requirements.txt

WORKDIR src/phase1

CMD [ "python", "./modele.py"]

EXPOSE 8888




