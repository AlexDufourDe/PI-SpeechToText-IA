FROM tensorflow/tensorflow




COPY src ./src
COPY requirements.txt  ./requirements.txt

RUN pip3 install -r requirements.txt

WORKDIR ./src/phase2
