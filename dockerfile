FROM tensorflow/tensorflow



COPY . /tf
WORKDIR /tf

RUN pip install -r requirement.txt

EXPOSE 8888


