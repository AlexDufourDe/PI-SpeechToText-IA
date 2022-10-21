FROM tensorflow/tensorflow




WORKDIR .
COPY src src/ requirements.txt .

RUN pip install -r requirements.txt

CMD ["python3", "test.py"]

EXPOSE 8888




