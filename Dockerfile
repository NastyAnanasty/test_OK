FROM python:3.9.10

ADD main.py .

RUN pip install pyarrow pandas

COPY . .

CMD ["python", "main.py", "alg2", "--host=0.0.0.0"]