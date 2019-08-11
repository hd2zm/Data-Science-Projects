FROM python:3.6-slim
COPY main.py /deploy/
COPY templates /deploy/templates/
COPY requirements.txt /deploy/
COPY finalized_model.p /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "main.py"]