FROM python:3.11-slim
WORKDIR /app
COPY . /app
ENV PYTHONPATH=/app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]"8080"]
