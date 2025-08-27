FROM python:3.9-slim

WORKDIR /app

ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY tests/ tests/

RUN mkdir -p artifacts

EXPOSE 8000

CMD ["bash", "-c", "python src/train.py && uvicorn src.app:app --host 0.0.0.0 --port 8000"]
