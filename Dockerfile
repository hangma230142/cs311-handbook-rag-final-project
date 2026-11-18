FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False \
    HOME=/tmp \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY run_rag.py app.py ./

ENTRYPOINT ["python", "run_rag.py"]
