FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY app.py data_pipeline.py ./
COPY historical_costs.csv* ./

RUN pip install --upgrade pip \
    && pip install \
        streamlit \
        pandas \
        numpy \
        plotly \
        python-dotenv \
        pymongo \
        google-generativeai

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
