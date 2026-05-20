FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-dev

COPY app ./app
COPY model/artifacts/mlflow_model_top30_optimized ./model/artifacts/mlflow_model_top30_optimized
COPY model/schema/top30_feature_schema.json ./model/schema/top30_feature_schema.json
COPY model/schema/top30_model_metadata.json ./model/schema/top30_model_metadata.json

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
