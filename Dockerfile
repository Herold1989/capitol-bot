FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt requirements-dev.txt ./
COPY bot ./bot
COPY tests ./tests
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-dev.txt

COPY config ./config
COPY README.md ./

CMD ["python", "-m", "bot.cli", "--config", "config/strategy.yaml", "paper-run"]
