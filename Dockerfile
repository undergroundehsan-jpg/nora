# NORA backend. Python 3.11 keeps the stdlib `audioop` module that the Twilio
# bridge uses for mu-law conversion (it was removed in 3.13).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ffmpeg is not required; torch needs libgomp for its CPU kernels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --upgrade pip && pip install -r requirements-deploy.txt

# Application code, the pre-recorded audio and the customer record.
COPY app/ ./app/
COPY audio_cache/ ./audio_cache/
COPY data/ ./data/

# Written at runtime; mount a Railway volume here to keep call history
# across deploys.
RUN mkdir -p /app/logs

EXPOSE 8080

# One worker only: sessions live in process memory, so the Twilio media
# stream and the REST call that started it must land on the same worker.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --ws websockets"]
