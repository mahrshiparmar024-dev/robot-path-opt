FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib (needed on slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Set matplotlib to non-interactive backend (no display needed on server)
ENV MPLBACKEND=Agg

EXPOSE 10000
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --timeout 120 --workers 1
