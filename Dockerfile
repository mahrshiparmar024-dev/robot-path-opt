FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --timeout 120
