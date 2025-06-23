FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Shell form para que ${PORT} se expanda
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port=${PORT:-8000}"]
