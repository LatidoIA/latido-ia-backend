# Usa Python “slim” como base
FROM python:3.10-slim

# 1) Instala FFmpeg del sistema
RUN apt-get update && apt-get install -y ffmpeg

# 2) Crea el directorio de trabajo
WORKDIR /app

# 3) Copia requirements y las instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copia el resto de tu código
COPY . .

# 5) Expone el puerto donde tu FastAPI escucha
EXPOSE 8080

# 6) Comando por defecto para arrancar tu API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
