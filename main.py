# main.py
from fastapi import FastAPI, Depends, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import joblib, os, io, base64, traceback, random
import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks
from pydub import AudioSegment
import matplotlib.pyplot as plt

# ---- BASE DE DATOS ----
SQLALCHEMY_DATABASE_URL = "sqlite:///./latido.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---- IMPORTA TUS MODELOS Y ROUTERS ----
# (Crea después un archivo `models.py` y otro `caregiver.py`)
import models
from caregiver import router as caregiver_router

# ---- INICIALIZA FASTAPI ----
app = FastAPI(
    title="Latido IA API",
    version="1.0.0",
    description="Procesamiento de audio de latidos + gestión de cuidadores"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crea tablas
Base.metadata.create_all(bind=engine)

# Monta tu router de cuidadores
app.include_router(caregiver_router, prefix="/caregiver", tags=["caregiver"])

# ---- CARGA MODELO EN ARRANQUE ----
LABELS = {0: "Normal", 1: "Bradicardia", 2: "Taquicardia"}
NORMAL_MESSAGES = [
    "Tu corazón es fuerte y saludable ❤️",
    "No se detectaron anomalías, ¡bien hecho!",
    "Latidos estables: tu corazón trabaja perfectamente",
    "¡Genial! Tu corazón late con normalidad"
]

@app.on_event("startup")
async def load_model():
    modelo = joblib.load("modelo_xgb_mfcc.pkl")
    app.state.modelo = modelo
    print("✅ Modelo cargado")

# ---- ENDPOINT DE ANÁLISIS DE AUDIO ----
@app.post("/analisis")
async def analizar_audio(
    audio: UploadFile = File(...),
    glucosa: float = Form(...),
    unidad: str = Form("mg/dl")
):
    # … (aquí copia íntegro tu código de conversión, predicción y generación de PNG)
    # Al final, devuelve JSONResponse({...})
    pass  # <-- reemplaza este pass con tu lógica actual

# ---- RUN ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
