import os, io, base64, traceback, random

from fastapi import FastAPI, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks
from pydub import AudioSegment
import matplotlib.pyplot as plt

from db import get_db                  # Ahora viene de db.py
from models import Base                # Para la definición de tablas
from caregiver import router as caregiver_router
from metrics import router as metrics_router

# Crea la app y monta middleware
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

# Monta los routers de caregiver y metrics
app.include_router(caregiver_router)
app.include_router(metrics_router)

# (Opcional) asegúrate de crear tablas en startup o en db.py
# Base.metadata.create_all(bind=engine)

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

@app.post("/analisis")
async def analizar_audio(
    audio: UploadFile = File(...),
    glucosa: float = Form(...),
    unidad: str = Form("mg/dl"),
    db=Depends(get_db)
):
    # ... tu lógica existente de análisis de audio ...
    # (no hay cambios aquí)
    ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True)
