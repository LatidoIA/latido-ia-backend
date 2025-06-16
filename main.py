from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import librosa
import numpy as np
import pickle
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga del modelo
import joblib
modelo = joblib.load("modelo_latido_rf_final.pkl")


@app.get("/")
def root():
    return {"mensaje": "API Latido IA activa"}

@app.post("/analisis")
async def analizar_audio(audio: UploadFile = File(...), glucosa: float = Form(...)):
    try:
        # Guardar archivo con la extensión original
        extension = audio.filename.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
            tmp.write(await audio.read())
            audio_path = tmp.name

        # Cargar audio
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Mezclar con glucosa
        entrada = np.append(mfcc_mean, glucosa).reshape(1, -1)
        pred = modelo.predict(entrada)[0]

        return {
            "resultado": int(pred),
            "mensaje": "Todo bien" if pred == 2 else "Riesgo detectado",
            "accion": "Sigue con tu rutina" if pred == 2 else "Recomendamos visitar un médico"
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
