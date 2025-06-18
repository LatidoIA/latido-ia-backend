from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import librosa
import joblib
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_model():
    # Verifica qué archivos hay en la raíz
    print("FILES AT STARTUP:", os.listdir("."))
    # Carga tu modelo XGBoost
    global modelo
    modelo = joblib.load("modelo_xgb_mfcc.pkl")
    print("✅ Modelo cargado en startup")

@app.get("/")
def root():
    return {"mensaje": "API Latido IA activa con XGBoost"}

@app.post("/analisis")
async def analizar_audio(audio: UploadFile = File(...), glucosa: float = Form(...)):
    try:
        # Guardar temporal
        data = await audio.read()
        with open("temp.wav", "wb") as f:
            f.write(data)

        # Cargar y extraer features
        y, sr = librosa.load("temp.wav", sr=16000, duration=5.0)
        mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        chroma   = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)

        feat = np.hstack([mfcc, chroma, contrast]).reshape(1, -1)
        pred = modelo.predict(feat)[0]

        return {
            "resultado": int(pred),
            "mensaje": "Todo bien" if pred == 2 else "Riesgo detectado",
            "accion": "Sigue con tu rutina" if pred == 2 else "Recomendamos visitar un médico"
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
