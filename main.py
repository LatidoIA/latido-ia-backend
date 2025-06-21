from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import joblib, os

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
    modelo_path = "modelo_xgb_mfcc.pkl"
    print("FILES AT STARTUP:", os.listdir("."))
    modelo = joblib.load(modelo_path)
    app.state.modelo = modelo
    print("✅ Modelo cargado en startup")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"mensaje": "API Latido IA activa con XGBoost"}

@app.post("/analisis")
async def analizar_audio(audio: UploadFile = File(...), glucosa: float = Form(...)):
    import numpy as np, librosa
    data = await audio.read()
    tmp = "temp.wav"
    with open(tmp, "wb") as f: f.write(data)
    try:
        y, sr = librosa.load(tmp, sr=16000, duration=5.0)
        mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        chroma   = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
        feat = np.hstack([mfcc, chroma, contrast]).reshape(1, -1)
        pred = app.state.modelo.predict(feat)[0]

        resultado = int(pred)
        mensaje   = "Todo bien" if pred == 2 else "Riesgo detectado"
        accion    = "Sigue con tu rutina" if pred == 2 else "Recomendamos visitar un médico"

        return {
            "resultado": resultado,
            "mensaje": mensaje,
            "accion": accion,
            "error": ""
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp): os.remove(tmp)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
