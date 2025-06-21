from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
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
async def analizar_audio(
    audio: UploadFile = File(...),
    glucosa: float = Form(...)
):
    import numpy as np
    import librosa
    from scipy.signal import butter, filtfilt
    from pydub import AudioSegment
    import traceback

    # 1) Guardar raw (3gp/…) en disco
    raw_ext = os.path.splitext(audio.filename)[1] or ".3gp"
    raw_tmp = f"temp_raw{raw_ext}"
    data = await audio.read()
    with open(raw_tmp, "wb") as f:
        f.write(data)

    # 2) Convertir a WAV mono 16 kHz
    wav_tmp = "temp.wav"
    try:
        sound = AudioSegment.from_file(raw_tmp)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(wav_tmp, format="wav")
    except Exception as e:
        tb = traceback.format_exc()
        print("❌ Error convirtiendo a WAV:\n", tb)
        return {"error": f"Conversión a WAV fallida: {e}"}
    finally:
        if os.path.exists(raw_tmp):
            os.remove(raw_tmp)

    try:
        # 3) Cargar WAV válido
        y, sr = librosa.load(wav_tmp, sr=16000, duration=10.0)

        # 4) Filtro pasa-banda 20–150 Hz
        nyq = 0.5 * sr
        b, a = butter(2, [20/nyq, 150/nyq], btype="band")
        y = filtfilt(b, a, y)

        # 5) Features y predicción
        mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        chroma   = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
        feat = np.hstack([mfcc, chroma, contrast]).reshape(1, -1)
        pred = app.state.modelo.predict(feat)[0]

        resultado = int(pred)
        mensaje   = "Todo bien" if pred == 2 else "Riesgo detectado"
        accion    = "Sigue con tu rutina" if pred == 2 else "Recomendamos visitar un médico"

        # 6) BPM
        try:
            tempos = librosa.beat.tempo(y=y, sr=sr)
            bpm = float(np.round(tempos[0], 1))
        except Exception:
            bpm = None

        return {
            "resultado": resultado,
            "mensaje": mensaje,
            "accion": accion,
            "bpm": bpm,
            "error": ""
        }

    except Exception as e:
        tb = traceback.format_exc()
        print("❌ Error interno en /analisis:\n", tb)
        return {"error": tb}

    finally:
        if os.path.exists(wav_tmp):
            os.remove(wav_tmp)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

