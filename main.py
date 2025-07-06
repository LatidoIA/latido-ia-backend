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

# Crea tablas (también lo hace db.py, pero no está de más)
# Base.metadata.create_all(bind=engine)

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
app.include_router(caregiver_router)

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
    # 1) Guardar raw
    raw_ext = os.path.splitext(audio.filename)[1] or ".3gp"
    raw_tmp = f"temp_raw{raw_ext}"
    data = await audio.read()
    with open(raw_tmp, "wb") as f:
        f.write(data)

    # 2) Convertir a WAV 16kHz mono
    wav_tmp = "temp.wav"
    try:
        sound = AudioSegment.from_file(raw_tmp)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(wav_tmp, format="wav")
    except Exception as e:
        tb = traceback.format_exc()
        os.remove(raw_tmp)
        return {"error": f"Conversión a WAV fallida: {e}", "waveform_png": None}
    finally:
        if os.path.exists(raw_tmp):
            os.remove(raw_tmp)

    try:
        # 3) Cargar WAV
        y, sr = librosa.load(wav_tmp, sr=16000, duration=10.0)
        # 4) Filtrar 20–150 Hz
        nyq = 0.5 * sr
        b, a = butter(2, [20/nyq, 150/nyq], btype="band")
        y = filtfilt(b, a, y)
        # 5) Extraer features y predecir
        mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        chroma   = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
        feat = np.hstack([mfcc, chroma, contrast]).reshape(1, -1)
        pred = int(app.state.modelo.predict(feat)[0])
        anomaly_type = LABELS.get(pred, "Desconocido")

        # 6) Cálculo de BPM
        env = np.abs(y)
        window = int(sr * 0.05)
        env_smooth = np.convolve(env, np.ones(window)/window, mode='same')
        peaks, _ = find_peaks(env_smooth,
                              distance=sr*0.4,
                              height=np.mean(env_smooth)*1.2)
        duration_sec = len(y) / sr
        bpm = float(np.round(len(peaks)/duration_sec*60, 1))
        beat_times = (peaks/sr).tolist()

        # 7) Reglas de consistencia
        inconsistente = False
        if pred == 1 and 60 <= bpm <= 100:
            pred = 0; anomaly_type = LABELS[0]; inconsistente = True
        elif pred == 2 and bpm <= 100:
            pred = 0; anomaly_type = LABELS[0]; inconsistente = True

        # 8) Mensajes
        if pred == 0:
            mensaje      = "Sin riesgo detectado"
            accion       = "Continúa tu rutina"
            encouragement = random.choice(NORMAL_MESSAGES)
        else:
            mensaje      = "Riesgo detectado"
            accion       = "Visita un médico para confirmar"
            encouragement = ""

        # 9) Gráfico
        times = np.arange(len(y)) / sr
        plt.figure(figsize=(8,3))
        plt.plot(times, y, linewidth=0.5)
        if beat_times:
            plt.vlines(beat_times, y.min(), y.max(), color='r', linewidth=1)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        waveform_png = base64.b64encode(buf.read()).decode('ascii')

        # 10) Respuesta
        return JSONResponse({
            "resultado": pred,
            "anomaly_type": anomaly_type,
            "mensaje": mensaje,
            "accion": accion,
            "bpm": bpm,
            "inconsistente": inconsistente,
            "encouragement": encouragement,
            "waveform_png": waveform_png,
            "error": ""
        })

    except Exception as e:
        tb = traceback.format_exc()
        return {"error": tb, "waveform_png": None}
    finally:
        if os.path.exists(wav_tmp):
            os.remove(wav_tmp)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT",8080)), reload=True)
