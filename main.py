from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib, os, io, base64, traceback, random

# Etiquetas de tu modelo
LABELS = {
    0: "Normal",
    1: "Bradicardia",
    2: "Taquicardia",
}

# Mensajes de ánimo para ritmo normal
NORMAL_MESSAGES = [
    "Tu corazón es fuerte y saludable ❤️",
    "No se detectaron anomalías, ¡bien hecho!",
    "Ritmo cardíaco dentro de lo esperado. Sigue así 💪",
    "Latidos estables: tu corazón trabaja perfectamente",
    "¡Genial! Tu corazón late con normalidad"
]

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
    modelo = joblib.load("modelo_xgb_mfcc.pkl")
    app.state.modelo = modelo
    print("✅ Modelo cargado en startup")

@app.post("/analisis")
async def analizar_audio(
    audio: UploadFile = File(...),
    glucosa: float = Form(...)
):
    import numpy as np, librosa
    from scipy.signal import butter, filtfilt
    from pydub import AudioSegment
    import matplotlib.pyplot as plt

    # 1) Guardar raw upload (.3gp u otro) en disco
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
        return {"error": f"Conversión a WAV fallida: {e}", "waveform_png": None}
    finally:
        os.remove(raw_tmp)

    try:
        # 3) Cargar WAV válido
        y, sr = librosa.load(wav_tmp, sr=16000, duration=10.0)

        # 4) Filtrar señal 20–150 Hz
        nyq = 0.5 * sr
        b, a = butter(2, [20/nyq, 150/nyq], btype="band")
        y = filtfilt(b, a, y)

        # 5) Features y predicción
        mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        chroma   = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
        feat = np.hstack([mfcc, chroma, contrast]).reshape(1, -1)

        pred = int(app.state.modelo.predict(feat)[0])
        anomaly_type = LABELS.get(pred, "Desconocido")

        # 6) Calcular BPM
        try:
            tempos, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(np.round(tempos, 1))
        except Exception:
            bpm = None
            beat_frames = []

        # 7) Mensajes según clase
        if pred == 0:
            mensaje = "Sin riesgo detectado"
            accion = "Continúa tu rutina"
            encouragement = random.choice(NORMAL_MESSAGES)
        else:
            mensaje = "Riesgo detectado"
            accion = "Visita un médico para confirmar"
            encouragement = ""

        # 8) Generar gráfico de forma de onda + beats
        times = np.arange(len(y)) / sr
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        plt.figure(figsize=(8, 3))
        plt.plot(times, y, linewidth=0.5)
        plt.vlines(beat_times, y.min(), y.max(), color='r', linewidth=1)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        waveform_png = base64.b64encode(buf.read()).decode('ascii')

        # 9) Responder JSON con imagen embedded
        return JSONResponse({
            "resultado": pred,
            "anomaly_type": anomaly_type,
            "mensaje": mensaje,
            "accion": accion,
            "bpm": bpm,
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
    uvicorn.run(app, host="0.0.0.0", port=8080)

