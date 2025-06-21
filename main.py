from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import joblib, os

app = FastAPI()
# ... middleware y startup ...

@app.post("/analisis")
async def analizar_audio(audio: UploadFile = File(...), glucosa: float = Form(...)):
    import numpy as np
    import librosa
    from scipy.signal import butter, filtfilt
    from pydub import AudioSegment
    import traceback

    # 1) Guardar el raw upload en .3gp (o el formato que venga)
    ext = os.path.splitext(audio.filename)[1] or ".3gp"
    raw_tmp = f"temp_raw{ext}"
    data = await audio.read()
    with open(raw_tmp, "wb") as f:
        f.write(data)

    # 2) Convertir a WAV mono 16 kHz con pydub + ffmpeg
    wav_tmp = "temp.wav"
    try:
        sound = AudioSegment.from_file(raw_tmp)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(wav_tmp, format="wav")
    except Exception as e:
        print("❌ Error convirtiendo a WAV:", traceback.format_exc())
        return {"error": f"Conversión a WAV fallida: {e}"}
    finally:
        if os.path.exists(raw_tmp):
            os.remove(raw_tmp)

    try:
        # 3) Cargar WAV ya válido
        y, sr = librosa.load(wav_tmp, sr=16000, duration=10.0)

        # 4) Filtro band-pass
        nyq = 0.5 * sr
        b, a = butter(2, [20/nyq, 150/nyq], btype="band")
        y = filtfilt(b, a, y)

        # 5) Extracción de features + predicción
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
        except Exception as e:
            print("⚠️ Error BPM:", traceback.format_exc())
            bpm = None

        # 7) Respuesta completa
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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )



