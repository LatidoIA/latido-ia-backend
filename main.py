from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import librosa
import numpy as np
import torch
import cv2
import io

app = FastAPI()

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.dropout = torch.nn.Dropout(0.25)

        dummy = torch.zeros(1, 1, 64, 64)
        dummy = self.pool(torch.relu(self.bn1(self.conv1(dummy))))
        dummy = self.pool(torch.relu(self.bn2(self.conv2(dummy))))
        self.flattened_size = dummy.numel()

        self.fc1 = torch.nn.Linear(self.flattened_size, 64)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, self.flattened_size)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN()
model.load_state_dict(torch.load('simple_cnn_heartsound.pth', map_location='cpu'))
model.eval()

def convertir_audio_a_librosa(file_bytes):
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    audio = audio.set_channels(1)          # Mono
    audio = audio.set_frame_rate(16000)    # 16 kHz
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer

def preprocess_audio(file_bytes):
    wav_buffer = convertir_audio_a_librosa(file_bytes)
    y, sr = librosa.load(wav_buffer, sr=16000, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    S_db_norm_resized = cv2.resize(S_db_norm, (64, 64))
    tensor = torch.tensor(S_db_norm_resized).unsqueeze(0).unsqueeze(0).float()
    return tensor

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        audio_tensor = preprocess_audio(contents)
        with torch.no_grad():
            output = model(audio_tensor)
            probs = torch.softmax(output, dim=1)
            prob, pred = torch.max(probs, 1)
        labels = ['Normal', 'Anormal']
        return JSONResponse(content={"prediction": labels[pred.item()], "probability": float(prob.item())})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
