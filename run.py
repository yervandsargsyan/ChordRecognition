import numpy as np
import torch
from model import ChordModel
from dataset import CHORDS, make_windows
import subprocess
import pandas as pd
from io import StringIO
from collections import Counter


WEIGHTS_PATH = "weights/final_weights.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChordModel(num_classes=24).to(device)
checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()


SONIC_ANNOTATOR = 'sonic-annotator' # make sure it's in your PATH

def audio_to_chroma(audio_path):
    result = subprocess.run([
        SONIC_ANNOTATOR,
        '-t', 'nnls.ttl',
        audio_path,
        '-w', 'csv',
        '--csv-stdout'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    df = pd.read_csv(StringIO(result.stdout), header=None)
    timestamps = df.iloc[:, 1].values.astype(float)
    chroma = df.iloc[:, 2:].values.astype(float)

    return timestamps, chroma



def smooth_chords(result, window=4.0):
    if not result:
        return result

    smoothed = []
    i = 0
    
    while i < len(result):
        t_start = result[i]["timestamp"]
        
        # собираем все аккорды в окне [t_start, t_start + window]
        window_chords = []
        j = i
        while j < len(result) and result[j]["timestamp"] < t_start + window:
            window_chords.append(result[j]["chord"])
            j += 1
        
        # берём моду
        most_common = Counter(window_chords).most_common(1)[0][0]
        smoothed.append({"timestamp": round(t_start, 2), "chord": most_common})
        
        i = j  # прыгаем на следующее окно
    
    return smoothed


def predict_chords(audio_path: str):
    timestamps, chroma = audio_to_chroma(audio_path)

    dummy_y = np.zeros(len(chroma), dtype=np.int64)
    X_windows, _ = make_windows(chroma, dummy_y)

    X_tensor = torch.tensor(X_windows, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    center = 7
    result = []
    for i, pred in enumerate(predictions):
        t = timestamps[i + center]
        chord = CHORDS[pred]
        result.append({"timestamp": round(float(t), 2), "chord": chord})

    return smooth_chords(result)


if __name__ == "__main__":
    result = predict_chords("song1.mp3")
    for item in result:
        print(f"{item['timestamp']:.2f}s  →  {item['chord']}")