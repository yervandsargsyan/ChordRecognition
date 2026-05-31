import pandas as pd
import numpy as np
import os
from tqdm import tqdm

ENHARMONIC = {
    'Db': 'C#', 'Eb': 'D#', 'Fb': 'E',
    'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#', 'Cb': 'B'
}

CHORDS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
          'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm']

CHORD_TO_IDX = {chord: idx for idx, chord in enumerate(CHORDS)}

def convert_chord(chord: str) -> str:
    """
    Convert chords MajMin vocabulary, 
    e.g. 'C:maj' -> 'C', 'D:min' -> 'Dm', 'Db:maj#' → 'C#'
    """
    if chord in ('N', 'X'):
        return None
    
    root = chord.split(':')[0]
    quality = chord.split(':')[1] if ':' in chord else 'maj'
    
    # enharmonic normalization
    root = ENHARMONIC.get(root, root)
    
    if quality.startswith('min'):
        return root + 'm'
    else:
        return root
    

def read_lab(lab_path: str) -> list:
    """
    Read .lab file and return list of chords in MajMin vocabulary
    """
    chords_with_timings = []
    with open(lab_path, 'r') as f:
        for line in f:
            try:
                start, end, chord = line.strip().split()
                chord = convert_chord(chord)
                if chord is not None:
                    chords_with_timings.append((float(start), float(end), chord))
            except: 
                continue
                
    return chords_with_timings


def read_chroma(chroma_path: str):
    df = pd.read_csv(chroma_path, header=None)
    timestamps = df.iloc[:, 1].values.astype(float)
    chroma = df.iloc[:, 2:].values.astype(float)
    return timestamps, chroma

def get_chord_at_time(t: float, chords_with_timings: list) -> str:
    for start, end, chord in chords_with_timings:
        if start <= t < end:
            return chord
    return None


def load_track(track_dir: str):
    """
    Load one track and return (chroma, chords) pairs
    """
    chroma_path = os.path.join(track_dir, 'bothchroma.csv')
    lab_path = os.path.join(track_dir, 'full.lab')
    
    timestamps, chroma = read_chroma(chroma_path)
    chords_with_timings = read_lab(lab_path)
    
    X = []  # chroma vectors
    y = []  # chords
    
    for i, t in enumerate(timestamps):
        chord = get_chord_at_time(t, chords_with_timings)
        if chord is not None:
            X.append(chroma[i])
            y.append(chord)
    
    return np.array(X), y


def load_dataset(data_dir: str):
    X_all = []
    y_all = []
    tracks = [t for t in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, t))]
    
    for track in tqdm(tracks, desc="Loading dataset"):
        track_dir = os.path.join(data_dir, track)
        try:
            X, y = load_track(track_dir)
            X_all.append(X)
            y_all.extend(y)
        except Exception as e:
            print(f"Skipping {track}: {e}")
            continue
    
    return np.concatenate(X_all, axis=0), y_all

def make_windows(X: np.ndarray, y: np.ndarray, window_size: int = 15):
    """
    Cut X and y into overlapping windows
    X: (N, 24) → (N - window_size, window_size, 24)
    y: take central chord for each window
    """
    X_windows = []
    y_windows = []
    
    center = window_size // 2  # central frame index
    
    for i in range(center, len(X) - center):
        window = X[i - center : i + center + 1]  # (window_size, 24)
        X_windows.append(window)
        y_windows.append(y[i])
    
    return np.array(X_windows), np.array(y_windows)

if __name__ == '__main__':
    X, y = load_dataset(r'McGill-Billboard')

    y = np.array([CHORD_TO_IDX[chord] for chord in y])

    X_w, y_w = make_windows(X, y)
    print(X_w.shape, y_w.shape)

    np.save('X.npy', X_w)
    np.save('y.npy', y_w)