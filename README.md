# Chord Recognition with NNLS-Chroma + PyTorch
A hybrid CNN–Transformer model for frame-level chord recognition using NNLS-Chroma features.
This project performs automatic chord recognition from audio using:
- NNLS-Chroma Vamp plugin (feature extraction)
- Sonic Annotator (audio analysis pipeline)
- PyTorch neural network model
- McGill Billboard dataset for training and evaluation

The system converts raw audio into chroma features and predicts chord labels over time.

---

## Project Structure
```text
chord_model/    
├── dataset.py # dataset builder  
├── model.py # PyTorch model definition   
├── train.py # training script    
├── run.py # inference pipeline (audio → chords)    
├── nnls.ttl # Vamp transform config for NNLS chroma    
├── McGill-Billboard/ # dataset (audio + annotations)   
├── weights/    
│ └── final_weights.pth # trained model weights   
├── requirements.txt    
```
---

## System Dependencies

### 1. Install [Sonic Annotator](https://github.com/sonic-visualiser/sonic-annotator/releases)

Add binary to PATH.


### 2. Install Vamp Plugin Pack (IMPORTANT)

NNLS-Chroma is NOT included in system packages.

Download [installer](https://www.vamp-plugins.org/pack.html)

During installation select:
Chordino and NNLS Chroma


### 3. Verify Installations

sonic-annotator -l | grep nnls

Expected output includes:

vamp:nnls-chroma:nnls-chroma:chroma
vamp:nnls-chroma:chordino:chordnotes


## Dataset (McGill Billboard)

This project uses the McGill [billboard-2.0-chordino](https://ddmal.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/) dataset:

Expected structure:
```text
McGill-Billboard/  
  ├── 0003/  
  │   ├── tuning.csv    
  │   ├── bothchroma.csv   
  │   ├── full.lab   
```
The chroma features are aligned with McGill baseline settings.

Feature Extraction

Extract chroma from audio:

```sonic-annotator -t nnls.ttl input.mp3 -w csv --csv-stdout```   

Output format:

timestamp, 24-dim chroma vector

This project uses 24-dimensional NNLS chroma, not 12D.

QUICK START: Run prediction on audio file: ```python run.py```     

For retraining and testing:

Modify McGill-Billboard dataset for training: ```python dataset.py```

Train model on McGill dataset: ```python train.py ```

The model learns:

24D chroma window → chord class

Chord labels include major/minor triads and extended set depending on CHORDS definition.

Inference (Chord Recognition)

Run prediction on audio file:

```python run.py``` 


## Training Pipeline

### 1. Data Preparation
- NNLS-Chroma features are extracted using Sonic Annotator
- Features are stored as:
  - X: chroma sequences
  - y: chord labels
- Data is cached as `.npy` files for fast reloading

### 2. Windowing
- Sliding window segmentation is applied
- Each training sample represents a fixed-length temporal context

### 3. Dataset Split
- 90% train / 10% validation
- Track-level split (no leakage between songs)

### 4. Model
Input:
- 15-frame chroma window
- 24-dimensional NNLS chroma features per frame

#### Architecture:
1. CNN Feature Extractor
   - Conv1D(24 → 64)
   - ReLU
   - Conv1D(64 → 128)
   - ReLU

2. Transformer Encoder
   - 2 encoder layers
   - 8 attention heads
   - hidden dimension: 128
   - feedforward dimension: 256

3. Classification Head
   - LayerNorm
   - Center-frame selection
   - Linear(128 → num_classes)

Output:
- Probability distribution over chord classes

### 5. Optimization
- Adam optimizer (lr=1e-3)
- Mixed precision training (torch.cuda.amp)
- Batch size: 64

## Key Design Decisions

### NNLS-Chroma instead of raw FFT
- Improves robustness to timbral variation
- Aligns with McGill Billboard baseline pipeline

### 24D chroma representation
- Captures richer harmonic structure than 12D chroma
- Preserves extended harmonic information

### Temporal smoothing
- Reduces frame-level jitter
- Improves musical consistency of predictions

## Why This Works

- CNN captures local harmonic structure in chroma space
- Transformer models temporal chord transitions
- Center-frame prediction ensures local temporal consistency

## Results

Evaluation is performed using a track-level split (no overlap between training and test songs).

- Validation accuracy (Billboard): ~86% after 10 epochs
- Performance decreases slightly on modern, out-of-domain music due to domain shift in production styles

---

## Example output:

12.45s → Am   
16.49s → C    
20.53s → Dm   
24.57s → G    
...
