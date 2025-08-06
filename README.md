# Music Structure Analysis with Transformer-Based Model

## 1. Project Summary

This project implements a **Transformer-based architecture** for **music boundary detection** and **segment classification** using mel-spectrograms and chroma features.

The architecture uses a **dual-input design**:
- **Encoder input:** Drum stem features (mel-spectrogram + chroma, concatenated to [T, 92])
- **Decoder input:** Full mix features (mel-spectrogram + chroma, concatenated to [T, 92])

The **encoder** focuses on rhythmic cues from the drum stem, while the **decoder** models the complete musical context from the full mix.  
A **cross-attention mechanism** allows the decoder to attend to encoder outputs.

Key components:
- **Custom CNN layers** and **Squeeze-and-Excitation (SE) blocks** for spectral feature extraction
- **Spectral and temporal masking** to control attention scope
- **Sinusoidal positional encoding** to preserve temporal order
- **Transformer encoder + decoder stacks** with multi-head attention
- Two output heads:
  - **Boundary detection:** Sigmoid output for frame-level boundaries
  - **Segment classification:** Softmax output for functional labels (7 classes)

The system processes datasets (**Beatles**, **SALAMI**, **Harmonix**) with both **original** and **augmented** versions.  
Augmentation includes pitch shifting and pre-emphasis filtering.

---

## 2. Installation

### Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
librosa
mir_eval
numpy
scipy
tensorflow
tqdm
```

---

## 3. Workflow Overview

The workflow consists of:

1. **Preprocessing** (Original & Augmented)
2. **Compression to NPZ**
3. **(Optional) Combining NPZ files**
4. **Training, Testing, and Evaluation**

---

## 4. Preprocessing

### 4.1 Original Data Preprocessing

- **Beatles Original:**
  - `beatles-original-preprocessing.py`  
  - `beatles-drums-original-preprocessing.py` (drum stem features)

- **SALAMI Original:** *(create analogous preprocessing scripts)*  
  - `salami-original-preprocessing.py`  
  - `salami-drums-original-preprocessing.py`

- **Harmonix Original:** *(create analogous preprocessing scripts)*  
  - `harmonix-original-preprocessing.py`  
  - `harmonix-drums-original-preprocessing.py`

These scripts:
- Extract mel-spectrogram & chroma features
- Align annotations to frame-level boundaries
- Save per-song numpy arrays (`_spec.npy`, `_chroma.npy`, `_boundary.npy`, `_function.npy`, `_section.npy`)

---

### 4.2 Augmented Data Preprocessing

- **Beatles Augmented:**
  - `beatles-aug-preprocessing.py`
  - `beatles-drums-aug-preprocessing.py`

- **SALAMI Augmented:** *(create analogous preprocessing scripts)*  
  - `salami-aug-preprocessing.py`  
  - `salami-drums-aug-preprocessing.py`

- **Harmonix Augmented:** *(create analogous preprocessing scripts)*  
  - `harmonix-aug-preprocessing.py`  
  - `harmonix-drums-aug-preprocessing.py`

Augmentation strategy:
- **Pitch shifts:** `-2, -1, 0, +1, +2` semitones
- **Pre-emphasis coefficients:** `0.7`, `0.97`
- Generates **10 versions per track**

---

## 5. Compression to NPZ

After preprocessing, compress processed features into `.npz` files.

- **Test (Original) compression:** `beatles-test-compression.py`  
- **Train (Augmented) compression:** `beatles-train-compression.py`

Repeat for **SALAMI** and **Harmonix**.

Output structure in `.npz`:
- `spec` : mel-spectrograms
- `chromagram` : chroma features
- `boundary` : boundary labels
- `function` : functional class labels
- `section` : section names

---

## 6. Combining Datasets

To combine any **two datasets** (e.g., original + augmented, or across datasets), use:
```bash
python combine-npz.py
```
Modify file paths inside to point to desired `.npz` files.

---

## 7. Training, Testing, and Evaluation

- **File:** `my_training_attemp1.py`
- Handles:
  - **Model building** (Transformer with spectral/temporal masking, CNN front-ends, drum encoder)
  - **Training loop** (BCE for boundaries, CE for segments)
  - **Evaluation** (mir_eval metrics, confusion matrix, per-epoch PNG visualizations)
  - **Saving models**:  
    - All epochs → `my_models/all_epochs/`  
    - Best epochs → `my_models/best_models/`

Run training:
```bash
python my_training_attemp1.py
```

If you have the provided **preprocessed data folders**:
- `my_salami_data/`
- `my_harmonix_data/`
- `my_beatles_data/`

…you can **skip preprocessing** and start directly from training.

---

## 8. Results

Example evaluation outputs are saved as `.png` in the results folder.  
They include per-epoch boundary plots, confusion matrices, and metric summaries.

---

## 9. Project Structure

```

.
├── my_training_attemp1.py                     # Main script: builds model, trains, tests, evaluates, saves results
│
├── beatles-original-preprocessing.py          # Preprocess Beatles original audio → mel & chroma features + labels
├── beatles-drums-original-preprocessing.py    # Same as above, but extracts features from drum stem
├── beatles-aug-preprocessing.py               # Preprocess Beatles augmented audio (pitch shift + pre-emphasis)
├── beatles-drums-aug-preprocessing.py         # Same as above, but for drum stem
│
├── salami-original-preprocessing.py           # Preprocess SALAMI original audio
├── salami-drums-original-preprocessing.py     # Preprocess SALAMI original drums
├── salami-aug-preprocessing.py                # Preprocess SALAMI augmented audio
├── salami-drums-aug-preprocessing.py          # Preprocess SALAMI augmented drums
│
├── harmonix-original-preprocessing.py         # Preprocess Harmonix original audio
├── harmonix-drums-original-preprocessing.py   # Preprocess Harmonix original drums
├── harmonix-aug-preprocessing.py              # Preprocess Harmonix augmented audio
├── harmonix-drums-aug-preprocessing.py        # Preprocess Harmonix augmented drums
│
├── beatles-test-compression.py                # Compress processed Beatles original files into test_data.npz
├── beatles-train-compression.py               # Compress processed Beatles augmented files into train_data.npz
├── salami-test-compression.py                 # Compress processed SALAMI original files into test_data.npz
├── salami-train-compression.py                # Compress processed SALAMI augmented files into train_data.npz
├── harmonix-test-compression.py               # Compress processed Harmonix original files into test_data.npz
├── harmonix-train-compression.py              # Compress processed Harmonix augmented files into train_data.npz
│
├── combine-npz.py                             # Combine any two datasets (.npz) into one for training
│
├── my_salami_data/                            # Preprocessed SALAMI data (ready-to-train NPZ files)
├── my_harmonix_data/                          # Preprocessed Harmonix data (ready-to-train NPZ files)
├── my_beatles_data/                           # Preprocessed Beatles data (ready-to-train NPZ files)
│
└── requirements.txt                           # Python dependencies for the project
│
├── my_models/all_epochs/                        # Saved checkpoints for all training epochs
├── my_models/best_models/                        # Saved checkpoints for best epochs (based on validation metrics)

```

---
## 10. Results
For my first attempt , the results are saved into results-for-beatles.png as I have used Beatles for testing purpose and Salami and Harmonix for training (original+augmented). 


## 11. Notes

- Ensure datasets follow the same structure before preprocessing.
- If using **augmented data**, match annotation files carefully to generated audio.
- The **model** expects mel + chroma inputs concatenated as `[T, 92]`.

---

**Author:** Aneeka Azmat  
**Purpose:** Research on music structure analysis with drum-aware Transformer models.

