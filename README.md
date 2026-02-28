# Sign Language Recognition App

A real-time sign language recognition system that uses **MediaPipe** for hand detection and a **TensorFlow LSTM model** for gesture classification. Point your webcam at your hands, and the app identifies the sign you're making.

## Supported Signs

| Static (alphabet) | Dynamic (words) | Special        |
|--------------------|-----------------|----------------|
| A, B, C, D, E     | Hello, ThankYou, Please | _Neutral (resting/no sign) |

## Requirements

- Python 3.10+
- Webcam

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: NumPy, OpenCV, MediaPipe, TensorFlow (CPU).

## Usage

### Collect Training Data

```bash
python -m src.collect_data
```

**Keyboard controls:**

| Key     | Action                    |
|---------|---------------------------|
| `SPACE` | Start/stop recording      |
| `N`     | Next sign                 |
| `P`     | Previous sign             |
| `R`     | Retry (delete last sample)|
| `0-9`   | Jump to sign by number    |
| `Q`     | Quit                      |

A 3-second countdown plays before each recording. Each recording captures 30 frames (1 second at 30 FPS) and saves a `.npy` sequence file to `data/sequences/<sign_name>/`.

### Test Hand Detection (Webcam)

Opens the webcam and verifies that MediaPipe correctly detects and labels your left/right hands:

```bash
python core/holistic_detector.py
```

### Test Feature Extraction (Webcam)

Opens the webcam and displays the live 126-dim feature vector, hand count, palm sizes, and non-zero feature count:

```bash
python core/landmark_extractor.py
```

### Test Visualization Overlay (Webcam)

Opens the webcam with the full UI overlay (landmarks, bounding boxes, info panel, progress bar) to verify drawing functions:

```bash
python utils/visualization.py
```

Press `Q` to quit any of the webcam tests.

### Verify Configuration

```bash
python config.py
```

## How It Works

1. **Detection** -- MediaPipe Holistic detects hand landmarks in each video frame.
2. **Feature Extraction** -- 21 landmarks per hand (x, y, z) are extracted and normalized (centered on wrist, scaled by palm width), producing a 126-dimensional feature vector per frame.
3. **Buffering** -- Frames are accumulated in a circular buffer. A 30-frame sequence represents one sign.
4. **Prediction** -- An LSTM model classifies the sequence. Predictions below 70% confidence or labeled as `_Neutral` are filtered out.

## Project Structure

```
config.py              # Central configuration (all tunable parameters)
core/
  holistic_detector.py # MediaPipe wrapper with hand-swap correction
  landmark_extractor.py# Landmark extraction and normalization
  frame_buffer.py      # Circular buffer for frame sequences
  predictor.py         # Model loading and inference
src/
  collect_data.py      # Data collection application
utils/
  preprocessing.py     # Normalization and data augmentation
  visualization.py     # OpenCV drawing helpers
data/sequences/        # Collected training data (.npy files)
models/                # Trained models and label mappings
```

## Roadmap

- **Phase 1 (current):** Hand-only recognition -- 126 features, 8 signs
- **Phase 2:** Add facial expressions -- 204 features (126 hand + 78 face)
- **Phase 3:** Real-time continuous recognition application
