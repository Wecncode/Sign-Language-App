# Setup Guide - Sign Language Recognition App

A step-by-step guide for downloading and running this project on your laptop.

## Prerequisites

Before you begin, make sure you have the following installed:

1. **Python 3.10 or higher** — Download from [python.org](https://www.python.org/downloads/)
   - During installation, check **"Add Python to PATH"**
   - To verify: open a terminal and run `python --version`

2. **Git** — Download from [git-scm.com](https://git-scm.com/downloads)
   - To verify: open a terminal and run `git --version`

3. **A webcam** — Built-in or external (required for data collection and testing)

## Step 1: Clone the Repository

Open a terminal (Command Prompt, PowerShell, or Git Bash on Windows; Terminal on Mac/Linux).

First, navigate to the folder where you want to download the project. For example:

```bash
cd D:\Projects
```

> Replace `D:\Projects` with whatever folder you prefer. On Mac/Linux it might look like `cd ~/Projects`.

Then clone the repository:

```bash
git clone https://github.com/Wecncode/Sign-Language-App.git
```

This creates a `Sign-Language-App` folder inside your chosen directory. Navigate into it:

```bash
cd Sign-Language-App
```

## Step 2: Create a Virtual Environment

A virtual environment keeps this project's packages separate from your system Python.

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt. This means the virtual environment is active.

> **Note:** Every time you open a new terminal to work on this project, you need to activate the virtual environment again using the activate command above.

## Step 3: Install Dependencies

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

This installs all required packages: NumPy, OpenCV, MediaPipe, TensorFlow, Matplotlib, etc.

## Step 4: Verify the Installation

Run the configuration test to make sure everything is working:

```bash
python config.py
```

You should see a printout of all configuration settings with no errors.

## Step 5: Test Your Webcam

Run the hand detection test to verify your webcam works with MediaPipe:

```bash
python core/holistic_detector.py
```

A window should open showing your webcam feed with hand detection. Hold up your hands and check that they are detected. Press **Q** to quit.

## Step 6: Obtain Model Files

The trained model files are **not included** in the repository due to their size. You need to obtain these files separately and place them in the `models/` folder:

```
models/
├── best_model.keras    (the trained LSTM model)
└── labels.pkl          (label mapping file)
```

> Ask your instructor or project lead for the model files or the link to download them.

If the `models/` folder doesn't exist, create it:

**Windows:**
```bash
mkdir models
```

**Mac / Linux:**
```bash
mkdir -p models
```

## Step 7: Run the Data Collection App

To start collecting training data:

```bash
python -m src.collect_data
```

### Keyboard Controls

| Key     | Action                     |
|---------|----------------------------|
| `SPACE` | Start/stop recording       |
| `N`     | Next sign                  |
| `P`     | Previous sign              |
| `R`     | Retry (delete last sample) |
| `0-9`   | Jump to sign by number     |
| `Q`     | Quit                       |

## Project Structure

```
Sign-Language-App/
├── config.py              # All settings and configuration
├── core/
│   ├── holistic_detector.py   # Hand detection (MediaPipe)
│   ├── landmark_extractor.py  # Feature extraction
│   ├── frame_buffer.py        # Frame sequence buffer
│   └── predictor.py           # Model inference
├── utils/
│   ├── preprocessing.py       # Normalization functions
│   └── visualization.py       # UI drawing functions
├── src/
│   └── collect_data.py        # Data collection app
├── data/sequences/            # Training data (created at runtime)
├── models/                    # Model files (not in repo)
├── requirements.txt           # Python dependencies
└── README.md
```

## Testing Individual Modules

Each module can be run independently to test it:

```bash
python core/holistic_detector.py    # Test hand detection (webcam)
python core/landmark_extractor.py   # Test feature extraction (webcam)
python core/frame_buffer.py         # Test frame buffering
python core/predictor.py            # Test model inference (needs model files)
python utils/preprocessing.py       # Test normalization
python utils/visualization.py       # Test UI overlay (webcam)
```

Press **Q** to quit any webcam test.

## Troubleshooting

### "python" is not recognized
- Make sure Python is added to your PATH. Reinstall Python and check "Add Python to PATH".
- On Mac/Linux, try `python3` instead of `python`.

### pip install fails with permission errors
- Make sure your virtual environment is activated (you should see `(.venv)` in your prompt).
- If still failing, try: `pip install --user -r requirements.txt`

### Webcam not opening
- Check that no other app is using your webcam (Zoom, Teams, etc.).
- If you have multiple cameras, change `CAMERA_INDEX` in `config.py` (try 1 or 2 instead of 0).

### MediaPipe not detecting hands
- Make sure you have good lighting.
- Keep your hands within the camera frame.
- Try lowering `MIN_DETECTION_CONFIDENCE` in `config.py` (e.g., from 0.7 to 0.5).

### ModuleNotFoundError
- Make sure your virtual environment is activated.
- Run `pip install -r requirements.txt` again.

### Model files not found
- You need to place `best_model.keras` and `labels.pkl` in the `models/` folder.
- These files are not included in the repository.
