"""
=============================================================================
CORE PACKAGE INITIALIZER
=============================================================================
"""

# Stage 1 imports
from core.holistic_detector import HolisticDetector

# Stage 2 imports (NOW AVAILABLE!)
from core.landmark_extractor import LandmarkExtractor
from core.frame_buffer import FrameBuffer

# Phase 3 imports (will be added later)
# from core.gesture_detector import GestureDetector
# from core.state_machine import StateMachine, State
# from core.predictor import Predictor
# from core.prediction_filter import PredictionFilter

__version__ = "1.0.0"
__author__ = "Sign Language App Team"