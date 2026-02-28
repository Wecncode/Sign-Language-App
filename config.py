"""
=============================================================================
CONFIGURATION FILE - Central Settings for Sign Language App
=============================================================================

This file contains ALL configurable settings for the project.
Instead of having magic numbers scattered across files, we keep them here.

WHY IS THIS IMPORTANT?
- Easy to tune parameters without searching through code
- All files use the same settings (consistency)
- Quick experimentation (change one number, affects entire app)

HOW TO USE:
    from config import Config
    
    print(Config.SEQUENCE_LENGTH)  # 30
    print(Config.MIN_CONFIDENCE)   # 0.7
=============================================================================
"""

import os


class Config:
    """
    Central configuration class.
    All settings are class attributes (no need to create an instance).
    
    Usage:
        from config import Config
        print(Config.SEQUENCE_LENGTH)  # Access directly
    """
    
    # =========================================================================
    # PROJECT PATHS
    # =========================================================================
    # These paths are relative to the project root folder
    
    # Where training data will be saved
    # Structure: DATA_DIR/sign_name/sequence_001.npy
    DATA_DIR = os.path.join("data", "sequences")
    
    # Where trained models will be stored
    MODEL_DIR = "models"
    
    # Specific model files (created after training in Phase 2)
    MODEL_PATH = os.path.join(MODEL_DIR, "unified_model.tflite")
    LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")
    
    # =========================================================================
    # LANDMARK SETTINGS
    # =========================================================================
    # MediaPipe Holistic detects multiple body parts.
    # For Phase 1, we only use hands. Phase 2 will add face.
    
    # Which body parts to use for recognition
    USE_HANDS = True      # Always True - hands are primary input
    USE_FACE = False      # Phase 2 feature - facial expressions
    USE_POSE = False      # Future feature - body pose
    
    # Number of landmarks per body part
    # MediaPipe provides these fixed numbers
    HAND_LANDMARKS = 21         # Each hand has 21 key points
    FACE_LANDMARKS = 468        # Full face mesh (we'll use subset)
    SELECTED_FACE_LANDMARKS = 26  # We only need 26 key points for expressions
    
    # Coordinates per landmark (x, y, z)
    COORDS_PER_LANDMARK = 3
    
    # =========================================================================
    # FEATURE VECTOR SIZE
    # =========================================================================
    # Calculate total features per frame
    # 
    # Phase 1 (hands only):
    #   21 landmarks × 3 coords × 2 hands = 126 features
    #
    # Phase 2 (hands + face):
    #   126 (hands) + 26 landmarks × 3 coords = 126 + 78 = 204 features
    
    # Features from hands: 21 × 3 × 2 = 126
    HAND_FEATURES = HAND_LANDMARKS * COORDS_PER_LANDMARK * 2  # 2 hands
    
    # Features from face: 26 × 3 = 78
    FACE_FEATURES = SELECTED_FACE_LANDMARKS * COORDS_PER_LANDMARK
    
    # Total features per frame (changes based on USE_FACE)
    @classmethod
    def get_features_per_frame(cls):
        """
        Calculate total features based on what body parts we're using.
        
        Returns:
            int: Number of features per frame (126 or 204)
        """
        total = 0
        if cls.USE_HANDS:
            total += cls.HAND_FEATURES  # 126
        if cls.USE_FACE:
            total += cls.FACE_FEATURES  # 78
        return total
    
    # For quick access (Phase 1 default)
    FEATURES_PER_FRAME = 126  # Update to 204 when USE_FACE = True
    
    # =========================================================================
    # LANDMARK INDICES (for normalization)
    # =========================================================================
    # MediaPipe hand landmark indices we need for normalization
    # 
    # Hand landmark map:
    #
    #         8   12  16  20      <- Fingertips
    #         |   |   |   |
    #         7   11  15  19
    #         |   |   |   |
    #         6   10  14  18
    #         |   |   |   |
    #         5   9   13  17      <- Knuckles (MCP joints)
    #          \  |   |  /
    #           \ |   | /
    #        4   \|   |/
    #        |    \   /
    #        3     \ /
    #        |      1
    #        2      |
    #         \     |
    #          \    |
    #           \   |
    #            \  |
    #             \ |
    #              \|
    #               0             <- WRIST
    #
    
    WRIST_INDEX = 0           # Wrist - center point for normalization
    THUMB_CMC_INDEX = 1       # Thumb base
    INDEX_MCP_INDEX = 5       # Index finger knuckle
    PINKY_MCP_INDEX = 17      # Pinky knuckle
    
    # =========================================================================
    # NORMALIZATION SETTINGS
    # =========================================================================
    # Normalization makes the model work regardless of:
    # - Hand position in frame (solved by centering on wrist)
    # - Hand size / distance from camera (solved by scaling)
    
    # Whether to apply each normalization step
    APPLY_CENTERING = True           # Subtract wrist position
    APPLY_SCALE_NORMALIZATION = True  # Divide by palm width
    
    # Minimum scale to prevent division by zero
    # If palm_width is smaller than this, we use this value instead
    MIN_SCALE = 0.001
    
    # =========================================================================
    # SEQUENCE SETTINGS
    # =========================================================================
    # Our LSTM model processes sequences of frames, not single frames.
    # This allows it to recognize motion (dynamic signs like "Hello")
    
    # Number of frames in one sequence
    # At 30 FPS, 30 frames = 1 second of video
    SEQUENCE_LENGTH = 30
    
    # How many frames to keep in memory
    # We keep 60 frames (2 seconds) to have buffer for motion analysis
    BUFFER_SIZE = 60
    
    # =========================================================================
    # GESTURE DETECTION SETTINGS
    # =========================================================================
    # We detect when the user has completed a sign by checking if
    # their hand has stopped moving (become "stable")
    
    # Motion threshold - if motion is below this, hand is considered stable
    # Lower = more sensitive (detects smaller movements)
    # Higher = less sensitive (ignores small jitters)
    STABILITY_THRESHOLD = 0.02
    
    # How many consecutive stable frames before triggering prediction
    # At 30 FPS, 10 frames = 0.33 seconds
    STABILITY_FRAMES = 10
    
    # Frames to wait after making a prediction
    # Prevents "AAAAA" spam when holding a sign
    # At 30 FPS, 15 frames = 0.5 seconds
    COOLDOWN_FRAMES = 15
    
    # =========================================================================
    # PREDICTION FILTERING
    # =========================================================================
    # Not all predictions should be shown to the user.
    # We filter out uncertain and neutral predictions.
    
    # Special class name for "not a sign" - MUST match training data folder
    NEUTRAL_LABEL = "_Neutral"
    
    # Minimum confidence to accept a prediction
    # 0.7 = 70% confident
    MIN_CONFIDENCE = 0.7
    
    # Classes to never display (even if confidence is high)
    IGNORE_CLASSES = ["_Neutral"]
    
    # =========================================================================
    # MEDIAPIPE SETTINGS
    # =========================================================================
    # Confidence thresholds for MediaPipe detection
    
    # Minimum confidence to detect a hand initially
    # Lower = detects hands more easily (but more false positives)
    MIN_DETECTION_CONFIDENCE = 0.7
    
    # Minimum confidence to track a hand between frames
    # Lower = keeps tracking even when hand is partially hidden
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # =========================================================================
    # CAMERA SETTINGS
    # =========================================================================
    
    # Which camera to use (0 = default/built-in webcam)
    # Change to 1, 2, etc. if you have multiple cameras
    CAMERA_INDEX = 0
    
    # Video resolution
    # Lower resolution = faster processing but less detail
    # 640×480 is a good balance for most laptops
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # Target frames per second
    # 30 FPS is standard, matches our SEQUENCE_LENGTH calculation
    FPS = 30
    
    # =========================================================================
    # UI SETTINGS
    # =========================================================================
    
    # Show landmark dots and connections on video
    SHOW_LANDMARKS = True
    
    # Show debug information (motion score, state, etc.)
    SHOW_DEBUG_INFO = True
    
    # Colors for drawing (BGR format - Blue, Green, Red)
    LANDMARK_COLOR = (0, 255, 0)       # Green
    CONNECTION_COLOR = (0, 255, 0)      # Green
    TEXT_COLOR = (255, 255, 255)        # White
    ALERT_COLOR = (0, 0, 255)           # Red
    SUCCESS_COLOR = (0, 255, 0)         # Green
    
    # Font settings
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    
    # =========================================================================
    # DATA COLLECTION SETTINGS
    # =========================================================================
    
    # Signs to collect data for
    # _Neutral MUST be first - it needs more samples
    SIGNS_TO_COLLECT = [
        "_Neutral",     # 200-300 samples (no sign / resting)
        "A", "B", "C", "D", "E",  # Static alphabet signs
        "Hello",        # Dynamic sign (waving)
        "ThankYou",     # Dynamic sign
       # "Yes",          # Dynamic sign 
       # "No",           # Dynamic sign 
        "Please",       # Dynamic sign
    ]
    
    # Target samples per sign
    SAMPLES_PER_SIGN = 50 #100 sample
    SAMPLES_FOR_NEUTRAL = 100  # 250 Neutral needs more samples
    
    # Recording countdown (seconds before recording starts)
    COUNTDOWN_SECONDS = 3
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    @classmethod
    def get_data_path(cls, sign_name):
        """
        Get the folder path for a specific sign's data.
        
        Args:
            sign_name: Name of the sign (e.g., "A", "Hello", "_Neutral")
            
        Returns:
            str: Full path to the sign's data folder
            
        Example:
            Config.get_data_path("A")  # Returns "data/sequences/A"
        """
        return os.path.join(cls.DATA_DIR, sign_name)
    
    @classmethod
    def ensure_directories(cls):
        """
        Create necessary directories if they don't exist.
        Call this at the start of data collection.
        
        Creates:
            - data/sequences/
            - data/sequences/_Neutral/
            - data/sequences/A/
            - ... (all signs)
            - models/
        """
        # Create main directories
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        
        # Create folder for each sign
        for sign in cls.SIGNS_TO_COLLECT:
            sign_path = cls.get_data_path(sign)
            os.makedirs(sign_path, exist_ok=True)
            
        print(f"✓ Created directories in {cls.DATA_DIR}")


# =============================================================================
# QUICK TEST
# =============================================================================
# Run this file directly to verify configuration

if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURATION TEST")
    print("=" * 60)
    
    print(f"\n📁 Paths:")
    print(f"   Data directory: {Config.DATA_DIR}")
    print(f"   Model path: {Config.MODEL_PATH}")
    
    print(f"\n🖐️ Landmark Settings:")
    print(f"   Hand landmarks: {Config.HAND_LANDMARKS}")
    print(f"   Features per frame: {Config.FEATURES_PER_FRAME}")
    
    print(f"\n🎬 Sequence Settings:")
    print(f"   Sequence length: {Config.SEQUENCE_LENGTH} frames")
    print(f"   Buffer size: {Config.BUFFER_SIZE} frames")
    
    print(f"\n📷 Camera Settings:")
    print(f"   Resolution: {Config.FRAME_WIDTH}×{Config.FRAME_HEIGHT}")
    print(f"   FPS: {Config.FPS}")
    
    print(f"\n✅ Signs to collect:")
    for sign in Config.SIGNS_TO_COLLECT:
        samples = Config.SAMPLES_FOR_NEUTRAL if sign == "_Neutral" else Config.SAMPLES_PER_SIGN
        print(f"   {sign}: {samples} samples")
    
    print("\n" + "=" * 60)
    print("Configuration loaded successfully!")
    print("=" * 60)