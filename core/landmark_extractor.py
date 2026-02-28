"""
=============================================================================
LANDMARK EXTRACTOR - Extract & Normalize Features from MediaPipe Results
=============================================================================

This module bridges MediaPipe results and our preprocessing functions.
It takes raw MediaPipe detection results and produces normalized feature vectors.

RESPONSIBILITIES:
-----------------
1. Extract hand landmarks from MediaPipe results
2. Convert MediaPipe format to numpy arrays
3. Apply 3-step normalization (center + scale)
4. Handle missing hands (fill with zeros)
5. Combine left/right hands into single feature vector

DATA FLOW:
----------
    MediaPipe Results → LandmarkExtractor → Feature Vector (126 values)
    
    ┌─────────────────┐     ┌────────────────┐     ┌─────────────┐
    │ results.left_   │     │                │     │             │
    │ hand_landmarks  │────▶│  Extractor     │────▶│  [126]      │
    │ results.right_  │     │  (normalize)   │     │  features   │
    │ hand_landmarks  │     │                │     │             │
    └─────────────────┘     └────────────────┘     └─────────────┘

USAGE:
------
    from core.landmark_extractor import LandmarkExtractor
    from config import Config
    
    extractor = LandmarkExtractor(Config)
    
    # In video loop:
    results = detector.detect(frame)
    features = extractor.extract_and_normalize(results)
    # features.shape = (126,)

=============================================================================
"""

import numpy as np
from config import Config
from utils.preprocessing import (
    normalize_hand_landmarks,
    landmarks_to_feature_vector,
    mediapipe_to_numpy,
    calculate_distance
)


class LandmarkExtractor:
    """
    Extracts and normalizes landmarks from MediaPipe results.
    
    This class:
    1. Takes MediaPipe Holistic results
    2. Extracts hand landmarks (and optionally face)
    3. Normalizes them using our preprocessing functions
    4. Returns a flat feature vector
    
    Attributes:
        config: Configuration object with settings
        features_per_frame: Number of features in output vector (126 for hands only)
    """
    
    def __init__(self, config=None):
        """
        Initialize the LandmarkExtractor.
        
        Args:
            config: Configuration object (uses default Config if not provided)
        
        Example:
            extractor = LandmarkExtractor()
            # or
            extractor = LandmarkExtractor(Config)
        """
        
        # Use provided config or default
        self.config = config if config is not None else Config
        
        # Calculate features per frame based on config
        # Phase 1: Hands only = 21 landmarks × 3 coords × 2 hands = 126
        # Phase 2: Hands + Face = 126 + 78 = 204
        self.features_per_frame = self.config.get_features_per_frame()
        
        # Store landmark indices for quick access
        self.wrist_idx = self.config.WRIST_INDEX
        self.index_mcp_idx = self.config.INDEX_MCP_INDEX
        self.pinky_mcp_idx = self.config.PINKY_MCP_INDEX
        
        print(f"✓ LandmarkExtractor initialized")
        print(f"  Features per frame: {self.features_per_frame}")
    
    def extract_and_normalize(self, results):
        """
        Extract landmarks from MediaPipe results and normalize them.
        
        This is the main method you'll use. It:
        1. Checks which hands are detected
        2. Converts MediaPipe format to numpy arrays
        3. Applies normalization
        4. Combines into single feature vector
        
        Args:
            results: MediaPipe Holistic results object
                     (returned by detector.detect())
        
        Returns:
            features: numpy array of shape (126,) for hands-only mode
                      Contains normalized [x, y, z] for all landmarks
        
        Example:
            results = detector.detect(frame)
            features = extractor.extract_and_normalize(results)
            
            print(features.shape)  # (126,)
            print(features[:3])    # First landmark [x, y, z]
        """
        
        # =====================================================================
        # EXTRACT LEFT HAND
        # =====================================================================
        if results.left_hand_landmarks is not None:
            # Convert MediaPipe landmarks to numpy array (21, 3)
            left_hand_np = mediapipe_to_numpy(results.left_hand_landmarks)
        else:
            # Hand not detected - will be filled with zeros later
            left_hand_np = None
        
        # =====================================================================
        # EXTRACT RIGHT HAND
        # =====================================================================
        if results.right_hand_landmarks is not None:
            # Convert MediaPipe landmarks to numpy array (21, 3)
            right_hand_np = mediapipe_to_numpy(results.right_hand_landmarks)
        else:
            # Hand not detected - will be filled with zeros later
            right_hand_np = None
        
        # =====================================================================
        # NORMALIZE AND COMBINE
        # =====================================================================
        # This function handles:
        # - Normalizing each hand (center on wrist, scale by palm width)
        # - Filling zeros for missing hands
        # - Combining into single vector
        features = landmarks_to_feature_vector(left_hand_np, right_hand_np)
        
        # =====================================================================
        # PHASE 2: ADD FACE FEATURES (Currently disabled)
        # =====================================================================
        if self.config.USE_FACE and results.face_landmarks is not None:
            # TODO: Implement face feature extraction
            # face_features = self._extract_face_features(results.face_landmarks)
            # features = np.concatenate([features, face_features])
            pass
        
        return features
    
    def get_raw_landmarks(self, results):
        """
        Get raw (un-normalized) landmarks as numpy arrays.
        
        Useful for:
        - Debugging
        - Visualization
        - Motion analysis (before normalization)
        
        Args:
            results: MediaPipe Holistic results
        
        Returns:
            dict: {
                'left_hand': numpy array (21, 3) or None,
                'right_hand': numpy array (21, 3) or None
            }
        
        Example:
            raw = extractor.get_raw_landmarks(results)
            if raw['left_hand'] is not None:
                print(f"Left wrist position: {raw['left_hand'][0]}")
        """
        
        raw = {
            'left_hand': None,
            'right_hand': None
        }
        
        if results.left_hand_landmarks is not None:
            raw['left_hand'] = mediapipe_to_numpy(results.left_hand_landmarks)
        
        if results.right_hand_landmarks is not None:
            raw['right_hand'] = mediapipe_to_numpy(results.right_hand_landmarks)
        
        return raw
    
    def get_wrist_position(self, results):
        """
        Get the wrist position for each detected hand.
        
        Useful for:
        - Tracking hand movement
        - Gesture detection
        - UI feedback
        
        Args:
            results: MediaPipe Holistic results
        
        Returns:
            dict: {
                'left': (x, y, z) or None,
                'right': (x, y, z) or None
            }
            
            Note: x, y are normalized (0-1), where:
                  x=0 is left edge, x=1 is right edge
                  y=0 is top edge, y=1 is bottom edge
        """
        
        wrist_positions = {
            'left': None,
            'right': None
        }
        
        if results.left_hand_landmarks is not None:
            wrist = results.left_hand_landmarks.landmark[self.wrist_idx]
            wrist_positions['left'] = (wrist.x, wrist.y, wrist.z)
        
        if results.right_hand_landmarks is not None:
            wrist = results.right_hand_landmarks.landmark[self.wrist_idx]
            wrist_positions['right'] = (wrist.x, wrist.y, wrist.z)
        
        return wrist_positions
    
    def hands_detected(self, results):
        """
        Check if at least one hand is detected.
        
        Args:
            results: MediaPipe Holistic results
        
        Returns:
            bool: True if any hand is detected
        """
        left = results.left_hand_landmarks is not None
        right = results.right_hand_landmarks is not None
        return left or right
    
    def get_hand_count(self, results):
        """
        Count how many hands are detected.
        
        Args:
            results: MediaPipe Holistic results
        
        Returns:
            int: 0, 1, or 2
        """
        count = 0
        if results.left_hand_landmarks is not None:
            count += 1
        if results.right_hand_landmarks is not None:
            count += 1
        return count
    
    def get_palm_size(self, results):
        """
        Get the palm size (used for scale normalization) for each hand.
        
        This shows how big the hand appears in the frame.
        Useful for debugging normalization issues.
        
        Args:
            results: MediaPipe Holistic results
        
        Returns:
            dict: {
                'left': float or None (palm width in normalized coords),
                'right': float or None
            }
        """
        
        palm_sizes = {
            'left': None,
            'right': None
        }
        
        if results.left_hand_landmarks is not None:
            landmarks = results.left_hand_landmarks.landmark
            index_mcp = np.array([
                landmarks[self.index_mcp_idx].x,
                landmarks[self.index_mcp_idx].y,
                landmarks[self.index_mcp_idx].z
            ])
            pinky_mcp = np.array([
                landmarks[self.pinky_mcp_idx].x,
                landmarks[self.pinky_mcp_idx].y,
                landmarks[self.pinky_mcp_idx].z
            ])
            palm_sizes['left'] = calculate_distance(index_mcp, pinky_mcp)
        
        if results.right_hand_landmarks is not None:
            landmarks = results.right_hand_landmarks.landmark
            index_mcp = np.array([
                landmarks[self.index_mcp_idx].x,
                landmarks[self.index_mcp_idx].y,
                landmarks[self.index_mcp_idx].z
            ])
            pinky_mcp = np.array([
                landmarks[self.pinky_mcp_idx].x,
                landmarks[self.pinky_mcp_idx].y,
                landmarks[self.pinky_mcp_idx].z
            ])
            palm_sizes['right'] = calculate_distance(index_mcp, pinky_mcp)
        
        return palm_sizes


# =============================================================================
# QUICK TEST
# =============================================================================
# Run this file directly to test landmark extraction

if __name__ == "__main__":
    import cv2
    from core.holistic_detector import HolisticDetector
    
    print("=" * 60)
    print("LANDMARK EXTRACTOR TEST")
    print("=" * 60)
    print("\nOpening webcam to test feature extraction...")
    print("Press 'q' to quit.\n")
    
    # Open webcam
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        exit(1)
    
    # Create detector and extractor
    detector = HolisticDetector()
    extractor = LandmarkExtractor(Config)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame (mirror view)
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            results = detector.detect(frame)
            
            # Extract features
            features = extractor.extract_and_normalize(results)
            
            # Get additional info
            hand_count = extractor.get_hand_count(results)
            wrist_pos = extractor.get_wrist_position(results)
            palm_sizes = extractor.get_palm_size(results)
            
            # Draw info on frame
            y = 30
            cv2.putText(frame, f"Feature shape: {features.shape}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            y += 30
            cv2.putText(frame, f"Hands detected: {hand_count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show feature values (first few)
            y += 30
            cv2.putText(frame, f"First 3 values: [{features[0]:.3f}, {features[1]:.3f}, {features[2]:.3f}]", 
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show palm sizes
            y += 30
            left_palm = f"{palm_sizes['left']:.4f}" if palm_sizes['left'] else "N/A"
            right_palm = f"{palm_sizes['right']:.4f}" if palm_sizes['right'] else "N/A"
            cv2.putText(frame, f"Palm sizes - L: {left_palm}, R: {right_palm}", 
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Check if features are non-zero (hand detected and processed)
            y += 30
            non_zero = np.count_nonzero(features)
            cv2.putText(frame, f"Non-zero features: {non_zero}/126", 
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Landmark Extractor Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        
        print("\n" + "=" * 60)
        print("Landmark extractor test completed!")
        print("=" * 60)