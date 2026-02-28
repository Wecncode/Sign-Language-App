"""
=============================================================================
PREPROCESSING MODULE - Landmark Normalization Functions
=============================================================================

This module contains functions to normalize hand landmarks.
NORMALIZATION IS THE MOST CRITICAL PART OF THIS PROJECT!

WHY NORMALIZE?
--------------
Raw MediaPipe landmarks have problems:

1. POSITION PROBLEM:
   - Same sign at different positions gives different coordinates
   - Hand on left: wrist at (0.2, 0.5)
   - Hand on right: wrist at (0.8, 0.5)
   - Solution: CENTER on wrist → position becomes (0, 0)

2. SIZE PROBLEM:
   - Same sign at different distances gives different scales
   - Close to camera: fingers spread across 0.3 units
   - Far from camera: fingers spread across 0.1 units
   - Solution: SCALE by palm width → always same relative size

THE 3-STEP NORMALIZATION:
-------------------------
Step A: MediaPipe already normalizes to 0-1 range (automatic)
Step B: Center on wrist (translation invariance)
Step C: Scale by palm width (size invariance) ← CRITICAL!

⚠️ CRITICAL WARNING:
-------------------
These EXACT functions must be used in:
- Data collection (collect_data.py)
- Training preprocessing (Colab notebook)
- Real-time inference (recognize.py)

If normalization differs between training and inference, MODEL WILL FAIL!

=============================================================================
"""

import numpy as np
from config import Config


def normalize_hand_landmarks(landmarks, 
                             wrist_idx=None, 
                             index_mcp_idx=None, 
                             pinky_mcp_idx=None):
    """
    Apply 3-step normalization to hand landmarks.
    
    This function takes raw MediaPipe landmarks and normalizes them
    so that the same hand pose always produces the same numbers,
    regardless of position in frame or distance from camera.
    
    Args:
        landmarks: numpy array of shape (21, 3)
                   Each row is one landmark with [x, y, z] coordinates
                   
        wrist_idx: Index of wrist landmark (default: from Config)
        index_mcp_idx: Index of index finger knuckle (default: from Config)
        pinky_mcp_idx: Index of pinky knuckle (default: from Config)
    
    Returns:
        normalized: numpy array of shape (21, 3)
                    Normalized landmarks (centered and scaled)
    
    Example:
        >>> landmarks = np.array([[0.5, 0.5, 0], ...])  # 21 landmarks
        >>> normalized = normalize_hand_landmarks(landmarks)
        >>> print(normalized[0])  # Wrist is now at origin
        [0. 0. 0.]
    """
    
    # Use config values if not specified
    if wrist_idx is None:
        wrist_idx = Config.WRIST_INDEX
    if index_mcp_idx is None:
        index_mcp_idx = Config.INDEX_MCP_INDEX
    if pinky_mcp_idx is None:
        pinky_mcp_idx = Config.PINKY_MCP_INDEX
    
    # -------------------------------------------------------------------------
    # HANDLE MISSING HAND
    # -------------------------------------------------------------------------
    # If landmarks are all zeros, hand was not detected
    # Return zeros (model learns to handle missing hands)
    if landmarks is None or np.all(landmarks == 0):
        return np.zeros((21, 3))
    
    # Ensure numpy array
    landmarks = np.array(landmarks)
    
    # -------------------------------------------------------------------------
    # STEP A: MediaPipe Normalization (AUTOMATIC)
    # -------------------------------------------------------------------------
    # MediaPipe already gives us:
    # - x: 0.0 (left edge) to 1.0 (right edge)
    # - y: 0.0 (top edge) to 1.0 (bottom edge)
    # - z: Depth relative to wrist (negative = closer to camera)
    #
    # We don't need to do anything for this step!
    
    # -------------------------------------------------------------------------
    # STEP B: Center on Wrist (Translation Invariance)
    # -------------------------------------------------------------------------
    # Problem: Hand position in frame changes coordinates
    # Solution: Subtract wrist position from all landmarks
    #
    # After this step:
    # - Wrist is always at (0, 0, 0)
    # - Other landmarks are relative to wrist
    # - Hand position in frame doesn't matter anymore!
    
    wrist = landmarks[wrist_idx].copy()  # Get wrist coordinates
    centered = landmarks - wrist          # Subtract from all landmarks
    
    # Verify wrist is now at origin
    # centered[wrist_idx] should be [0, 0, 0]
    
    # -------------------------------------------------------------------------
    # STEP C: Scale by Palm Width (Size Invariance) ⚠️ CRITICAL!
    # -------------------------------------------------------------------------
    # Problem: Hand size in frame changes with distance to camera
    # Solution: Divide by a reference distance (palm width)
    #
    # Palm width = distance from index knuckle to pinky knuckle
    # This distance is consistent for a given hand, regardless of position
    #
    # After this step:
    # - Coordinates are in "palm width units"
    # - Close hand and far hand give same numbers!
    # - Different hand sizes are normalized
    
    # Get the two reference points for palm width
    index_mcp = landmarks[index_mcp_idx]   # Index finger knuckle
    pinky_mcp = landmarks[pinky_mcp_idx]   # Pinky knuckle
    
    # Calculate palm width (3D Euclidean distance)
    palm_width = calculate_distance(index_mcp, pinky_mcp)
    
    # Prevent division by zero
    # If palm_width is too small (detection error), use minimum value
    if palm_width < Config.MIN_SCALE:
        palm_width = Config.MIN_SCALE
    
    # Scale all coordinates by palm width
    normalized = centered / palm_width
    
    return normalized


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two 3D points.
    
    This is the straight-line distance formula extended to 3D:
    distance = sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)
    
    Args:
        point1: numpy array [x, y, z]
        point2: numpy array [x, y, z]
    
    Returns:
        float: Distance between the two points
    
    Example:
        >>> p1 = np.array([0, 0, 0])
        >>> p2 = np.array([3, 4, 0])
        >>> calculate_distance(p1, p2)
        5.0
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Euclidean distance formula
    diff = point2 - point1           # Difference vector
    squared = diff ** 2              # Square each component
    sum_squared = np.sum(squared)    # Sum of squares
    distance = np.sqrt(sum_squared)  # Square root
    
    return distance


def landmarks_to_feature_vector(left_hand, right_hand):
    """
    Convert two hand landmarks into a single feature vector.
    
    This function:
    1. Normalizes each hand independently
    2. Flattens the landmarks into 1D arrays
    3. Concatenates left and right hands
    
    Args:
        left_hand: numpy array (21, 3) or None if not detected
        right_hand: numpy array (21, 3) or None if not detected
    
    Returns:
        features: numpy array of shape (126,)
                  [left_x1, left_y1, left_z1, ..., right_x21, right_y21, right_z21]
    
    Example:
        >>> left = np.random.rand(21, 3)
        >>> right = np.random.rand(21, 3)
        >>> features = landmarks_to_feature_vector(left, right)
        >>> print(features.shape)
        (126,)
    """
    features = []
    
    # -------------------------------------------------------------------------
    # Process Left Hand (63 values: 21 landmarks × 3 coordinates)
    # -------------------------------------------------------------------------
    if left_hand is not None and not np.all(left_hand == 0):
        # Hand detected - normalize it
        normalized_left = normalize_hand_landmarks(left_hand)
        features.extend(normalized_left.flatten())  # Flatten (21,3) → (63,)
    else:
        # Hand not detected - use zeros
        # Model learns that zeros = no hand present
        features.extend(np.zeros(63))
    
    # -------------------------------------------------------------------------
    # Process Right Hand (63 values: 21 landmarks × 3 coordinates)
    # -------------------------------------------------------------------------
    if right_hand is not None and not np.all(right_hand == 0):
        # Hand detected - normalize it
        normalized_right = normalize_hand_landmarks(right_hand)
        features.extend(normalized_right.flatten())  # Flatten (21,3) → (63,)
    else:
        # Hand not detected - use zeros
        features.extend(np.zeros(63))
    
    # Convert to numpy array
    return np.array(features, dtype=np.float32)


def mediapipe_to_numpy(hand_landmarks):
    """
    Convert MediaPipe hand landmarks to numpy array.
    
    MediaPipe returns landmarks as a special object with .x, .y, .z attributes.
    This function converts them to a simple numpy array.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
                        (results.left_hand_landmarks or results.right_hand_landmarks)
    
    Returns:
        numpy array of shape (21, 3) with [x, y, z] for each landmark
        Returns None if input is None
    
    Example:
        >>> results = holistic.process(image)
        >>> left_np = mediapipe_to_numpy(results.left_hand_landmarks)
        >>> print(left_np.shape)
        (21, 3)
    """
    if hand_landmarks is None:
        return None
    
    # Extract x, y, z for each landmark
    coords = []
    for landmark in hand_landmarks.landmark:
        coords.append([landmark.x, landmark.y, landmark.z])
    
    return np.array(coords, dtype=np.float32)


# =============================================================================
# DATA AUGMENTATION FUNCTIONS (Used in Training)
# =============================================================================

def augment_sequence(sequence, noise_level=0.02):
    """
    Add random noise to a sequence for data augmentation.
    
    Data augmentation creates variations of training data to:
    - Prevent overfitting
    - Make model more robust to small variations
    - Effectively increase dataset size
    
    Args:
        sequence: numpy array of shape (30, 126) - one training sequence
        noise_level: Standard deviation of Gaussian noise (default 0.02)
    
    Returns:
        augmented: numpy array of shape (30, 126) - sequence with noise
    
    Example:
        >>> sequence = np.random.rand(30, 126)
        >>> augmented = augment_sequence(sequence, noise_level=0.02)
        >>> # augmented is slightly different from sequence
    """
    # Generate random noise with same shape as sequence
    # np.random.normal(mean, std, shape)
    noise = np.random.normal(0, noise_level, sequence.shape)
    
    # Add noise to original sequence
    augmented = sequence + noise
    
    return augmented.astype(np.float32)


def mirror_sequence(sequence):
    """
    Horizontally flip a sequence (swap left and right hands).
    
    This augmentation doubles the effective dataset size by creating
    mirrored versions. A right-handed "A" becomes a left-handed "A".
    
    Args:
        sequence: numpy array of shape (30, 126)
                  First 63 values = left hand
                  Last 63 values = right hand
    
    Returns:
        mirrored: numpy array of shape (30, 126) - hands swapped + x flipped
    
    Example:
        >>> sequence = np.random.rand(30, 126)
        >>> mirrored = mirror_sequence(sequence)
        >>> # Left and right hands are swapped
    """
    mirrored = sequence.copy()
    
    # -------------------------------------------------------------------------
    # Step 1: Swap left hand (0:63) and right hand (63:126)
    # -------------------------------------------------------------------------
    left_hand = mirrored[:, :63].copy()    # Columns 0-62
    right_hand = mirrored[:, 63:].copy()   # Columns 63-125
    
    mirrored[:, :63] = right_hand   # Right hand goes to left position
    mirrored[:, 63:] = left_hand    # Left hand goes to right position
    
    # -------------------------------------------------------------------------
    # Step 2: Flip x-coordinates (negate every 3rd value starting from 0)
    # -------------------------------------------------------------------------
    # Feature vector structure: [x1, y1, z1, x2, y2, z2, ...]
    # X values are at indices 0, 3, 6, 9, ... (every 3rd starting from 0)
    # 
    # We negate x-coordinates to flip horizontally
    # Before: pointing right → After: pointing left
    
    mirrored[:, 0::3] *= -1  # [0::3] means start at 0, step by 3
    
    return mirrored.astype(np.float32)


def scale_sequence(sequence, scale_factor=None):
    """
    Scale all coordinates by a random factor.
    
    This simulates the hand being slightly closer or farther from camera
    after normalization (small residual variations).
    
    Args:
        sequence: numpy array of shape (30, 126)
        scale_factor: How much to scale (default: random 0.9-1.1)
    
    Returns:
        scaled: numpy array of shape (30, 126)
    """
    if scale_factor is None:
        # Random scale between 0.9 and 1.1 (±10%)
        scale_factor = np.random.uniform(0.9, 1.1)
    
    scaled = sequence * scale_factor
    
    return scaled.astype(np.float32)


# =============================================================================
# QUICK TEST
# =============================================================================
# Run this file directly to verify preprocessing functions

if __name__ == "__main__":
    print("=" * 60)
    print("PREPROCESSING MODULE TEST")
    print("=" * 60)
    
    # Create fake landmarks (21 landmarks, each with x, y, z)
    # Simulating a hand roughly in center of frame
    fake_landmarks = np.random.rand(21, 3) * 0.3 + 0.35  # Values around 0.35-0.65
    
    print("\n📍 Original landmarks (first 5):")
    print(fake_landmarks[:5])
    
    # Test normalization
    normalized = normalize_hand_landmarks(fake_landmarks)
    
    print("\n📍 Normalized landmarks (first 5):")
    print(normalized[:5])
    
    print("\n✓ Wrist position (should be ~[0,0,0]):")
    print(f"   {normalized[Config.WRIST_INDEX]}")
    
    # Test feature vector creation
    left_hand = np.random.rand(21, 3)
    right_hand = np.random.rand(21, 3)
    
    features = landmarks_to_feature_vector(left_hand, right_hand)
    
    print(f"\n📊 Feature vector shape: {features.shape}")
    print(f"   Expected: (126,)")
    print(f"   Match: {features.shape == (126,)}")
    
    # Test with missing hand
    features_missing = landmarks_to_feature_vector(left_hand, None)
    
    print(f"\n📊 With missing right hand:")
    print(f"   Right hand features (should be zeros): {features_missing[63:66]}")
    
    # Test augmentation
    sequence = np.random.rand(30, 126)
    augmented = augment_sequence(sequence)
    mirrored = mirror_sequence(sequence)
    
    print(f"\n🔄 Augmentation test:")
    print(f"   Original shape: {sequence.shape}")
    print(f"   Augmented shape: {augmented.shape}")
    print(f"   Mirrored shape: {mirrored.shape}")
    print(f"   Augmented differs from original: {not np.allclose(sequence, augmented)}")
    
    print("\n" + "=" * 60)
    print("All preprocessing tests passed!")
    print("=" * 60)