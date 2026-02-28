"""
=============================================================================
HOLISTIC DETECTOR - MediaPipe Wrapper for Hand Detection
=============================================================================

FIXED: Left/Right hand swap issue when using mirrored video.

=============================================================================
"""

import cv2
import mediapipe as mp
from config import Config


class HolisticDetector:
    """
    Wrapper class for MediaPipe Holistic.
    """
    
    def __init__(self, 
                 min_detection_confidence=None, 
                 min_tracking_confidence=None,
                 static_image_mode=False,
                 flip_hands=True):  # NEW PARAMETER
        """
        Initialize MediaPipe Holistic detector.
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            static_image_mode: If True, treats each image independently
            flip_hands: If True, swap left/right hand labels (for mirrored video)
        """
        # Use config values if not specified
        if min_detection_confidence is None:
            min_detection_confidence = Config.MIN_DETECTION_CONFIDENCE
        if min_tracking_confidence is None:
            min_tracking_confidence = Config.MIN_TRACKING_CONFIDENCE
        
        # Store the flip setting
        self.flip_hands = flip_hands
        
        # Store references to MediaPipe modules
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize the Holistic model
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False
        )
        
        # Store settings for reference
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        print(f"✓ MediaPipe Holistic initialized")
        print(f"  Detection confidence: {min_detection_confidence}")
        print(f"  Tracking confidence: {min_tracking_confidence}")
        print(f"  Flip hands (for mirror view): {flip_hands}")
    
    def detect(self, frame):
        """
        Process a video frame and detect hands/face/pose.
        
        Args:
            frame: OpenCV image in BGR format
        
        Returns:
            results: MediaPipe results with left/right hands swapped if flip_hands=True
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Performance optimization
        rgb_frame.flags.writeable = False
        
        # Run detection
        results = self.holistic.process(rgb_frame)
        
        # Re-enable writeable
        rgb_frame.flags.writeable = True
        
        # =====================================================================
        # FIX: Swap left/right hands for mirrored video
        # =====================================================================
        if self.flip_hands:
            results = self._swap_hands(results)
        
        return results
    
    def _swap_hands(self, results):
        """
        Swap left and right hand landmarks.
        
        When video is mirrored (flipped horizontally), MediaPipe's left/right
        labels are backwards. This function swaps them to match user's perspective.
        
        Args:
            results: MediaPipe Holistic results
        
        Returns:
            results: Same results with left/right hands swapped
        """
        # Swap the hand landmarks
        # We use a simple swap: left becomes right, right becomes left
        left_temp = results.left_hand_landmarks
        results.left_hand_landmarks = results.right_hand_landmarks
        results.right_hand_landmarks = left_temp
        
        return results
    
    def hands_detected(self, results):
        """Check if any hands are detected."""
        left_detected = results.left_hand_landmarks is not None
        right_detected = results.right_hand_landmarks is not None
        return left_detected or right_detected
    
    def get_hand_count(self, results):
        """Count how many hands are detected."""
        count = 0
        if results.left_hand_landmarks is not None:
            count += 1
        if results.right_hand_landmarks is not None:
            count += 1
        return count
    
    def get_detection_info(self, results):
        """Get detailed information about what was detected."""
        return {
            'left_hand': results.left_hand_landmarks is not None,
            'right_hand': results.right_hand_landmarks is not None,
            'face': results.face_landmarks is not None,
            'pose': results.pose_landmarks is not None,
            'hand_count': self.get_hand_count(results)
        }
    
    def release(self):
        """Release MediaPipe resources."""
        self.holistic.close()
        print("✓ MediaPipe Holistic released")
    
    def __enter__(self):
        """Support for 'with' statement."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting 'with' block."""
        self.release()
        return False


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HOLISTIC DETECTOR TEST (with hand swap fix)")
    print("=" * 60)
    print("\nOpening webcam... Press 'q' to quit.\n")
    print("Raise your RIGHT hand - should show 'Right Hand: YES'")
    print("Raise your LEFT hand - should show 'Left Hand: YES'\n")
    
    # Open webcam
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        exit(1)
    
    # Create detector with flip_hands=True (for mirrored view)
    detector = HolisticDetector(flip_hands=True)
    
    # For FPS calculation
    import time
    prev_time = time.time()
    fps = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally (mirror view)
            frame = cv2.flip(frame, 1)
            
            # Detect hands (will swap left/right automatically)
            results = detector.detect(frame)
            
            # Get detection info
            info = detector.get_detection_info(results)
            
            # Draw landmarks
            if info['left_hand']:
                detector.mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    detector.mp_holistic.HAND_CONNECTIONS,
                    detector.mp_drawing_styles.get_default_hand_landmarks_style(),
                    detector.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            if info['right_hand']:
                detector.mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    detector.mp_holistic.HAND_CONNECTIONS,
                    detector.mp_drawing_styles.get_default_hand_landmarks_style(),
                    detector.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Draw info on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show hand detection status
            left_color = (0, 255, 0) if info['left_hand'] else (0, 0, 255)
            right_color = (0, 255, 0) if info['right_hand'] else (0, 0, 255)
            
            cv2.putText(frame, f"Left Hand: {'YES' if info['left_hand'] else 'NO'}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
            
            cv2.putText(frame, f"Right Hand: {'YES' if info['right_hand'] else 'NO'}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
            
            # Instructions
            cv2.putText(frame, "Raise RIGHT hand - should show 'Right Hand: YES'", 
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Raise LEFT hand - should show 'Left Hand: YES'", 
                        (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", 
                        (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Holistic Detector Test", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)