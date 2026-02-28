"""
=============================================================================
DATA COLLECTION APPLICATION
=============================================================================

This script records sign language samples for training the recognition model.

HOW IT WORKS:
-------------
1. Shows webcam feed with hand landmarks
2. User selects which sign to record
3. Press SPACE to record a 30-frame sequence (1 second)
4. Sequence is saved as .npy file
5. Repeat until enough samples collected

DATA SAVED:
-----------
    data/sequences/
    ├── _Neutral/
    │   ├── sequence_001.npy    # Shape: (30, 126)
    │   ├── sequence_002.npy
    │   └── ...
    ├── A/
    │   ├── sequence_001.npy
    │   └── ...
    └── Hello/
        ├── sequence_001.npy
        └── ...

CONTROLS:
---------
    SPACE     : Start/stop recording
    N         : Next sign
    P         : Previous sign
    R         : Retry (delete last sample)
    Q         : Quit
    0-9       : Jump to sign by number

TIPS FOR GOOD DATA:
-------------------
1. Record in different positions (left, center, right)
2. Vary your distance from camera
3. Record at different angles
4. Use good lighting
5. Record _Neutral with various resting positions

=============================================================================
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

# Import our modules
from config import Config
from core.holistic_detector import HolisticDetector
from core.landmark_extractor import LandmarkExtractor
from core.frame_buffer import FrameBuffer
from utils.visualization import (
    draw_landmarks,
    draw_sign_label,
    draw_recording_indicator,
    draw_countdown,
    draw_progress_bar,
    draw_instructions,
    draw_message
)


class DataCollector:
    """
    Main application class for collecting sign language data.
    
    This class handles:
    - Webcam capture
    - Hand detection
    - Recording sequences
    - Saving data files
    - User interface
    """
    
    def __init__(self):
        """
        Initialize the data collector.
        """
        
        print("=" * 60)
        print("SIGN LANGUAGE DATA COLLECTOR")
        print("=" * 60)
        
        # =====================================================================
        # SETUP DIRECTORIES
        # =====================================================================
        print("\n📁 Setting up directories...")
        Config.ensure_directories()
        
        # =====================================================================
        # INITIALIZE COMPONENTS
        # =====================================================================
        print("\n🔧 Initializing components...")
        
        # MediaPipe detector
        self.detector = HolisticDetector()
        
        # Landmark extractor
        self.extractor = LandmarkExtractor(Config)
        
        # Frame buffer for recording
        self.buffer = FrameBuffer(
            size=Config.SEQUENCE_LENGTH * 2,  # 2x for safety
            features_per_frame=Config.FEATURES_PER_FRAME
        )
        
        # =====================================================================
        # WEBCAM SETUP
        # =====================================================================
        print("\n📷 Opening webcam...")
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.FPS)
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam! Check if it's connected.")
        
        # =====================================================================
        # STATE VARIABLES
        # =====================================================================
        
        # Signs to collect
        self.signs = Config.SIGNS_TO_COLLECT
        self.current_sign_index = 0
        
        # Recording state
        self.is_recording = False
        self.countdown_active = False
        self.countdown_start = 0
        self.recording_start = 0
        
        # Messages
        self.message = ""
        self.message_type = "info"
        self.message_time = 0
        
        # Sample counts (loaded from existing files)
        self.sample_counts = self._count_existing_samples()
        
        # FPS calculation
        self.fps = 0
        self.prev_time = time.time()
        
        print("\n✅ Data collector ready!")
        print(f"   Signs to collect: {len(self.signs)}")
        print(f"   Press SPACE to start recording")
    
    def _count_existing_samples(self):
        """
        Count how many samples already exist for each sign.
        
        Returns:
            dict: {sign_name: count}
        """
        
        counts = {}
        
        for sign in self.signs:
            sign_path = Config.get_data_path(sign)
            
            if os.path.exists(sign_path):
                # Count .npy files
                files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
                counts[sign] = len(files)
            else:
                counts[sign] = 0
        
        return counts
    
    def _get_next_filename(self, sign):
        """
        Get the next available filename for a sign.
        
        Args:
            sign: Sign name (e.g., "A", "Hello")
        
        Returns:
            str: Full path like "data/sequences/A/sequence_047.npy"
        """
        
        sign_path = Config.get_data_path(sign)
        count = self.sample_counts.get(sign, 0) + 1
        
        # Keep incrementing until we find an unused filename
        while True:
            filename = f"sequence_{count:03d}.npy"
            filepath = os.path.join(sign_path, filename)
            
            if not os.path.exists(filepath):
                return filepath
            
            count += 1
    
    def _get_target_samples(self, sign):
        """
        Get target sample count for a sign.
        
        Args:
            sign: Sign name
        
        Returns:
            int: Target number of samples
        """
        
        if sign == Config.NEUTRAL_LABEL:
            return Config.SAMPLES_FOR_NEUTRAL
        else:
            return Config.SAMPLES_PER_SIGN
    
    @property
    def current_sign(self):
        """Get the currently selected sign name."""
        return self.signs[self.current_sign_index]
    
    def _show_message(self, text, msg_type="info", duration=2.0):
        """
        Show a temporary message on screen.
        
        Args:
            text: Message text
            msg_type: "info", "success", "warning", or "error"
            duration: How long to show (seconds)
        """
        
        self.message = text
        self.message_type = msg_type
        self.message_time = time.time() + duration
    
    def _start_countdown(self):
        """Start countdown before recording."""
        
        self.countdown_active = True
        self.countdown_start = time.time()
        self.buffer.clear()  # Clear buffer before recording
    
    def _start_recording(self):
        """Start actual recording after countdown."""
        
        self.countdown_active = False
        self.is_recording = True
        self.recording_start = time.time()
        self.buffer.clear()  # Ensure buffer is empty
        self._show_message("Recording...", "info", 2.0)
    
    def _stop_recording(self, save=True):
        """
        Stop recording and optionally save the sequence.
        
        Args:
            save: Whether to save the recorded sequence
        """
        
        self.is_recording = False
        
        if save and self.buffer.is_ready():
            # Get the recorded sequence
            sequence = self.buffer.get_last_n(Config.SEQUENCE_LENGTH)
            
            if sequence is not None:
                # Save to file
                filepath = self._get_next_filename(self.current_sign)
                np.save(filepath, sequence)
                
                # Update count
                self.sample_counts[self.current_sign] = \
                    self.sample_counts.get(self.current_sign, 0) + 1
                
                count = self.sample_counts[self.current_sign]
                target = self._get_target_samples(self.current_sign)
                
                self._show_message(
                    f"Saved! ({count}/{target})",
                    "success",
                    1.5
                )
            else:
                self._show_message("Not enough frames!", "error", 2.0)
        
        # Clear buffer for next recording
        self.buffer.clear()
    
    def _delete_last_sample(self):
        """Delete the most recently saved sample for current sign."""
        
        sign_path = Config.get_data_path(self.current_sign)
        
        if not os.path.exists(sign_path):
            self._show_message("No samples to delete!", "warning", 2.0)
            return
        
        # Find most recent file
        files = sorted([f for f in os.listdir(sign_path) if f.endswith('.npy')])
        
        if not files:
            self._show_message("No samples to delete!", "warning", 2.0)
            return
        
        # Delete last file
        last_file = os.path.join(sign_path, files[-1])
        os.remove(last_file)
        
        # Update count
        self.sample_counts[self.current_sign] = max(0, 
            self.sample_counts.get(self.current_sign, 0) - 1)
        
        self._show_message(f"Deleted {files[-1]}", "warning", 2.0)
    
    def _next_sign(self):
        """Move to next sign."""
        
        self.current_sign_index = (self.current_sign_index + 1) % len(self.signs)
        self.buffer.clear()
        self._show_message(f"Now recording: {self.current_sign}", "info", 2.0)
    
    def _prev_sign(self):
        """Move to previous sign."""
        
        self.current_sign_index = (self.current_sign_index - 1) % len(self.signs)
        self.buffer.clear()
        self._show_message(f"Now recording: {self.current_sign}", "info", 2.0)
    
    def _jump_to_sign(self, index):
        """Jump to a specific sign by index."""
        
        if 0 <= index < len(self.signs):
            self.current_sign_index = index
            self.buffer.clear()
            self._show_message(f"Now recording: {self.current_sign}", "info", 2.0)
    
    def _process_frame(self, frame):
        """
        Process a single frame: detect hands and extract features.
        
        Args:
            frame: OpenCV BGR frame
        
        Returns:
            results: MediaPipe detection results
            features: Extracted features (126,) or None
        """
        
        # Detect hands
        results = self.detector.detect(frame)
        
        # Extract features if hand detected
        if self.extractor.hands_detected(results):
            features = self.extractor.extract_and_normalize(results)
        else:
            features = None
        
        return results, features
    
    def _update_recording(self, features):
        """
        Update recording state and buffer.
        
        Args:
            features: Extracted features or None
        """
        
        # =====================================================================
        # COUNTDOWN PHASE
        # =====================================================================
        if self.countdown_active:
            elapsed = time.time() - self.countdown_start
            
            if elapsed >= Config.COUNTDOWN_SECONDS:
                # Countdown finished, start recording
                self._start_recording()
        
        # =====================================================================
        # RECORDING PHASE
        # =====================================================================
        elif self.is_recording:
            if features is not None:
                # Add frame to buffer
                self.buffer.add(features)
                
                # Check if we have enough frames
                if len(self.buffer) >= Config.SEQUENCE_LENGTH:
                    self._stop_recording(save=True)
            else:
                # Lost hand tracking during recording
                # Continue recording, will fill with zeros or retry
                # For simplicity, we add zeros
                zero_features = np.zeros(Config.FEATURES_PER_FRAME, dtype=np.float32)
                self.buffer.add(zero_features)
                
                if len(self.buffer) >= Config.SEQUENCE_LENGTH:
                    self._stop_recording(save=True)
    
    def _draw_ui(self, frame, results):
        """
        Draw all UI elements on the frame.
        
        Args:
            frame: OpenCV BGR frame
            results: MediaPipe detection results
        
        Returns:
            frame: Frame with UI drawn
        """
        
        height, width = frame.shape[:2]
        
        # =====================================================================
        # DRAW LANDMARKS
        # =====================================================================
        if Config.SHOW_LANDMARKS:
            frame = draw_landmarks(frame, results)
        
        # =====================================================================
        # DRAW SIGN LABEL
        # =====================================================================
        target = self._get_target_samples(self.current_sign)
        count = self.sample_counts.get(self.current_sign, 0)
        frame = draw_sign_label(frame, self.current_sign, count, target)
        
        # =====================================================================
        # DRAW RECORDING INDICATOR
        # =====================================================================
        frame = draw_recording_indicator(frame, self.is_recording)
        
        # =====================================================================
        # DRAW COUNTDOWN
        # =====================================================================
        if self.countdown_active:
            elapsed = time.time() - self.countdown_start
            remaining = Config.COUNTDOWN_SECONDS - elapsed
            if remaining > 0:
                frame = draw_countdown(frame, remaining)
        
        # =====================================================================
        # DRAW PROGRESS BAR (during recording)
        # =====================================================================
        if self.is_recording:
            progress = len(self.buffer) / Config.SEQUENCE_LENGTH
            frame = draw_progress_bar(
                frame,
                progress,
                position=(width // 2 - 100, height - 120),
                label="Recording",
                width=200
            )
        
        # =====================================================================
        # DRAW MESSAGE
        # =====================================================================
        if time.time() < self.message_time:
            frame = draw_message(frame, self.message, self.message_type)
        
        # =====================================================================
        # DRAW INSTRUCTIONS
        # =====================================================================
        instructions = [
            f"[SPACE] Record | [N] Next | [P] Prev | [R] Retry | [Q] Quit | Current: {self.current_sign_index + 1}/{len(self.signs)}"
        ]
        frame = draw_instructions(frame, instructions)
        
        # =====================================================================
        # DRAW FPS
        # =====================================================================
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (width - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # =====================================================================
        # DRAW HAND STATUS
        # =====================================================================
        hand_status = "Hands: "
        if results.left_hand_landmarks:
            hand_status += "L "
        if results.right_hand_landmarks:
            hand_status += "R"
        if not self.extractor.hands_detected(results):
            hand_status += "None"
        
        cv2.putText(
            frame,
            hand_status,
            (width - 100, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1
        )
        
        return frame
    
    def _handle_input(self, key):
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
        
        Returns:
            bool: False if should quit, True otherwise
        """
        
        if key == ord('q') or key == ord('Q'):
            # Quit
            return False
        
        elif key == ord(' '):
            # Spacebar - toggle recording
            if self.is_recording:
                self._stop_recording(save=False)
                self._show_message("Recording cancelled", "warning", 1.5)
            elif self.countdown_active:
                self.countdown_active = False
                self._show_message("Countdown cancelled", "warning", 1.5)
            else:
                self._start_countdown()
        
        elif key == ord('n') or key == ord('N'):
            # Next sign
            if not self.is_recording and not self.countdown_active:
                self._next_sign()
        
        elif key == ord('p') or key == ord('P'):
            # Previous sign
            if not self.is_recording and not self.countdown_active:
                self._prev_sign()
        
        elif key == ord('r') or key == ord('R'):
            # Retry (delete last)
            if not self.is_recording and not self.countdown_active:
                self._delete_last_sample()
        
        elif ord('0') <= key <= ord('9'):
            # Jump to sign by number
            if not self.is_recording and not self.countdown_active:
                index = key - ord('0')
                self._jump_to_sign(index)
        
        return True
    
    def run(self):
        """
        Main loop - run the data collection application.
        """
        
        print("\n" + "=" * 60)
        print("Starting data collection...")
        print("Press 'Q' to quit")
        print("=" * 60 + "\n")
        
        try:
            while True:
                # =============================================================
                # READ FRAME
                # =============================================================
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Could not read frame!")
                    break
                
                # Flip horizontally (mirror view - more intuitive)
                frame = cv2.flip(frame, 1)
                
                # =============================================================
                # CALCULATE FPS
                # =============================================================
                current_time = time.time()
                self.fps = 1 / max(current_time - self.prev_time, 0.001)
                self.prev_time = current_time
                
                # =============================================================
                # PROCESS FRAME
                # =============================================================
                results, features = self._process_frame(frame)
                
                # =============================================================
                # UPDATE RECORDING
                # =============================================================
                self._update_recording(features)
                
                # =============================================================
                # DRAW UI
                # =============================================================
                frame = self._draw_ui(frame, results)
                
                # =============================================================
                # DISPLAY
                # =============================================================
                cv2.imshow("Sign Language Data Collector", frame)
                
                # =============================================================
                # HANDLE INPUT
                # =============================================================
                key = cv2.waitKey(1) & 0xFF
                
                if not self._handle_input(key):
                    break
        
        finally:
            # =================================================================
            # CLEANUP
            # =================================================================
            self.cap.release()
            cv2.destroyAllWindows()
            self.detector.release()
            
            # Print summary
            print("\n" + "=" * 60)
            print("DATA COLLECTION SUMMARY")
            print("=" * 60)
            
            total_samples = 0
            for sign in self.signs:
                count = self.sample_counts.get(sign, 0)
                target = self._get_target_samples(sign)
                status = "✓" if count >= target else "○"
                print(f"  {status} {sign}: {count}/{target}")
                total_samples += count
            
            print("-" * 60)
            print(f"  Total samples: {total_samples}")
            print("=" * 60)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main function to run the data collector."""
    
    try:
        collector = DataCollector()
        collector.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise


if __name__ == "__main__":
    main()