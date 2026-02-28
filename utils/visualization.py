"""
=============================================================================
VISUALIZATION MODULE - Drawing Functions for UI
=============================================================================

This module contains functions to draw on video frames:
- Hand landmarks (dots and connections)
- Information panels (status, predictions, instructions)
- Progress bars and indicators

WHY SEPARATE VISUALIZATION?
---------------------------
Keeping drawing code separate from logic code makes both easier to:
- Read and understand
- Modify (change colors, positions without touching logic)
- Test (logic works even without visualization)

OPENCV DRAWING BASICS:
----------------------
OpenCV draws on images using these functions:
- cv2.circle(): Draw a dot
- cv2.line(): Draw a line
- cv2.rectangle(): Draw a box
- cv2.putText(): Draw text

All coordinates are (x, y) where:
- x = 0 is LEFT edge
- y = 0 is TOP edge
- Colors are (Blue, Green, Red) format, values 0-255

=============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
from config import Config


# =============================================================================
# MEDIAPIPE DRAWING SETUP
# =============================================================================
# MediaPipe provides built-in drawing utilities
# We'll use these for drawing hand landmarks

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def draw_landmarks(frame, results, show_hands=True, show_face=False):
    """
    Draw detected landmarks on the video frame.
    
    This function draws:
    - Colored dots on each landmark point
    - Lines connecting the landmarks (skeleton)
    
    Args:
        frame: OpenCV image (numpy array) to draw on
               This is modified in-place!
        results: MediaPipe Holistic detection results
        show_hands: Whether to draw hand landmarks (default: True)
        show_face: Whether to draw face landmarks (default: False)
    
    Returns:
        frame: The same frame with landmarks drawn on it
    
    Example:
        results = detector.detect(frame)
        frame = draw_landmarks(frame, results)
        cv2.imshow("Window", frame)
    
    Visual result:
        ┌─────────────────────────────┐
        │                             │
        │         ●───●───●           │
        │        /    │               │
        │       ●     ●               │
        │      /                      │
        │     ●──●                    │
        │     Hand with landmarks     │
        └─────────────────────────────┘
    """
    
    # =========================================================================
    # DRAW HAND LANDMARKS
    # =========================================================================
    if show_hands:
        # Draw LEFT hand (if detected)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.left_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                # Landmark style (the dots)
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),      # Green color (BGR)
                    thickness=2,             # Dot thickness
                    circle_radius=3          # Dot size
                ),
                # Connection style (the lines between dots)
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 200, 0),       # Slightly darker green
                    thickness=2              # Line thickness
                )
            )
        
        # Draw RIGHT hand (if detected)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.right_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),       # Green
                    thickness=2,
                    circle_radius=3
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 200, 0),
                    thickness=2
                )
            )
    
    # =========================================================================
    # DRAW FACE LANDMARKS (Phase 2 - currently disabled by default)
    # =========================================================================
    if show_face and results.face_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,  # Don't draw individual points (too many!)
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
    
    return frame


def draw_hand_box(frame, results):
    """
    Draw a bounding box around detected hands.
    
    This helps users see where their hands are in the frame
    and whether they're being detected properly.
    
    Args:
        frame: OpenCV image to draw on
        results: MediaPipe Holistic detection results
    
    Returns:
        frame: Frame with bounding boxes drawn
    
    Visual result:
        ┌─────────────────────────────┐
        │                             │
        │      ┌─────────┐            │
        │      │  ✋     │            │
        │      │         │            │
        │      └─────────┘            │
        │      Bounding box           │
        └─────────────────────────────┘
    """
    
    # Get frame dimensions
    height, width, _ = frame.shape
    
    # Process each hand
    for hand_landmarks, label in [
        (results.left_hand_landmarks, "Left"),
        (results.right_hand_landmarks, "Right")
    ]:
        if hand_landmarks is None:
            continue
        
        # Find min/max coordinates to create bounding box
        x_coords = []
        y_coords = []
        
        for landmark in hand_landmarks.landmark:
            # Convert normalized (0-1) to pixel coordinates
            x_coords.append(int(landmark.x * width))
            y_coords.append(int(landmark.y * height))
        
        # Calculate box corners with padding
        padding = 20  # Pixels of padding around hand
        x_min = max(0, min(x_coords) - padding)
        x_max = min(width, max(x_coords) + padding)
        y_min = max(0, min(y_coords) - padding)
        y_max = min(height, max(y_coords) + padding)
        
        # Draw rectangle
        cv2.rectangle(
            frame,
            (x_min, y_min),          # Top-left corner
            (x_max, y_max),          # Bottom-right corner
            (0, 255, 0),             # Green color
            2                         # Line thickness
        )
        
        # Draw label
        cv2.putText(
            frame,
            f"{label} Hand",
            (x_min, y_min - 10),      # Position above box
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,                       # Font scale
            (0, 255, 0),               # Green
            1                          # Thickness
        )
    
    return frame


def draw_info_panel(frame, info_dict, position="top"):
    """
    Draw an information panel on the frame.
    
    This displays useful information like:
    - Current status
    - FPS
    - Detection info
    - Instructions
    
    Args:
        frame: OpenCV image to draw on
        info_dict: Dictionary with info to display
                   Example: {"Status": "Recording", "FPS": 30}
        position: Where to draw - "top" or "bottom"
    
    Returns:
        frame: Frame with info panel drawn
    
    Visual result:
        ┌─────────────────────────────┐
        │ ┌─────────────────────────┐ │
        │ │ Status: Recording       │ │
        │ │ FPS: 30                 │ │
        │ │ Samples: 45/100         │ │
        │ └─────────────────────────┘ │
        │                             │
        │         Video area          │
        │                             │
        └─────────────────────────────┘
    """
    
    height, width, _ = frame.shape
    
    # Calculate panel dimensions
    line_height = 25           # Pixels per line of text
    padding = 10               # Padding inside panel
    num_lines = len(info_dict)
    panel_height = (num_lines * line_height) + (padding * 2)
    panel_width = 300
    
    # Determine panel position
    if position == "top":
        panel_x = 10
        panel_y = 10
    else:  # bottom
        panel_x = 10
        panel_y = height - panel_height - 10
    
    # Draw semi-transparent background
    # Create overlay for transparency effect
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (0, 0, 0),               # Black background
        -1                        # -1 means filled rectangle
    )
    
    # Blend overlay with original frame (0.7 = 70% opacity)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw border
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (100, 100, 100),         # Gray border
        1                         # Border thickness
    )
    
    # Draw each info line
    y_offset = panel_y + padding + 15  # Starting y position for text
    
    for key, value in info_dict.items():
        text = f"{key}: {value}"
        
        cv2.putText(
            frame,
            text,
            (panel_x + padding, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,                  # Font scale
            (255, 255, 255),      # White text
            1                     # Thickness
        )
        
        y_offset += line_height
    
    return frame


def draw_progress_bar(frame, progress, position, label="Progress", 
                      width=200, height=20, color=(0, 255, 0)):
    """
    Draw a progress bar on the frame.
    
    Args:
        frame: OpenCV image to draw on
        progress: Float from 0.0 to 1.0 (0% to 100%)
        position: Tuple (x, y) - top-left corner of progress bar
        label: Text label to show (default: "Progress")
        width: Width of progress bar in pixels
        height: Height of progress bar in pixels
        color: Fill color (BGR tuple)
    
    Returns:
        frame: Frame with progress bar drawn
    
    Visual result:
        Progress: ████████████░░░░░░░░ 60%
    """
    
    x, y = position
    
    # Clamp progress to valid range
    progress = max(0.0, min(1.0, progress))
    
    # Draw label
    cv2.putText(
        frame,
        f"{label}:",
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    # Draw background (empty bar)
    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (50, 50, 50),            # Dark gray background
        -1                        # Filled
    )
    
    # Draw filled portion
    filled_width = int(width * progress)
    if filled_width > 0:
        cv2.rectangle(
            frame,
            (x, y),
            (x + filled_width, y + height),
            color,                # Fill color
            -1                    # Filled
        )
    
    # Draw border
    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (255, 255, 255),         # White border
        1                         # Border thickness
    )
    
    # Draw percentage text
    percentage = int(progress * 100)
    cv2.putText(
        frame,
        f"{percentage}%",
        (x + width + 10, y + height - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    return frame


def draw_recording_indicator(frame, is_recording):
    """
    Draw a recording indicator (red dot when recording).
    
    Args:
        frame: OpenCV image to draw on
        is_recording: Boolean - True if currently recording
    
    Returns:
        frame: Frame with recording indicator
    
    Visual result:
        ┌─────────────────────────────┐
        │                        ● REC│  (red when recording)
        │                             │
        └─────────────────────────────┘
    """
    
    height, width, _ = frame.shape
    
    # Position in top-right corner
    x = width - 80
    y = 30
    
    if is_recording:
        # Draw red circle (recording)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)  # Red, filled
        cv2.putText(frame, "REC", (x + 20, y + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        # Draw gray circle (not recording)
        cv2.circle(frame, (x, y), 10, (100, 100, 100), -1)  # Gray, filled
        cv2.putText(frame, "IDLE", (x + 20, y + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return frame


def draw_countdown(frame, seconds_remaining):
    """
    Draw a large countdown number in the center of the frame.
    
    Used before recording starts to give user time to prepare.
    
    Args:
        frame: OpenCV image to draw on
        seconds_remaining: Integer countdown number (3, 2, 1...)
    
    Returns:
        frame: Frame with countdown displayed
    
    Visual result:
        ┌─────────────────────────────┐
        │                             │
        │             3               │
        │                             │
        │       Get ready...          │
        └─────────────────────────────┘
    """
    
    height, width, _ = frame.shape
    
    # Draw semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw countdown number
    text = str(int(seconds_remaining))
    
    # Get text size to center it
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    thickness = 10
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Center position
    x = (width - text_width) // 2
    y = (height + text_height) // 2
    
    # Draw text with outline for visibility
    # Black outline
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 5)
    # White text
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    # Draw "Get ready..." text below
    ready_text = "Get ready..."
    (ready_width, _), _ = cv2.getTextSize(ready_text, font, 1, 2)
    ready_x = (width - ready_width) // 2
    cv2.putText(frame, ready_text, (ready_x, y + 60), font, 1, (255, 255, 255), 2)
    
    return frame


def draw_instructions(frame, instructions, position="bottom"):
    """
    Draw instruction text at the bottom of the frame.
    
    Args:
        frame: OpenCV image to draw on
        instructions: List of instruction strings
        position: "top" or "bottom"
    
    Returns:
        frame: Frame with instructions displayed
    
    Visual result:
        ┌─────────────────────────────┐
        │                             │
        │         Video area          │
        │                             │
        │─────────────────────────────│
        │ [SPACE] Record  [N] Next    │
        │ [Q] Quit                    │
        └─────────────────────────────┘
    """
    
    height, width, _ = frame.shape
    
    line_height = 25
    padding = 10
    num_lines = len(instructions)
    panel_height = (num_lines * line_height) + (padding * 2)
    
    # Position at bottom
    if position == "bottom":
        panel_y = height - panel_height - 10
    else:
        panel_y = 10
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (10, panel_y),
        (width - 10, panel_y + panel_height),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw each instruction line
    y_offset = panel_y + padding + 15
    
    for instruction in instructions:
        cv2.putText(
            frame,
            instruction,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        y_offset += line_height
    
    return frame


def draw_sign_label(frame, sign_name, samples_collected, samples_target):
    """
    Draw the current sign being collected with sample count.
    
    Args:
        frame: OpenCV image to draw on
        sign_name: Name of current sign (e.g., "A", "Hello")
        samples_collected: How many samples collected so far
        samples_target: Target number of samples
    
    Returns:
        frame: Frame with sign label displayed
    
    Visual result:
        ┌─────────────────────────────┐
        │  ╔═══════════════════════╗  │
        │  ║  Sign: A              ║  │
        │  ║  Samples: 45/100      ║  │
        │  ╚═══════════════════════╝  │
        │                             │
        └─────────────────────────────┘
    """
    
    height, width, _ = frame.shape
    
    # Panel dimensions
    panel_width = 250
    panel_height = 70
    panel_x = (width - panel_width) // 2  # Center horizontally
    panel_y = 10
    
    # Draw background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (50, 50, 50),
        -1
    )
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Draw border
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (0, 255, 0),
        2
    )
    
    # Draw sign name
    cv2.putText(
        frame,
        f"Sign: {sign_name}",
        (panel_x + 15, panel_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    # Draw sample count
    cv2.putText(
        frame,
        f"Samples: {samples_collected}/{samples_target}",
        (panel_x + 15, panel_y + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    return frame


def draw_message(frame, message, message_type="info"):
    """
    Draw a message overlay on the frame.
    
    Args:
        frame: OpenCV image to draw on
        message: Text message to display
        message_type: "info", "success", "warning", or "error"
    
    Returns:
        frame: Frame with message displayed
    """
    
    height, width, _ = frame.shape
    
    # Choose color based on message type
    colors = {
        "info": (255, 200, 0),      # Yellow/orange
        "success": (0, 255, 0),     # Green
        "warning": (0, 165, 255),   # Orange
        "error": (0, 0, 255)        # Red
    }
    color = colors.get(message_type, colors["info"])
    
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(message, font, font_scale, thickness)
    
    # Position at center-bottom
    x = (width - text_width) // 2
    y = height - 80
    
    # Draw background
    padding = 10
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + padding),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(frame, message, (x, y), font, font_scale, color, thickness)
    
    return frame


# =============================================================================
# QUICK TEST
# =============================================================================
# Run this file directly to test visualization functions

if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZATION MODULE TEST")
    print("=" * 60)
    print("\nOpening webcam to test drawing functions...")
    print("Press 'q' to quit.\n")
    
    import time
    from core.holistic_detector import HolisticDetector
    
    # Open webcam
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        exit(1)
    
    # Create detector
    detector = HolisticDetector()
    
    # Test variables
    sample_count = 0
    is_recording = False
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame (mirror view)
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            results = detector.detect(frame)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1 / max(elapsed, 0.001)
            start_time = time.time()
            
            # Toggle recording with spacebar (for testing)
            # We check in the key handler below
            
            # Draw all visualizations
            frame = draw_landmarks(frame, results)
            frame = draw_hand_box(frame, results)
            frame = draw_recording_indicator(frame, is_recording)
            frame = draw_sign_label(frame, "A", sample_count, 100)
            
            # Draw info panel
            info = {
                "FPS": f"{fps:.1f}",
                "Left Hand": "Yes" if results.left_hand_landmarks else "No",
                "Right Hand": "Yes" if results.right_hand_landmarks else "No"
            }
            frame = draw_info_panel(frame, info, position="top")
            
            # Draw progress bar
            frame = draw_progress_bar(
                frame, 
                sample_count / 100,
                position=(10, frame.shape[0] - 100),
                label="Samples"
            )
            
            # Draw instructions
            instructions = [
                "[SPACE] Toggle recording  |  [N] Next sign  |  [Q] Quit"
            ]
            frame = draw_instructions(frame, instructions)
            
            # Show frame
            cv2.imshow("Visualization Test", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar
                is_recording = not is_recording
                if is_recording:
                    sample_count += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        
        print("\n" + "=" * 60)
        print("Visualization test completed!")
        print("=" * 60)