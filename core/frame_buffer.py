"""
FRAME BUFFER - Circular Buffer for Storing Landmark Sequences
"""

import numpy as np

# Import Config carefully to avoid circular imports
try:
    from config import Config
except ImportError:
    # Fallback defaults if Config not available
    class Config:
        BUFFER_SIZE = 60
        FEATURES_PER_FRAME = 126
        SEQUENCE_LENGTH = 30


class FrameBuffer:
    """
    Circular buffer for storing landmark feature sequences.
    """
    
    def __init__(self, size=None, features_per_frame=None):
        """
        Initialize the frame buffer.
        
        Args:
            size: Maximum frames to store (default: 60)
            features_per_frame: Features per frame (default: 126)
        """
        # Use config values if not specified
        if size is None:
            size = Config.BUFFER_SIZE
        if features_per_frame is None:
            features_per_frame = Config.FEATURES_PER_FRAME
            
        self.size = size
        self.features_per_frame = features_per_frame
        
        # Initialize empty buffer
        self.buffer = []
        
        # Track total frames added
        self.frame_count = 0
        
        print(f"✓ FrameBuffer initialized (size: {self.size})")
    
    def add(self, frame):
        """
        Add a frame to the buffer.
        """
        # Convert to numpy array
        frame = np.array(frame, dtype=np.float32)
        
        # Add frame to buffer
        if len(self.buffer) < self.size:
            self.buffer.append(frame)
        else:
            self.buffer.pop(0)
            self.buffer.append(frame)
        
        self.frame_count += 1
    
    def get_last_n(self, n):
        """
        Get the last n frames as a numpy array.
        """
        if len(self.buffer) < n:
            return None
        
        last_n = self.buffer[-n:]
        return np.array(last_n, dtype=np.float32)
    
    def get_all(self):
        """
        Get all frames currently in buffer.
        """
        if len(self.buffer) == 0:
            return None
        return np.array(self.buffer, dtype=np.float32)
    
    def get_motion_score(self, n_frames=5):
        """
        Calculate how much the hand has moved in recent frames.
        """
        if len(self.buffer) < n_frames + 1:
            return 0.0
        
        recent = self.buffer[-(n_frames + 1):]
        
        distances = []
        for i in range(len(recent) - 1):
            frame1 = np.array(recent[i])
            frame2 = np.array(recent[i + 1])
            diff = frame2 - frame1
            distance = np.sqrt(np.sum(diff ** 2))
            distances.append(distance)
        
        return np.mean(distances)
    
    def is_ready(self, required_frames=None):
        """
        Check if buffer has enough frames for prediction.
        """
        if required_frames is None:
            required_frames = Config.SEQUENCE_LENGTH
        return len(self.buffer) >= required_frames
    
    def clear(self):
        """
        Clear all frames from buffer.
        """
        self.buffer = []
    
    def __len__(self):
        """
        Get current number of frames in buffer.
        """
        return len(self.buffer)
    
    def get_stats(self):
        """
        Get buffer statistics.
        """
        return {
            'current_size': len(self.buffer),
            'max_size': self.size,
            'total_frames_added': self.frame_count,
            'is_full': len(self.buffer) >= self.size,
            'is_ready': self.is_ready()
        }


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing FrameBuffer...")
    
    buffer = FrameBuffer(size=5, features_per_frame=126)
    
    for i in range(7):
        fake_frame = np.ones(126) * i
        buffer.add(fake_frame)
        print(f"Added frame {i}, buffer length: {len(buffer)}")
    
    print(f"Buffer stats: {buffer.get_stats()}")
    print("Test completed!")