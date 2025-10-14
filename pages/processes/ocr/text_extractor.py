"""
OCR Text Extraction Module
Extracts on-screen text from video frames using EasyOCR.
This module is independent of transcription and analysis modules.
"""

import os
import cv2
import tempfile
from typing import List, Dict, Tuple
import numpy as np


class VideoTextExtractor:
    """
    Extract text from video frames using OCR.
    Designed to work independently without affecting existing modules.
    """
    
    def __init__(self, languages=['en'], gpu=False):
        """
        Initialize the OCR reader.
        
        Args:
            languages: List of language codes (e.g., ['en', 'es', 'zh'])
            gpu: Whether to use GPU (False for CPU mode)
        """
        self._reader = None
        self.languages = languages
        self.gpu = gpu
        
    def _get_reader(self):
        """Lazy load the EasyOCR reader (downloads models on first use)."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader
    
    def extract_frames(self, video_path: str, fps: float = 1.0) -> List[np.ndarray]:
        """
        Extract frames from video at specified FPS.
        
        Args:
            video_path: Path to video file
            fps: Frames per second to extract (1.0 = one frame per second)
            
        Returns:
            List of frame images as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames.append(frame)
                
            frame_count += 1
        
        cap.release()
        return frames
    
    def extract_text_from_frame(self, frame: np.ndarray) -> List[Tuple[str, float]]:
        """
        Extract text from a single frame.
        
        Args:
            frame: Image as numpy array (OpenCV format)
            
        Returns:
            List of (text, confidence) tuples
        """
        reader = self._get_reader()
        results = reader.readtext(frame)
        
        # Filter out low confidence detections
        filtered = [(text, conf) for (bbox, text, conf) in results if conf > 0.3]
        return filtered
    
    def extract_text_from_video(
        self, 
        video_path: str, 
        sample_fps: float = 1.0,
        min_confidence: float = 0.5
    ) -> Dict[str, any]:
        """
        Extract all text from video by sampling frames.
        
        Args:
            video_path: Path to video file
            sample_fps: Frames per second to sample (lower = faster but might miss text)
            min_confidence: Minimum confidence threshold for text detection
            
        Returns:
            Dictionary with:
                - all_text: Combined text from all frames
                - unique_text: Unique text phrases found
                - frame_count: Number of frames processed
                - detections: List of all detections with metadata
        """
        frames = self.extract_frames(video_path, fps=sample_fps)
        
        all_detections = []
        all_text_list = []
        
        for frame_idx, frame in enumerate(frames):
            detections = self.extract_text_from_frame(frame)
            
            for text, conf in detections:
                if conf >= min_confidence:
                    all_text_list.append(text)
                    all_detections.append({
                        'frame_idx': frame_idx,
                        'text': text,
                        'confidence': round(conf, 3)
                    })
        
        # Combine and deduplicate
        combined_text = ' '.join(all_text_list)
        unique_text = list(set(all_text_list))
        
        return {
            'all_text': combined_text,
            'unique_text': unique_text,
            'frame_count': len(frames),
            'detections': all_detections,
            'detection_count': len(all_detections)
        }


# Convenience function for simple use cases
def extract_text_from_video_simple(
    video_path: str, 
    languages=['en'], 
    sample_fps=1.0
) -> str:
    """
    Simple interface: returns just the combined text string.
    
    Args:
        video_path: Path to video file
        languages: OCR languages to use
        sample_fps: Sampling rate (frames per second)
        
    Returns:
        Combined text string from all frames
    """
    extractor = VideoTextExtractor(languages=languages, gpu=False)
    result = extractor.extract_text_from_video(video_path, sample_fps=sample_fps)
    return result['all_text']