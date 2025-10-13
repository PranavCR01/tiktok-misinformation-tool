"""
Multimodal Content Extraction Module
Combines audio transcription (ASR) with on-screen text (OCR).
This module integrates but doesn't modify existing modules.
"""

import os
from typing import Dict, Optional
from pages.processes.transcription import _transcribe_local_faster_whisper
from pages.processes.ocr.text_extractor import VideoTextExtractor


class MultimodalExtractor:
    """
    Extract both audio (speech) and visual (on-screen text) content from videos.
    Keeps both modalities separate but accessible in a unified structure.
    """
    
    def __init__(self, ocr_languages=['en'], ocr_sample_fps=1.0, use_gpu=False):
        """
        Initialize multimodal extractor.
        
        Args:
            ocr_languages: Languages for OCR detection
            ocr_sample_fps: Frame sampling rate for OCR (1.0 = 1 frame/sec)
            use_gpu: Use GPU for OCR (False = CPU mode)
        """
        self.ocr_extractor = VideoTextExtractor(
            languages=ocr_languages, 
            gpu=use_gpu
        )
        self.ocr_sample_fps = ocr_sample_fps
    
    def extract_audio_transcript(self, video_path: str) -> str:
        """
        Extract audio transcript using existing transcription module.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Transcript text string
        """
        return _transcribe_local_faster_whisper(video_path)
    
    def extract_visual_text(self, video_path: str, min_confidence=0.5) -> Dict:
        """
        Extract on-screen text using OCR module.
        
        Args:
            video_path: Path to video file
            min_confidence: Minimum OCR confidence threshold
            
        Returns:
            Dictionary with OCR results
        """
        return self.ocr_extractor.extract_text_from_video(
            video_path, 
            sample_fps=self.ocr_sample_fps,
            min_confidence=min_confidence
        )
    
    def extract_all(
        self, 
        video_path: str, 
        include_audio=True, 
        include_visual=True,
        ocr_confidence=0.5
    ) -> Dict[str, any]:
        """
        Extract all content (audio + visual) from video.
        
        Args:
            video_path: Path to video file
            include_audio: Whether to extract audio transcript
            include_visual: Whether to extract on-screen text
            ocr_confidence: Minimum confidence for OCR detections
            
        Returns:
            Dictionary containing:
                - audio_transcript: Speech-to-text transcript
                - visual_text: On-screen text (combined)
                - visual_text_unique: Unique on-screen phrases
                - visual_detections: Detailed OCR results
                - combined_content: All text combined (audio + visual)
                - modalities_used: List of modalities extracted
        """
        result = {
            'audio_transcript': '',
            'visual_text': '',
            'visual_text_unique': [],
            'visual_detections': [],
            'combined_content': '',
            'modalities_used': [],
            'metadata': {}
        }
        
        # Extract audio
        if include_audio:
            result['audio_transcript'] = self.extract_audio_transcript(video_path)
            result['modalities_used'].append('audio')
            result['metadata']['audio_length_chars'] = len(result['audio_transcript'])
        
        # Extract visual
        if include_visual:
            ocr_result = self.extract_visual_text(video_path, min_confidence=ocr_confidence)
            result['visual_text'] = ocr_result['all_text']
            result['visual_text_unique'] = ocr_result['unique_text']
            result['visual_detections'] = ocr_result['detections']
            result['modalities_used'].append('visual')
            result['metadata']['visual_length_chars'] = len(result['visual_text'])
            result['metadata']['visual_detection_count'] = ocr_result['detection_count']
            result['metadata']['frames_processed'] = ocr_result['frame_count']
        
        # Combine all content
        parts = []
        if result['audio_transcript']:
            parts.append(f"[AUDIO TRANSCRIPT]\n{result['audio_transcript']}")
        if result['visual_text']:
            parts.append(f"[ON-SCREEN TEXT]\n{result['visual_text']}")
        
        result['combined_content'] = '\n\n'.join(parts)
        result['metadata']['total_content_length'] = len(result['combined_content'])
        
        return result


# Convenience function for simple use
def extract_multimodal_content(
    video_path: str,
    include_audio=True,
    include_visual=True,
    ocr_languages=['en']
) -> Dict:
    """
    Simple interface for multimodal extraction.
    
    Args:
        video_path: Path to video file
        include_audio: Extract speech transcript
        include_visual: Extract on-screen text
        ocr_languages: Languages for OCR
        
    Returns:
        Dictionary with all extracted content
    """
    extractor = MultimodalExtractor(ocr_languages=ocr_languages)
    return extractor.extract_all(
        video_path,
        include_audio=include_audio,
        include_visual=include_visual
    )