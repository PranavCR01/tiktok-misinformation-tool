##pages/processes/multimodal.py

"""
Multimodal Content Extraction Module
Combines audio transcription (ASR) with on-screen text (OCR).
This module integrates but doesn't modify existing modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pages.processes.transcription import _transcribe_local_faster_whisper
from pages.processes.ocr.text_extractor import VideoTextExtractor


class MultimodalExtractor:
    """
    Extract both audio (speech) and visual (on-screen text) content from videos.
    Keeps both modalities separate but accessible in a unified structure.
    """

    def __init__(
        self,
        ocr_languages: Optional[List[str]] = None,
        ocr_sample_fps: float = 1.0,
        use_gpu: bool = False,
    ):
        """
        Initialize multimodal extractor.

        Args:
            ocr_languages: Languages for OCR detection (e.g., ["en", "es"])
            ocr_sample_fps: Frame sampling rate for OCR (1.0 = 1 frame/sec)
            use_gpu: Use GPU for OCR (False = CPU mode)
        """
        if ocr_languages is None:
            ocr_languages = ["en", "es"]

        self.ocr_extractor = VideoTextExtractor(
            languages=ocr_languages,
            gpu=use_gpu,
        )
        self.ocr_sample_fps = float(ocr_sample_fps)

    def extract_audio_transcript(self, video_path: str) -> str:
        """
        Extract audio transcript using existing transcription module.
        """
        return _transcribe_local_faster_whisper(video_path)

    def extract_visual_text(self, video_path: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Extract on-screen text using OCR module.

        Returns:
            Dictionary with OCR results from VideoTextExtractor.extract_text_from_video
        """
        return self.ocr_extractor.extract_text_from_video(
            video_path,
            sample_fps=self.ocr_sample_fps,
            min_confidence=float(min_confidence),
        )

    def extract_all(
        self,
        video_path: str,
        include_audio: bool = True,
        include_visual: bool = True,
        ocr_confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Extract all content (audio + visual) from video.

        Returns a dictionary containing:
            - audio_transcript: Speech-to-text transcript
            - visual_text: On-screen text (combined)
            - visual_text_unique: Unique on-screen phrases
            - visual_detections: Detailed OCR results
            - combined_content: All text combined (audio + visual)
            - modalities_used: List of modalities extracted
            - metadata: lengths + counts
        """
        result: Dict[str, Any] = {
            "audio_transcript": "",
            "visual_text": "",
            "visual_text_unique": [],
            "visual_detections": [],
            "combined_content": "",
            "modalities_used": [],
            "metadata": {},
        }

        # Extract audio
        if include_audio:
            result["audio_transcript"] = self.extract_audio_transcript(video_path) or ""
            result["modalities_used"].append("audio")
            result["metadata"]["audio_length_chars"] = len(result["audio_transcript"])

        # Extract visual
        if include_visual:
            ocr_result = self.extract_visual_text(video_path, min_confidence=ocr_confidence) or {}

            # Be defensive in case OCR module returns slightly different keys
            all_text = ocr_result.get("all_text", "") or ""
            unique_text = ocr_result.get("unique_text", []) or []
            detections = ocr_result.get("detections", []) or []

            result["visual_text"] = all_text
            result["visual_text_unique"] = unique_text
            result["visual_detections"] = detections

            result["modalities_used"].append("visual")
            result["metadata"]["visual_length_chars"] = len(result["visual_text"])
            result["metadata"]["visual_detection_count"] = int(ocr_result.get("detection_count", len(detections)) or 0)
            result["metadata"]["frames_processed"] = int(ocr_result.get("frame_count", 0) or 0)

            # Keep OCR params in metadata so experiments are reproducible
            result["metadata"]["ocr_sample_fps"] = self.ocr_sample_fps
            result["metadata"]["ocr_confidence_threshold"] = float(ocr_confidence)

        # Combine all content
        parts: List[str] = []
        if result["audio_transcript"]:
            parts.append(f"[AUDIO TRANSCRIPT]\n{result['audio_transcript']}")
        if result["visual_text"]:
            parts.append(f"[ON-SCREEN TEXT]\n{result['visual_text']}")

        result["combined_content"] = "\n\n".join(parts)
        result["metadata"]["total_content_length"] = len(result["combined_content"])

        return result


# Convenience function for simple use
def extract_multimodal_content(
    video_path: str,
    include_audio: bool = True,
    include_visual: bool = True,
    ocr_languages: Optional[List[str]] = None,
    ocr_sample_fps: float = 1.0,
    use_gpu: bool = False,
    ocr_confidence: float = 0.5,
) -> Dict[str, Any]:
    """
    Simple interface for multimodal extraction.

    Args:
        video_path: Path to video file
        include_audio: Extract speech transcript
        include_visual: Extract on-screen text
        ocr_languages: Languages for OCR (e.g., ["en","es"])
        ocr_sample_fps: Frame sampling rate for OCR
        use_gpu: Use GPU for OCR
        ocr_confidence: Minimum confidence for OCR detections

    Returns:
        Dictionary with all extracted content
    """
    extractor = MultimodalExtractor(
        ocr_languages=ocr_languages,
        ocr_sample_fps=ocr_sample_fps,
        use_gpu=use_gpu,
    )
    return extractor.extract_all(
        video_path,
        include_audio=include_audio,
        include_visual=include_visual,
        ocr_confidence=ocr_confidence,
    )
