"""
Package: src.ocr
AI Challenge TP.HCM 2026 - OCR & Spatial Text Retrieval System
"""

from .preprocessor import ImagePreprocessor, FrameMetadataMapper
from .detector import TextDetector
from .recognizer import BatchTextRecognizer
from .postprocessor import VietnamesePostProcessor
from .deduplicator import TemporalDeduplicator
from .indexer import OCRIndexer, OCRIndexSearcher
from .agent_tool import SpatialOCRTool
from .pipeline import OCRPipeline

__all__ = [
    "ImagePreprocessor",
    "FrameMetadataMapper",
    "TextDetector",
    "BatchTextRecognizer",
    "VietnamesePostProcessor",
    "TemporalDeduplicator",
    "OCRIndexer",
    "OCRIndexSearcher",
    "SpatialOCRTool",
    "OCRPipeline"
]
