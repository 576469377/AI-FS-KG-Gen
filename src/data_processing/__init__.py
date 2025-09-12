"""Data processing module for AI-FS-KG-Gen"""

from .text_cleaner import TextCleaner, clean_text_batch

# Optional imports
try:
    from .llm_processor import LLMProcessor, process_text_batch
    _llm_available = True
except ImportError:
    LLMProcessor = None
    process_text_batch = None
    _llm_available = False

try:
    from .vlm_processor import VLMProcessor, process_image_batch
    _vlm_available = True
except ImportError:
    VLMProcessor = None
    process_image_batch = None
    _vlm_available = False

__all__ = [
    "TextCleaner",
    "clean_text_batch"
]

if _llm_available:
    __all__.extend(["LLMProcessor", "process_text_batch"])

if _vlm_available:
    __all__.extend(["VLMProcessor", "process_image_batch"])
