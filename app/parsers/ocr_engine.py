"""
app/parsers/ocr_engine.py
─────────────────────────────────────────────────────────────────────────────
OCR abstraction layer for scanned / image-only PDF pages.

Strategy:
  1. Primary   → Tesseract (pytesseract)  — fast, CPU-only, good accuracy
  2. Fallback  → EasyOCR                 — slower, GPU-optional, better on
                                           non-standard fonts / rotated text
  3. Skip      → if neither is installed, log a warning and return ""

Auto-detection of which engine to use:
  engine="auto"   → try Tesseract first, fall back to EasyOCR
  engine="tesseract" → Tesseract only
  engine="easyocr"   → EasyOCR only
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OCREngineType(str, Enum):
    AUTO = "auto"
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    NONE = "none"


class OCRResult:
    """Holds the raw text and confidence from an OCR pass."""

    def __init__(self, text: str, confidence: float = 0.0, engine: str = ""):
        self.text = text
        self.confidence = confidence   # 0.0–1.0
        self.engine = engine

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class OCREngine:
    """
    Unified OCR interface. Converts a PIL Image (or numpy array) to text.

    Usage:
        engine = OCREngine(engine="auto", languages=["en"])
        result = engine.run(pil_image)
        print(result.text)
    """

    def __init__(
        self,
        engine: str = "auto",
        languages: Optional[List[str]] = None,
        dpi: int = 300,
        gpu: bool = False,
    ):
        self.engine_type = OCREngineType(engine)
        self.languages = languages or ["en"]
        self.dpi = dpi
        self.gpu = gpu

        self._tesseract_available = self._check_tesseract()
        self._easyocr_available = self._check_easyocr()
        self._easyocr_reader = None   # lazy-initialised (slow to load)

        # Resolve "auto" at construction time
        if self.engine_type == OCREngineType.AUTO:
            if self._tesseract_available:
                self._resolved = OCREngineType.TESSERACT
            elif self._easyocr_available:
                self._resolved = OCREngineType.EASYOCR
            else:
                self._resolved = OCREngineType.NONE
                logger.warning(
                    "No OCR engine available. Install pytesseract+Tesseract "
                    "or easyocr for scanned PDF support."
                )
        else:
            self._resolved = self.engine_type

        logger.info("OCREngine resolved to: %s", self._resolved.value)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, image) -> OCRResult:
        """
        Run OCR on a PIL Image or numpy array.
        Returns an OCRResult (empty if no engine available).
        """
        if self._resolved == OCREngineType.NONE:
            return OCRResult("", engine="none")

        if self._resolved == OCREngineType.TESSERACT:
            result = self._run_tesseract(image)
            # Cascade to EasyOCR if Tesseract produces garbage
            if not result and self._easyocr_available:
                logger.debug("Tesseract returned empty — cascading to EasyOCR")
                result = self._run_easyocr(image)
            return result

        if self._resolved == OCREngineType.EASYOCR:
            return self._run_easyocr(image)

        return OCRResult("", engine="none")

    @property
    def is_available(self) -> bool:
        return self._resolved != OCREngineType.NONE

    # ── Tesseract ─────────────────────────────────────────────────────────────

    def _run_tesseract(self, image) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image as PILImage

            # Accept numpy arrays too
            if isinstance(image, np.ndarray):
                image = PILImage.fromarray(image)

            # Config: OEM 3 (LSTM), PSM 3 (auto page segmentation)
            config = "--oem 3 --psm 3"
            lang = "+".join(self.languages)

            text = pytesseract.image_to_string(image, lang=lang, config=config)

            # Get confidence data
            data = pytesseract.image_to_data(
                image, lang=lang, config=config,
                output_type=pytesseract.Output.DICT,
            )
            confidences = [
                c for c in data["conf"]
                if isinstance(c, (int, float)) and c > 0
            ]
            avg_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0

            return OCRResult(text=text, confidence=avg_conf, engine="tesseract")

        except Exception as exc:
            logger.warning("Tesseract OCR failed: %s", exc)
            return OCRResult("", engine="tesseract")

    # ── EasyOCR ───────────────────────────────────────────────────────────────

    def _run_easyocr(self, image) -> OCRResult:
        try:
            import easyocr
            from PIL import Image as PILImage

            # Lazy-load the reader (takes ~2-3 s first time)
            if self._easyocr_reader is None:
                logger.info("Loading EasyOCR reader (first-time load)…")
                self._easyocr_reader = easyocr.Reader(
                    self.languages, gpu=self.gpu, verbose=False
                )

            # EasyOCR accepts numpy arrays natively
            if isinstance(image, PILImage.Image):
                image = np.array(image)

            results = self._easyocr_reader.readtext(image, detail=1)
            # results: List[(bbox, text, confidence)]

            if not results:
                return OCRResult("", engine="easyocr")

            lines = [text for (_, text, _) in results]
            avg_conf = sum(conf for (_, _, conf) in results) / len(results)
            full_text = "\n".join(lines)

            return OCRResult(text=full_text, confidence=avg_conf, engine="easyocr")

        except Exception as exc:
            logger.warning("EasyOCR failed: %s", exc)
            return OCRResult("", engine="easyocr")

    # ── Availability checks ───────────────────────────────────────────────────

    @staticmethod
    def _check_tesseract() -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @staticmethod
    def _check_easyocr() -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False
