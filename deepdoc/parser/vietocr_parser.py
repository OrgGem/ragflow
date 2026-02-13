#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""VietOCR-based PDF parser for Vietnamese documents.

This parser uses the built-in text detector from deepdoc for text region detection,
combined with VietOCR for Vietnamese text recognition. It is optimized for
documents containing Vietnamese text.
"""

from __future__ import annotations

import logging
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pdfplumber
from PIL import Image

try:
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
except Exception:

    class RAGFlowPdfParser:
        pass


SectionTuple = tuple[str, ...]
TableTuple = tuple[str, ...]
ParseResult = tuple[list[SectionTuple], list[TableTuple]]


class VietOCRParser(RAGFlowPdfParser):
    """Parser for PDF documents using VietOCR for Vietnamese text recognition.

    Uses deepdoc's built-in TextDetector for text region detection and
    VietOCR for recognition. VietOCR processes single-line or few-line
    text images, so each detected text box is individually recognized.
    """

    _ZOOMIN = 3

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "vgg_transformer",
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self._vietocr = None
        self.page_images: list[Image.Image] = []
        self.page_from = 0

    @property
    def vietocr(self):
        """Lazy initialization of VietOCR pipeline."""
        if self._vietocr is None:
            from deepdoc.vision.vietocr import VietOCR

            self._vietocr = VietOCR(device=self.device, model_name=self.model_name)
        return self._vietocr

    def check_installation(self) -> tuple[bool, str]:
        """Check if VietOCR dependencies are available."""
        try:
            import vietocr  # noqa: F401
        except ImportError:
            return False, "[VietOCR] vietocr package not installed. Install with: pip install vietocr"

        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "[VietOCR] torch package not installed. Install with: pip install torch"

        return True, ""

    def parse_pdf(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable[[float, str], None]] = None,
        *,
        parse_method: str = "raw",
        from_page: int = 0,
        to_page: int = 100000,
        **kwargs: Any,
    ) -> ParseResult:
        """Parse PDF document using VietOCR.

        Args:
            filepath: Path to the PDF file.
            binary: Optional binary content of the PDF.
            callback: Optional progress callback function.
            parse_method: Parsing method ('raw', 'manual', 'paper').
            from_page: Start page number (0-indexed).
            to_page: End page number (exclusive).

        Returns:
            Tuple of (sections, tables) where sections contain recognized text.
        """
        ok, reason = self.check_installation()
        if not ok:
            raise RuntimeError(reason)

        if callback:
            callback(0.1, "[VietOCR] loading document")

        # Load page images
        page_images = self._load_page_images(filepath, binary, from_page, to_page)
        self.page_images = page_images

        if not page_images:
            self.logger.warning("[VietOCR] No pages found in document")
            return [], []

        if callback:
            callback(0.2, f"[VietOCR] processing {len(page_images)} pages")

        sections: list[SectionTuple] = []
        total_pages = len(page_images)

        for page_idx, page_img in enumerate(page_images):
            if callback:
                progress = 0.2 + 0.7 * (page_idx / max(total_pages, 1))
                callback(progress, f"[VietOCR] processing page {page_idx + 1}/{total_pages}")

            # Convert PIL Image to numpy array (BGR for OpenCV)
            img_array = np.array(page_img)
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            # Convert RGB to BGR
            img_array = img_array[:, :, ::-1].copy()

            # Run OCR pipeline (detect + recognize)
            result = self.vietocr(img_array)
            if result is None:
                continue

            # Process results
            for box, (text, score) in result:
                if not text or not text.strip():
                    continue

                # Calculate bounding box
                box_array = np.array(box)
                x0 = int(np.min(box_array[:, 0])) // self._ZOOMIN
                x1 = int(np.max(box_array[:, 0])) // self._ZOOMIN
                top = int(np.min(box_array[:, 1])) // self._ZOOMIN
                bottom = int(np.max(box_array[:, 1])) // self._ZOOMIN

                actual_page = from_page + page_idx + 1
                tag = f"@@{actual_page}\t{x0}\t{x1}\t{top}\t{bottom}##"

                if parse_method == "manual":
                    sections.append((text.strip(), "", tag))
                elif parse_method == "paper":
                    sections.append((text.strip() + tag, ""))
                else:
                    sections.append((text.strip(), tag))

        if callback:
            callback(0.95, f"[VietOCR] done, sections: {len(sections)}")

        tables: list[TableTuple] = []

        if callback:
            callback(1.0, f"[VietOCR] complete, sections: {len(sections)}, tables: {len(tables)}")

        return sections, tables

    def _load_page_images(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None,
        from_page: int,
        to_page: int,
    ) -> list[Image.Image]:
        """Load PDF pages as images."""
        self.page_from = from_page
        try:
            if binary is not None:
                if isinstance(binary, (bytes, bytearray)):
                    source = BytesIO(binary)
                else:
                    source = binary
            else:
                source = str(filepath)

            with pdfplumber.open(source) as pdf:
                pages = pdf.pages[from_page:to_page]
                images = []
                for page in pages:
                    img = page.to_image(resolution=72 * self._ZOOMIN, antialias=True).original
                    images.append(img)
                return images

        except Exception as e:
            self.logger.exception(f"[VietOCR] Failed to load page images: {e}")
            return []
