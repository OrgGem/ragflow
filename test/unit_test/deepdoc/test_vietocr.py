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
"""Unit tests for VietOCR integration.

Tests the VietOCR recognizer, parser, OCR model registration, and constants.
Heavy transitive dependencies are stubbed via conftest.py.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Test: VietOCR Constants (no heavy imports needed)
# ---------------------------------------------------------------------------

class TestVietOCRConstants:
    """Test cases for VietOCR constants."""

    def test_env_keys_defined(self):
        """Test that VIETOCR_ENV_KEYS are defined."""
        from common.constants import VIETOCR_ENV_KEYS

        assert "VIETOCR_DEVICE" in VIETOCR_ENV_KEYS
        assert "VIETOCR_MODEL_NAME" in VIETOCR_ENV_KEYS

    def test_default_config_defined(self):
        """Test that VIETOCR_DEFAULT_CONFIG has correct defaults."""
        from common.constants import VIETOCR_DEFAULT_CONFIG

        assert VIETOCR_DEFAULT_CONFIG["VIETOCR_DEVICE"] == "cpu"
        assert VIETOCR_DEFAULT_CONFIG["VIETOCR_MODEL_NAME"] == "vgg_transformer"

    def test_env_keys_match_defaults(self):
        """Test that every key in VIETOCR_ENV_KEYS is present in VIETOCR_DEFAULT_CONFIG."""
        from common.constants import VIETOCR_ENV_KEYS, VIETOCR_DEFAULT_CONFIG

        for key in VIETOCR_ENV_KEYS:
            assert key in VIETOCR_DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Test: VietOCR Recognizer
# ---------------------------------------------------------------------------

class TestVietOCRRecognizer:
    """Test cases for VietOCRRecognizer (deepdoc/vision/vietocr.py)."""

    def _get_recognizer_cls(self):
        # Import directly; the vietocr.py module only imports cv2/numpy/PIL
        # which are available, and deepdoc.vision.ocr.OCR lazily inside VietOCR.
        from deepdoc.vision.vietocr import VietOCRRecognizer
        return VietOCRRecognizer

    def test_init_defaults(self):
        """Test VietOCRRecognizer initialization with default parameters."""
        cls = self._get_recognizer_cls()
        recognizer = cls(device="cpu", model_name="vgg_transformer")
        assert recognizer.device == "cpu"
        assert recognizer.model_name == "vgg_transformer"
        assert recognizer._predictor is None  # Lazy init

    def test_init_custom(self):
        """Test VietOCRRecognizer initialization with custom parameters."""
        cls = self._get_recognizer_cls()
        recognizer = cls(device="cuda", model_name="vgg_seq2seq")
        assert recognizer.device == "cuda"
        assert recognizer.model_name == "vgg_seq2seq"

    @patch("deepdoc.vision.vietocr._load_vietocr_predictor")
    def test_recognize_single(self, mock_load):
        """Test single text-line recognition."""
        cls = self._get_recognizer_cls()
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = ("Xin chào", 0.95)
        mock_load.return_value = mock_predictor

        recognizer = cls()
        img_crop = np.zeros((48, 320, 3), dtype=np.uint8)
        text, score = recognizer.recognize(img_crop)

        assert text == "Xin chào"
        assert score == 0.95

    @patch("deepdoc.vision.vietocr._load_vietocr_predictor")
    def test_recognize_batch(self, mock_load):
        """Test batch recognition."""
        cls = self._get_recognizer_cls()
        mock_predictor = MagicMock()
        mock_predictor.predict_batch.return_value = (
            ["Xin chào", "Việt Nam"],
            [0.95, 0.88],
        )
        mock_load.return_value = mock_predictor

        recognizer = cls()
        imgs = [np.zeros((48, 320, 3), dtype=np.uint8)] * 2
        results = recognizer.recognize_batch(imgs)

        assert len(results) == 2
        assert results[0] == ("Xin chào", 0.95)
        assert results[1] == ("Việt Nam", 0.88)

    @patch("deepdoc.vision.vietocr._load_vietocr_predictor")
    def test_recognize_batch_empty(self, mock_load):
        """Test batch recognition with empty input."""
        cls = self._get_recognizer_cls()
        recognizer = cls()
        assert recognizer.recognize_batch([]) == []


# ---------------------------------------------------------------------------
# Test: VietOCR full pipeline
# ---------------------------------------------------------------------------

class TestVietOCRPipeline:
    """Test cases for the VietOCR class (detect + recognize)."""

    def _make_vietocr(self):
        from deepdoc.vision.vietocr import VietOCR, VietOCRRecognizer
        obj = VietOCR.__new__(VietOCR)
        obj._base_ocr = MagicMock()
        obj._recognizer = MagicMock(spec=VietOCRRecognizer)
        obj.drop_score = 0.5
        return obj

    def test_call_none_image(self):
        """VietOCR(None) should return a None-like sentinel."""
        vietocr = self._make_vietocr()
        result = vietocr(None)
        # Returns (None, None, time_dict) when img is None
        assert result is None or (isinstance(result, tuple) and result[0] is None)

    @patch("deepdoc.vision.vietocr._load_vietocr_predictor")
    def test_recognize_delegates(self, mock_load):
        """recognize() should delegate cropping + recognition."""
        from deepdoc.vision.vietocr import VietOCR, VietOCRRecognizer

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = ("Tiếng Việt", 0.9)
        mock_load.return_value = mock_predictor

        obj = VietOCR.__new__(VietOCR)
        obj._base_ocr = MagicMock()
        obj._base_ocr.get_rotate_crop_image.return_value = np.zeros((48, 320, 3), dtype=np.uint8)
        obj._recognizer = VietOCRRecognizer()
        obj.drop_score = 0.5

        box = np.array([[0, 0], [100, 0], [100, 30], [0, 30]], dtype=np.float32)
        text = obj.recognize(np.zeros((100, 200, 3), dtype=np.uint8), box)
        assert text == "Tiếng Việt"

    @patch("deepdoc.vision.vietocr._load_vietocr_predictor")
    def test_recognize_drops_low_score(self, mock_load):
        """Low-confidence results should be dropped."""
        from deepdoc.vision.vietocr import VietOCR, VietOCRRecognizer

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = ("bad", 0.2)
        mock_load.return_value = mock_predictor

        obj = VietOCR.__new__(VietOCR)
        obj._base_ocr = MagicMock()
        obj._base_ocr.get_rotate_crop_image.return_value = np.zeros((48, 320, 3), dtype=np.uint8)
        obj._recognizer = VietOCRRecognizer()
        obj.drop_score = 0.5

        box = np.array([[0, 0], [100, 0], [100, 30], [0, 30]], dtype=np.float32)
        text = obj.recognize(np.zeros((100, 200, 3), dtype=np.uint8), box)
        assert text == ""

    def test_recognize_batch_delegates(self):
        """recognize_batch() should delegate to the recognizer."""
        vietocr = self._make_vietocr()
        vietocr._recognizer.recognize_batch.return_value = [
            ("Xin", 0.9),
            ("chào", 0.3),
        ]

        imgs = [np.zeros((48, 100, 3), dtype=np.uint8)] * 2
        texts = vietocr.recognize_batch(imgs)
        assert texts[0] == "Xin"
        assert texts[1] == ""  # below drop_score


# ---------------------------------------------------------------------------
# Test: VietOCR Parser
# ---------------------------------------------------------------------------

class TestVietOCRParser:
    """Test cases for VietOCRParser (deepdoc/parser/vietocr_parser.py)."""

    def _get_parser_cls(self):
        from deepdoc.parser.vietocr_parser import VietOCRParser
        return VietOCRParser

    def test_init(self):
        cls = self._get_parser_cls()
        parser = cls(device="cpu", model_name="vgg_transformer")
        assert parser.device == "cpu"
        assert parser.model_name == "vgg_transformer"
        assert parser._vietocr is None

    def test_check_installation_returns_tuple(self):
        cls = self._get_parser_cls()
        parser = cls()
        ok, reason = parser.check_installation()
        assert isinstance(ok, bool)
        assert isinstance(reason, str)

    @patch("deepdoc.parser.vietocr_parser.pdfplumber")
    def test_parse_pdf_returns_sections_and_tables(self, mock_pdfplumber):
        cls = self._get_parser_cls()

        mock_page = MagicMock()
        mock_page_img = MagicMock()
        mock_page_img.original = Image.new("RGB", (100, 100), "white")
        mock_page.to_image.return_value = mock_page_img
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        parser = cls()
        parser._vietocr = MagicMock()
        parser._vietocr.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], ("Xin chào Việt Nam", 0.95)),
        ]

        with patch.object(parser, "check_installation", return_value=(True, "")):
            sections, tables = parser.parse_pdf(filepath="/tmp/test.pdf")

        assert isinstance(sections, list)
        assert isinstance(tables, list)
        assert len(sections) == 1
        assert "Xin chào Việt Nam" in sections[0][0]

    @patch("deepdoc.parser.vietocr_parser.pdfplumber")
    def test_parse_pdf_empty_document(self, mock_pdfplumber):
        cls = self._get_parser_cls()

        mock_pdf = MagicMock()
        mock_pdf.pages = []
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        parser = cls()
        with patch.object(parser, "check_installation", return_value=(True, "")):
            sections, tables = parser.parse_pdf(filepath="/tmp/empty.pdf")
        assert sections == []
        assert tables == []

    def test_parse_pdf_raises_when_not_installed(self):
        cls = self._get_parser_cls()
        parser = cls()
        with patch.object(parser, "check_installation", return_value=(False, "[VietOCR] vietocr package not installed")):
            with pytest.raises(RuntimeError, match="vietocr package not installed"):
                parser.parse_pdf(filepath="/tmp/test.pdf")

    @patch("deepdoc.parser.vietocr_parser.pdfplumber")
    def test_parse_pdf_callback_progress(self, mock_pdfplumber):
        cls = self._get_parser_cls()

        mock_pdf = MagicMock()
        mock_pdf.pages = []
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        parser = cls()
        callback = MagicMock()
        with patch.object(parser, "check_installation", return_value=(True, "")):
            parser.parse_pdf(filepath="/tmp/test.pdf", callback=callback)
        assert callback.call_count >= 1

    @patch("deepdoc.parser.vietocr_parser.pdfplumber")
    def test_parse_pdf_manual_mode(self, mock_pdfplumber):
        cls = self._get_parser_cls()

        mock_page = MagicMock()
        mock_page_img = MagicMock()
        mock_page_img.original = Image.new("RGB", (100, 100), "white")
        mock_page.to_image.return_value = mock_page_img
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        parser = cls()
        parser._vietocr = MagicMock()
        parser._vietocr.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], ("Dữ liệu", 0.9)),
        ]

        with patch.object(parser, "check_installation", return_value=(True, "")):
            sections, tables = parser.parse_pdf(filepath="/tmp/test.pdf", parse_method="manual")

        assert len(sections) == 1
        assert len(sections[0]) == 3  # (text, label, tag)
        assert sections[0][0] == "Dữ liệu"

    @patch("deepdoc.parser.vietocr_parser.pdfplumber")
    def test_parse_pdf_paper_mode(self, mock_pdfplumber):
        cls = self._get_parser_cls()

        mock_page = MagicMock()
        mock_page_img = MagicMock()
        mock_page_img.original = Image.new("RGB", (100, 100), "white")
        mock_page.to_image.return_value = mock_page_img
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        parser = cls()
        parser._vietocr = MagicMock()
        parser._vietocr.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], ("Bài báo", 0.85)),
        ]

        with patch.object(parser, "check_installation", return_value=(True, "")):
            sections, tables = parser.parse_pdf(filepath="/tmp/test.pdf", parse_method="paper")

        assert len(sections) == 1
        assert len(sections[0]) == 2  # (text+tag, label)
        assert "Bài báo" in sections[0][0]
        assert "@@" in sections[0][0]  # tag is appended


# ---------------------------------------------------------------------------
# Test: VietOCR OCR Model (factory registration)
# ---------------------------------------------------------------------------

class TestVietOCROcrModel:
    """Test cases for VietOCROcrModel in rag/llm/ocr_model.py."""

    def _get_model_cls(self):
        from rag.llm.ocr_model import VietOCROcrModel
        return VietOCROcrModel

    def test_factory_name(self):
        cls = self._get_model_cls()
        assert cls._FACTORY_NAME == "VietOCR"

    def test_init_defaults(self):
        cls = self._get_model_cls()
        model = cls(key="", model_name="vietocr-test")
        assert model.vietocr_device == "cpu"
        assert model.vietocr_model_name == "vgg_transformer"

    def test_init_json_config(self):
        cls = self._get_model_cls()
        cfg = json.dumps({"VIETOCR_DEVICE": "cuda", "VIETOCR_MODEL_NAME": "vgg_seq2seq"})
        model = cls(key=cfg, model_name="vietocr-test")
        assert model.vietocr_device == "cuda"
        assert model.vietocr_model_name == "vgg_seq2seq"

    def test_init_env_vars(self):
        cls = self._get_model_cls()
        with patch.dict(os.environ, {"VIETOCR_DEVICE": "cuda:1", "VIETOCR_MODEL_NAME": "resnet_transformer"}):
            model = cls(key="", model_name="vietocr-test")
            assert model.vietocr_device == "cuda:1"
            assert model.vietocr_model_name == "resnet_transformer"

    def test_check_available(self):
        cls = self._get_model_cls()
        model = cls(key="", model_name="vietocr-test")
        ok, reason = model.check_available()
        assert isinstance(ok, bool)

    def test_parse_pdf_raises_when_unavailable(self):
        cls = self._get_model_cls()
        model = cls(key="", model_name="vietocr-test")
        with patch.object(model, "check_installation", return_value=(False, "[VietOCR] vietocr package not installed")):
            with pytest.raises(RuntimeError, match="VietOCR not available"):
                model.parse_pdf(filepath="/tmp/test.pdf")

    def test_factory_registered(self):
        """VietOCR must be discoverable in OcrModel dict."""
        # Importing rag.llm triggers the factory auto-discovery
        from rag.llm import OcrModel
        assert "VietOCR" in OcrModel
