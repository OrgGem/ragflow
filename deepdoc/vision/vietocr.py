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
"""VietOCR integration for Vietnamese text recognition.

VietOCR is an open-source Vietnamese OCR engine based on transformer models.
It processes single-line or few-line text images, so it requires a text detector
(e.g., the built-in TextDetector from deepdoc) to first locate text regions,
then recognizes text within each cropped box.
"""

import copy
import logging
import time

import cv2
import numpy as np
from PIL import Image


def _load_vietocr_predictor(device: str = "cpu", model_name: str = "vgg_transformer"):
    """Lazily load VietOCR Predictor. Requires torch and vietocr packages."""
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    config = Cfg.load_config_from_name(model_name)
    config["device"] = device
    config["cnn"]["pretrained"] = False
    config["predictor"]["beamsearch"] = False
    return Predictor(config)


class VietOCRRecognizer:
    """Vietnamese text recognizer using VietOCR.

    This class wraps VietOCR's Predictor and provides an interface compatible
    with the deepdoc OCR pipeline. It processes cropped text-line images
    (numpy BGR arrays or PIL Images) and returns recognized Vietnamese text.
    """

    def __init__(self, device: str = "cpu", model_name: str = "vgg_transformer"):
        self.device = device
        self.model_name = model_name
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            logging.info(f"Loading VietOCR model '{self.model_name}' on device '{self.device}'")
            self._predictor = _load_vietocr_predictor(self.device, self.model_name)
        return self._predictor

    def recognize(self, img_crop: np.ndarray) -> tuple[str, float]:
        """Recognize text from a single cropped text-line image.

        Args:
            img_crop: BGR numpy array of a cropped text region.

        Returns:
            Tuple of (recognized_text, confidence_score).
        """
        pil_img = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        text, prob = self.predictor.predict(pil_img, return_prob=True)
        score = prob if prob is not None else 0.0
        return text.strip(), float(score)

    def recognize_batch(self, img_crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Recognize text from a batch of cropped text-line images.

        Args:
            img_crops: List of BGR numpy arrays of cropped text regions.

        Returns:
            List of (recognized_text, confidence_score) tuples.
        """
        if not img_crops:
            return []

        pil_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in img_crops]
        texts, probs = self.predictor.predict_batch(pil_imgs, return_prob=True)

        results = []
        for text, prob in zip(texts, probs):
            score = prob if prob is not None else 0.0
            results.append((text.strip(), float(score)))
        return results


class VietOCR:
    """Full OCR pipeline for Vietnamese text: detection + VietOCR recognition.

    Uses the built-in TextDetector from deepdoc for text region detection,
    and VietOCR for text recognition on each detected box. This is suitable
    for Vietnamese documents where the built-in CTC recognizer may not
    perform well.
    """

    def __init__(self, model_dir=None, device: str = "cpu", model_name: str = "vgg_transformer"):
        from deepdoc.vision.ocr import OCR

        self._base_ocr = OCR(model_dir=model_dir)
        self._recognizer = VietOCRRecognizer(device=device, model_name=model_name)
        self.drop_score = 0.5

    def get_rotate_crop_image(self, img, points):
        """Crop and rotate text region from image using perspective transform."""
        return self._base_ocr.get_rotate_crop_image(img, points)

    def sorted_boxes(self, dt_boxes):
        """Sort text boxes top-to-bottom, left-to-right."""
        return self._base_ocr.sorted_boxes(dt_boxes)

    def detect(self, img, device_id: int | None = None):
        """Detect text regions using the built-in TextDetector."""
        return self._base_ocr.detect(img, device_id=device_id)

    def recognize(self, ori_im, box) -> str:
        """Recognize text in a single box using VietOCR."""
        img_crop = self.get_rotate_crop_image(ori_im, box)
        text, score = self._recognizer.recognize(img_crop)
        if score < self.drop_score:
            return ""
        return text

    def recognize_batch(self, img_list) -> list[str]:
        """Recognize text in a batch of cropped images using VietOCR."""
        if not img_list:
            return []
        results = self._recognizer.recognize_batch(img_list)
        texts = []
        for text, score in results:
            if score < self.drop_score:
                text = ""
            texts.append(text)
        return texts

    def __call__(self, img, device_id=0, cls=True):
        """Full OCR pipeline: detect text regions + recognize with VietOCR.

        Args:
            img: Input image as numpy array (BGR).
            device_id: GPU device ID for text detection.
            cls: Not used, kept for API compatibility.

        Returns:
            List of (box_coords, (text, score)) tuples, or None on failure.
        """
        time_dict = {"det": 0, "rec": 0, "cls": 0, "all": 0}
        if device_id is None:
            device_id = 0

        if img is None:
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self._base_ocr.text_detector[device_id](img)
        time_dict["det"] = elapse

        if dt_boxes is None:
            end = time.time()
            time_dict["all"] = end - start
            return None, None, time_dict

        img_crop_list = []
        dt_boxes = self.sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_results = self._recognizer.recognize_batch(img_crop_list)
        time_dict["rec"] = time.time() - start - time_dict["det"]

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_results):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        end = time.time()
        time_dict["all"] = end - start

        return list(zip([a.tolist() for a in filter_boxes], filter_rec_res))
