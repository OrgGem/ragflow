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
"""
Standalone OCR service endpoint.

Upload an image or PDF file and get OCR results back.

Usage:
    POST /api/v1/ocr
    Headers:
        Authorization: Bearer <api_key>
    Form data:
        file: <file>                         (required) image or PDF to OCR
        ocr_engine: vietocr|paddleocr|mineru  (optional, default: vietocr)
        parse_method: raw|paper|manual        (optional, default: raw)
        from_page: int                        (optional, default: 0)
        to_page: int                          (optional, default: 100000)
        lang: string                          (optional, default: Vietnamese)

Response:
    {
        "code": 0,
        "data": {
            "sections": [...],
            "tables": [...]
        }
    }
"""
import logging
import os

from quart import request

OCR_ONLY_MODE = os.getenv("RAGFLOW_OCR_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
if OCR_ONLY_MODE:
    from rag.llm.ocr_model import MinerUOcrModel, PaddleOCROcrModel, VietOCROcrModel
    LLMBundle = None
    TenantLLMService = None
else:
    from api.db.services.llm_service import LLMBundle
    from api.db.services.tenant_llm_service import TenantLLMService
from api.utils.api_utils import (
    get_result,
    server_error_response,
    token_required,
)
from common.constants import LLMType, RetCode

SUPPORTED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
}

ENGINE_FACTORY_MAP = {
    "vietocr": "VietOCR",
    "paddleocr": "PaddleOCR",
    "mineru": "MinerU",
}

ENGINE_ENV_ENSURE = {
    "vietocr": "ensure_vietocr_from_env",
    "paddleocr": "ensure_paddleocr_from_env",
    "mineru": "ensure_mineru_from_env",
}

EMPTY_OCR_CONFIG = "{}"


def _resolve_ocr_model_name(tenant_id: str, engine: str) -> str | None:
    """Resolve the LLM model name for the given OCR engine and tenant.

    ``engine`` must be a key in ENGINE_FACTORY_MAP (validated by the caller).
    In OCR-only mode, a synthetic env-backed model name is returned.
    """
    if OCR_ONLY_MODE:
        return f"{engine}-from-env"

    factory = ENGINE_FACTORY_MAP.get(engine)
    if not factory:
        return None

    # Try to auto-provision from env variables first
    ensure_method = ENGINE_ENV_ENSURE.get(engine)
    if ensure_method and hasattr(TenantLLMService, ensure_method):
        try:
            env_name = getattr(TenantLLMService, ensure_method)(tenant_id)
        except Exception as e:
            logging.warning("OCR env auto-provision failed for %s: %s", engine, e)
            env_name = None
    else:
        env_name = None

    # Look up configured models for this tenant + factory
    candidates = TenantLLMService.query(
        tenant_id=tenant_id,
        llm_factory=factory,
        model_type=LLMType.OCR,
    )
    if candidates:
        return candidates[0].llm_name
    if env_name:
        return env_name
    return None


def _build_ocr_model_for_ocr_only(engine: str):
    if engine not in ENGINE_FACTORY_MAP:
        raise ValueError(f"Unsupported OCR engine: {engine}")
    model_name = f"{engine}-from-env"
    if engine == "vietocr":
        return VietOCROcrModel(EMPTY_OCR_CONFIG, model_name)
    if engine == "paddleocr":
        return PaddleOCROcrModel(EMPTY_OCR_CONFIG, model_name)
    return MinerUOcrModel(EMPTY_OCR_CONFIG, model_name)


def _identity_auth_decorator(func):
    return func


auth_required = token_required if not OCR_ONLY_MODE else _identity_auth_decorator


# `manager` is a Blueprint injected by the auto-discovery system in
# ``api/apps/__init__.py:register_page``.
@manager.route("/ocr", methods=["POST"])  # noqa: F821
@auth_required
async def ocr(tenant_id=None):
    """
    Perform OCR on an uploaded file and return extracted text.
    ---
    tags:
      - OCR
    security:
      - ApiKeyAuth: []
    parameters:
      - in: header
        name: Authorization
        type: string
        required: true
        description: Bearer token for authentication.
      - in: formData
        name: file
        type: file
        required: true
        description: Image or PDF file to perform OCR on.
      - in: formData
        name: ocr_engine
        type: string
        required: false
        description: OCR engine to use (vietocr, paddleocr, mineru). Default is vietocr.
      - in: formData
        name: parse_method
        type: string
        required: false
        description: Parse method (raw, paper, manual). Default is raw.
      - in: formData
        name: from_page
        type: integer
        required: false
        description: Start page (for PDFs). Default is 0.
      - in: formData
        name: to_page
        type: integer
        required: false
        description: End page (for PDFs). Default is 100000.
      - in: formData
        name: lang
        type: string
        required: false
        description: Language hint. Default is Vietnamese.
    responses:
      200:
        description: OCR results
    """
    try:
        files = await request.files
        form = await request.form

        # --- validate file ---
        if "file" not in files:
            return get_result(
                code=RetCode.ARGUMENT_ERROR,
                message="No file uploaded. Send a file with form field name 'file'.",
            )

        file_obj = files["file"]
        if not file_obj or not file_obj.filename:
            return get_result(
                code=RetCode.ARGUMENT_ERROR,
                message="Empty file. Please upload a valid image or PDF.",
            )

        filename = file_obj.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return get_result(
                code=RetCode.ARGUMENT_ERROR,
                message=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

        # --- parse form params ---
        ocr_engine = (form.get("ocr_engine") or "vietocr").strip().lower()
        if ocr_engine not in ENGINE_FACTORY_MAP:
            return get_result(
                code=RetCode.ARGUMENT_ERROR,
                message=f"Unknown ocr_engine '{ocr_engine}'. Supported: {', '.join(ENGINE_FACTORY_MAP)}",
            )

        parse_method = (form.get("parse_method") or "raw").strip().lower()
        if parse_method not in ("raw", "paper", "manual"):
            return get_result(
                code=RetCode.ARGUMENT_ERROR,
                message="parse_method must be one of: raw, paper, manual",
            )

        try:
            from_page = int(form.get("from_page", 0))
        except (ValueError, TypeError):
            from_page = 0

        try:
            to_page = int(form.get("to_page", 100000))
        except (ValueError, TypeError):
            to_page = 100000

        lang = form.get("lang", "Vietnamese").strip()

        # --- resolve OCR model ---
        # --- read file into memory ---
        binary = await file_obj.read()

        # --- run OCR ---
        if OCR_ONLY_MODE:
            parser = _build_ocr_model_for_ocr_only(ocr_engine)
        else:
            llm_name = _resolve_ocr_model_name(tenant_id, ocr_engine)
            if not llm_name:
                return get_result(
                    code=RetCode.DATA_ERROR,
                    message=(
                        f"No '{ocr_engine}' OCR model configured for this tenant. "
                        f"Please add a {ENGINE_FACTORY_MAP[ocr_engine]} model in Settings → Model Providers, "
                        f"or set the corresponding environment variables."
                    ),
                )
            parser = LLMBundle(
                tenant_id=tenant_id,
                llm_type=LLMType.OCR,
                llm_name=llm_name,
                lang=lang,
            ).mdl

        sections, tables = parser.parse_pdf(
            filepath=filename,  # used as a display name; actual data is in `binary`
            binary=binary,
            callback=None,
            parse_method=parse_method,
            from_page=from_page,
            to_page=to_page,
        )

        # --- format response ---
        section_list = []
        if sections:
            for sec in sections:
                if isinstance(sec, (list, tuple)):
                    # sections are typically (text, tag) or (text, extra, tag) tuples
                    section_list.append({
                        "text": sec[0] if len(sec) > 0 else "",
                        "metadata": sec[1:] if len(sec) > 1 else [],
                    })
                elif isinstance(sec, str):
                    section_list.append({"text": sec, "metadata": []})
                else:
                    section_list.append({"text": str(sec), "metadata": []})

        table_list = []
        if tables:
            for tbl in tables:
                if isinstance(tbl, (list, tuple)):
                    table_list.append({
                        "content": tbl[0] if len(tbl) > 0 else "",
                        "metadata": tbl[1:] if len(tbl) > 1 else [],
                    })
                elif isinstance(tbl, str):
                    table_list.append({"content": tbl, "metadata": []})
                else:
                    table_list.append({"content": str(tbl), "metadata": []})

        return get_result(data={
            "ocr_engine": ocr_engine,
            "filename": filename,
            "sections": section_list,
            "tables": table_list,
        })

    except AssertionError as e:
        logging.exception("OCR model loading failed")
        return get_result(
            code=RetCode.DATA_ERROR,
            message=f"Failed to load OCR model: {e}",
        )
    except Exception as e:
        logging.exception("OCR processing failed")
        return server_error_response(e)
