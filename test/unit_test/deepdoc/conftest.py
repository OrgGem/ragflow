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
"""conftest.py for deepdoc unit tests.

Uses aggressive sys.modules stubbing to break the heavy import chains
so that individual modules under deepdoc can be imported in isolation.

The strategy:
1. Stub `common.settings` entirely (it imports ES, infinity, opensearch, minio,
   memory, rag.nlp, etc. — all of which need massive dependency graphs).
2. Stub `common.token_utils` (tiktoken needs network).
3. Stub `rag.nlp` (transitively depends on tiktoken).
4. Stub all `rag.utils.*` and `memory.*` connectors.
5. Stub `deepdoc.vision.__init__` and `deepdoc.parser.__init__` so that
   importing sub-modules (like vietocr.py, vietocr_parser.py) skips the
   __init__ files that import the full collection.
6. Stub `rag.llm` module-scanning __init__ so we can import ocr_model directly.
"""

import importlib
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock


def _make_module_stub(name: str) -> ModuleType:
    """Create a real module stub that Python's import system will accept."""
    mod = ModuleType(name)
    mod.__path__ = [os.path.join(os.path.dirname(__file__), "__stub_path__")]
    mod.__spec__ = None
    mod.__loader__ = None
    mod.__package__ = name
    return mod


# --- Stage 1: stub impossible-to-install packages ---
for _m in [
    "opensearchpy",
    "infinity", "infinity.common",
    "memory", "memory.utils", "memory.utils.es_conn",
    "memory.utils.infinity_conn", "memory.utils.ob_conn",
]:
    sys.modules.setdefault(_m, _make_module_stub(_m))

# --- Stage 2: stub modules that need network or unavailable resources ---
_token_utils_stub = _make_module_stub("common.token_utils")
_token_utils_stub.num_tokens_from_string = lambda *a, **kw: 0
sys.modules.setdefault("common.token_utils", _token_utils_stub)

# --- Stage 3: stub common.settings (avoids the massive import chain) ---
_settings_stub = _make_module_stub("common.settings")
# Provide commonly accessed attributes as None/empty
for _attr in [
    "LLM", "LLM_FACTORY", "LLM_BASE_URL", "CHAT_MDL", "EMBEDDING_MDL",
    "RERANK_MDL", "ASR_MDL", "IMAGE2TEXT_MDL", "FACTORY_LLM_INFOS",
    "DOC_ENGINE", "docStoreConn", "retrievaler", "kg_retrievaler",
]:
    setattr(_settings_stub, _attr, None)
sys.modules.setdefault("common.settings", _settings_stub)

# --- Stage 4: stub rag.utils.* connector modules ---
for _m in [
    "rag.utils", "rag.utils.es_conn", "rag.utils.infinity_conn",
    "rag.utils.ob_conn", "rag.utils.opensearch_conn",
    "rag.utils.azure_sas_conn", "rag.utils.azure_spn_conn",
    "rag.utils.gcs_conn", "rag.utils.minio_conn",
    "rag.utils.opendal_conn", "rag.utils.s3_conn", "rag.utils.oss_conn",
]:
    sys.modules.setdefault(_m, _make_module_stub(_m))

# --- Stage 5: stub rag.nlp (depends on tiktoken + settings) ---
_rag_nlp = _make_module_stub("rag.nlp")
_rag_nlp.rag_tokenizer = MagicMock()
_rag_nlp.search = MagicMock()
sys.modules.setdefault("rag.nlp", _rag_nlp)

# --- Stage 6: stub deepdoc.vision and deepdoc.parser __init__ modules ---
# This prevents their __init__.py from running (which imports all parsers/
# recognizers and pulls in the entire dep graph).  We then manually load
# just the sub-modules we need (vietocr.py, vietocr_parser.py).
_dv = _make_module_stub("deepdoc.vision")
sys.modules.setdefault("deepdoc.vision", _dv)

_dp = _make_module_stub("deepdoc.parser")
sys.modules.setdefault("deepdoc.parser", _dp)

# --- Stage 7: now load our target sub-modules directly ---
# deepdoc.vision.vietocr  (needs cv2, numpy, PIL — all available)
# deepdoc.parser.vietocr_parser (needs pdfplumber — available, PIL — available)
# These will be registered as sub-modules of the stubs above.

# For vietocr.py, it imports from deepdoc.vision.ocr lazily in VietOCR.__init__,
# so the import of the module itself doesn't need OCR to be importable.
spec_vietocr = importlib.util.spec_from_file_location(
    "deepdoc.vision.vietocr",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "deepdoc", "vision", "vietocr.py"),
)
_vietocr_mod = importlib.util.module_from_spec(spec_vietocr)
sys.modules["deepdoc.vision.vietocr"] = _vietocr_mod
spec_vietocr.loader.exec_module(_vietocr_mod)

# For vietocr_parser.py, it tries to import RAGFlowPdfParser which goes through
# deepdoc.parser.pdf_parser → heavy deps. The parser has a try/except fallback.
# But we need to ensure the deepdoc.parser stub allows the import to work.
# The vietocr_parser.py already has a try/except for the import.
spec_parser = importlib.util.spec_from_file_location(
    "deepdoc.parser.vietocr_parser",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "deepdoc", "parser", "vietocr_parser.py"),
)
_parser_mod = importlib.util.module_from_spec(spec_parser)
sys.modules["deepdoc.parser.vietocr_parser"] = _parser_mod
spec_parser.loader.exec_module(_parser_mod)

# --- Stage 8: stub rag.llm __init__ to avoid its module scanner ---
# The rag.llm.__init__ scans all files and imports them, pulling in cv_model
# which needs openai, etc.  We stub it to only expose what we need.
_rag_llm = _make_module_stub("rag.llm")
sys.modules.setdefault("rag.llm", _rag_llm)

# Stub MinerUParser and PaddleOCRParser (imported by ocr_model.py)
_mineru_parser = _make_module_stub("deepdoc.parser.mineru_parser")
_mineru_parser.MinerUParser = type("MinerUParser", (), {})
sys.modules.setdefault("deepdoc.parser.mineru_parser", _mineru_parser)

_paddleocr_parser = _make_module_stub("deepdoc.parser.paddleocr_parser")
_paddleocr_parser.PaddleOCRParser = type("PaddleOCRParser", (), {})
sys.modules.setdefault("deepdoc.parser.paddleocr_parser", _paddleocr_parser)

# Now load ocr_model directly
spec_ocr_model = importlib.util.spec_from_file_location(
    "rag.llm.ocr_model",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "rag", "llm", "ocr_model.py"),
)
_ocr_model_mod = importlib.util.module_from_spec(spec_ocr_model)
sys.modules["rag.llm.ocr_model"] = _ocr_model_mod
spec_ocr_model.loader.exec_module(_ocr_model_mod)

# Populate rag.llm.OcrModel from what ocr_model discovered
import inspect as _inspect
_OcrModel = {}
_base_class = getattr(_ocr_model_mod, "Base", None)
if _base_class is not None:
    for _, _obj in _inspect.getmembers(_ocr_model_mod):
        if _inspect.isclass(_obj) and issubclass(_obj, _base_class) and _obj is not _base_class and hasattr(_obj, "_FACTORY_NAME"):
            if isinstance(_obj._FACTORY_NAME, list):
                for _fn in _obj._FACTORY_NAME:
                    _OcrModel[_fn] = _obj
            else:
                _OcrModel[_obj._FACTORY_NAME] = _obj
_rag_llm.OcrModel = _OcrModel
