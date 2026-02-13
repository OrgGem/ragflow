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
"""Unit tests for the OCR service endpoint logic.

These tests validate the helper functions and input validation of the
OCR endpoint without requiring a running server or database.
"""

import os
import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy dependencies so the module can be imported in isolation.
# ---------------------------------------------------------------------------

def _make_module_stub(name: str) -> ModuleType:
    mod = ModuleType(name)
    mod.__path__ = [os.path.join(os.path.dirname(__file__), "__stub_path__")]
    mod.__spec__ = None
    mod.__loader__ = None
    mod.__package__ = name
    return mod


# Stub modules that would pull in the full dependency graph
for _m in [
    "opensearchpy",
    "infinity", "infinity.common",
    "memory", "memory.utils", "memory.utils.es_conn",
    "memory.utils.infinity_conn", "memory.utils.ob_conn",
]:
    sys.modules.setdefault(_m, _make_module_stub(_m))

# Stub common.settings (avoids massive import chain)
_settings_stub = _make_module_stub("common.settings")
for _attr in [
    "LLM", "LLM_FACTORY", "LLM_BASE_URL", "CHAT_MDL", "EMBEDDING_MDL",
    "RERANK_MDL", "ASR_MDL", "IMAGE2TEXT_MDL", "FACTORY_LLM_INFOS",
    "DOC_ENGINE", "docStoreConn", "retrievaler", "kg_retrievaler",
]:
    setattr(_settings_stub, _attr, None)
sys.modules.setdefault("common.settings", _settings_stub)

# Stub token_utils
_token_utils_stub = _make_module_stub("common.token_utils")
_token_utils_stub.num_tokens_from_string = lambda *a, **kw: 0
sys.modules.setdefault("common.token_utils", _token_utils_stub)

# Stub rag.utils.*
for _m in [
    "rag.utils", "rag.utils.es_conn", "rag.utils.infinity_conn",
    "rag.utils.ob_conn", "rag.utils.opensearch_conn",
    "rag.utils.azure_sas_conn", "rag.utils.azure_spn_conn",
    "rag.utils.gcs_conn", "rag.utils.minio_conn",
    "rag.utils.opendal_conn", "rag.utils.s3_conn", "rag.utils.oss_conn",
]:
    sys.modules.setdefault(_m, _make_module_stub(_m))

# Stub rag.nlp
_rag_nlp = _make_module_stub("rag.nlp")
_rag_nlp.rag_tokenizer = MagicMock()
_rag_nlp.search = MagicMock()
sys.modules.setdefault("rag.nlp", _rag_nlp)


# ---------------------------------------------------------------------------
# Now we can safely import the constants and utility functions we test.
# We don't import the full endpoint module (it needs Quart), but we can
# test the standalone helpers and constants.
# ---------------------------------------------------------------------------
from common.constants import LLMType  # noqa: E402


class TestOCREndpointConstants(unittest.TestCase):
    """Verify the OCR endpoint module has correct constants."""

    def _get_module_attrs(self):
        """Import the OCR module's constants without running the route registration."""
        # We can't import the module directly because @manager.route needs
        # the Blueprint. Instead, parse the constants from the source.
        import ast

        ocr_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "api", "apps", "sdk", "ocr.py"
        )
        with open(ocr_path) as f:
            tree = ast.parse(f.read())

        # Extract top-level assignments
        attrs = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                name = getattr(node.targets[0], "id", None)
                if name:
                    try:
                        attrs[name] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        return attrs

    def test_supported_extensions(self):
        attrs = self._get_module_attrs()
        exts = attrs["SUPPORTED_EXTENSIONS"]
        self.assertIn(".pdf", exts)
        self.assertIn(".png", exts)
        self.assertIn(".jpg", exts)
        self.assertIn(".jpeg", exts)
        self.assertIn(".tiff", exts)

    def test_engine_factory_map(self):
        attrs = self._get_module_attrs()
        engine_map = attrs["ENGINE_FACTORY_MAP"]
        self.assertEqual(engine_map["vietocr"], "VietOCR")
        self.assertEqual(engine_map["paddleocr"], "PaddleOCR")
        self.assertEqual(engine_map["mineru"], "MinerU")

    def test_engine_env_ensure(self):
        attrs = self._get_module_attrs()
        env_map = attrs["ENGINE_ENV_ENSURE"]
        self.assertIn("vietocr", env_map)
        self.assertIn("paddleocr", env_map)
        self.assertIn("mineru", env_map)


class TestOCRResolveModelName(unittest.TestCase):
    """Test the _resolve_ocr_model_name helper."""

    def _get_resolve_fn(self):
        """Dynamically load the helper from the source file."""
        import importlib.util

        ocr_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "api", "apps", "sdk", "ocr.py"
        )
        # Create a temporary module with stubs for the decorator-dependent code
        spec = importlib.util.spec_from_file_location("_ocr_test_mod", ocr_path)
        mod = importlib.util.module_from_spec(spec)
        # Provide the `manager` Blueprint mock before exec
        mod.manager = MagicMock()
        sys.modules["_ocr_test_mod"] = mod

        # Mock the imports that would fail
        with patch.dict(sys.modules, {
            "quart": MagicMock(),
            "api.db.services.llm_service": MagicMock(),
            "api.db.services.tenant_llm_service": MagicMock(),
            "api.utils.api_utils": MagicMock(),
        }):
            # We need real constants
            import common.constants
            mod_constants = MagicMock()
            mod_constants.LLMType = common.constants.LLMType
            mod_constants.RetCode = common.constants.RetCode
            sys.modules["common.constants"] = mod_constants
            spec.loader.exec_module(mod)
            # Restore
            sys.modules["common.constants"] = common.constants

        del sys.modules["_ocr_test_mod"]
        return mod._resolve_ocr_model_name, mod

    def test_resolve_returns_candidate_llm_name(self):
        resolve_fn, mod = self._get_resolve_fn()

        mock_candidate = MagicMock()
        mock_candidate.llm_name = "vietocr-model-1"

        with patch.object(mod, "TenantLLMService") as mock_svc:
            mock_svc.query.return_value = [mock_candidate]
            mock_svc.ensure_vietocr_from_env.return_value = None

            result = resolve_fn("tenant-123", "vietocr")
            self.assertEqual(result, "vietocr-model-1")

    def test_resolve_returns_env_name_when_no_candidates(self):
        resolve_fn, mod = self._get_resolve_fn()

        with patch.object(mod, "TenantLLMService") as mock_svc:
            mock_svc.query.return_value = []
            mock_svc.ensure_vietocr_from_env.return_value = "vietocr-from-env-1"

            result = resolve_fn("tenant-123", "vietocr")
            self.assertEqual(result, "vietocr-from-env-1")

    def test_resolve_returns_none_when_nothing_configured(self):
        resolve_fn, mod = self._get_resolve_fn()

        with patch.object(mod, "TenantLLMService") as mock_svc:
            mock_svc.query.return_value = []
            mock_svc.ensure_vietocr_from_env.return_value = None

            result = resolve_fn("tenant-123", "vietocr")
            self.assertIsNone(result)


class TestOCREndpointFileExists(unittest.TestCase):
    """Basic structural checks."""

    def test_ocr_py_exists(self):
        ocr_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "api", "apps", "sdk", "ocr.py"
        )
        self.assertTrue(os.path.exists(ocr_path), "api/apps/sdk/ocr.py must exist")

    def test_ocr_py_has_route_decorator(self):
        ocr_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "api", "apps", "sdk", "ocr.py"
        )
        with open(ocr_path) as f:
            content = f.read()
        self.assertIn('@manager.route("/ocr"', content)
        self.assertIn("@token_required", content)
        self.assertIn("async def ocr(", content)

    def test_ocr_py_has_docstring(self):
        ocr_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "api", "apps", "sdk", "ocr.py"
        )
        with open(ocr_path) as f:
            content = f.read()
        self.assertIn("POST /api/v1/ocr", content)


if __name__ == "__main__":
    unittest.main()
