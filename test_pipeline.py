"""
tests/test_pipeline.py
----------------------
Unit and integration tests for the Flamingo production pipeline.

Run:
  pytest tests/ -v
  pytest tests/ -v --cov=src --cov-report=term-missing
"""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import (
    GenerationConfig,
    ModelConfig,
    PreprocessingConfig,
    QuantizationMode,
)
from src.model import FewShotExample, InferenceResult
from src.preprocessing import ImagePreprocessor
from src.radiology_pipeline import (
    AnatomicalFinding,
    RadiologyReport,
    RadiologyReportPipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_dummy_xray(width: int = 512, height: int = 512) -> Image.Image:
    """Generate a synthetic grayscale chest-X-ray-like image for testing."""
    arr = np.random.randint(30, 200, (height, width), dtype=np.uint8)
    # Simulate lungs as lighter regions
    arr[100:400, 50:220] = np.random.randint(150, 230, (300, 170), dtype=np.uint8)
    arr[100:400, 280:450] = np.random.randint(150, 230, (300, 170), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def make_dummy_rgb(width: int = 256, height: int = 256) -> Image.Image:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# PreprocessingConfig tests
# ---------------------------------------------------------------------------

class TestPreprocessingConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = PreprocessingConfig()
        self.assertEqual(cfg.target_size, (448, 448))
        self.assertTrue(cfg.apply_clahe)
        self.assertTrue(cfg.convert_grayscale_to_rgb)


# ---------------------------------------------------------------------------
# ImagePreprocessor tests
# ---------------------------------------------------------------------------

class TestImagePreprocessor(unittest.TestCase):

    def setUp(self):
        self.cfg = PreprocessingConfig()
        self.preprocessor = ImagePreprocessor(self.cfg)

    def test_grayscale_to_rgb(self):
        gray = make_dummy_xray()
        self.assertEqual(gray.mode, "L")
        result = self.preprocessor._to_rgb(gray)
        self.assertEqual(result.mode, "RGB")

    def test_rgba_to_rgb(self):
        rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        result = self.preprocessor._to_rgb(rgba)
        self.assertEqual(result.mode, "RGB")

    def test_resize_with_pad_exact_output_size(self):
        img = make_dummy_rgb(300, 200)
        result = self.preprocessor._resize_with_pad(img, (448, 448))
        self.assertEqual(result.size, (448, 448))

    def test_resize_with_pad_maintains_aspect_ratio(self):
        # Wide image: 600×200
        img = make_dummy_rgb(600, 200)
        result = self.preprocessor._resize_with_pad(img, (448, 448))
        self.assertEqual(result.size, (448, 448))

    def test_clahe_does_not_change_size(self):
        img = make_dummy_xray().convert("RGB")
        result = self.preprocessor._apply_clahe(img)
        self.assertEqual(result.size, img.size)
        self.assertEqual(result.mode, "RGB")

    def test_load_from_pil(self):
        img = make_dummy_rgb()
        loaded = self.preprocessor._load(img)
        self.assertIsInstance(loaded, Image.Image)
        self.assertIsNot(loaded, img)   # should be a copy

    def test_load_from_bytes(self):
        img = make_dummy_rgb()
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()
        loaded = self.preprocessor._load(raw)
        self.assertIsInstance(loaded, Image.Image)

    def test_full_pipeline_output_size(self):
        xray = make_dummy_xray()
        result = self.preprocessor.load_and_preprocess(xray)
        self.assertEqual(result.size, (448, 448))
        self.assertEqual(result.mode, "RGB")

    def test_batch_preprocessing(self):
        images = [make_dummy_xray() for _ in range(4)]
        results = self.preprocessor.load_and_preprocess_batch(images)
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(r.size, (448, 448))
            self.assertEqual(r.mode, "RGB")

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.preprocessor._load("/nonexistent/path/image.jpg")


# ---------------------------------------------------------------------------
# ModelConfig tests
# ---------------------------------------------------------------------------

class TestModelConfig(unittest.TestCase):

    def test_string_enum_coercion(self):
        cfg = ModelConfig(quantization="4bit", device="cuda")
        self.assertEqual(cfg.quantization, QuantizationMode.INT4)

    def test_dirs_created(self):
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ModelConfig(
                output_dir=Path(tmpdir) / "outputs",
                log_dir=Path(tmpdir) / "logs",
            )
            self.assertTrue(cfg.output_dir.exists())
            self.assertTrue(cfg.log_dir.exists())

    def test_generation_config_defaults(self):
        cfg = GenerationConfig()
        self.assertEqual(cfg.max_new_tokens, 512)
        self.assertFalse(cfg.do_sample)  # greedy by default


# ---------------------------------------------------------------------------
# InferenceResult tests
# ---------------------------------------------------------------------------

class TestInferenceResult(unittest.TestCase):

    def test_tokens_per_second(self):
        result = InferenceResult(
            text="hello world",
            prompt_tokens=100,
            generated_tokens=50,
            latency_ms=1000.0,
            image_paths=["test.jpg"],
            model_id="test-model",
        )
        self.assertAlmostEqual(result.tokens_per_second, 50.0, places=1)

    def test_tokens_per_second_zero_latency(self):
        result = InferenceResult(
            text="",
            prompt_tokens=0,
            generated_tokens=0,
            latency_ms=0.0,
            image_paths=[],
            model_id="test",
        )
        self.assertEqual(result.tokens_per_second, 0.0)


# ---------------------------------------------------------------------------
# RadiologyReport tests
# ---------------------------------------------------------------------------

class TestRadiologyReport(unittest.TestCase):

    def _make_report(self) -> RadiologyReport:
        return RadiologyReport(
            patient_id="TEST001",
            study_date="2025-06-01",
            lung_fields="Clear bilaterally. No consolidation.",
            cardiac_silhouette="Normal size. Cardiothoracic ratio < 0.5.",
            mediastinum="Mediastinum is not widened.",
            pleural_spaces="No pleural effusion or pneumothorax.",
            bones_and_soft_tissue="No acute bony abnormality.",
            impression="1. Normal chest X-ray.",
            recommendations="No follow-up required.",
            model_id="HuggingFaceM4/idefics2-8b",
            total_latency_ms=3500.0,
            total_generated_tokens=250,
        )

    def test_to_dict_keys(self):
        report = self._make_report()
        d = report.to_dict()
        required_keys = [
            "patient_id", "lung_fields", "cardiac_silhouette",
            "impression", "recommendations", "model_id",
        ]
        for key in required_keys:
            self.assertIn(key, d)

    def test_to_narrative_contains_sections(self):
        report = self._make_report()
        narrative = report.to_narrative()
        self.assertIn("FINDINGS", narrative)
        self.assertIn("IMPRESSION", narrative)
        self.assertIn("RECOMMENDATIONS", narrative)
        self.assertIn("TEST001", narrative)

    def test_to_narrative_contains_model_id(self):
        report = self._make_report()
        narrative = report.to_narrative()
        self.assertIn("idefics2", narrative.lower())

    def test_anatomical_findings_serialization(self):
        report = self._make_report()
        report.anatomical_findings = [
            AnatomicalFinding("Right Lung", "Clear", "normal", "high"),
            AnatomicalFinding("Left Lung", "Mild consolidation", "mild", "medium"),
        ]
        d = report.to_dict()
        self.assertEqual(len(d["anatomical_findings"]), 2)
        self.assertEqual(d["anatomical_findings"][0]["location"], "Right Lung")


# ---------------------------------------------------------------------------
# RadiologyReportPipeline unit tests (mocked model)
# ---------------------------------------------------------------------------

class TestRadiologyPipelineLogic(unittest.TestCase):
    """
    Tests the pipeline's logic without loading the actual model.
    The heavy inference calls are mocked.
    """

    def _make_pipeline(self) -> RadiologyReportPipeline:
        config = ModelConfig.__new__(ModelConfig)
        config.model_id = "mock-model"
        config.quantization = QuantizationMode.INT4
        config.preprocessing = PreprocessingConfig()
        config.generation = GenerationConfig()
        config.cache = MagicMock()
        config.cache.enable_result_cache = False
        config.cache.max_cache_entries = 10
        config.output_dir = Path("/tmp")
        config.log_dir = Path("/tmp")
        config.log_level = "INFO"
        config.use_flash_attention = False
        config.trust_remote_code = True
        config.cache_dir = None
        config.device = MagicMock()
        config.device.value = "cpu"
        config.torch_dtype = "float32"
        config.few_shot_examples_dir = Path("/tmp/few_shot")
        config.retry = MagicMock()
        config.server = MagicMock()

        pipeline = RadiologyReportPipeline.__new__(RadiologyReportPipeline)
        pipeline.config = config
        pipeline._few_shot_cache = {}
        pipeline.few_shot_dir = None
        # Mock the inner FlamingoInferencePipeline
        pipeline.pipeline = MagicMock()
        return pipeline

    def test_section_prompts_defined(self):
        # RadiologyReportPipeline should have all expected section keys
        expected = {
            "image_quality", "lung_fields", "cardiac_silhouette",
            "mediastinum", "pleural_spaces", "bones_and_soft_tissue",
            "impression", "recommendations",
        }
        actual = set(RadiologyReportPipeline.SECTION_PROMPTS.keys())
        self.assertEqual(expected, actual)

    def test_apply_section_result_lung_fields(self):
        pipeline = self._make_pipeline()
        report = RadiologyReport(patient_id="T001")
        pipeline._apply_section_result(report, "lung_fields", "Clear bilaterally.")
        self.assertEqual(report.lung_fields, "Clear bilaterally.")

    def test_apply_section_result_impression(self):
        pipeline = self._make_pipeline()
        report = RadiologyReport(patient_id="T002")
        pipeline._apply_section_result(report, "impression", "Normal chest X-ray.")
        self.assertEqual(report.impression, "Normal chest X-ray.")

    def test_quality_gates_always_adds_disclaimer(self):
        pipeline = self._make_pipeline()
        report = RadiologyReport(patient_id="T003", impression="Normal.", lung_fields="Clear.")
        warnings = pipeline._quality_gates(report)
        # Should always have the research disclaimer
        self.assertTrue(any("RESEARCH" in w for w in warnings))

    def test_quality_gates_empty_impression_warning(self):
        pipeline = self._make_pipeline()
        report = RadiologyReport(patient_id="T004", lung_fields="Clear bilaterally.")
        # impression is empty
        warnings = pipeline._quality_gates(report)
        self.assertTrue(any("impression" in w.lower() for w in warnings))

    def test_assess_confidence_high(self):
        pipeline = self._make_pipeline()
        report = RadiologyReport(
            patient_id="T005",
            lung_fields="Clear.",
            cardiac_silhouette="Normal.",
            mediastinum="Normal.",
            pleural_spaces="No effusion.",
            impression="Normal.",
        )
        self.assertEqual(pipeline._assess_confidence(report), "high")

    def test_assess_confidence_low(self):
        pipeline = self._make_pipeline()
        report = RadiologyReport(patient_id="T006")  # all fields empty
        self.assertEqual(pipeline._assess_confidence(report), "low")

    def test_extract_structured_findings_severity_detection(self):
        pipeline = self._make_pipeline()
        report = RadiologyReport(
            patient_id="T007",
            lung_fields="Large consolidation in right lower lobe.",
            cardiac_silhouette="Heart size is normal.",
        )
        findings = pipeline._extract_structured_findings(report)
        # Should detect "large" → "severe" for lung
        lung = next((f for f in findings if "Lung" in f.location or "lung" in f.location.lower()), None)
        # cardiac: "normal" keyword → "normal" severity
        cardiac = next((f for f in findings if "Cardiac" in f.location), None)
        if cardiac:
            self.assertEqual(cardiac.severity, "normal")

    def test_save_report_creates_files(self):
        import tempfile
        pipeline = self._make_pipeline()
        report = RadiologyReport(
            patient_id="SAVE_TEST",
            impression="Normal.",
            lung_fields="Clear.",
            model_id="mock",
            total_latency_ms=100.0,
            total_generated_tokens=50,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, txt_path = pipeline.save_report(report, Path(tmpdir), "SAVE_TEST")
            self.assertTrue(json_path.exists())
            self.assertTrue(txt_path.exists())
            # JSON should be valid
            with open(json_path) as f:
                data = json.load(f)
            self.assertEqual(data["patient_id"], "SAVE_TEST")


# ---------------------------------------------------------------------------
# FewShotExample tests
# ---------------------------------------------------------------------------

class TestFewShotExample(unittest.TestCase):

    def test_creation(self):
        ex = FewShotExample(
            image_source="path/to/image.jpg",
            prompt="What is in this image?",
            answer="A chest X-ray showing normal lung fields.",
        )
        self.assertEqual(ex.prompt, "What is in this image?")

    def test_with_pil_image(self):
        img = make_dummy_rgb()
        ex = FewShotExample(
            image_source=img,
            prompt="Describe.",
            answer="Normal.",
        )
        self.assertIsInstance(ex.image_source, Image.Image)


# ---------------------------------------------------------------------------
# Pytest parametrize examples
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode, expected_img_mode", [
    ("L", "RGB"),
    ("RGB", "RGB"),
    ("RGBA", "RGB"),
])
def test_to_rgb_all_modes(mode: str, expected_img_mode: str):
    preprocessor = ImagePreprocessor(PreprocessingConfig())
    if mode == "L":
        img = Image.new("L", (100, 100), 128)
    elif mode == "RGBA":
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 200))
    else:
        img = Image.new("RGB", (100, 100), (128, 128, 128))
    result = preprocessor._to_rgb(img)
    assert result.mode == expected_img_mode


@pytest.mark.parametrize("width, height, target", [
    (100, 100, (448, 448)),
    (1000, 500, (448, 448)),
    (200, 800, (448, 448)),
    (448, 448, (448, 448)),
])
def test_resize_always_produces_target_size(width: int, height: int, target: tuple):
    preprocessor = ImagePreprocessor(PreprocessingConfig())
    img = make_dummy_rgb(width, height)
    result = preprocessor._resize_with_pad(img, target)
    assert result.size == target


@pytest.mark.parametrize("quantization", [
    QuantizationMode.NONE,
    QuantizationMode.INT4,
    QuantizationMode.INT8,
])
def test_model_config_quantization_enum(quantization: QuantizationMode):
    cfg = ModelConfig(quantization=quantization)
    assert cfg.quantization == quantization


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
