"""
config.py
---------
Centralized configuration for the Flamingo/IDEFICS2 production pipeline.
All tuneable knobs live here — no magic numbers scattered in business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class QuantizationMode(str, Enum):
    NONE  = "none"   # full precision (bf16 / fp16)
    INT8  = "8bit"   # bitsandbytes LLM.int8()
    INT4  = "4bit"   # bitsandbytes NF4 (QLoRA-style)


class DeviceTarget(str, Enum):
    CUDA = "cuda"
    CPU  = "cpu"
    MPS  = "mps"     # Apple Silicon


class ReportSectionEnum(str, Enum):
    CLINICAL_INFO    = "clinical_information"
    TECHNIQUE        = "technique"
    FINDINGS         = "findings"
    IMPRESSION       = "impression"
    RECOMMENDATIONS  = "recommendations"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Controls how the LLM decodes tokens."""
    max_new_tokens: int       = 512
    min_new_tokens: int       = 32
    temperature: float        = 0.2        # low = more deterministic (good for clinical)
    top_p: float              = 0.9
    top_k: int                = 50
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 4
    do_sample: bool           = False      # greedy by default for reproducibility
    num_beams: int            = 1          # beam search (>1 is slower but more coherent)
    length_penalty: float     = 1.0


@dataclass
class PreprocessingConfig:
    """Image preprocessing parameters."""
    target_size: tuple         = (448, 448)   # IDEFICS2 native resolution
    apply_clahe: bool          = True          # contrast enhancement for X-rays
    clahe_clip_limit: float    = 2.0
    clahe_tile_grid: tuple     = (8, 8)
    normalize_mean: tuple      = (0.485, 0.456, 0.406)
    normalize_std: tuple       = (0.229, 0.224, 0.225)
    convert_grayscale_to_rgb: bool = True     # X-rays are grayscale


@dataclass
class RetryConfig:
    """Tenacity retry settings for transient failures."""
    max_attempts: int   = 3
    wait_min_sec: float = 1.0
    wait_max_sec: float = 8.0


@dataclass
class ServerConfig:
    """FastAPI server settings."""
    host: str          = "0.0.0.0"
    port: int          = 8080
    workers: int       = 1          # single worker — model loaded once in memory
    reload: bool       = False
    log_level: str     = "info"
    timeout_keep_alive: int = 60


@dataclass
class CacheConfig:
    """Optional KV-cache / result cache."""
    enable_result_cache: bool   = True
    cache_ttl_seconds: int      = 3600      # 1 hour
    max_cache_entries: int      = 256


# ---------------------------------------------------------------------------
# Master ModelConfig
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Top-level configuration passed to FlamingoInferencePipeline.

    Example
    -------
    config = ModelConfig(
        model_id="HuggingFaceM4/idefics2-8b",
        quantization=QuantizationMode.INT4,
        device=DeviceTarget.CUDA,
    )
    """

    # ---- Model identity ----
    model_id: str = "HuggingFaceM4/idefics2-8b"

    # ---- Runtime ----
    device: DeviceTarget           = DeviceTarget.CUDA
    quantization: QuantizationMode = QuantizationMode.INT4
    torch_dtype: str               = "bfloat16"   # "float16" on older GPUs
    trust_remote_code: bool        = True
    use_flash_attention: bool      = True          # requires flash-attn package

    # ---- Paths ----
    cache_dir: Optional[Path]      = None          # HF_HOME override
    few_shot_examples_dir: Path    = Path("data/few_shot_examples")
    output_dir: Path               = Path("outputs")
    log_dir: Path                  = Path("logs")

    # ---- Sub-configs ----
    generation: GenerationConfig   = field(default_factory=GenerationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    retry: RetryConfig             = field(default_factory=RetryConfig)
    server: ServerConfig           = field(default_factory=ServerConfig)
    cache: CacheConfig             = field(default_factory=CacheConfig)

    # ---- Logging ----
    log_level: str = "INFO"

    def __post_init__(self):
        # Coerce string enums if plain strings were passed
        if isinstance(self.device, str):
            self.device = DeviceTarget(self.device)
        if isinstance(self.quantization, str):
            self.quantization = QuantizationMode(self.quantization)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
