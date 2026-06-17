"""
model.py
--------
Production-grade IDEFICS2 (best open Flamingo-lineage model) inference pipeline.

Features:
  - 4-bit / 8-bit quantization via bitsandbytes
  - Flash Attention 2 support
  - Few-shot in-context learning (the Flamingo paradigm)
  - Retry logic for transient failures
  - Structured output with confidence metadata
  - Batch inference support
  - Context manager for safe resource cleanup
"""

from __future__ import annotations

import gc
import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import torch
from loguru import logger
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Idefics2ForConditionalGeneration,
)

from .config import DeviceTarget, ModelConfig, QuantizationMode
from .preprocessing import ImagePreprocessor


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """
    Encapsulates a single model response with rich metadata.

    Attributes
    ----------
    text : str
        Raw decoded text from the model.
    prompt_tokens : int
        Number of tokens in the input prompt.
    generated_tokens : int
        Number of tokens generated.
    latency_ms : float
        Wall-clock inference time in milliseconds.
    image_paths : List[str]
        Paths/URLs of images used in this call.
    model_id : str
        Model identifier for auditability.
    metadata : Dict[str, Any]
        Extensible bag for domain-specific extras (e.g. report sections).
    """
    text: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float
    image_paths: List[str]
    model_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms == 0:
            return 0.0
        return self.generated_tokens / (self.latency_ms / 1000)


@dataclass
class FewShotExample:
    """A single (image, prompt, answer) demonstration for in-context learning."""
    image_source: Union[str, Path, Image.Image]
    prompt: str
    answer: str


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class FlamingoInferencePipeline:
    """
    Production inference pipeline for IDEFICS2-8B (best open Flamingo-lineage model).

    The pipeline follows the Flamingo paradigm:
      1. Encode few-shot (image, text) demonstrations into a single interleaved prompt
      2. Append the query image + question
      3. Let the LLM autoregressively generate the answer

    Usage
    -----
    with FlamingoInferencePipeline(config) as pipeline:
        result = pipeline.generate(
            image_path="xray.jpg",
            prompt="Describe findings in this chest X-ray.",
            few_shot_examples=[...],
        )
        print(result.text)
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.preprocessor = ImagePreprocessor(config.preprocessing)
        self.model: Optional[Idefics2ForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None
        self._result_cache: Dict[str, InferenceResult] = {}
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load IDEFICS2 model and processor with configured precision / quantization."""
        logger.info(f"Loading model: {self.config.model_id}")
        logger.info(f"Quantization: {self.config.quantization}  |  Device: {self.config.device}")

        # ---- BitsAndBytes quantization config ----
        bnb_config = self._build_bnb_config()

        # ---- Torch dtype ----
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        # ---- Model kwargs ----
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.cache_dir:
            model_kwargs["cache_dir"] = str(self.config.cache_dir)
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self.config.device.value

        if self.config.use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 enabled.")
            except ImportError:
                logger.warning("flash-attn not installed — falling back to eager attention.")

        # ---- Load ----
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model = Idefics2ForConditionalGeneration.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )
        self.model.eval()
        elapsed = time.time() - t0
        logger.info(f"Model loaded in {elapsed:.1f}s")

    def _build_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """Build BitsAndBytesConfig based on quantization setting."""
        if self.config.quantization == QuantizationMode.INT4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,      # QLoRA double quantization
                bnb_4bit_quant_type="nf4",            # Normal Float 4 — best for LLMs
            )
        if self.config.quantization == QuantizationMode.INT8:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def generate(
        self,
        image_path: Union[str, Path, Image.Image],
        prompt: str,
        few_shot_examples: Optional[List[FewShotExample]] = None,
        system_context: Optional[str] = None,
        enhance_contrast: Optional[bool] = None,
    ) -> InferenceResult:
        """
        Run few-shot visual inference on a single image.

        Parameters
        ----------
        image_path :
            Path, URL, or PIL Image of the query image.
        prompt :
            The question / instruction for the query image.
        few_shot_examples :
            List of FewShotExample demonstrations prepended to the prompt.
            Following the Flamingo paradigm — no gradient updates needed.
        system_context :
            Optional system-level instruction (e.g. "You are a radiologist...").
        enhance_contrast :
            Override CLAHE setting per call.

        Returns
        -------
        InferenceResult
        """
        # ---- Cache check ----
        cache_key = self._make_cache_key(image_path, prompt, few_shot_examples)
        if self.config.cache.enable_result_cache and cache_key in self._result_cache:
            logger.debug("Cache hit — returning cached result.")
            return self._result_cache[cache_key]

        # ---- Preprocess images ----
        few_shot = few_shot_examples or []
        all_images: List[Image.Image] = []

        for ex in few_shot:
            all_images.append(self.preprocessor.load_and_preprocess(ex.image_source, enhance_contrast))
        query_image = self.preprocessor.load_and_preprocess(image_path, enhance_contrast)
        all_images.append(query_image)

        # ---- Build interleaved prompt ----
        messages = self._build_messages(
            few_shot_examples=few_shot,
            query_prompt=prompt,
            system_context=system_context,
        )

        # ---- Tokenize ----
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=all_images,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[-1]

        # ---- Generate ----
        result = self._generate_with_retry(inputs, prompt_tokens, image_path, few_shot)

        # ---- Cache store ----
        if self.config.cache.enable_result_cache:
            if len(self._result_cache) >= self.config.cache.max_cache_entries:
                # Evict oldest entry (simple FIFO)
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            self._result_cache[cache_key] = result

        return result

    @retry(
        retry=retry_if_exception_type((RuntimeError, torch.cuda.OutOfMemoryError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _generate_with_retry(
        self,
        inputs: Dict[str, torch.Tensor],
        prompt_tokens: int,
        image_path: Any,
        few_shot: List[FewShotExample],
    ) -> InferenceResult:
        """Inner generation call wrapped with retry logic."""
        gen_cfg = self.config.generation
        t_start = time.perf_counter()

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_cfg.max_new_tokens,
                min_new_tokens=gen_cfg.min_new_tokens,
                temperature=gen_cfg.temperature if gen_cfg.do_sample else None,
                top_p=gen_cfg.top_p if gen_cfg.do_sample else None,
                top_k=gen_cfg.top_k if gen_cfg.do_sample else None,
                do_sample=gen_cfg.do_sample,
                repetition_penalty=gen_cfg.repetition_penalty,
                no_repeat_ngram_size=gen_cfg.no_repeat_ngram_size,
                num_beams=gen_cfg.num_beams,
                length_penalty=gen_cfg.length_penalty,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Decode only newly generated tokens (strip the prompt)
        new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        generated_tokens = new_tokens.shape[-1]

        image_paths = [str(ex.image_source) for ex in few_shot] + [str(image_path)]

        logger.info(
            f"Generated {generated_tokens} tokens in {latency_ms:.0f}ms "
            f"({generated_tokens / (latency_ms/1000):.1f} tok/s)"
        )
        return InferenceResult(
            text=decoded,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            latency_ms=latency_ms,
            image_paths=image_paths,
            model_id=self.config.model_id,
        )

    def generate_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[InferenceResult]:
        """
        Process multiple (image, prompt) pairs sequentially.
        Each item dict: {"image_path": ..., "prompt": ..., "few_shot_examples": [...]}
        """
        results = []
        for i, item in enumerate(items):
            logger.info(f"Batch item {i+1}/{len(items)}")
            try:
                result = self.generate(
                    image_path=item["image_path"],
                    prompt=item["prompt"],
                    few_shot_examples=item.get("few_shot_examples"),
                    system_context=item.get("system_context"),
                    enhance_contrast=item.get("enhance_contrast"),
                )
                results.append(result)
            except Exception as exc:
                logger.error(f"Batch item {i} failed: {exc}")
                results.append(InferenceResult(
                    text=f"[ERROR: {exc}]",
                    prompt_tokens=0,
                    generated_tokens=0,
                    latency_ms=0,
                    image_paths=[str(item.get("image_path", ""))],
                    model_id=self.config.model_id,
                ))
        return results

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        few_shot_examples: List[FewShotExample],
        query_prompt: str,
        system_context: Optional[str],
    ) -> List[Dict]:
        """
        Build the IDEFICS2 chat-template message list with interleaved images.

        Structure (following the Flamingo in-context learning pattern):
            [SYSTEM]
            [Image1] <Few-shot question 1>
            Assistant: <Answer 1>
            [Image2] <Few-shot question 2>
            Assistant: <Answer 2>
            ...
            [Query Image] <Query question>
        """
        messages = []

        # System message
        if system_context:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_context}]
            })

        # Few-shot demonstrations
        for ex in few_shot_examples:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex.prompt},
                ]
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": ex.answer}]
            })

        # Query (the actual input we want answered)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query_prompt},
            ]
        })

        return messages

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(
        image_path: Any,
        prompt: str,
        few_shot_examples: Optional[List[FewShotExample]],
    ) -> str:
        parts = [str(image_path), prompt]
        if few_shot_examples:
            for ex in few_shot_examples:
                parts.extend([str(ex.image_source), ex.prompt, ex.answer])
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear the result cache and free GPU memory."""
        self._result_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        logger.info("Cache cleared.")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "FlamingoInferencePipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.clear_cache()
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Pipeline resources released.")
