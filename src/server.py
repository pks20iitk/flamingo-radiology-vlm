"""
server.py
---------
FastAPI REST API exposing the Flamingo radiology report pipeline.

Endpoints:
  POST /report/generate   - Generate radiology report from uploaded image
  POST /vqa               - General visual question answering
  GET  /health            - Health check with GPU stats
  GET  /model/info        - Model metadata

Usage:
  python -m src.server
  # or
  uvicorn src.server:app --host 0.0.0.0 --port 8080 --workers 1
"""

from __future__ import annotations

import base64
import io
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from .config import ModelConfig, QuantizationMode
from .model import FewShotExample, FlamingoInferencePipeline
from .radiology_pipeline import RadiologyReport, RadiologyReportPipeline


# ---------------------------------------------------------------------------
# Global pipeline instance (loaded once at startup)
# ---------------------------------------------------------------------------

_pipeline: Optional[RadiologyReportPipeline] = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup, release at shutdown."""
    global _pipeline, _start_time
    _start_time = time.time()
    logger.info("Server starting — loading Flamingo/IDEFICS2 pipeline...")

    config = ModelConfig(
        model_id="HuggingFaceM4/idefics2-8b",
        quantization=QuantizationMode.INT4,
    )
    _pipeline = RadiologyReportPipeline(config)
    logger.info("Pipeline loaded. Server ready.")
    yield
    # Cleanup
    if _pipeline:
        del _pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Server shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Flamingo VLM API",
    description=(
        "Production REST API for IDEFICS2-8B (Flamingo-lineage) inference. "
        "Supports radiology report generation and general VQA."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RadiologyRequest(BaseModel):
    """Request body for /report/generate (JSON mode)."""
    image_base64: str = Field(..., description="Base64-encoded image (JPEG/PNG)")
    patient_id: str   = Field(default="UNKNOWN")
    study_date: str   = Field(default="")
    clinical_info: str = Field(default="")
    modality: str     = Field(default="Chest X-Ray")
    sections: Optional[List[str]] = Field(
        default=None,
        description="Subset of sections to run. None = all sections."
    )


class VQARequest(BaseModel):
    """Request body for /vqa."""
    image_base64: str = Field(..., description="Base64-encoded image")
    question: str     = Field(..., description="Question about the image")
    few_shot_examples: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=(
            "List of {image_base64, question, answer} dicts for few-shot prompting. "
            "Following the Flamingo in-context learning paradigm."
        ),
    )
    system_context: Optional[str] = None
    enhance_contrast: bool = False


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_id: str
    cuda_available: bool
    gpu_allocated_gb: Optional[float]
    gpu_reserved_gb: Optional[float]


class VQAResponse(BaseModel):
    request_id: str
    answer: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(b64: str) -> Image.Image:
    """Decode base64 image string to PIL Image."""
    try:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")


def _get_pipeline() -> RadiologyReportPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not yet loaded.")
    return _pipeline


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check server health and GPU stats."""
    pipeline = _get_pipeline()
    mem = pipeline.pipeline.get_memory_stats()
    return HealthResponse(
        status="ok",
        uptime_seconds=time.time() - _start_time,
        model_id=pipeline.config.model_id,
        cuda_available=mem.get("cuda_available", False),
        gpu_allocated_gb=mem.get("allocated_gb"),
        gpu_reserved_gb=mem.get("reserved_gb"),
    )


@app.get("/model/info", tags=["System"])
async def model_info():
    """Return model metadata."""
    pipeline = _get_pipeline()
    return {
        "model_id": pipeline.config.model_id,
        "quantization": pipeline.config.quantization.value,
        "device": pipeline.config.device.value,
        "torch_dtype": pipeline.config.torch_dtype,
        "max_new_tokens": pipeline.config.generation.max_new_tokens,
        "supported_sections": list(pipeline.SECTION_PROMPTS.keys()),
    }


@app.post("/report/generate", tags=["Radiology"])
async def generate_report(request: RadiologyRequest) -> JSONResponse:
    """
    Generate a structured radiology report from a chest X-ray.

    Accepts a base64-encoded image and returns a full structured report
    with findings per anatomical region, impression, and recommendations.

    Following the Flamingo few-shot paradigm: no fine-tuning required —
    the model generalises from in-context examples.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Radiology report request — patient={request.patient_id}")

    pipeline = _get_pipeline()
    image = _decode_image(request.image_base64)

    try:
        run_all = request.sections is None
        report: RadiologyReport = pipeline.generate_report(
            image_path=image,
            patient_id=request.patient_id,
            study_date=request.study_date,
            clinical_info=request.clinical_info,
            modality=request.modality,
            run_all_sections=run_all,
            sections=request.sections,
        )
    except Exception as exc:
        logger.error(f"[{request_id}] Report generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    response = report.to_dict()
    response["request_id"] = request_id
    response["narrative"] = report.to_narrative()
    return JSONResponse(content=response)


@app.post("/report/generate/upload", tags=["Radiology"])
async def generate_report_upload(
    file: UploadFile = File(...),
    patient_id: str = Form(default="UNKNOWN"),
    study_date: str = Form(default=""),
    clinical_info: str = Form(default=""),
    modality: str = Form(default="Chest X-Ray"),
) -> JSONResponse:
    """
    Generate a radiology report from a multipart-uploaded image file.
    Alternative to the base64 JSON endpoint.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Upload-based report request — patient={patient_id}")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {exc}")

    pipeline = _get_pipeline()
    try:
        report = pipeline.generate_report(
            image_path=image,
            patient_id=patient_id,
            study_date=study_date,
            clinical_info=clinical_info,
            modality=modality,
        )
    except Exception as exc:
        logger.error(f"[{request_id}] Failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    response = report.to_dict()
    response["request_id"] = request_id
    response["narrative"] = report.to_narrative()
    return JSONResponse(content=response)


@app.post("/vqa", response_model=VQAResponse, tags=["VQA"])
async def visual_question_answering(request: VQARequest) -> VQAResponse:
    """
    General visual question answering endpoint.

    Supports few-shot in-context learning following the Flamingo paradigm:
    provide a list of (image, question, answer) demonstrations in `few_shot_examples`
    and the model will use them without any gradient updates.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] VQA request: {request.question[:80]}")

    pipeline = _get_pipeline()
    query_image = _decode_image(request.image_base64)

    # Parse few-shot examples
    few_shot: List[FewShotExample] = []
    if request.few_shot_examples:
        for ex in request.few_shot_examples:
            ex_image = _decode_image(ex["image_base64"])
            few_shot.append(FewShotExample(
                image_source=ex_image,
                prompt=ex["question"],
                answer=ex["answer"],
            ))

    try:
        result = pipeline.pipeline.generate(
            image_path=query_image,
            prompt=request.question,
            few_shot_examples=few_shot,
            system_context=request.system_context,
            enhance_contrast=request.enhance_contrast,
        )
    except Exception as exc:
        logger.error(f"[{request_id}] VQA failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return VQAResponse(
        request_id=request_id,
        answer=result.text,
        prompt_tokens=result.prompt_tokens,
        generated_tokens=result.generated_tokens,
        latency_ms=result.latency_ms,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        log_level="info",
    )
