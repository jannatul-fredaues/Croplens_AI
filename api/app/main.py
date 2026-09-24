"""
FastAPI entrypoint. Run locally with:

    uvicorn app.main:app --reload --port 8000

or `make run` from the repo root.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.inference import InvalidImageError, is_loaded, load_model, predict
from app.knowledge import load_knowledge
from app.schemas import ErrorResponse, HealthResponse, PredictResponse

settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("croplens.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the model and care data once, not per request.
    # In app_env=test, tests stub app.inference directly and skip this.
    if settings.app_env != "test":
        load_model()
        load_knowledge()
    yield
    # No teardown needed yet.


app = FastAPI(
    title="CropLens API",
    description="Predicts crop/plant class from a photo and returns care guidance.",
    version=settings.model_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=settings.model_version,
        model_loaded=is_loaded(),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or unreadable image"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported content type"},
    },
    tags=["prediction"],
)
async def predict_endpoint(file: UploadFile = File(...)) -> PredictResponse:
    if file.content_type not in settings.content_types_list:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'. "
            f"Allowed: {', '.join(settings.content_types_list)}",
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_mb} MB limit.",
        )

    try:
        return predict(file_bytes)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        # Model not loaded - a server-side problem, not a client error.
        logger.exception("Prediction failed: model not ready")
        raise HTTPException(status_code=503, detail="Model not ready.") from e
