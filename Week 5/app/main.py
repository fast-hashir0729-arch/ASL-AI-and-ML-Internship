"""FastAPI application: routes only. The logic lives in the other modules."""

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import FRONTEND_FILE, MODEL_NAME, MODEL_VERSION
from app.errors import (
    http_error_handler,
    unexpected_error_handler,
    validation_error_handler,
)
from app.model_loader import load_metadata, load_model
from app.prediction_logger import log_prediction
from app.predictor import predict_passengers
from app.schemas import BatchRequest, BatchResponse, Passenger, PredictionResponse
from app.security import check_access

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE when the server starts, not on every request."""
    try:
        app.state.model = load_model()
        app.state.metadata = load_metadata()
    except Exception as error:
        logger.error("Could not load model: %s", error)
        app.state.model = None
        app.state.metadata = {}
    yield


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Serves the tuned Gradient Boosting model from Week 4.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(StarletteHTTPException, http_error_handler)
app.add_exception_handler(Exception, unexpected_error_handler)

router = APIRouter()


def get_model(request: Request):
    """Give an endpoint the loaded model, or return 503 if loading failed."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return model


@router.get("/health")
def health(request: Request):
    """Report whether the model loaded successfully."""
    model_loaded = request.app.state.model is not None
    body = {
        "status": "ok" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }
    status_code = 200 if model_loaded else 503
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model-info")
def model_info(request: Request):
    """Show what the model was trained on and how well it scored."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "metadata": request.app.state.metadata,
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(check_access)],
)
def predict(passenger: Passenger, model=Depends(get_model)):
    """Predict survival for one passenger."""
    result = predict_passengers(model, [passenger])[0]
    log_prediction("/predict", passenger.model_dump(), result)
    return {**result, "model_name": MODEL_NAME, "model_version": MODEL_VERSION}


@router.post(
    "/predict/batch",
    response_model=BatchResponse,
    dependencies=[Depends(check_access)],
)
def predict_batch(batch: BatchRequest, model=Depends(get_model)):
    """Predict survival for a list of passengers (bonus feature)."""
    results = predict_passengers(model, batch.records)
    inputs = [passenger.model_dump() for passenger in batch.records]
    log_prediction("/predict/batch", inputs, results)
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "count": len(results),
        "predictions": results,
    }


# Same routes at /predict (simple) and /v1/predict (stable, versioned path)
app.include_router(router)
app.include_router(router, prefix="/v1", include_in_schema=False)


@app.get("/", include_in_schema=False)
def home():
    """Serve the small HTML form (bonus front-end)."""
    return FileResponse(FRONTEND_FILE)
