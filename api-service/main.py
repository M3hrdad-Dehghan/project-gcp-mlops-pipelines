import os
import datetime
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from forecast_core import ForecastCore
from forecast_core import BQMLTrainer
from google.auth.exceptions import DefaultCredentialsError
from fastapi.staticfiles import StaticFiles

# ============================
# Main App (for STATIC UI)
# ============================
app = FastAPI(title="Taxi Forecast App UI")

# ============================
# API Router
# ============================
api = FastAPI(
    title="Taxi Demand Forecasting API",
    version="1.0.0"
)

PROJECT_ID = os.getenv("PROJECT_ID", "ml-ai-portfolio")
DATASET_ID = os.getenv("DATASET_ID", "taxi_forecasting")
MODEL_PATH = os.getenv("MODEL_PATH",
    "ml-ai-portfolio.taxi_forecasting.daily_arima_default_model_v1")


class ForecastRequest(BaseModel):
    start_date: str
    horizon: int = Field(..., ge=1, le=30)

    @validator("start_date")
    def validate_date(cls, v):
        # Validate format and ensure date is within November 2022
        d = datetime.datetime.strptime(v, "%Y-%m-%d").date()
        min_date = datetime.date(2022, 11, 1)
        max_date = datetime.date(2022, 11, 30)
        if d < min_date or d > max_date:
            raise ValueError("start_date must be within November 2022 (2022-11-01 to 2022-11-30)")
        return v


class ForecastResponse(BaseModel):
    meta: dict
    data: list
    error: str | None = None


@api.get("/health")
def health_check():
    return {"status": "ok", "project_id": PROJECT_ID}


@api.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    # Mock mode: return deterministic sample data for local UI testing
    mock_mode = os.getenv("MOCK_FORECAST", "false").lower() in ("1", "true", "yes")
    if mock_mode:
        # Simple 3-day horizon example starting at start_date
        start_date = datetime.datetime.strptime(request.start_date, "%Y-%m-%d").date()
        results = []
        for i in range(request.horizon):
            dt = start_date + datetime.timedelta(days=i)
            results.append({"date": dt.isoformat(), "forecast": float(100 + i)})
        return ForecastResponse(meta={"model": MODEL_PATH, "start_date": request.start_date}, data=results)
    try:
        fc = ForecastCore(MODEL_PATH, PROJECT_ID)
        results = fc.forecast(request.start_date, request.horizon)
        # Build meta with last_query_stats to help the UI show why empty
        meta = {"model": MODEL_PATH, "start_date": request.start_date}
        try:
            if hasattr(fc, "last_query_stats") and fc.last_query_stats:
                meta.update(fc.last_query_stats)
        except Exception:
            pass
        return ForecastResponse(meta=meta, data=results)
    except DefaultCredentialsError as e:
        # Make this explicit and helpful to users testing locally without ADC
        return ForecastResponse(meta={}, data=[], error=(
            "Google Application Default Credentials not found. "
            "If running locally, set GOOGLE_APPLICATION_CREDENTIALS or mount a service account JSON. "
            f"Original error: {e}"
        ))
    except Exception as e:
        # Ensure we always return a JSON body the client can parse
        return ForecastResponse(meta={}, data=[], error=str(e))


class RetrainRequest(BaseModel):
    model_name: str = Field(...)
    source_table: str = Field(...)
    dataset_id: Optional[str] = Field(None)
    horizon: int = Field(30, ge=1)
    cutoff_date: Optional[str] = None


@api.post('/retrain')
def retrain(request: RetrainRequest):
    try:
        dataset = request.dataset_id if request.dataset_id else DATASET_ID
        trainer = BQMLTrainer(PROJECT_ID, dataset)
        cutoff_date = request.cutoff_date
        if not cutoff_date:
            cutoff_date = trainer.get_max_date(request.source_table)
        model_path = trainer.train_arima(request.source_table, request.model_name, horizon=request.horizon)
        return {
            "status": "ok",
            "model_path": model_path,
            "cutoff_date_used": cutoff_date,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================
# Mount API + Static
# ============================
app.mount("/api", api)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
