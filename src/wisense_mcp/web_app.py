"""Starlette application for the browser dashboard and MCP HTTP endpoint."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src.settings import load_settings
from src.wisense_mcp.mcp_server import (
    diagnostics_engine,
    inference_engine,
    mcp,
    repository,
    sample_catalog,
)


settings = load_settings()
PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
CACHE_DIR = settings.path("paths", "cache_dir")

for required in ("index.html", "app.css", "app.js"):
    path = STATIC_DIR / required
    if not path.exists():
        raise RuntimeError(f"Required dashboard file is missing: {path}")

signal_mmaps: dict[str, np.ndarray] = {}
inference_lock = asyncio.Lock()


def json_error(message: str, status_code: int = 400) -> JSONResponse:
    """Create a consistent JSON error response."""

    return JSONResponse({"error": message}, status_code=status_code)


async def home(request: Request) -> FileResponse:
    """Serve the single-page dashboard."""

    del request
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


async def api_health(request: Request) -> JSONResponse:
    """Return lightweight service and accelerator status."""

    del request
    status = inference_engine.get_status()
    return JSONResponse(
        {
            "status": "online",
            "service": settings.get("project", "name"),
            "mcp_endpoint": "/mcp",
            "model_loaded": status["model_loaded"],
            "device": status["device"],
            "cuda_available": status["cuda_available"],
            "gpu": status["gpu"],
        }
    )


async def api_overview(request: Request) -> JSONResponse:
    """Return dataset, final metrics, model status, and sample-filter metadata."""

    del request
    return JSONResponse(
        {
            "dataset": repository.get_dataset_summary(),
            "model": repository.get_final_model_metrics(),
            "analysis": repository.get_error_analysis_summary(),
            "inference": inference_engine.get_status(),
            "sample_filters": sample_catalog.filters("test"),
            "external_input": {
                "formats": settings.get_list("server", "external_sample_formats"),
                "max_upload_mb": settings.getfloat("server", "external_sample_max_mb"),
                "num_channels": settings.getint("dataset", "num_channels"),
                "target_length": settings.getint("preprocessing", "target_length"),
            },
        }
    )


async def api_samples(request: Request) -> JSONResponse:
    """Return filtered sample cards for the explorer page."""

    params = request.query_params
    try:
        result = sample_catalog.list_samples(
            split=params.get("split", "test"),
            limit=int(params.get("limit", 50)),
            offset=int(params.get("offset", 0)),
            environment=params.get("environment"),
            band=None if params.get("band") in (None, "") else float(params["band"]),
            user_count=None if params.get("users") in (None, "") else int(params["users"]),
            search=params.get("search"),
        )
        return JSONResponse(result)
    except (ValueError, KeyError) as error:
        return json_error(str(error))


async def api_sample(request: Request) -> JSONResponse:
    """Return metadata for one sample alias or dataset ID."""

    try:
        return JSONResponse(
            sample_catalog.resolve(
                request.path_params["identifier"], request.query_params.get("split", "test")
            )
        )
    except ValueError as error:
        return json_error(str(error), 404)


async def api_predict(request: Request) -> JSONResponse:
    """Run live ensemble inference for one selected sample."""

    try:
        payload = await request.json()
        identifier = payload.get("sample") or payload.get("sample_id")
        if not identifier:
            return json_error("A sample identifier is required.")
        split = payload.get("split", "test")
        sample = sample_catalog.resolve(identifier, split)
        async with inference_lock:
            prediction = await asyncio.to_thread(
                inference_engine.predict_sample,
                sample["dataset_id"],
                split,
                int(payload.get("top_k", settings.getint("server", "top_k_activities"))),
            )
        prediction["sample"] = sample
        prediction["technical"] = {
            "dataset_id": sample["dataset_id"],
            "row_index": prediction["row_index"],
        }
        prediction.pop("sample_id", None)
        return JSONResponse(prediction)
    except (ValueError, KeyError, FileNotFoundError) as error:
        return json_error(str(error))


async def api_predict_upload(request: Request) -> JSONResponse:
    """Run inference on a compatible CSI sample uploaded from the browser."""

    filename = request.query_params.get("filename", "uploaded_sample").strip()
    suffix = Path(filename).suffix.lower()
    allowed_formats = {
        item.lower()
        for item in settings.get_list("server", "external_sample_formats")
    }
    if suffix not in allowed_formats:
        supported = ", ".join(sorted(allowed_formats))
        return json_error(
            f"Unsupported sample format '{suffix or 'unknown'}'. Supported formats: {supported}."
        )

    max_bytes = int(settings.getfloat("server", "external_sample_max_mb") * 1024 * 1024)
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                return json_error("The uploaded sample is larger than the configured size limit.", 413)
        except ValueError:
            pass

    payload = await request.body()
    if not payload:
        return json_error("The uploaded sample is empty.")
    if len(payload) > max_bytes:
        return json_error("The uploaded sample is larger than the configured size limit.", 413)

    try:
        top_k = int(
            request.query_params.get(
                "top_k",
                settings.getint("server", "top_k_activities"),
            )
        )
        async with inference_lock:
            prediction = await asyncio.to_thread(
                inference_engine.predict_uploaded_sample,
                payload,
                filename,
                top_k,
            )
        return JSONResponse(prediction)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
        return json_error(str(error))


def signal_preview(identifier: str, split: str, points: int) -> dict:
    """Downsample the cached CSI RMS trace for browser visualization."""

    sample = sample_catalog.resolve(identifier, split)
    row_index = int(sample["row_index"])
    if split not in signal_mmaps:
        signal_mmaps[split] = np.load(CACHE_DIR / f"{split}_x.npy", mmap_mode="r")

    tensor = np.asarray(signal_mmaps[split][row_index], dtype=np.float32)
    energy = np.sqrt(np.mean(tensor * tensor, axis=0))
    points = max(50, min(int(points), 500))
    downsampled = np.array([float(group.mean()) for group in np.array_split(energy, points)])
    lower, upper = np.percentile(downsampled, [5, 95])
    normalized = (
        np.clip((downsampled - lower) / (upper - lower), 0.0, 1.0)
        if upper > lower
        else np.zeros_like(downsampled)
    )
    duration = settings.getfloat("dataset", "duration_seconds")
    time_seconds = np.linspace(0.0, duration, len(normalized))
    return {
        "sample": sample,
        "signal_type": "RMS across globally standardized CSI-amplitude channels",
        "duration_seconds": duration,
        "points": [
            {"t": float(time_value), "value": float(value)}
            for time_value, value in zip(time_seconds, normalized, strict=True)
        ],
    }


async def api_signal(request: Request) -> JSONResponse:
    """Return a compact temporal CSI trace for the live-analysis chart."""

    identifier = request.query_params.get("sample")
    if not identifier:
        return json_error("sample query parameter is required.")
    try:
        result = await asyncio.to_thread(
            signal_preview,
            identifier,
            request.query_params.get("split", "test"),
            int(request.query_params.get("points", settings.getint("server", "signal_preview_points"))),
        )
        return JSONResponse(result)
    except (ValueError, FileNotFoundError) as error:
        return json_error(str(error))


async def api_analytics(request: Request) -> JSONResponse:
    """Return final-test activity, confusion, and subgroup analytics."""

    del request
    try:
        return JSONResponse(
            {
                "overall": repository.get_final_model_metrics(),
                "activities": repository.get_activity_performance(),
                "confusions": repository.get_top_confusions(10),
                "bands": repository.get_band_performance(),
                "environments": repository.get_environment_performance(),
                "user_counts": repository.get_user_count_performance(),
                "users": repository.get_user_index_performance(),
            }
        )
    except FileNotFoundError as error:
        return json_error(str(error))


async def api_model(request: Request) -> JSONResponse:
    """Return architecture, checkpoint, and final-metric information for the model page."""

    del request
    status = inference_engine.get_status()
    model_config = None
    existing_seed = next(
        (seed for seed in inference_engine.seeds if inference_engine._checkpoint_path(seed).exists()),
        None,
    )
    if existing_seed is not None:
        import torch

        checkpoint = torch.load(
            inference_engine._checkpoint_path(existing_seed), map_location="cpu", weights_only=False
        )
        model_config = checkpoint.get("model_config")

    return JSONResponse(
        {
            "name": "Temporal-Pyramid Ensemble",
            "architecture": "Temporal CNN + packed BiLSTM + global/segment attention",
            "task": "Multi-user presence detection and user-specific activity recognition",
            "presence_threshold": inference_engine.presence_threshold,
            "input": {
                "shape": [
                    settings.getint("dataset", "num_channels"),
                    settings.getint("preprocessing", "target_length"),
                ],
                "duration_seconds": settings.getfloat("dataset", "duration_seconds"),
                "representation": "Globally standardized WiFi CSI amplitude",
            },
            "outputs": {
                "presence": f"[{settings.num_users}] sigmoid",
                "activity": f"[{settings.num_users}, {settings.num_activities}] softmax",
            },
            "ensemble": {
                "size": len(inference_engine.seeds),
                "seeds": inference_engine.seeds,
                "rule": "equal-weight probability average",
            },
            "checkpoint_epochs": list(status["checkpoint_epochs"].values()),
            "model_config": model_config,
            "final_test": repository.get_final_model_metrics(),
            "inference": status,
        }
    )


async def api_sample_diagnostics(request: Request) -> JSONResponse:
    """Run live CSI and prediction diagnostics for one sample."""

    try:
        payload = await request.json()
        identifier = payload.get("sample")
        if not identifier:
            return json_error("sample is required.")
        async with inference_lock:
            result = await asyncio.to_thread(
                diagnostics_engine.analyze_sample, identifier, payload.get("split", "test")
            )
        return JSONResponse(result)
    except (ValueError, KeyError, FileNotFoundError) as error:
        return json_error(str(error))


async def api_model_errors(request: Request) -> JSONResponse:
    """Return frozen test samples matching an error category and optional metadata filters."""

    params = request.query_params
    try:
        result = await asyncio.to_thread(
            diagnostics_engine.find_model_errors,
            params.get("error_type", "any"),
            int(params.get("limit", 30)),
            params.get("environment"),
            None if params.get("band") in (None, "") else float(params["band"]),
            None if params.get("users") in (None, "") else int(params["users"]),
        )
        return JSONResponse(result)
    except (ValueError, FileNotFoundError) as error:
        return json_error(str(error))


async def api_difficult_samples(request: Request) -> JSONResponse:
    """Return the highest-ranked retrospective uncertainty samples."""

    params = request.query_params
    try:
        result = await asyncio.to_thread(
            diagnostics_engine.find_difficult_samples,
            int(params.get("limit", 30)),
            params.get("environment"),
            None if params.get("band") in (None, "") else float(params["band"]),
            None if params.get("users") in (None, "") else int(params["users"]),
        )
        return JSONResponse(result)
    except (ValueError, FileNotFoundError) as error:
        return json_error(str(error))


mcp_http_app = mcp.streamable_http_app(
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
)


@asynccontextmanager
async def lifespan(app: Starlette):
    """Run the MCP session manager for the lifetime of the parent Starlette app."""

    del app
    async with mcp.session_manager.run():
        yield


app = Starlette(
    debug=False,
    routes=[
        Route("/", home),
        Route("/api/health", api_health),
        Route("/api/overview", api_overview),
        Route("/api/samples", api_samples),
        Route("/api/samples/{identifier}", api_sample),
        Route("/api/predict", api_predict, methods=["POST"]),
        Route("/api/predict-upload", api_predict_upload, methods=["POST"]),
        Route("/api/signal", api_signal),
        Route("/api/analytics", api_analytics),
        Route("/api/model", api_model),
        Route("/api/diagnostics/sample", api_sample_diagnostics, methods=["POST"]),
        Route("/api/diagnostics/errors", api_model_errors),
        Route("/api/diagnostics/difficult", api_difficult_samples),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR), check_dir=True), name="static"),
        Mount("/mcp", app=mcp_http_app),
    ],
    lifespan=lifespan,
)
