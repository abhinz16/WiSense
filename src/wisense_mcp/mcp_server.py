"""MCP interface for the completed WiFi CSI sensing project."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from src.wisense_mcp.diagnostics import WiSenseDiagnostics
from src.wisense_mcp.inference import WiSenseInferenceEngine
from src.wisense_mcp.repository import WiSenseRepository
from src.wisense_mcp.sample_catalog import SampleCatalog


repository = WiSenseRepository()
inference_engine = WiSenseInferenceEngine()
sample_catalog = SampleCatalog()
diagnostics_engine = WiSenseDiagnostics(inference_engine, sample_catalog)


mcp = MCPServer(
    "WiSense-MCP",
    instructions=(
        "Read-only access to a WiFi CSI multi-user people-sensing project built on WiMANS. "
        "The production model combines a temporal CNN, packed bidirectional LSTM, temporal-pyramid "
        "attention, and an equal-weight seed ensemble. Tools expose dataset metadata, locked final "
        "metrics, frozen predictions, live inference, subgroup analysis, and CSI diagnostics. "
        "The test set is reporting-only and is not used for tuning."
    ),
)


@mcp.tool(structured_output=True)
def get_dataset_summary() -> dict[str, Any]:
    """Return dataset labels, environments, WiFi bands, and split counts."""

    return repository.get_dataset_summary()


@mcp.tool(structured_output=True)
def get_final_model_metrics() -> dict[str, Any]:
    """Return the locked final test metrics and model-lock metadata."""

    return repository.get_final_model_metrics()


@mcp.tool(structured_output=True)
def get_activity_performance(activity: str | None = None) -> dict[str, Any]:
    """Return per-activity final-test performance for one or all activity classes."""

    try:
        result = repository.get_activity_performance(activity)
        return {"activity": activity, "result": result} if activity else {"results": result}
    except (ValueError, FileNotFoundError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def get_top_activity_confusions(limit: int = 5) -> dict[str, Any]:
    """Return the largest directional activity confusions on the final test split."""

    try:
        return {"limit": int(limit), "confusions": repository.get_top_confusions(limit)}
    except (ValueError, FileNotFoundError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def compare_wifi_bands() -> dict[str, Any]:
    """Compare final model performance between available WiFi frequency bands."""

    try:
        return {"bands": repository.get_band_performance()}
    except FileNotFoundError as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def compare_environments() -> dict[str, Any]:
    """Compare final model performance across sensing environments."""

    try:
        return {"environments": repository.get_environment_performance()}
    except FileNotFoundError as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def compare_user_counts() -> dict[str, Any]:
    """Compare final model performance by simultaneous user count."""

    try:
        return {"user_counts": repository.get_user_count_performance()}
    except FileNotFoundError as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def get_user_performance(user_index: int | None = None) -> dict[str, Any]:
    """Return final presence and activity performance by anonymized user slot."""

    try:
        result = repository.get_user_index_performance(user_index)
        return {"user_index": user_index, "result": result} if user_index else {"results": result}
    except (ValueError, FileNotFoundError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def get_error_analysis_summary() -> dict[str, Any]:
    """Summarize final activity, false-presence, and missed-presence errors."""

    try:
        return repository.get_error_analysis_summary()
    except FileNotFoundError as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def list_samples(
    split: str = "test",
    limit: int = 20,
    offset: int = 0,
    environment: str | None = None,
    band: float | None = None,
    user_count: int | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """List human-readable samples with optional metadata filters."""

    try:
        return sample_catalog.list_samples(
            split=split,
            limit=limit,
            offset=offset,
            environment=environment,
            band=band,
            user_count=user_count,
            search=search,
        )
    except ValueError as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def get_sample(identifier: str, split: str = "test") -> dict[str, Any]:
    """Resolve a human-readable alias or raw dataset ID to sample metadata."""

    try:
        return sample_catalog.resolve(identifier, split)
    except ValueError as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def get_frozen_prediction(identifier: str) -> dict[str, Any]:
    """Return the saved final ensemble probabilities for one test sample."""

    try:
        record = sample_catalog.resolve(identifier, "test")
        result = repository.get_frozen_prediction(int(record["row_index"]))
        result["sample"] = record
        return result
    except (ValueError, FileNotFoundError, IndexError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def predict_sample(identifier: str, split: str = "test", top_k: int = 3) -> dict[str, Any]:
    """Run live temporal-pyramid ensemble inference for one cached CSI sample."""

    try:
        record = sample_catalog.resolve(identifier, split)
        result = inference_engine.predict_sample(record["dataset_id"], split, top_k)
        result["sample"] = record
        return result
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def verify_live_prediction(identifier: str, split: str = "test") -> dict[str, Any]:
    """Check live inference against the stored ensemble probabilities for the same sample."""

    try:
        record = sample_catalog.resolve(identifier, split)
        result = inference_engine.verify_live_prediction(record["dataset_id"], split)
        result["sample"] = record
        return result
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def get_model_status() -> dict[str, Any]:
    """Return model-loading, checkpoint, device, and result-artifact status."""

    return inference_engine.get_status()


@mcp.tool(structured_output=True)
def diagnose_sample(identifier: str, split: str = "test") -> dict[str, Any]:
    """Inspect cached CSI integrity and compare live predictions with ground truth."""

    try:
        return diagnostics_engine.analyze_sample(identifier, split)
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def find_model_errors(
    error_type: str = "any",
    limit: int = 20,
    environment: str | None = None,
    band: float | None = None,
    user_count: int | None = None,
) -> dict[str, Any]:
    """Search frozen test predictions for selected error types."""

    try:
        return diagnostics_engine.find_model_errors(
            error_type, limit, environment, band, user_count
        )
    except (ValueError, FileNotFoundError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def find_difficult_samples(
    limit: int = 20,
    environment: str | None = None,
    band: float | None = None,
    user_count: int | None = None,
) -> dict[str, Any]:
    """Rank frozen test samples using the retrospective uncertainty heuristic."""

    try:
        return diagnostics_engine.find_difficult_samples(
            limit, environment, band, user_count
        )
    except (ValueError, FileNotFoundError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(structured_output=True)
def get_capabilities() -> dict[str, Any]:
    """Describe the dataset, model outputs, and read-only analysis capabilities."""

    return {
        "project": "WiSense-MCP",
        "tasks": ["multi-user presence detection", "user-specific activity recognition"],
        "model": "Temporal CNN + packed BiLSTM + temporal-pyramid attention + seed ensemble",
        "read_only": True,
        "test_set_used_for_tuning": False,
        "tools": 19,
    }
