"""Verify live production inference against saved ensemble probabilities."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.settings import load_settings
from src.wisense_mcp.inference import WiSenseInferenceEngine
from src.wisense_mcp.sample_catalog import SampleCatalog


def main() -> None:
    """Check a small configured sample set from validation and test splits."""

    settings = load_settings()
    engine = WiSenseInferenceEngine(settings)
    catalog = SampleCatalog(settings)
    count = settings.getint("evaluation", "verification_samples_per_split")
    tolerance = settings.getfloat("evaluation", "verification_tolerance")

    checked = 0
    passed = 0
    for split in ("validation", "test"):
        samples = catalog.records.get(split, [])[:count]
        print(f"\n{split.upper()}")
        for sample in samples:
            result = engine.verify_live_prediction(
                sample["dataset_id"], split=split, atol=tolerance
            )
            checked += 1
            passed += int(result["verified"])
            print(
                f"{sample['display_name']}: {result['verified']} | "
                f"presence error {result['max_presence_probability_error']:.8f} | "
                f"activity error {result['max_activity_probability_error']:.8f} | "
                f"{result['inference_ms']:.2f} ms"
            )

    print(f"\nVerified {passed}/{checked} samples")
    if passed != checked:
        raise RuntimeError("Production inference did not match all saved ensemble predictions.")


if __name__ == "__main__":
    main()
