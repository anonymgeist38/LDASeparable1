#!/usr/bin/env python3
"""RunPod HTTP service for SegMend nnU-Net inference.

This intentionally keeps the HTTP surface small and aligned with the Electron
app: GET /health, GET /models, and POST /predict with base64 NIfTI payloads.
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request


DATASET_ID = "1001"
DATASET_NAME = "Dataset1001_AneurysmTOFMRA18"
DEFAULT_CONFIG = "3d_fullres"
WORKSPACE_ROOT = Path(os.environ.get("SEGMEND_WORKSPACE_ROOT", "/workspace"))

app = Flask(__name__)


def first_existing_dir(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def find_dataset_dir() -> Path:
    configured_results = os.environ.get("NNUNET_RESULTS") or os.environ.get("nnUNet_results")
    candidates: list[Path] = []
    if configured_results:
        candidates.append(Path(configured_results) / DATASET_NAME)

    candidates.extend(
        [
            WORKSPACE_ROOT / "aneurysm" / "workspace" / "nnUNet_results" / DATASET_NAME,
            WORKSPACE_ROOT / "nnUNet_results" / DATASET_NAME,
            WORKSPACE_ROOT / "workspace" / "nnUNet_results" / DATASET_NAME,
        ]
    )

    found = first_existing_dir(candidates)
    if found:
        return found

    for path in WORKSPACE_ROOT.rglob(DATASET_NAME):
        if path.is_dir():
            return path

    raise FileNotFoundError(f"{DATASET_NAME} not found under {WORKSPACE_ROOT}")


def configure_nnunet_environment() -> Path:
    dataset_dir = find_dataset_dir()
    results_dir = dataset_dir.parent

    raw_dir = Path(
        os.environ.get("NNUNET_RAW_DATA_BASE")
        or os.environ.get("nnUNet_raw")
        or str(WORKSPACE_ROOT / "nnUNet_raw")
    )
    preprocessed_dir = Path(
        os.environ.get("NNUNET_PREPROCESSED")
        or os.environ.get("nnUNet_preprocessed")
        or str(WORKSPACE_ROOT / "nnUNet_preprocessed")
    )
    mpl_dir = Path(os.environ.get("MPLCONFIGDIR", str(WORKSPACE_ROOT / ".matplotlib")))

    raw_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)

    os.environ["NNUNET_RESULTS"] = str(results_dir)
    os.environ["nnUNet_results"] = str(results_dir)
    os.environ["NNUNET_PREPROCESSED"] = str(preprocessed_dir)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_dir)
    os.environ["NNUNET_RAW_DATA_BASE"] = str(raw_dir)
    os.environ["nnUNet_raw"] = str(raw_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ.setdefault("NNUNET_USE_MIRRORING", "0")
    os.environ.setdefault("TORCH_NUM_THREADS", "4")
    os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "2")

    return dataset_dir


def model_artifacts_for(dataset_dir: Path, configuration: str) -> tuple[str, str]:
    trainer_dirs = sorted(dataset_dir.glob(f"*__nnUNetPlans__{configuration}"))
    for trainer_dir in trainer_dirs:
        fold_dir = trainer_dir / "fold_0"
        trainer_name = trainer_dir.name.split("__", 1)[0]
        if (fold_dir / "checkpoint_final.pth").exists():
            return trainer_name, "checkpoint_final.pth"
        if (fold_dir / "checkpoint_best.pth").exists():
            return trainer_name, "checkpoint_best.pth"
        if (fold_dir / "checkpoint_latest.pth").exists():
            return trainer_name, "checkpoint_latest.pth"
    return "nnUNetTrainer", "checkpoint_final.pth"


@app.get("/health")
def health():
    try:
        dataset_dir = configure_nnunet_environment()
        return jsonify(
            {
                "status": "healthy",
                "service": "SegMend RunPod nnU-Net",
                "model": DATASET_NAME,
                "results": os.environ["NNUNET_RESULTS"],
                "datasetDir": str(dataset_dir),
            }
        )
    except Exception as error:
        return jsonify({"status": "unhealthy", "error": str(error)}), 500


@app.get("/models")
def models():
    try:
        dataset_dir = configure_nnunet_environment()
        configurations = sorted(
            {
                path.name.rsplit("__", 1)[-1]
                for path in dataset_dir.iterdir()
                if path.is_dir() and "__nnUNetPlans__" in path.name
            }
        )
        return jsonify(
            {
                "success": True,
                "count": 1,
                "models": [
                    {
                        "name": DATASET_NAME,
                        "datasetId": DATASET_ID,
                        "configurations": configurations or [DEFAULT_CONFIG],
                    }
                ],
            }
        )
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 500


@app.post("/predict")
def predict():
    started_at = time.monotonic()
    data = request.get_json(force=True, silent=True) or {}
    input_data = data.get("inputDataBase64")
    input_filename = Path(data.get("inputFilename") or "scan.nii.gz").name
    configuration = data.get("configuration") or DEFAULT_CONFIG

    if not input_data:
        return jsonify({"success": False, "error": "Missing inputDataBase64"}), 400

    try:
        dataset_dir = configure_nnunet_environment()
        trainer, checkpoint = model_artifacts_for(dataset_dir, configuration)
        request_id = uuid.uuid4().hex[:10]

        with tempfile.TemporaryDirectory(prefix="segmend_") as temp_root:
            temp_root_path = Path(temp_root)
            input_dir = temp_root_path / "input"
            output_dir = temp_root_path / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            case_id = f"case_{request_id}"
            input_path = input_dir / f"{case_id}_0000.nii.gz"
            if input_filename.endswith(".nii") and not input_filename.endswith(".nii.gz"):
                input_path = input_dir / f"{case_id}_0000.nii"
            input_path.write_bytes(base64.b64decode(input_data))

            command = [
                "nnUNetv2_predict",
                "-d",
                DATASET_ID,
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-f",
                "0",
                "-c",
                configuration,
                "-tr",
                trainer,
                "-chk",
                checkpoint,
            ]
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=int(os.environ.get("SEGMEND_PREDICT_TIMEOUT", "3600")),
            )
            if completed.returncode != 0:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": (completed.stderr or completed.stdout)[-4000:],
                            "command": command,
                        }
                    ),
                    500,
                )

            prediction_path = output_dir / f"{case_id}.nii.gz"
            if not prediction_path.exists():
                predictions = sorted(output_dir.glob("*.nii.gz"))
                if not predictions:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "nnUNetv2_predict finished but created no .nii.gz output",
                                "stdout": completed.stdout[-4000:],
                            }
                        ),
                        500,
                    )
                prediction_path = predictions[0]

            return jsonify(
                {
                    "success": True,
                    "model": DATASET_NAME,
                    "configuration": configuration,
                    "trainer": trainer,
                    "checkpoint": checkpoint,
                    "outputFilename": prediction_path.name,
                    "outputDataBase64": base64.b64encode(prediction_path.read_bytes()).decode("ascii"),
                    "inferenceTimeMs": round((time.monotonic() - started_at) * 1000),
                }
            )
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8001, type=int)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
