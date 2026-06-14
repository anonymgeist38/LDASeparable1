#!/usr/bin/env python3
"""Audit nnU-Net image/label folders before training or deployment."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def load_array(path):
    return np.asanyarray(nib.load(str(path)).dataobj)


def load_image(path):
    return nib.load(str(path))


def case_id_from_image(path):
    name = path.name
    if name.endswith("_0000.nii.gz"):
        return name[:-12]
    if name.endswith("_0000.nii"):
        return name[:-9]
    return path.stem


def label_path_for_image(label_dir, image_path):
    return label_dir / f"{case_id_from_image(image_path)}.nii.gz"


def summarize_labels(label_files, image_dir=None):
    rows = []
    for label_file in label_files:
        label_img = load_image(label_file)
        data = np.asanyarray(label_img.dataobj)
        nonzero = int(np.count_nonzero(data))
        labels = [int(v) for v in np.unique(data)]
        image_file = None
        affine_matches = None
        shape_matches = None
        foreground_intensity_median = None
        image_intensity_median = None
        image_intensity_p99 = None
        if image_dir is not None:
            image_file = image_dir / f"{label_file.name[:-7]}_0000.nii.gz"
            if image_file.exists():
                image_img = load_image(image_file)
                image_data = np.asanyarray(image_img.dataobj)
                affine_matches = bool(np.allclose(image_img.affine, label_img.affine))
                shape_matches = tuple(image_data.shape) == tuple(data.shape)
                if shape_matches:
                    foreground = image_data[data > 0]
                    foreground_intensity_median = float(np.median(foreground)) if foreground.size else None
                    image_intensity_median = float(np.median(image_data))
                    image_intensity_p99 = float(np.percentile(image_data, 99))
        rows.append({
            "case": label_file.name,
            "shape": list(data.shape),
            "nonzero": nonzero,
            "labels": labels,
            "foreground_fraction": float(nonzero / data.size) if data.size else 0.0,
            "image": image_file.name if image_file and image_file.exists() else None,
            "shape_matches_image": shape_matches,
            "affine_matches_image": affine_matches,
            "foreground_intensity_median": foreground_intensity_median,
            "image_intensity_median": image_intensity_median,
            "image_intensity_p99": image_intensity_p99,
        })
    return rows


def audit(dataset_dir, min_nonzero, min_fraction):
    dataset_dir = Path(dataset_dir)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"

    image_files = sorted(images_tr.glob("*_0000.nii.gz"))
    label_files = sorted(labels_tr.glob("*.nii.gz"))
    test_files = sorted(images_ts.glob("*_0000.nii.gz"))
    rows = summarize_labels(label_files, images_tr)

    missing_labels = [
        image.name for image in image_files
        if not label_path_for_image(labels_tr, image).exists()
    ]

    tiny_labels = [
        row for row in rows
        if row["nonzero"] < min_nonzero or row["foreground_fraction"] < min_fraction
    ]
    empty_labels = [row for row in rows if row["nonzero"] == 0]
    geometry_mismatches = [
        row["case"] for row in rows
        if row["shape_matches_image"] is False or row["affine_matches_image"] is False
    ]

    summary = {
        "dataset_dir": str(dataset_dir),
        "training_images": len(image_files),
        "training_labels": len(label_files),
        "test_images": len(test_files),
        "missing_labels": missing_labels,
        "empty_labels": [row["case"] for row in empty_labels],
        "tiny_labels": [row["case"] for row in tiny_labels],
        "geometry_mismatches": geometry_mismatches,
        "nonzero_min": min((row["nonzero"] for row in rows), default=0),
        "nonzero_max": max((row["nonzero"] for row in rows), default=0),
        "nonzero_median": float(np.median([row["nonzero"] for row in rows])) if rows else 0.0,
        "foreground_fraction_median": float(np.median([row["foreground_fraction"] for row in rows])) if rows else 0.0,
        "labels": rows,
    }

    failures = []
    if not image_files:
        failures.append("No training images found in imagesTr.")
    if not label_files:
        failures.append("No labels found in labelsTr.")
    if missing_labels:
        failures.append(f"{len(missing_labels)} training images are missing matching labels.")
    if empty_labels:
        failures.append(f"{len(empty_labels)} labels are entirely background.")
    if tiny_labels:
        failures.append(
            f"{len(tiny_labels)} labels are below the minimum foreground threshold "
            f"({min_nonzero} voxels or fraction {min_fraction})."
        )
    if geometry_mismatches:
        failures.append(f"{len(geometry_mismatches)} labels do not match image shape/affine.")

    summary["ok"] = not failures
    summary["failures"] = failures
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Audit nnU-Net DatasetXXX folders for usable segmentation labels."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Path to DatasetXXX folder")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--min-nonzero", type=int, default=50, help="Minimum foreground voxels per label")
    parser.add_argument("--min-fraction", type=float, default=0.0001, help="Minimum foreground fraction per label")
    args = parser.parse_args()

    summary = audit(args.dataset, args.min_nonzero, args.min_fraction)

    print(f"Training images: {summary['training_images']}")
    print(f"Training labels: {summary['training_labels']}")
    print(f"Test images: {summary['test_images']}")
    print(f"Label nonzero min/median/max: {summary['nonzero_min']} / {summary['nonzero_median']} / {summary['nonzero_max']}")
    print(f"Median foreground fraction: {summary['foreground_fraction_median']:.8f}")

    if summary["ok"]:
        print("Audit passed.")
    else:
        print("Audit failed:")
        for failure in summary["failures"]:
            print(f"  - {failure}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote {args.output}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
