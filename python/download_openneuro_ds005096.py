#!/usr/bin/env python3
"""Download compact OpenNeuro ds005096 aneurysm masks and prepare image/mask pairs."""

import argparse
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def run_aws(args):
    return subprocess.run(
        ["aws", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def parse_s3_ls(output):
    entries = []
    for line in output.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4 and parts[2].isdigit():
            entries.append({"size": int(parts[2]), "key": parts[3]})
    return entries


def case_key_from_path(key):
    parts = key.split("/")
    subject = next((part for part in reversed(parts) if re.fullmatch(r"sub-\d+", part)), None)
    session = next((part for part in parts if re.fullmatch(r"ses-\d+", part)), None)
    if not subject or not session:
        return None
    return f"{subject}_{session}"


def is_source_image(key):
    return key.endswith("_angio.nii.gz") and "/anat/" in key and "/derivatives/" not in key


def is_aneurysm_label(key, include_parent=False):
    if "/Slicer/" not in key or not key.endswith("-label.nrrd"):
        return False
    name = Path(key).name.lower()
    if not name.startswith("segmentation-aneurysm"):
        return False
    if not include_parent and "parent" in name:
        return False
    return True


def is_consistent_derivative_path(key, case_id):
    parts = key.split("/")
    if "derivatives" not in parts:
        return True
    index = parts.index("derivatives")
    return index + 1 < len(parts) and parts[index + 1] == case_id.split("_", 1)[0]


def local_download_path(output_root, key):
    prefix = "ds005096/"
    rel = key[len(prefix):] if key.startswith(prefix) else key
    return output_root / "downloads" / rel


def s3_cp(bucket_uri, key, destination, region):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    run_aws([
        "s3",
        "cp",
        "--no-sign-request",
        "--region",
        region,
        f"{bucket_uri.rstrip('/')}/{key.split('ds005096/', 1)[-1]}",
        str(destination),
    ])


def images_match(image, mask):
    return (
        image.GetSize() == mask.GetSize()
        and np.allclose(image.GetSpacing(), mask.GetSpacing(), rtol=0, atol=1e-4)
    )


def convert_case(case_id, image_path, label_paths, image_out_dir, mask_out_dir):
    image = sitk.ReadImage(str(image_path))
    merged = None
    label_stats = []

    for label_path in label_paths:
        label = sitk.ReadImage(str(label_path))
        if not images_match(image, label):
            print(
                f"Geometry mismatch for {case_id}: image size/spacing "
                f"{image.GetSize()} {image.GetSpacing()} vs mask "
                f"{label.GetSize()} {label.GetSpacing()} ({label_path}); skipping"
            )
            continue
        array = sitk.GetArrayFromImage(label) > 0
        label_stats.append({
            "source": str(label_path),
            "nonzero": int(np.count_nonzero(array)),
        })
        merged = array if merged is None else np.logical_or(merged, array)

    if merged is None:
        raise RuntimeError(f"No geometry-compatible labels were provided for {case_id}")

    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    image_out = image_out_dir / f"{case_id}.nii.gz"
    mask_out = mask_out_dir / f"{case_id}.nii.gz"

    shutil.copy2(image_path, image_out)
    mask = sitk.GetImageFromArray(merged.astype(np.uint8))
    mask.CopyInformation(image)
    sitk.WriteImage(mask, str(mask_out), True)

    return {
        "case": case_id,
        "image": str(image_out),
        "mask": str(mask_out),
        "label_files": label_stats,
        "merged_nonzero": int(np.count_nonzero(merged)),
        "shape_xyz": list(image.GetSize()),
        "spacing_xyz": list(image.GetSpacing()),
    }


def build_manifest(entries, include_parent, max_source_mib=None):
    sources = {}
    labels = defaultdict(list)
    for entry in entries:
        key = entry["key"]
        case_id = case_key_from_path(key)
        if not case_id:
            continue
        if is_source_image(key):
            if max_source_mib and entry["size"] > max_source_mib * 1024 * 1024:
                continue
            sources.setdefault(case_id, key)
        elif is_aneurysm_label(key, include_parent) and is_consistent_derivative_path(key, case_id):
            labels[case_id].append(key)

    cases = []
    for case_id in sorted(set(sources) & set(labels)):
        cases.append({
            "case": case_id,
            "image_key": sources[case_id],
            "label_keys": sorted(labels[case_id]),
        })
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenNeuro ds005096 TOF-MRA images and compact dense aneurysm masks."
    )
    parser.add_argument("--bucket-uri", default="s3://openneuro.org/ds005096")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--output-root", type=Path, default=Path("data/openneuro/ds005096"))
    parser.add_argument("--prepared-root", type=Path, default=Path("data/openneuro/ds005096_pairs"))
    parser.add_argument("--max-cases", type=int, help="Download only the first N paired cases")
    parser.add_argument("--target-cases", type=int, help="Continue candidates until this many compatible cases are prepared")
    parser.add_argument("--max-source-mib", type=int, default=90, help="Skip source images larger than this many MiB")
    parser.add_argument("--include-parent", action="store_true", help="Use aneurysm-with-parent-artery labels too")
    args = parser.parse_args()

    print("Listing OpenNeuro ds005096...")
    listing = run_aws([
        "s3",
        "ls",
        "--no-sign-request",
        "--region",
        args.region,
        args.bucket_uri.rstrip("/") + "/",
        "--recursive",
    ])
    entries = parse_s3_ls(listing)
    cases = build_manifest(entries, args.include_parent, args.max_source_mib)
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        raise RuntimeError("No image/mask pairs found in the OpenNeuro listing.")

    image_out_dir = args.prepared_root / "images"
    mask_out_dir = args.prepared_root / "masks"
    summaries = []

    for index, case in enumerate(cases, start=1):
        if args.target_cases and len(summaries) >= args.target_cases:
            break
        case_id = case["case"]
        print(f"[{index}/{len(cases)}] {case_id}")
        image_path = local_download_path(args.output_root, case["image_key"])
        s3_cp(args.bucket_uri, case["image_key"], image_path, args.region)

        label_paths = []
        for label_key in case["label_keys"]:
            label_path = local_download_path(args.output_root, label_key)
            s3_cp(args.bucket_uri, label_key, label_path, args.region)
            label_paths.append(label_path)

        try:
            summaries.append(convert_case(case_id, image_path, label_paths, image_out_dir, mask_out_dir))
        except RuntimeError as error:
            print(f"Skipping {case_id}: {error}")

    if not summaries:
        raise RuntimeError("No geometry-compatible image/mask pairs were prepared.")

    args.prepared_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.prepared_root / "openneuro_ds005096_summary.json"
    summary_path.write_text(json.dumps({
        "source": args.bucket_uri,
        "cases": summaries,
        "num_cases": len(summaries),
        "mask_mode": "aneurysm-with-parent" if args.include_parent else "aneurysm-only",
    }, indent=2) + "\n")
    print(f"Wrote {summary_path}")
    print(f"Images: {image_out_dir}")
    print(f"Masks: {mask_out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
