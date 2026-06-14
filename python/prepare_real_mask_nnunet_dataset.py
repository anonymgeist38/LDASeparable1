#!/usr/bin/env python3
"""Prepare an nnU-Net dataset from paired NIfTI images and dense masks."""

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def strip_nii_suffix(path):
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def normalized_case_id(path):
    stem = strip_nii_suffix(path)
    if stem.endswith("_0000"):
        stem = stem[:-5]
    for suffix in ["_mask", "_seg", "_label", "-mask", "-seg", "-label"]:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def nifti_files(folder):
    files = sorted(folder.glob("*.nii")) + sorted(folder.glob("*.nii.gz"))
    return sorted(files, key=lambda item: item.name)


def load_shape(path):
    return nib.load(str(path)).shape


def write_binary_mask(source_path, reference_image_path, output_path):
    mask_image = nib.load(str(source_path))
    reference_image = nib.load(str(reference_image_path))
    data = np.asanyarray(mask_image.dataobj)
    binary = (data > 0).astype(np.uint8)
    header = reference_image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(binary, reference_image.affine, header), str(output_path))
    return int(np.count_nonzero(binary)), list(binary.shape)


def copy_image(source_path, output_path):
    shutil.copy2(source_path, output_path)
    return list(load_shape(output_path))


def split_pairs(pairs, train_split):
    train_count = int(round(len(pairs) * train_split))
    train_count = max(1, min(train_count, len(pairs))) if pairs else 0
    return pairs[:train_count], pairs[train_count:]


def prepare_dataset(args):
    image_dir = args.images
    mask_dir = args.masks
    output_base = args.output_base
    dataset_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_dir = output_base / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"

    image_files = nifti_files(image_dir)
    mask_files = nifti_files(mask_dir)
    masks_by_id = {normalized_case_id(path): path for path in mask_files}

    pairs = []
    missing_masks = []
    for image_path in image_files:
        case_id = normalized_case_id(image_path)
        mask_path = masks_by_id.get(case_id)
        if mask_path is None:
            missing_masks.append(image_path.name)
            continue
        pairs.append((case_id, image_path, mask_path))

    if not pairs:
        raise RuntimeError("No paired image/mask NIfTI files found.")

    if missing_masks:
        print(f"Warning: {len(missing_masks)} images had no matching mask and were skipped.")

    pairs = sorted(pairs, key=lambda item: item[0])
    if args.max_cases:
        pairs = pairs[: args.max_cases]
    train_pairs, test_pairs = split_pairs(pairs, args.train_split)

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    training_entries = []
    label_stats = []

    for index, (case_id, image_path, mask_path) in enumerate(train_pairs):
        nnunet_case = f"{args.case_prefix}{index:03d}"
        image_out = images_tr / f"{nnunet_case}_0000.nii.gz"
        label_out = labels_tr / f"{nnunet_case}.nii.gz"

        image_shape = copy_image(image_path, image_out)
        nonzero, mask_shape = write_binary_mask(mask_path, image_path, label_out)
        if tuple(image_shape) != tuple(mask_shape):
            raise RuntimeError(
                f"Shape mismatch for {case_id}: image {image_shape}, mask {mask_shape}"
            )

        label_stats.append({"case": nnunet_case, "source": case_id, "nonzero": nonzero})
        training_entries.append({
            "image": [f"imagesTr/{image_out.name}"],
            "label": f"labelsTr/{label_out.name}",
        })

    test_entries = []
    for index, (case_id, image_path, _mask_path) in enumerate(test_pairs):
        nnunet_case = f"{args.case_prefix}{index + len(train_pairs):03d}"
        image_out = images_ts / f"{nnunet_case}_0000.nii.gz"
        copy_image(image_path, image_out)
        test_entries.append(f"imagesTs/{image_out.name}")

    dataset_json = {
        "name": args.dataset_name,
        "description": "Aneurysm segmentation dataset with dense masks",
        "reference": args.reference,
        "license": args.license,
        "release": "1.0",
        "channel_names": {"0": args.channel_name},
        "labels": {"background": 0, "aneurysm": 1},
        "numTraining": len(training_entries),
        "numTest": len(test_entries),
        "file_ending": ".nii.gz",
        "training": training_entries,
        "test": test_entries,
    }

    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")
    (dataset_dir / "label_stats.json").write_text(json.dumps(label_stats, indent=2) + "\n")

    tiny = [row for row in label_stats if row["nonzero"] < args.min_nonzero]
    if tiny:
        tiny_cases = ", ".join(row["case"] for row in tiny[:10])
        raise RuntimeError(
            f"{len(tiny)} training masks have fewer than {args.min_nonzero} foreground voxels. "
            f"Examples: {tiny_cases}"
        )

    print(f"Prepared {dataset_dir}")
    print(f"Training pairs: {len(training_entries)}")
    print(f"Test images: {len(test_entries)}")
    print(f"Label nonzero min/median/max: {min(row['nonzero'] for row in label_stats)} / "
          f"{float(np.median([row['nonzero'] for row in label_stats]))} / "
          f"{max(row['nonzero'] for row in label_stats)}")
    return dataset_dir


def main():
    parser = argparse.ArgumentParser(
        description="Build a DatasetXXX folder from paired NIfTI images and real dense masks."
    )
    parser.add_argument("--images", required=True, type=Path, help="Folder containing image .nii/.nii.gz files")
    parser.add_argument("--masks", required=True, type=Path, help="Folder containing matching mask .nii/.nii.gz files")
    parser.add_argument("--output-base", required=True, type=Path, help="nnUNet raw data base folder")
    parser.add_argument("--dataset-id", type=int, default=1000)
    parser.add_argument("--dataset-name", default="AneurysmSeg")
    parser.add_argument("--case-prefix", default="Aneurysm")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--max-cases", type=int, help="Limit the number of paired cases for tiny smoke runs")
    parser.add_argument("--min-nonzero", type=int, default=50)
    parser.add_argument("--channel-name", default="CT", help="nnU-Net channel name, for example CT or MRA")
    parser.add_argument("--reference", default="Local dense aneurysm mask dataset")
    parser.add_argument("--license", default="Verify before sharing")
    args = parser.parse_args()

    if not args.images.exists():
        raise FileNotFoundError(f"Image folder not found: {args.images}")
    if not args.masks.exists():
        raise FileNotFoundError(f"Mask folder not found: {args.masks}")
    if not (0 < args.train_split <= 1):
        raise ValueError("--train-split must be in (0, 1]")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be at least 1")

    prepare_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
