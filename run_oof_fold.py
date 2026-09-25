#!/usr/bin/env python3
"""Run out-of-fold nnDetection inference for one cross-validation fold."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample

IMAGE_DIRS = [
    "images/transverse-t2-prostate-mri",
    "images/transverse-adc-prostate-mri",
    "images/transverse-hbv-prostate-mri",
]
TASK = "Task2201_picai_baseline"
MODEL = "RetinaUNetV001_D3V001_3d"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--gc-cases-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--combined-dir",
        type=Path,
        default=None,
        help="Optional pooled OOF output directory (one map per case).",
    )
    return parser.parse_args()


def scratch_layout(results_dir: Path) -> dict[str, Path]:
    scratch = results_dir.parent
    return {
        "scratch": scratch,
        "results": results_dir,
        "inp": scratch / "nndet" / "input",
        "out": scratch / "nndet" / "output",
        "workdir": scratch / "work",
        "det_data": scratch / "det_data",
    }


def strip_metadata(img: sitk.Image) -> None:
    for key in img.GetMetaDataKeys():
        img.EraseMetaData(key)


def load_subject_list(splits_dir: Path, fold: int) -> list[str]:
    split_file = splits_dir / f"ds-config-valid-fold-{fold}.json"
    with split_file.open() as fp:
        return json.load(fp)["subject_list"]


def case_image_paths(case_dir: Path) -> list[Path]:
    paths = []
    for rel_dir in IMAGE_DIRS:
        matches = sorted((case_dir / rel_dir).glob("*.mha"))
        if not matches:
            raise FileNotFoundError(f"No .mha files in {case_dir / rel_dir}")
        paths.append(matches[0])
    return paths


def clear_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def reset_nndet_scratch(layout: dict[str, Path]):
    """Clear per-case nnDetection preprocessing state from prior predictions."""
    task_data = layout["det_data"] / TASK
    if task_data.is_dir():
        for item in task_data.iterdir():
            if item.name == "dataset.json":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    test_predictions = (
        layout["results"] / "nnDet" / TASK / MODEL / "consolidated" / "test_predictions"
    )
    if test_predictions.is_dir():
        shutil.rmtree(test_predictions)


def configure_nndet_env(layout: dict[str, Path]):
    os.environ["RESULTS_FOLDER"] = str(layout["results"])
    os.environ["det_data"] = str(layout["det_data"])
    os.environ["det_models"] = str(layout["results"] / "nnDet")


def ensure_dataset_json(task: str, layout: dict[str, Path]):
    path_src = layout["results"] / "nnDet" / task / "dataset.json"
    if not path_src.is_file():
        raise FileNotFoundError(f"Missing dataset.json: {path_src}")

    for path_dst in (
        layout["workdir"] / "nnDet_raw_data" / task / "dataset.json",
        layout["det_data"] / task / "dataset.json",
    ):
        if path_dst.is_file():
            continue
        path_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path_src, path_dst)


def prepare_single_fold_checkpoint(fold: int, results_dir: Path):
    """Keep only model_fold{N}.ckpt in consolidated/ for single-fold OOF."""
    consolidated = results_dir / "nnDet" / TASK / MODEL / "consolidated"
    if not consolidated.is_dir():
        raise FileNotFoundError(f"Missing consolidated models: {consolidated}")

    keep = consolidated / f"model_fold{fold}.ckpt"
    if not keep.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {keep}")

    for ckpt in consolidated.glob("model_fold*.ckpt"):
        if ckpt != keep:
            ckpt.unlink()

    print(f"Using single-fold checkpoint: {keep.name}")


def predict_case(layout: dict[str, Path], image_paths: list[Path]) -> tuple[sitk.Image, float]:
    configure_nndet_env(layout)
    reset_nndet_scratch(layout)

    clear_dir(layout["inp"])
    clear_dir(layout["out"])

    sample = Sample(scans=[sitk.ReadImage(str(path)) for path in image_paths])
    sample.preprocess()
    for i, scan in enumerate(sample.scans):
        atomic_image_write(scan, layout["inp"] / f"scan_{i:04d}.nii.gz")

    subprocess.check_call(
        [
            "nndet",
            "predict",
            TASK,
            MODEL,
            str(layout["workdir"]),
            "--results",
            str(layout["results"] / "nnDet"),
            "--input",
            str(layout["inp"]),
            "--output",
            str(layout["out"]),
            "--fold",
            "-1",
            "--check",
        ]
    )

    subprocess.check_call(
        [
            "python",
            "/opt/code/nndet_generate_detection_maps.py",
            "--input",
            str(layout["out"]),
            "--output",
            str(layout["out"]),
        ]
    )

    detection_map = sitk.ReadImage(str(layout["out"] / "scan_detection_map.nii.gz"))
    strip_metadata(detection_map)
    max_val = float(np.max(sitk.GetArrayFromImage(detection_map)))
    return detection_map, max_val


def main():
    args = parse_args()
    layout = scratch_layout(args.results_dir)
    print(f"Fold {args.fold} on {'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'}")

    configure_nndet_env(layout)
    ensure_dataset_json(TASK, layout)
    prepare_single_fold_checkpoint(args.fold, args.results_dir)

    fold_output_dir = args.output_dir / f"fold_{args.fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    if args.combined_dir is not None:
        args.combined_dir.mkdir(parents=True, exist_ok=True)

    subject_list = load_subject_list(args.splits_dir, args.fold)

    processed = 0
    skipped = 0
    for case_id in subject_list:
        det_map_path = fold_output_dir / f"{case_id}_detection_map.mha"
        if det_map_path.exists():
            skipped += 1
            continue

        case_dir = args.gc_cases_dir / case_id
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Missing GC case directory: {case_dir}")

        image_paths = case_image_paths(case_dir)
        det_map, case_score = predict_case(layout, image_paths)
        atomic_image_write(det_map, det_map_path)

        score_path = fold_output_dir / f"{case_id}_case_level_likelihood.json"
        score_path.write_text(json.dumps(case_score) + "\n")

        if args.combined_dir is not None:
            combined_path = args.combined_dir / f"{case_id}_detection_map.mha"
            atomic_image_write(det_map, combined_path)

        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{len(subject_list)} cases (skipped {skipped})")

    print(
        f"Done fold {args.fold}: processed={processed}, skipped={skipped}, "
        f"total={len(subject_list)}"
    )


if __name__ == "__main__":
    main()
