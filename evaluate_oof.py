#!/usr/bin/env python3
"""Evaluate pooled out-of-fold nnDetection predictions with picai_eval."""

import argparse
import json
from pathlib import Path

import SimpleITK as sitk
from picai_eval import evaluate_folder

RESAMPLED_DIR = Path("csPCa_lesion_delineations/human_expert/resampled")
POOCH25_DIR = Path("csPCa_lesion_delineations/human_expert/Pooch25")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory with {case_id}_detection_map.mha files.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        required=True,
        help="Path to data/picai_labels.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Directory with ds-config-valid-fold-*.json files.",
    )
    parser.add_argument(
        "--prepared-labels-dir",
        type=Path,
        default=None,
        help="Optional cache for binarized {case_id}_label.nii.gz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save metrics.json (default: predictions-dir/metrics.json).",
    )
    return parser.parse_args()


def load_oof_subject_list(splits_dir: Path) -> list[str]:
    subjects = []
    for fold in range(5):
        split_file = splits_dir / f"ds-config-valid-fold-{fold}.json"
        with split_file.open() as fp:
            subjects.extend(json.load(fp)["subject_list"])
    return subjects


def find_label_source(labels_root: Path, case_id: str) -> Path | None:
    for rel_dir in (RESAMPLED_DIR, POOCH25_DIR):
        candidate = labels_root / rel_dir / f"{case_id}.nii.gz"
        if candidate.is_file():
            return candidate
    return None


def prepare_binary_label(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    label = sitk.GetArrayFromImage(sitk.ReadImage(str(src)))
    binary = (label >= 2).astype("uint8")
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(sitk.ReadImage(str(src)))
    sitk.WriteImage(out, str(dst), useCompression=True)


def main():
    args = parse_args()
    subject_list = load_oof_subject_list(args.splits_dir)
    prepared_labels_dir = args.prepared_labels_dir or (
        args.predictions_dir / "_labels_binary"
    )
    prepared_labels_dir.mkdir(parents=True, exist_ok=True)

    missing_labels = []
    for case_id in subject_list:
        src = find_label_source(args.labels_root, case_id)
        if src is None:
            missing_labels.append(case_id)
            continue
        prepare_binary_label(src, prepared_labels_dir / f"{case_id}_label.nii.gz")

    if missing_labels:
        raise SystemExit(
            f"Missing labels for {len(missing_labels)} cases. "
            f"Examples: {missing_labels[:5]}"
        )

    metrics = evaluate_folder(
        y_det_dir=args.predictions_dir,
        y_true_dir=prepared_labels_dir,
        subject_list=subject_list,
    )

    output_path = args.output or (args.predictions_dir / "metrics.json")
    metrics.save(output_path)

    print(f"Cases evaluated: {len(subject_list)}")
    print(f"AP:           {metrics.AP:.4f}")
    print(f"AUROC:        {metrics.auroc:.4f}")
    print(f"PI-CAI score: {metrics.score:.4f}")
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
