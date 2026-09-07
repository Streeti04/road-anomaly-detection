"""Shared robust-statistics helpers (torch-free, unit-tested).

Extracted from part_b/part_c, which both flag CSV-record outliers with a
median/MAD robust z — the same shape for every stage, differing only in the
metric columns and the stage/reason labels written to flags.csv.
"""

import numpy as np


def robust_z(values: list) -> np.ndarray | None:
    """Robust z = 0.6745 * (x - median) / MAD; None if MAD == 0 (constant metric)."""
    arr = np.asarray(values, dtype=np.float64)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad == 0:
        return None
    return 0.6745 * (arr - med) / mad


def outlier_flag_rows(records: list, metrics: tuple, *, stage: str, reason: str,
                      z_thresh: float = 3.0) -> list:
    """One flags.csv row per record/metric whose robust z exceeds z_thresh.

    `records` are dicts keyed by image_id plus every name in `metrics`.
    A constant metric (MAD == 0) yields no flags. Stable order: input record
    order, then `metrics` tuple order.
    """
    if not records:
        return []
    z = {metric: robust_z([float(r[metric]) for r in records])
         for metric in metrics}
    rows = []
    for i, record in enumerate(records):
        for metric in metrics:
            if z[metric] is None or abs(z[metric][i]) <= z_thresh:
                continue
            rows.append({
                "image_id": record["image_id"],
                "stage": stage,
                "reason": reason,
                "metric": metric,
                "value": float(record[metric]),
                "z": float(z[metric][i]),
            })
    return rows
