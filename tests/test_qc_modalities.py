"""Unit tests for image and signal quality control.

Exercises ``qc_images`` and ``qc_signals`` end to end: builds small,
deterministic numpy inputs that trigger each QC branch (class imbalance,
mixed sizes, constant images; NaNs, flatline segments, and clipping), then
verifies the populated ``QCReport`` summary/warnings plus ``render`` and
``to_dict``. All inputs are offline and seeded.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.qc.images import qc_images
from ddm4bio.qc.report import QCReport
from ddm4bio.qc.signals import qc_signals

# --------------------------------------------------------------------------- #
# qc_images
# --------------------------------------------------------------------------- #


def test_qc_images_grayscale_stack_with_class_imbalance():
    # (N, H, W) stack; five class-0 images and one class-1 image -> 5:1 ratio.
    rng = np.random.default_rng(0)
    images = rng.random((6, 4, 4))
    labels = [0, 0, 0, 0, 0, 1]

    report = qc_images(images, labels)

    assert isinstance(report, QCReport)
    assert report.modality == "images"

    summary = report.summary
    assert summary["count"] == 6

    # Uniform shape -> size-consistent, single float dtype.
    assert summary["size_consistent"] is True
    assert summary["shapes"] == ["(4, 4)"]
    assert summary["dtype"] == ["float64"]

    # Intensity stats populated and internally consistent (rng in [0, 1)).
    intensity = summary["intensity"]
    assert 0.0 <= intensity["min"] <= intensity["mean"] <= intensity["max"] <= 1.0

    # Per-class counts and the imbalance warning.
    assert summary["per_class_counts"] == {"0": 5, "1": 1}
    assert summary["class_imbalance_ratio"] == 5.0
    assert any("imbalance" in w.lower() for w in report.warnings)


def test_qc_images_color_stack_is_size_consistent():
    # (N, H, W, C) stack, all identical shape -> no size/dtype warnings.
    rng = np.random.default_rng(1)
    images = rng.random((3, 5, 5, 3))

    report = qc_images(images)

    summary = report.summary
    assert summary["count"] == 3
    assert summary["size_consistent"] is True
    assert summary["shapes"] == ["(5, 5, 3)"]
    assert summary["dtype"] == ["float64"]
    assert summary["intensity"]["min"] <= summary["intensity"]["max"]
    # No labels supplied -> no per-class summary.
    assert "per_class_counts" not in summary
    # No defects -> no warnings.
    assert report.warnings == []


def test_qc_images_mixed_sizes_and_constant_image_warns():
    # A constant image plus two differently sized images -> mixed-size warning.
    constant = np.full((4, 4), 7.0)
    images = [constant, np.zeros((5, 5)), np.ones((4, 4))]

    report = qc_images(images)

    summary = report.summary
    assert summary["count"] == 3
    assert summary["size_consistent"] is False
    # Two distinct shapes present.
    assert set(summary["shapes"]) == {"(4, 4)", "(5, 5)"}

    # Overall intensity spans the constant (7.0) and the zeros image.
    assert summary["intensity"]["min"] == 0.0
    assert summary["intensity"]["max"] == 7.0

    assert any("mixed sizes" in w.lower() for w in report.warnings)


def test_qc_images_empty_input():
    report = qc_images()
    assert report.modality == "images"
    assert report.summary["count"] == 0
    assert report.summary["size_consistent"] is True
    assert report.summary["dtype"] is None
    assert report.summary["intensity"] == {"min": None, "max": None, "mean": None}
    assert report.warnings == []


# --------------------------------------------------------------------------- #
# qc_signals
# --------------------------------------------------------------------------- #


def _make_defective_signal() -> np.ndarray:
    # 1-D signal with:
    #   * two NaN values,
    #   * one flatline segment (3.0 repeated three times),
    #   * clipped/saturated samples at the observed min (0) and max (10).
    return np.array(
        [0.0, 1.0, 3.0, 3.0, 3.0, 7.0, 10.0, np.nan, np.nan, 0.0, 10.0, 5.0]
    )


def test_qc_signals_records_all_defects_and_warns():
    x = _make_defective_signal()
    report = qc_signals(x, fs=100.0)

    assert isinstance(report, QCReport)
    assert report.modality == "signals"

    summary = report.summary

    # A 1-D signal is treated as a single channel.
    assert summary["n_channels"] == 1
    assert summary["length"] == 12
    assert summary["sampling_rate"] == 100.0
    assert summary["duration_s"] == 12 / 100.0

    # NaN accounting.
    assert summary["nan_count"] == 2

    # One flatline (constant) run: the three consecutive 3.0 samples.
    assert summary["flatline_segments"] == 1

    # Clipping fraction: samples at min (0.0 x2) or max (10.0 x2) of the ten
    # finite samples -> 4 / 10.
    assert summary["clipping_fraction"] == 0.4

    warning_text = " ".join(report.warnings).lower()
    assert "nan" in warning_text
    assert "flatline" in warning_text
    assert "clipping" in warning_text or "saturation" in warning_text


def test_qc_signals_clean_signal_has_no_defect_warnings():
    # A smooth sine has no NaNs, no long flat runs, and few extreme samples.
    fs = 200.0
    t = np.arange(0, 1.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 5 * t)

    report = qc_signals(x, fs=fs)
    summary = report.summary

    assert summary["length"] == t.size
    assert summary["sampling_rate"] == fs
    assert summary["nan_count"] == 0

    # No NaN or flatline warnings for a clean signal.
    joined = " ".join(report.warnings).lower()
    assert "nan" not in joined
    assert "flatline" not in joined


def test_qc_signals_empty_input():
    report = qc_signals()
    assert report.modality == "signals"
    assert report.summary["length"] == 0
    assert report.summary["sampling_rate"] is None
    assert report.warnings == []


# --------------------------------------------------------------------------- #
# QCReport.render / to_dict for both modalities
# --------------------------------------------------------------------------- #


def test_images_report_render_and_to_dict_round_trip():
    rng = np.random.default_rng(2)
    report = qc_images(rng.random((6, 4, 4)), [0, 0, 0, 0, 0, 1])

    text = report.render()
    assert isinstance(text, str)
    assert "images" in text
    # Summary and warning content surface in the rendered text.
    assert "count" in text
    assert "imbalance" in text.lower()

    d = report.to_dict()
    assert d["modality"] == "images"
    assert d["summary"] == report.summary
    assert d["warnings"] == report.warnings

    rebuilt = QCReport(**d)
    assert rebuilt.modality == report.modality
    assert rebuilt.summary == report.summary
    assert rebuilt.warnings == report.warnings


def test_signals_report_render_and_to_dict_round_trip():
    report = qc_signals(_make_defective_signal(), fs=100.0)

    text = report.render()
    assert isinstance(text, str)
    assert "signals" in text
    assert "nan_count" in text

    d = report.to_dict()
    assert d["modality"] == "signals"
    assert d["summary"] == report.summary
    assert d["warnings"] == report.warnings

    rebuilt = QCReport(**d)
    assert rebuilt.modality == report.modality
    assert rebuilt.summary == report.summary
    assert rebuilt.warnings == report.warnings
