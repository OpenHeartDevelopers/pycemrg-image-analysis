"""
cardiac_phase_inspector.py

Inspects a folder of DICOM files and reports the cardiac phase(s)
encoded in the headers.

Usage:
    python cardiac_phase_inspector.py /path/to/dicom/folder

Architecture follows the Modular Systems Manifest:
  - Utility tier  : pure, stateless functions for tag extraction and classification
  - Logic tier    : SlicePhaseReader (stateless engine) and SeriesPhaseAnalyser
  - Orchestrator  : main() handles all I/O, path resolution, and reporting
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import Optional

import pydicom


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DicomPhaseRecord:
    """Contract representing the cardiac phase data extracted from one DICOM file."""
    filepath: str
    instance_number: Optional[int]
    trigger_time_ms: Optional[float]
    rr_interval_ms: Optional[float]
    cardiac_cycle_fraction: Optional[float]   # explicit tag, 0.0-1.0


@dataclass(frozen=True)
class SlicePhaseResult:
    """Contract representing the fully resolved phase for one slice."""
    record: DicomPhaseRecord
    estimated_phase_pct: Optional[float]
    phase_label: str


@dataclass(frozen=True)
class SeriesPhaseSummary:
    """Contract representing the aggregate analysis of a full DICOM series."""
    total_files_scanned: int
    files_with_trigger_data: int
    is_gated: bool
    is_multi_phase: bool
    min_trigger_ms: Optional[float]
    max_trigger_ms: Optional[float]
    spread_ms: Optional[float]
    slice_results: list[SlicePhaseResult]


# ---------------------------------------------------------------------------
# Utility tier — pure, stateless functions
# ---------------------------------------------------------------------------

def extract_optional_float(ds: pydicom.Dataset, group: int, element: int) -> Optional[float]:
    """Return a DICOM tag value as float, or None if the tag is absent."""
    tag = (group, element)
    if tag in ds:
        try:
            return float(ds[tag].value)
        except (ValueError, TypeError):
            return None
    return None


def extract_optional_int(ds: pydicom.Dataset, attribute: str) -> Optional[int]:
    """Return a named DICOM attribute as int, or None if absent."""
    if attribute in ds:
        try:
            return int(getattr(ds, attribute))
        except (ValueError, TypeError):
            return None
    return None


def compute_phase_pct(trigger_time_ms: float, rr_interval_ms: float) -> float:
    """Calculate cardiac phase as a percentage of the RR interval."""
    if rr_interval_ms <= 0:
        raise ValueError("RR interval must be a positive value.")
    return (trigger_time_ms / rr_interval_ms) * 100.0


def classify_cardiac_phase(phase_pct: float) -> str:
    """
    Map a numeric phase percentage to a human-readable anatomical label.
    Thresholds are approximate and based on standard cardiac CT conventions.
    """
    if phase_pct < 10:
        return "End-systole onset"
    if phase_pct < 40:
        return "Systole"
    if phase_pct < 65:
        return "Early diastole"
    if phase_pct <= 80:
        return "Mid-diastole (optimal window for coronaries)"
    return "Late diastole"


def is_multi_phase_series(records: list[DicomPhaseRecord], spread_threshold_ms: float = 50.0) -> bool:
    """
    A series is considered multi-phase if trigger times vary by more than
    the given threshold across the sampled files.
    """
    times = [r.trigger_time_ms for r in records if r.trigger_time_ms is not None]
    if len(times) < 2:
        return False
    return (max(times) - min(times)) > spread_threshold_ms


# ---------------------------------------------------------------------------
# Logic tier — stateless engines
# ---------------------------------------------------------------------------

class SlicePhaseReader:
    """
    Stateless engine that reads a single DICOM file and returns a DicomPhaseRecord.
    Accepts no filesystem assumptions; the caller provides the resolved path.
    """

    def read(self, filepath: str) -> DicomPhaseRecord:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)

        return DicomPhaseRecord(
            filepath=filepath,
            instance_number=extract_optional_int(ds, "InstanceNumber"),
            trigger_time_ms=extract_optional_float(ds, 0x0018, 0x1060),   # Trigger Time
            rr_interval_ms=extract_optional_float(ds, 0x0018, 0x1062),    # Nominal Interval
            cardiac_cycle_fraction=extract_optional_float(ds, 0x0018, 0x9169),  # Cardiac Cycle Fraction
        )


class SeriesPhaseAnalyser:
    """
    Stateless engine that resolves and classifies phase data across a
    collection of DicomPhaseRecords.

    Dependencies are injected at instantiation; no globals or singletons.
    """

    def __init__(self, reader: SlicePhaseReader) -> None:
        self._reader = reader

    def analyse(self, filepaths: list[str]) -> SeriesPhaseSummary:
        records: list[DicomPhaseRecord] = []
        for path in filepaths:
            records.append(self._reader.read(path))

        records_with_data = [r for r in records if r.trigger_time_ms is not None]
        is_gated = len(records_with_data) > 0
        multi_phase = is_multi_phase_series(records) if is_gated else False

        trigger_times = [r.trigger_time_ms for r in records_with_data]
        min_t = min(trigger_times) if trigger_times else None
        max_t = max(trigger_times) if trigger_times else None
        spread = (max_t - min_t) if (min_t is not None and max_t is not None) else None

        slice_results = [self._resolve_slice(r) for r in records]
        slice_results.sort(key=lambda s: s.record.instance_number or 0)

        return SeriesPhaseSummary(
            total_files_scanned=len(records),
            files_with_trigger_data=len(records_with_data),
            is_gated=is_gated,
            is_multi_phase=multi_phase,
            min_trigger_ms=min_t,
            max_trigger_ms=max_t,
            spread_ms=spread,
            slice_results=slice_results,
        )

    def _resolve_slice(self, record: DicomPhaseRecord) -> SlicePhaseResult:
        phase_pct: Optional[float] = None

        # Prefer explicit cardiac cycle fraction tag when available
        if record.cardiac_cycle_fraction is not None:
            phase_pct = record.cardiac_cycle_fraction * 100.0
        elif record.trigger_time_ms is not None and record.rr_interval_ms is not None:
            phase_pct = compute_phase_pct(record.trigger_time_ms, record.rr_interval_ms)

        if phase_pct is not None:
            label = classify_cardiac_phase(phase_pct)
        elif record.trigger_time_ms is not None:
            label = f"Trigger at {record.trigger_time_ms:.1f} ms (no RR interval — cannot compute %)"
        else:
            label = "No cardiac gating data found"

        return SlicePhaseResult(
            record=record,
            estimated_phase_pct=round(phase_pct, 1) if phase_pct is not None else None,
            phase_label=label,
        )


# ---------------------------------------------------------------------------
# Orchestrator — all I/O lives here
# ---------------------------------------------------------------------------

def collect_dicom_files(folder: str, sample_n: int) -> list[str]:
    """Resolve and sample DICOM file paths from a directory."""
    patterns = [
        os.path.join(folder, "*.dcm"),
        os.path.join(folder, "*.DCM"),
        os.path.join(folder, "*"),          # scanners often omit extension
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for f in found:
        if f not in seen and os.path.isfile(f):
            seen.add(f)
            unique.append(f)

    unique.sort()

    if not unique:
        return []

    step = max(1, len(unique) // sample_n)
    return unique[::step]


def print_summary(summary: SeriesPhaseSummary) -> None:
    """Render the SeriesPhaseSummary to stdout."""
    print()
    print("=" * 60)
    print("  CARDIAC PHASE INSPECTION REPORT")
    print("=" * 60)
    print(f"  Files scanned       : {summary.total_files_scanned}")
    print(f"  Files with ECG data : {summary.files_with_trigger_data}")
    print(f"  ECG-gated series    : {'Yes' if summary.is_gated else 'No'}")

    if not summary.is_gated:
        print()
        print("  No cardiac gating tags detected.")
        print("  This is likely a non-gated CT acquisition.")
        print("=" * 60)
        return

    print(f"  Multi-phase series  : {'Yes' if summary.is_multi_phase else 'No'}")
    if summary.min_trigger_ms is not None:
        print(f"  Trigger time range  : {summary.min_trigger_ms:.1f} ms "
              f"- {summary.max_trigger_ms:.1f} ms  "
              f"(spread: {summary.spread_ms:.1f} ms)")

    print()

    if summary.is_multi_phase:
        print("  Per-slice phase breakdown:")
        print(f"  {'Instance':>10}  {'Trigger (ms)':>14}  {'Phase %':>8}  Label")
        print("  " + "-" * 56)
        for s in summary.slice_results:
            inst = str(s.record.instance_number) if s.record.instance_number is not None else "?"
            trig = f"{s.record.trigger_time_ms:.1f}" if s.record.trigger_time_ms is not None else "n/a"
            pct  = f"{s.estimated_phase_pct:.1f}%" if s.estimated_phase_pct is not None else "n/a"
            print(f"  {inst:>10}  {trig:>14}  {pct:>8}  {s.phase_label}")
    else:
        # Single-phase: all slices should agree; report the first resolved slice
        representative = next(
            (s for s in summary.slice_results if s.estimated_phase_pct is not None),
            summary.slice_results[0] if summary.slice_results else None,
        )
        if representative:
            print("  Single-phase series — all slices share the same cardiac phase.")
            print()
            r = representative.record
            if r.trigger_time_ms is not None:
                print(f"  Trigger time  : {r.trigger_time_ms:.1f} ms")
            if r.rr_interval_ms is not None:
                print(f"  RR interval   : {r.rr_interval_ms:.1f} ms")
            if representative.estimated_phase_pct is not None:
                print(f"  Phase         : {representative.estimated_phase_pct:.1f}% of RR interval")
            print(f"  Interpretation: {representative.phase_label}")

    print("=" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect cardiac phase information from a DICOM series folder."
    )
    parser.add_argument("folder", help="Path to the folder containing DICOM files.")
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        metavar="N",
        help="Maximum number of files to sample for the survey (default: 20).",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {folder}")
    filepaths = collect_dicom_files(folder, sample_n=args.sample)

    if not filepaths:
        print("No DICOM files found in the specified folder.", file=sys.stderr)
        sys.exit(1)

    print(f"Sampled {len(filepaths)} file(s) from the series.")

    reader = SlicePhaseReader()
    analyser = SeriesPhaseAnalyser(reader=reader)
    summary = analyser.analyse(filepaths)

    print_summary(summary)


if __name__ == "__main__":
    main()
