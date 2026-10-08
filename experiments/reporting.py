"""Builds defense tables, figure PDFs, PDF-derived PNGs, chart data, semantic repair outputs, and summary catalogs.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import csv
import math
import re
import shutil
from decimal import Decimal, InvalidOperation, getcontext
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

getcontext().prec = 50

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from dataset_tools.serialization import save_json
from dataset_tools.simple_xlsx import save_rows_as_xlsx
from experiments.pipeline import save_rows_as_csv

FULL_PRECISION_SIGNIFICANT_DIGITS = 17
ANNOTATION_FONT_SIZE = 5.6
GROUPED_ANNOTATION_FONT_SIZE = 4.8
PDF_PLOT_DPI = 300
PNG_PLOT_DPI = 220
NEAR_EQUAL_RELATIVE_SPAN_THRESHOLD = 2e-3
NEAR_EQUAL_ABSOLUTE_SPAN_THRESHOLD = 1e-9


ALGORITHM_DISPLAY_NAMES: Dict[str, str] = {
    "gnn": "GNN",
    "heft": "HEFT",
    "nsga_iii": "NSGA-III",
    "proposed_gat_float32": "Proposed GAT (FP32)",
    "proposed_gat_quantization_int8": "Proposed GAT (QAT INT8)",
    "original_fp32_reference": "Original FP32 Reference",
    "qat_int8_pdf": "QAT INT8 - PDF Method",
    "weight_only_int8_per_channel": "Weight-Only INT8 Per-Channel",
    "fp16_mixed_precision": "FP16 Mixed Precision",
    "vd_edf": "VD-EDF",
    "with_uncertainty": "With Uncertainty",
    "without_uncertainty": "Without Uncertainty",
}

PASTEL_COLORS: Dict[str, str] = {
    "gnn": "#B8E1FF",
    "heft": "#FFD6A5",
    "nsga_iii": "#CDEAC0",
    "proposed_gat_float32": "#FFCAD4",
    "proposed_gat_quantization_int8": "#E2C6FF",
    "original_fp32_reference": "#C7CEEA",
    "qat_int8_pdf": "#FFCAD4",
    "weight_only_int8_per_channel": "#CDEAC0",
    "fp16_mixed_precision": "#FFD6A5",
    "vd_edf": "#C7CEEA",
    "with_uncertainty": "#B8E1FF",
    "without_uncertainty": "#FFD6A5",
}

ALGORITHM_HATCHES: Dict[str, str] = {
    "gnn": "////",
    "heft": "\\\\",
    "nsga_iii": "xxxx",
    "proposed_gat_float32": "....",
    "proposed_gat_quantization_int8": "----",
    "original_fp32_reference": "",
    "qat_int8_pdf": "....",
    "weight_only_int8_per_channel": "xxxx",
    "fp16_mixed_precision": "\\",
    "vd_edf": "++++",
    "with_uncertainty": "////",
    "without_uncertainty": "\\\\",
}

ALGORITHM_MARKERS: Dict[str, str] = {
    "gnn": "s",
    "heft": "^",
    "nsga_iii": "D",
    "proposed_gat_float32": "o",
    "proposed_gat_quantization_int8": "P",
    "original_fp32_reference": "o",
    "qat_int8_pdf": "P",
    "weight_only_int8_per_channel": "D",
    "fp16_mixed_precision": "^",
    "vd_edf": "X",
    "with_uncertainty": "o",
    "without_uncertainty": "s",
}

ALGORITHM_LINESTYLES: Dict[str, str] = {
    "gnn": "-",
    "heft": "--",
    "nsga_iii": "-.",
    "proposed_gat_float32": "-",
    "proposed_gat_quantization_int8": ":",
    "original_fp32_reference": "-",
    "qat_int8_pdf": "-",
    "weight_only_int8_per_channel": "-.",
    "fp16_mixed_precision": ":",
    "vd_edf": "--",
    "with_uncertainty": "-",
    "without_uncertainty": "--",
}

BASELINE_METRICS = [
    "mean_deadline_miss_ratio",
    "mean_task_deadline_miss_ratio",
    "mean_service_loss_ratio",
    "mean_dropped_lo_ratio",
    "mean_completed_lo_ratio",
    "mean_utilization",
    "mean_energy",
    "mean_mode_switch_probability",
    "mean_mode_switch_count",
    "mean_mode_switch_time_overhead",
    "mean_mode_switch_energy_overhead",
    "mean_average_cvar",
    "mean_aging_index",
    "mean_average_reliability",
    "mean_completed_hi_ratio",
]

METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "mean_deadline_miss_ratio": "Deadline Miss Ratio",
    "mean_application_deadline_miss_ratio": "Application-level DMR",
    "mean_task_deadline_miss_ratio": "Task-level DMR",
    "mean_service_loss_ratio": "Service Loss",
    "mean_dropped_lo_ratio": "Dropped LO",
    "mean_completed_lo_ratio": "Completed LO",
    "mean_completed_task_ratio": "Completed Tasks",
    "mean_utilization": "Utilization",
    "mean_energy": "Energy",
    "mean_energy_proportional_workload_total": "Total Energy",
    "mean_mode_switch_probability": "Mode Switch Probability",
    "mean_mode_switch_count": "Mode Switch Count",
    "mean_mode_switch_time_overhead": "Mode-Switch Time Overhead",
    "mean_mode_switch_energy_overhead": "Mode-Switch Energy Overhead",
    "mean_average_cvar": "CVaR",
    "mean_aging_index": "Aging",
    "mean_average_reliability": "Reliability",
    "mean_completed_hi_ratio": "Completed HI",
    "parameter_count": "Parameters",
    "model_size_bytes": "Model Size",
    "predictor_size_bytes": "Predictor Size",
    "policy_size_bytes": "Policy Size",
    "scheduler_size_bytes": "Scheduler Size",
    "scheduler_latency_ms": "Legacy Scheduler Latency",
    "scheduler_decision_latency_ms": "Scheduler Decision Latency",
    "end_to_end_latency_ms": "End-to-End Scheduling Latency",
    "predictor_latency_ms": "Predictor Latency",
    "policy_latency_ms": "Policy Latency",
    "prediction_mae": "Prediction MAE",
    "MAE": "MAE",
    "MSE": "MSE",
    "RMSE": "RMSE",
    "MAPE": "MAPEε",
    "MAPE_raw": "Raw MAPE",
    "MAPE_epsilon": "MAPEε",
    "sMAPE": "sMAPE",
    "energy_before_dvfs": "Energy Before DVFS",
    "energy_after_dvfs": "Energy After DVFS",
    "power_before_dvfs": "Power Before DVFS",
    "power_after_dvfs": "Power After DVFS",
    "utilization_per_core": "Per-Core Utilization",
}


METRIC_UNITS: Dict[str, str] = {
    "mean_deadline_miss_ratio": "ratio; deadline misses over total tasks",
    "deadline_miss_ratio": "ratio; deadline misses over total tasks",
    "mean_application_deadline_miss_ratio": "ratio; application-level",
    "application_deadline_miss_ratio": "ratio; application-level",
    "mean_task_deadline_miss_ratio": "ratio; task-level sub-deadline",
    "task_deadline_miss_ratio": "ratio; task-level sub-deadline",
    "mean_service_loss_ratio": "ratio",
    "service_loss_ratio": "ratio",
    "mean_dropped_lo_ratio": "ratio",
    "dropped_lo_ratio": "ratio",
    "mean_completed_lo_ratio": "ratio",
    "completed_lo_ratio": "ratio",
    "mean_completed_task_ratio": "ratio",
    "completed_task_ratio": "ratio",
    "mean_utilization": "ratio",
    "utilization": "ratio",
    "mean_energy": "relative energy unit; Σ(C·V²·GHz·time)",
    "mean_energy_proportional_workload_total": "relative energy unit; total under proportional workload",
    "energy": "relative energy unit; Σ(C·V²·GHz·time)",
    "mean_mode_switch_probability": "probability",
    "mode_switch_probability": "probability",
    "mean_mode_switch_count": "count",
    "mode_switch_count": "count",
    "mean_mode_switch_time_overhead": "sim_time_unit",
    "mode_switch_time_overhead": "sim_time_unit",
    "mean_mode_switch_energy_overhead": "relative_energy_unit",
    "mode_switch_energy_overhead": "relative_energy_unit",
    "mode_switch_power_overhead": "relative_power_unit",
    "mean_average_cvar": "time units",
    "average_cvar": "time units",
    "mean_aging_index": "aging proxy; Σλ(t)ρ(t)",
    "aging_index": "aging proxy; Σλ(t)ρ(t)",
    "mean_average_reliability": "probability",
    "average_reliability": "probability",
    "mean_completed_hi_ratio": "ratio",
    "completed_hi_ratio": "ratio",
    "parameter_count": "count",
    "model_size_bytes": "bytes",
    "predictor_size_bytes": "bytes",
    "policy_size_bytes": "bytes",
    "scheduler_size_bytes": "bytes",
    "predictor_latency_ms": "milliseconds",
    "policy_latency_ms": "milliseconds",
    "scheduler_latency_ms": "milliseconds",
    "prediction_mae": "time units",
    "average_mean_absolute_prediction_error": "time units",
    "average_relative_size_reduction": "ratio",
    "MAE": "time units",
    "MSE": "squared time units",
    "RMSE": "time units",
    "MAPE": "ratio",
    "energy_before_dvfs": "relative energy unit",
    "energy_after_dvfs": "relative energy unit",
    "power_before_dvfs": "relative power unit",
    "power_after_dvfs": "relative power unit",
    "utilization_per_core": "ratio",
}

CONFIG_PATTERN = re.compile(
    r"graphs_(?P<graph_count>\d+)/density_(?P<density>\d+_\d+)/regularity_(?P<regularity>\d+_\d+)/width_(?P<width>\d+_\d+)"
)


def load_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    """Loads a CSV file as a list of dictionaries."""

    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def count_csv_data_rows(path: str | Path) -> int:
    """Counts data rows in a CSV file without loading it into memory."""

    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as file:
        # subtract one header row when the file is non-empty
        line_count = sum(1 for _ in file)
    return max(0, line_count - 1)


def sample_csv_rows(path: str | Path, limit: int = 5000) -> List[Dict[str, str]]:
    """Reads at most ``limit`` rows from a CSV file for preview/report tables."""

    path = Path(path)
    if not path.exists() or limit <= 0:
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            if index >= limit:
                break
            rows.append(dict(row))
    return rows


def detailed_output_manifest_rows(paths: Dict[str, Path]) -> List[Dict[str, float | str]]:
    """Summarizes detailed CSV artifacts without duplicating huge files."""

    manifest: List[Dict[str, float | str]] = []
    for name, path in paths.items():
        manifest.append(
            {
                "artifact": name,
                "csv_path": str(path),
                "rows": count_csv_data_rows(path),
                "status": "exists" if path.exists() else "missing",
            }
        )
    return manifest



def _safe_relative_path(path: Path, base: Path) -> str:
    """Returns a readable path without changing any actual output address."""

    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def _file_size_mb(path: Path) -> float:
    """Internal helper for file size mb."""
    if not path.exists() or not path.is_file():
        return 0.0
    return round(path.stat().st_size / (1024 * 1024), 6)


def _csv_header(path: Path) -> str:
    """Internal helper for csv header."""
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".csv":
        return ""
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            first_line = file.readline().strip()
        return first_line[:5000]
    except Exception as exc:
        return f"<read-error: {exc}>"


def _catalog_files(paths: Sequence[Path], *, base: Path, label: str) -> List[Dict[str, float | str]]:
    """Internal helper for catalog files."""
    rows: List[Dict[str, float | str]] = []
    for root in paths:
        root = Path(root)
        if not root.exists():
            rows.append({
                "group": label,
                "path": str(root),
                "relative_path": "",
                "extension": "",
                "size_mb": 0.0,
                "rows": 0,
                "status": "missing_root",
                "header_preview": "",
            })
            continue
        for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
            suffix = file_path.suffix.lower()
            rows.append({
                "group": label,
                "path": str(file_path),
                "relative_path": _safe_relative_path(file_path, base),
                "extension": suffix,
                "size_mb": _file_size_mb(file_path),
                "rows": count_csv_data_rows(file_path) if suffix == ".csv" else "",
                "status": "exists",
                "header_preview": _csv_header(file_path),
            })
    return rows


def _copy_summary_files(files: Sequence[Path], destination_dir: Path, *, max_size_mb: float = 25.0) -> List[Dict[str, float | str]]:
    """Internal helper for copy summary files."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float | str]] = []
    used_names: Dict[str, int] = defaultdict(int)
    for source in sorted(Path(p) for p in files if Path(p).exists() and Path(p).is_file()):
        size_mb = _file_size_mb(source)
        base_name = source.name
        used_names[base_name] += 1
        target_name = base_name if used_names[base_name] == 1 else f"{source.stem}_{used_names[base_name]}{source.suffix}"
        target = destination_dir / target_name
        error_message = ""
        if size_mb <= max_size_mb:
            try:
                shutil.copy2(source, target)
                status = "copied"
            except (PermissionError, OSError) as exc:
                # On Windows, Excel/Explorer/antivirus may transiently lock XLSX/PDF files.
                # Summary creation should not fail the whole defense report; the source
                # path is still recorded in the catalog so the file is not lost.
                status = "skipped_locked_or_copy_error"
                error_message = str(exc)
        else:
            status = "skipped_too_large"
        rows.append({
            "source_path": str(source),
            "copied_to": str(target) if status == "copied" else "",
            "file_name": source.name,
            "size_mb": size_mb,
            "status": status,
            "error": error_message,
        })
    return rows


def build_defense_summary_folder(
    *,
    all_config_output_root: str | Path,
    structural_output_root: str | Path | None,
    output_root: str | Path,
) -> Dict[str, str]:
    """Creates a compact summary folder without moving the defense report address.

    The full detailed CSV files remain in their original locations.  This function
    creates catalogs, copies small Excel files for quick inspection, and records
    where every large artifact lives.
    """

    all_config_output_root = Path(all_config_output_root)
    structural_output_root = Path(structural_output_root) if structural_output_root is not None else None
    output_root = Path(output_root)
    summary_dir = output_root / "summary"
    excel_dir = summary_dir / "excel_files"
    chart_excel_dir = summary_dir / "chart_data_excel"
    summary_dir.mkdir(parents=True, exist_ok=True)
    excel_dir.mkdir(parents=True, exist_ok=True)
    chart_excel_dir.mkdir(parents=True, exist_ok=True)

    roots_to_catalog = [output_root, all_config_output_root]
    if structural_output_root is not None:
        roots_to_catalog.append(structural_output_root)

    catalog_rows = _catalog_files(roots_to_catalog, base=Path.cwd(), label="project_output")
    save_rows_as_csv(catalog_rows, summary_dir / "all_output_files_catalog.csv", xlsx_row_limit=100_000)

    excel_files = [Path(row["path"]) for row in catalog_rows if row.get("extension") == ".xlsx" and Path(str(row.get("path", ""))).exists()]
    chart_excel_files = [p for p in excel_files if "chart_data" in str(p).lower()]
    non_chart_excel_files = [p for p in excel_files if p not in chart_excel_files]

    copied_excel_rows = _copy_summary_files(non_chart_excel_files, excel_dir)
    copied_chart_excel_rows = _copy_summary_files(chart_excel_files, chart_excel_dir)
    save_rows_as_csv(copied_excel_rows, summary_dir / "copied_excel_files_catalog.csv", xlsx_row_limit=50_000)
    save_rows_as_csv(copied_chart_excel_rows, summary_dir / "copied_chart_data_excel_catalog.csv", xlsx_row_limit=50_000)

    plot_rows = [row for row in catalog_rows if row.get("extension") in {".pdf", ".png"}]
    save_rows_as_csv(plot_rows, summary_dir / "plot_files_catalog.csv", xlsx_row_limit=100_000)

    aggregate_root = all_config_output_root / "_aggregate"
    large_detail_paths = {
        "task_level_results_aggregate": aggregate_root / "task_level_results_aggregate.csv",
        "core_level_results_aggregate": aggregate_root / "core_level_results_aggregate.csv",
        "schedule_summary_extended_aggregate": aggregate_root / "schedule_summary_extended_aggregate.csv",
        "energy_power_summary_aggregate": aggregate_root / "energy_power_summary_aggregate.csv",
        "prediction_metrics_aggregate": aggregate_root / "prediction_metrics_aggregate.csv",
        "prediction_detail_results_aggregate": aggregate_root / "prediction_detail_results_aggregate.csv",
        "power_trace_aggregate": aggregate_root / "power_trace_aggregate.csv",
        "training_loss_curves_aggregate": aggregate_root / "training_loss_curves_aggregate.csv",
    }
    large_manifest = detailed_output_manifest_rows(large_detail_paths)
    save_rows_as_csv(large_manifest, summary_dir / "large_detailed_outputs_manifest.csv", xlsx_row_limit=10_000)

    readme = """# خلاصه خروجی‌های دفاع

این پوشه فقط یک خلاصه و ایندکس از خروجی‌هاست و مسیر اصلی گزارش دفاع را تغییر نمی‌دهد.

- `all_output_files_catalog.csv/xlsx`: فهرست همه فایل‌های خروجی، جدول‌ها، نمودارها و گزارش‌ها.
- `plot_files_catalog.csv/xlsx`: فهرست نمودارهای PDF و PNG.
- `copied_excel_files_catalog.csv/xlsx`: فهرست فایل‌های Excel کوچک که داخل `excel_files` کپی شده‌اند.
- `copied_chart_data_excel_catalog.csv/xlsx`: فهرست Excelهای داده نمودار که داخل `chart_data_excel` کپی شده‌اند.
- `large_detailed_outputs_manifest.csv/xlsx`: فهرست خروجی‌های بزرگ ریزدانه؛ خود فایل‌های بزرگ در مسیر اصلی باقی می‌مانند.

خروجی‌های بزرگ مثل task-level با چند میلیون ردیف به XLSX کامل تبدیل نمی‌شوند، چون Excel و حافظه سیستم برای این حجم مناسب نیستند. CSV کامل، خروجی مرجع است.
"""
    (summary_dir / "README_FA.md").write_text(readme, encoding="utf-8")

    return {
        "summary_dir": str(summary_dir),
        "excel_dir": str(excel_dir),
        "chart_excel_dir": str(chart_excel_dir),
        "catalog_csv": str(summary_dir / "all_output_files_catalog.csv"),
    }


def _to_float(value: str | float | int | None) -> float:
    """Converts a scalar-like value to float and suppresses non-finite artifacts."""

    if value is None or value == "":
        return 0.0
    try:
        if isinstance(value, (float, int)):
            number = float(value)
        else:
            number = float(str(value).strip())
    except Exception:
        return 0.0
    return number if math.isfinite(number) else 0.0


def _display_name(algorithm_name: str) -> str:
    """Returns a presentation-friendly algorithm name."""

    return ALGORITHM_DISPLAY_NAMES.get(algorithm_name, algorithm_name)



def _metric_unit(metric_name: str) -> str:
    """Returns the explicit unit used for a metric in plots and tables."""

    return METRIC_UNITS.get(metric_name, "unitless")


def _metric_label(metric_name: str) -> str:
    """Returns a metric label with its unit for report outputs."""

    display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    unit = _metric_unit(metric_name)
    if unit in {"", "unitless"}:
        return display_name
    return f"{display_name} [{unit}]"


def _to_decimal(value: str | float | int | None) -> Decimal:
    """Converts scalar values to Decimal for high-precision report aggregation."""

    if value is None or value == "":
        return Decimal("0")
    try:
        if isinstance(value, float):
            return Decimal(format(value, ".17g"))
        if isinstance(value, int):
            return Decimal(value)
        text = str(value).strip()
        if not text:
            return Decimal("0")
        number = Decimal(text)
        if not number.is_finite():
            return Decimal("0")
        return number
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _decimal_mean(values: Iterable[str | float | int | None]) -> Decimal:
    """Computes an arithmetic mean with Decimal precision for report tables."""

    decimals = [_to_decimal(value) for value in values]
    if not decimals:
        return Decimal("0")
    return sum(decimals, Decimal("0")) / Decimal(len(decimals))


def _decimal_to_output(value: Decimal) -> str:
    """Serializes Decimal values without presentation rounding."""

    if not value.is_finite():
        return str(value)
    return format(value, "f")


def _parse_configuration_metadata(configuration: str) -> Dict[str, float]:
    """Extracts graph_count/density/regularity/width from a configuration path."""

    match = CONFIG_PATTERN.search(configuration.replace("\\", "/"))
    if match is None:
        return {
            "graph_count": 0.0,
            "density": 0.0,
            "regularity": 0.0,
            "width": 0.0,
        }
    payload = match.groupdict()
    return {
        "graph_count": float(payload["graph_count"]),
        "density": float(payload["density"].replace("_", ".")),
        "regularity": float(payload["regularity"].replace("_", ".")),
        "width": float(payload["width"].replace("_", ".")),
    }


def _decorate_rows_with_configuration_axes(rows: Sequence[Dict[str, str]]) -> List[Dict[str, float | str]]:
    """Adds parsed configuration axes to flat aggregate rows."""

    output: List[Dict[str, float | str]] = []
    for row in rows:
        decorated = dict(row)
        decorated.update(_parse_configuration_metadata(str(row.get("configuration", ""))))
        output.append(decorated)
    return output


def _mean_by_group(rows: Sequence[Dict[str, str | float]], group_key: str, metric_keys: Sequence[str]) -> List[Dict[str, float | str]]:
    """Computes grouped means over a flat row collection."""

    grouped: Dict[str, List[Dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)

    output: List[Dict[str, float | str]] = []
    for group_value, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary_row: Dict[str, float | str] = {group_key: group_value}
        for metric_key in metric_keys:
            summary_row[metric_key] = _decimal_to_output(_decimal_mean(row.get(metric_key) for row in group_rows))
        output.append(summary_row)
    return output


def _mean_by_two_keys(
    rows: Sequence[Dict[str, str | float]],
    key_1: str,
    key_2: str,
    metric_keys: Sequence[str],
) -> List[Dict[str, float | str]]:
    """Computes grouped means over two grouping keys."""

    grouped: Dict[tuple[str, str], List[Dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row[key_1]), str(row[key_2]))].append(row)

    output: List[Dict[str, float | str]] = []
    for (group_1, group_2), group_rows in sorted(grouped.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))):
        summary_row: Dict[str, float | str] = {key_1: group_1, key_2: group_2}
        for metric_key in metric_keys:
            summary_row[metric_key] = _decimal_to_output(_decimal_mean(row.get(metric_key) for row in group_rows))
        output.append(summary_row)
    return output


def _sort_key(value: str | float | int) -> tuple[int, float | str]:
    """Sorts numeric strings numerically and falls back to lexical ordering."""

    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


def _algorithm_order(rows: Sequence[Dict[str, float | str]], key: str = "algorithm") -> List[str]:
    """Returns algorithms in a stable order for plots and tables."""

    canonical_order = [
        "gnn",
        "heft",
        "nsga_iii",
        "proposed_gat_float32",
        "proposed_gat_quantization_int8",
        "original_fp32_reference",
        "qat_int8_pdf",
        "weight_only_int8_per_channel",
        "fp16_mixed_precision",
        "vd_edf",
        "with_uncertainty",
        "without_uncertainty",
    ]
    present = {str(row[key]) for row in rows}
    ordered = [item for item in canonical_order if item in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _format_value_label(value: float) -> str:
    """Formats a numeric value without presentation rounding.

    Python floats are IEEE-754 double precision values. Seventeen significant
    digits are enough for round-trip reconstruction of the stored float, so this
    formatter preserves the numerical value used by the experiment outputs.
    """

    number = float(value)
    if not math.isfinite(number):
        return str(number)
    if abs(number) < 1e-300:
        return "0"
    return format(number, f".{FULL_PRECISION_SIGNIFICANT_DIGITS}g")


def _apply_bar_axis_padding(ax: plt.Axes, values: Sequence[float]) -> tuple[float, float]:
    """Expands y-limits moderately so annotations remain inside the axes."""

    finite_values = [float(v) for v in values if math.isfinite(float(v))]
    if not finite_values:
        ax.set_ylim(0.0, 1.0)
        return (0.0, 1.0)

    lower = min(0.0, min(finite_values))
    upper = max(0.0, max(finite_values))
    span = upper - lower
    if span <= 1e-12:
        span = max(1.0, abs(upper), abs(lower))
    lower_pad = 0.06 * span if lower < 0.0 else 0.0
    upper_pad = 0.18 * span + 1e-9
    ax.set_ylim(lower - lower_pad, upper + upper_pad)
    return ax.get_ylim()


def _should_use_symlog(values: Sequence[float], threshold_ratio: float = 30.0) -> tuple[bool, float]:
    """Decides whether a symlog y-axis would improve readability for wide dynamic ranges."""

    positive = [abs(float(v)) for v in values if math.isfinite(float(v)) and abs(float(v)) > 0.0]
    if len(positive) < 2:
        return (False, 1e-6)
    min_positive = min(positive)
    max_positive = max(positive)
    ratio = max_positive / max(min_positive, 1e-12)
    if ratio < threshold_ratio:
        return (False, max(min_positive, 1e-6))
    linthresh = max(min_positive * 2.0, 1e-6)
    return (True, linthresh)


def _maybe_apply_symlog_y(ax: plt.Axes, values: Sequence[float]) -> bool:
    """Applies a symlog y-scale when orders of magnitude differ strongly."""

    use_symlog, linthresh = _should_use_symlog(values)
    if use_symlog:
        ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0, base=10)
        ax.text(
            0.99,
            0.98,
            "symlog y-scale",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#9ca3af", "lw": 0.4, "alpha": 0.85},
            zorder=10,
        )
    return use_symlog


def _set_post_scale_limits(ax: plt.Axes, values: Sequence[float], used_symlog: bool) -> None:
    """Applies safe y-limits after scale selection."""

    if used_symlog:
        finite_values = [float(v) for v in values if math.isfinite(float(v))]
        if not finite_values:
            ax.set_ylim(0.0, 1.0)
            return
        lower = min(0.0, min(finite_values))
        upper = max(0.0, max(finite_values))
        span = upper - lower
        if span <= 1e-12:
            span = max(1.0, abs(upper), abs(lower))
        upper_pad = 0.35 * span + 1e-9
        lower_pad = 0.08 * span if lower < 0.0 else 0.0
        ax.set_ylim(lower - lower_pad, upper + upper_pad)
    else:
        _apply_bar_axis_padding(ax, values)


def _power10_order_exponent(max_abs_value: float) -> int:
    """Returns the 10^k order used to display raw values on a comparable visual scale.

    Chart rows keep the raw value for labels and CSV/XLSX exports.  The plotted
    ``scaled_value`` is ``raw_value / 10^k`` and the category label carries
    ``×10^k`` so mixed-unit aggregate charts are readable without normalization.
    """

    if not math.isfinite(max_abs_value) or max_abs_value <= 0.0:
        return 0
    return int(math.floor(math.log10(max_abs_value)))


def _scale_label(base_label: str, exponent: int) -> str:
    """Returns a two-line metric label with the 10^k scale shown under it."""

    return f"{base_label}\n×10^{exponent}"


def _build_scaled_metric_matrix(
    rows: Sequence[Dict[str, float | str]],
    metric_keys: Sequence[str],
    key_name: str = "algorithm",
) -> List[Dict[str, float | str]]:
    """Builds metric-specific 10^k scaled rows without normalizing values.

    This is used instead of normalized overview charts.  It preserves the raw
    metric numbers in ``raw_value`` while plotting each metric in its own
    decade-scale bucket so different units can appear on one aggregate chart.
    """

    output: List[Dict[str, float | str]] = []
    for metric_key in metric_keys:
        raw_values = [_to_float(row.get(metric_key)) for row in rows]
        max_abs = max([abs(value) for value in raw_values], default=0.0)
        if max_abs <= 0.0:
            continue
        exponent = _power10_order_exponent(max_abs)
        divisor = 10.0 ** exponent if exponent != 0 else 1.0
        metric_label = _scale_label(_metric_label(metric_key), exponent)
        for row in rows:
            raw_value = _to_float(row.get(metric_key))
            output.append({
                key_name: str(row[key_name]),
                "metric": metric_key,
                "metric_display_name": metric_label,
                "unit": _metric_unit(metric_key),
                "raw_value": raw_value,
                "scaled_value": raw_value / divisor,
                "scale_exponent": float(exponent),
                "scale_factor": divisor,
                "scale_note": f"displayed_value = raw_value / 10^{exponent}",
            })
    return output


def _build_uncertainty_scaled_rows(uncertainty_overall: Sequence[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    """Builds one merged uncertainty comparison with metric-specific 10^k scaling."""

    metric_specs = [
        "mean_mode_switch_probability",
        "mean_deadline_miss_ratio",
        "mean_utilization",
        "mean_energy",
        "mean_average_cvar",
        "mean_aging_index",
        "mean_average_reliability",
        "mean_completed_hi_ratio",
    ]
    by_algorithm = {str(row["algorithm"]): dict(row) for row in uncertainty_overall}
    source_rows: List[Dict[str, float | str]] = []
    for algorithm_name in ("with_uncertainty", "without_uncertainty"):
        if algorithm_name in by_algorithm:
            source_rows.append(by_algorithm[algorithm_name])
    if not source_rows:
        return []
    return _build_scaled_metric_matrix(source_rows, metric_specs, key_name="algorithm")


def _structural_proportional_subset_rows(structural_summary_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keeps the no-rerun B subset: min base_node_count scaled proportionally with core count.

    With current outputs this selects cases like 4/50, 8/100, 16/200,
    32/400, 64/800.  These rows are valid for scalability under proportional
    workload, not for the isolated effect of core count.
    """

    if not structural_summary_rows:
        return []
    scaled_rows = [row for row in structural_summary_rows if _to_float(row.get("scale_nodes_with_cores", 0.0)) >= 0.5]
    source_rows = scaled_rows or list(structural_summary_rows)
    base_values = [_to_float(row.get("base_node_count", row.get("node_count"))) for row in source_rows]
    positive_bases = [value for value in base_values if value > 0.0]
    if not positive_bases:
        return []
    target_base = min(positive_bases)
    subset = [row for row in source_rows if abs(_to_float(row.get("base_node_count", row.get("node_count"))) - target_base) <= 1e-9]
    for row in subset:
        row.setdefault("scalability_interpretation", "proportional_workload_subset")
        row.setdefault("workload_scaling_note", "node/task count scales with core count; chart is not isolated core-count effect")
    return subset


def _core_task_pair_label(row: Dict[str, float | str]) -> str:
    """Builds the defense-facing proportional core/task label, e.g. 4 core + 50 task."""

    core_count = int(round(_to_float(row.get("core_count", 0.0))))
    node_count = int(round(_to_float(row.get("node_count", 0.0))))
    return f"{core_count} core + {node_count} task"


def _add_core_task_pair_labels(rows: Sequence[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    """Adds explicit proportional-workload labels to core-count chart rows."""

    labeled_rows: List[Dict[str, float | str]] = []
    for row in rows:
        labeled_row = dict(row)
        labeled_row["core_task_pair"] = _core_task_pair_label(labeled_row)
        labeled_rows.append(labeled_row)
    return labeled_rows


def _core_task_pair_order(rows: Sequence[Dict[str, float | str]]) -> List[str]:
    """Returns core/task pair labels ordered by numeric core count."""

    ordered: List[str] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: _to_float(item.get("core_count", 0.0))):
        label = str(row.get("core_task_pair", _core_task_pair_label(row)))
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    return ordered


def _add_proportional_total_energy(rows: Sequence[Dict[str, str]]) -> List[Dict[str, float | str]]:
    """Adds a defense-facing total-energy metric for proportional workload plots.

    Older structural outputs increased node_count with core_count but kept
    total_utilization fixed.  In those cached runs, raw ``mean_energy`` is closer
    to energy for a constant total workload.  For the B path used in the defense
    chart (4/50, 8/100, ..., 64/800), the intended interpretation is increasing
    platform size together with increasing workload size.  Therefore the chart
    uses a workload-adjusted total energy proxy for old outputs.

    New v44+ structural runs store ``effective_total_utilization`` larger than
    ``base_total_utilization``; in that case the simulation already scaled the
    workload demand and the raw mean_energy is used directly.
    """

    enriched: List[Dict[str, float | str]] = []
    for row in rows:
        new_row: Dict[str, float | str] = dict(row)
        mean_energy = _to_float(row.get("mean_energy"))
        base_nodes = _to_float(row.get("base_node_count", row.get("node_count", 0.0)))
        node_count = _to_float(row.get("node_count", 0.0))
        explicit_scale = _to_float(row.get("workload_scale_factor", 0.0))
        node_scale = node_count / base_nodes if base_nodes > 0.0 else 1.0
        workload_scale = explicit_scale if explicit_scale > 0.0 else max(1.0, node_scale)
        base_u = _to_float(row.get("base_total_utilization", row.get("utilization_level", 0.0)))
        effective_u = _to_float(row.get("effective_total_utilization", row.get("utilization_level", 0.0)))
        already_scaled = base_u > 0.0 and effective_u > base_u * 1.000001
        total_energy = mean_energy if already_scaled else mean_energy * workload_scale
        new_row["mean_energy_proportional_workload_total"] = total_energy
        new_row["workload_adjustment_factor_for_energy"] = 1.0 if already_scaled else workload_scale
        new_row["energy_plot_note"] = (
            "raw simulated energy from v44+ scaled-utilization run"
            if already_scaled
            else "workload-adjusted total energy proxy from cached constant-utilization sweep"
        )
        enriched.append(new_row)
    return enriched


def _pdf_sibling_path(output_path: Path) -> Path:
    """Returns the PDF version of a PNG plot under a sibling plots_pdf directory."""

    output_path = Path(output_path)
    parts = list(output_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "plots":
            parts[index] = "plots_pdf"
            return Path(*parts).with_suffix(".pdf")
    return output_path.with_suffix(".pdf")


def _chart_data_sibling_path(output_path: Path) -> Path:
    """Returns the XLSX data file path corresponding to one chart output."""

    output_path = Path(output_path)
    parts = list(output_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "plots":
            parts[index] = "chart_data_xlsx"
            return Path(*parts).with_suffix(".xlsx")
    return output_path.with_suffix(".xlsx")


def _save_chart_data(rows: Sequence[Dict[str, float | str]], output_path: Path) -> None:
    """Stores the exact chart data as an XLSX workbook next to plot outputs."""

    if not rows:
        return
    save_rows_as_xlsx([dict(row) for row in rows], _chart_data_sibling_path(output_path), sheet_name=output_path.stem[:31])


def _should_zoom_near_equal_axis(values: Sequence[float]) -> bool:
    """Detects near-equal high-precision values that should not be plotted from zero."""

    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite_values) < 2:
        return False
    minimum = min(finite_values)
    maximum = max(finite_values)
    span = maximum - minimum
    if span <= 0.0:
        return False
    scale = max(abs(maximum), abs(minimum), 1.0)
    relative_span = span / scale
    return span > NEAR_EQUAL_ABSOLUTE_SPAN_THRESHOLD and relative_span <= NEAR_EQUAL_RELATIVE_SPAN_THRESHOLD


def _apply_zoomed_near_equal_axis(ax: plt.Axes, values: Sequence[float]) -> bool:
    """Zooms the y-axis for near-equal values and annotates that the axis is zoomed."""

    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values or not _should_zoom_near_equal_axis(finite_values):
        return False
    minimum = min(finite_values)
    maximum = max(finite_values)
    span = max(maximum - minimum, NEAR_EQUAL_ABSOLUTE_SPAN_THRESHOLD)
    pad = max(span * 0.55, NEAR_EQUAL_ABSOLUTE_SPAN_THRESHOLD)
    ax.set_ylim(minimum - pad, maximum + pad)
    ax.text(
        0.99,
        0.98,
        "zoomed y-axis",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#9ca3af", "lw": 0.4, "alpha": 0.88},
        zorder=10,
    )
    return True


def _delta_rows_from_min(
    rows: Sequence[Dict[str, float | str]],
    label_key: str,
    value_key: str,
    delta_label: str = "delta_from_min",
) -> List[Dict[str, float | str]]:
    """Builds rows that show the distance from the minimum value for precision-sensitive charts."""

    if not rows:
        return []
    values = [_to_float(row.get(value_key)) for row in rows]
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return []
    base = min(finite_values)
    output: List[Dict[str, float | str]] = []
    for row, value in zip(rows, values):
        new_row = dict(row)
        new_row[delta_label] = value - base
        new_row["base_value"] = base
        new_row["original_value"] = value
        output.append(new_row)
    return output


def _zoomed_delta_companion_chart(
    rows: Sequence[Dict[str, float | str]],
    label_key: str,
    value_key: str,
    output_path: Path,
    title: str,
    xlabel: str,
    unit_label: str,
) -> None:
    """Creates a companion delta chart for near-equal values."""

    if not rows:
        return
    values = [_to_float(row.get(value_key)) for row in rows]
    if not _should_zoom_near_equal_axis(values):
        return
    delta_rows = _delta_rows_from_min(rows, label_key=label_key, value_key=value_key)
    if not delta_rows:
        return
    _bar_chart(
        delta_rows,
        label_key=label_key,
        value_key="delta_from_min",
        output_path=output_path,
        title=f"{title}: Difference from Minimum",
        xlabel=xlabel,
        ylabel=f"Δ {METRIC_DISPLAY_NAMES.get(value_key, value_key)} [{unit_label}]",
    )

def _prepare_plot(figsize: tuple[float, float] = (11.2, 6.4)) -> tuple[plt.Figure, plt.Axes]:
    """Creates a consistent paper-style figure and axis."""

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=11.5)
    return fig, ax


def _render_pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = PNG_PLOT_DPI) -> None:
    """Rasterizes the first page of a PDF chart to PNG.

    The project treats PDF as the source-of-truth chart format.  PNG files are
    generated from the saved PDF so the raster and vector versions stay visually
    identical.  PyMuPDF is preferred because it works on Windows without a
    separate Poppler installation.  Poppler/ImageMagick are kept as fallbacks.
    """

    import shutil
    import subprocess
    import tempfile

    pdf_path = Path(pdf_path)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # Preferred path: pure Python dependency available through pip install PyMuPDF.
    try:
        import fitz  # type: ignore

        document = fitz.open(str(pdf_path))
        try:
            page = document.load_page(0)
            zoom = float(dpi) / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pixmap.save(str(png_path))
        finally:
            document.close()
        return
    except Exception as fitz_error:
        last_error: Exception = fitz_error

    # Fallback 1: Poppler's pdftoppm.
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                stem = Path(tmpdir) / "page"
                subprocess.run(
                    [pdftoppm, "-f", "1", "-l", "1", "-singlefile", "-png", "-r", str(dpi), str(pdf_path), str(stem)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                generated = stem.with_suffix(".png")
                if generated.exists():
                    png_path.write_bytes(generated.read_bytes())
                    return
        except Exception as poppler_error:
            last_error = poppler_error

    # Fallback 2: ImageMagick.
    magick = shutil.which("magick")
    if not magick:
        convert_candidate = shutil.which("convert")
        if convert_candidate:
            convert_path = Path(convert_candidate)
            if convert_path.name.lower() == "convert.exe" and "system32" in str(convert_path.parent).lower():
                convert_candidate = None
        magick = convert_candidate
    if magick:
        try:
            subprocess.run(
                [magick, "-density", str(dpi), f"{pdf_path}[0]", "-background", "white", "-alpha", "remove", "-alpha", "off", str(png_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if png_path.exists():
                return
        except Exception as magick_error:
            last_error = magick_error

    raise RuntimeError(f"Could not rasterize PDF chart to PNG: {pdf_path}") from last_error


def _finalize_plot(fig: plt.Figure, output_path: Path) -> None:
    """Finalizes a chart by saving both PDF and PNG.

    The PDF remains the vector/canonical artifact.  PNG generation first tries
    to rasterize the PDF, so PNG/PDF pairs stay visually consistent when a
    real PDF rasterizer is available.  On Windows, however, ``convert.exe`` can
    resolve to the system disk-conversion utility instead of ImageMagick.  If
    rasterization fails for any reason, the PNG is saved directly from the
    still-open Matplotlib figure and report generation continues.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_output_path = _pdf_sibling_path(output_path)
    pdf_output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(pdf_output_path, format="pdf", dpi=PDF_PLOT_DPI, facecolor="white", edgecolor="white")
        try:
            _render_pdf_to_png(pdf_output_path, output_path, dpi=PNG_PLOT_DPI)
        except Exception as raster_error:
            print(
                f"WARNING: PDF rasterization failed for {pdf_output_path}; "
                f"saving PNG directly from Matplotlib. Reason: {raster_error}"
            )
            fig.savefig(output_path, format="png", dpi=PNG_PLOT_DPI, facecolor="white", edgecolor="white")
    finally:
        plt.close(fig)

def _png_pdf_sibling_path(png_path: Path) -> Path:
    """Returns the expected PDF sibling for a PNG chart."""

    png_path = Path(png_path)
    parts = list(png_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "plots":
            parts[index] = "plots_pdf"
            return Path(*parts).with_suffix(".pdf")
    return png_path.with_suffix(".pdf")


def _pdf_png_sibling_path(pdf_path: Path) -> Path:
    """Returns the expected PNG sibling for a PDF chart."""

    pdf_path = Path(pdf_path)
    parts = list(pdf_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "plots_pdf":
            parts[index] = "plots"
            return Path(*parts).with_suffix(".png")
    return pdf_path.with_suffix(".png")


def _is_plot_pdf(path: Path) -> bool:
    """Internal helper for is plot pdf."""
    parts = set(Path(path).parts)
    return "plots_pdf" in parts or "plots" in parts


def _raster_png_to_pdf(png_path: Path, pdf_path: Path) -> None:
    """Last-resort compatibility fallback for orphan PNGs.

    New charts should never need this path: ``_finalize_plot`` saves PDF first and
    renders PNG from that PDF.  This fallback only handles older/non-matplotlib
    PNG files so the audit can report complete pairs instead of silently leaving
    them unmatched.
    """

    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    image = plt.imread(str(png_path))
    height_px, width_px = image.shape[:2]
    dpi = 300.0
    fig_width = max(1.0, width_px / dpi)
    fig_height = max(1.0, height_px / dpi)
    fig = plt.figure(figsize=(fig_width, fig_height), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis("off")
    try:
        fig.savefig(pdf_path, format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=0)
    finally:
        plt.close(fig)


def ensure_pdf_for_png_outputs(output_root: str | Path, overwrite: bool = False) -> Dict[str, int]:
    """Ensures plot PDF/PNG pairs using PDF as the canonical source.

    For new outputs, PDF charts under ``plots_pdf`` are converted to PNG siblings
    under ``plots``.  Orphan PNGs are treated as a compatibility case and get a
    raster PDF copy, but the audit exposes their count so they can be noticed.
    """

    output_root = Path(output_root)
    pdf_paths = sorted(path for path in output_root.rglob("*.pdf") if _is_plot_pdf(path))
    png_created_from_pdf = 0
    png_existing_from_pdf = 0
    png_failed_from_pdf = 0
    for pdf_path in pdf_paths:
        png_path = _pdf_png_sibling_path(pdf_path)
        if png_path.exists() and not overwrite:
            png_existing_from_pdf += 1
            continue
        try:
            _render_pdf_to_png(pdf_path, png_path, dpi=PNG_PLOT_DPI)
            png_created_from_pdf += 1
        except Exception:
            png_failed_from_pdf += 1

    png_paths = sorted(output_root.rglob("*.png"))
    orphan_png_files = 0
    orphan_pdf_created = 0
    orphan_pdf_existing = 0
    orphan_pdf_failed = 0
    for png_path in png_paths:
        pdf_path = _png_pdf_sibling_path(png_path)
        if pdf_path.exists():
            orphan_pdf_existing += 1
            continue
        orphan_png_files += 1
        try:
            _raster_png_to_pdf(png_path, pdf_path)
            orphan_pdf_created += 1
        except Exception:
            orphan_pdf_failed += 1

    return {
        "pdf_files": len(pdf_paths),
        "png_files": len(sorted(output_root.rglob("*.png"))),
        "png_existing_from_pdf": png_existing_from_pdf,
        "png_created_from_pdf": png_created_from_pdf,
        "png_failed_from_pdf": png_failed_from_pdf,
        "orphan_png_files": orphan_png_files,
        "orphan_pdf_existing": orphan_pdf_existing,
        "orphan_pdf_created_from_png": orphan_pdf_created,
        "orphan_pdf_failed_from_png": orphan_pdf_failed,
        # Backward-compatible fields used by earlier sanity scripts.
        "pdf_existing": orphan_pdf_existing,
        "pdf_created": orphan_pdf_created,
        "pdf_failed": png_failed_from_pdf + orphan_pdf_failed,
    }


def _annotate_bars(ax: plt.Axes, bars: Iterable, values: Sequence[float], font_size: float = ANNOTATION_FONT_SIZE) -> None:
    """Writes full-precision values on bars while minimizing visual overlap.

    The labels are rotated vertically and rendered with a small font. This keeps
    exact values visible even when grouped bars are dense.
    """

    values_list = [float(v) for v in values if math.isfinite(float(v))]
    if not values_list:
        return

    y_min, y_max = ax.get_ylim()
    y_span = max(1e-12, y_max - y_min)
    y_offset = 0.012 * y_span

    for bar, value in zip(bars, values):
        value = float(value)
        if not math.isfinite(value):
            continue
        height = float(bar.get_height())
        x = bar.get_x() + bar.get_width() / 2.0
        if height >= 0.0:
            y = min(height + y_offset, y_max - 0.015 * y_span)
            va = "bottom"
        else:
            y = max(height - y_offset, y_min + 0.015 * y_span)
            va = "top"
        ax.text(
            x,
            y,
            _format_value_label(value),
            ha="center",
            va=va,
            rotation=90,
            fontsize=font_size,
            bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "#6b7280", "lw": 0.25, "alpha": 0.72},
            zorder=6,
            clip_on=True,
        )


def _bar_chart(rows: Sequence[Dict[str, float | str]], label_key: str, value_key: str, output_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    """Creates a patterned pastel bar chart with exact value annotations."""

    if not rows:
        return

    labels = [str(row.get(label_key, "")) for row in rows]
    values = [_to_float(row.get(value_key)) for row in rows]
    fig_width = min(48.0, max(12.0, 2.35 * len(labels)))
    fig, ax = _prepare_plot((fig_width, 8.8))

    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        bar = ax.bar(
            index,
            value,
            width=0.68,
            color=PASTEL_COLORS.get(label, "#D9D9D9"),
            edgecolor="#2f2f2f",
            linewidth=0.95,
            hatch=ALGORITHM_HATCHES.get(label, "//"),
            zorder=3,
        )
        bars.extend(bar)

    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels([_display_name(label) for label in labels], rotation=12, ha="right")
    used_symlog = False
    if not _apply_zoomed_near_equal_axis(ax, values):
        used_symlog = _maybe_apply_symlog_y(ax, values)
        _set_post_scale_limits(ax, values, used_symlog)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(f"{ylabel} (symlog)" if used_symlog else ylabel, fontsize=12)
    _annotate_bars(ax, bars, values)
    _save_chart_data([dict(row) for row in rows], output_path)
    _finalize_plot(fig, output_path)


def _grouped_bar_chart(
    rows: Sequence[Dict[str, float | str]],
    category_key: str,
    series_key: str,
    value_key: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    category_label_formatter=None,
    annotate: bool = True,
    category_order: Sequence[str] | None = None,
    series_order: Sequence[str] | None = None,
    annotation_key: str | None = None,
) -> None:
    """Creates an academic grouped bar chart with hatches and exact labels."""

    if not rows:
        return

    if category_order is None:
        categories = sorted({str(row[category_key]) for row in rows}, key=_sort_key)
    else:
        present_categories = {str(row[category_key]) for row in rows}
        categories = [str(category) for category in category_order if str(category) in present_categories]
        categories.extend(sorted(present_categories - set(categories), key=_sort_key))

    if series_order is None:
        series_names = _algorithm_order(rows, key=series_key)
    else:
        present_series = {str(row[series_key]) for row in rows}
        series_names = [str(series_name) for series_name in series_order if str(series_name) in present_series]
        series_names.extend([name for name in _algorithm_order(rows, key=series_key) if name not in series_names])

    row_map: Dict[tuple[str, str], float] = {
        (str(row[category_key]), str(row[series_key])): _to_float(row[value_key])
        for row in rows
    }
    annotation_map: Dict[tuple[str, str], float] = {
        (str(row[category_key]), str(row[series_key])): _to_float(row[annotation_key])
        for row in rows
        if annotation_key is not None and annotation_key in row
    }

    n_categories = len(categories)
    n_series = max(1, len(series_names))
    fig_width = min(72.0, max(14.0, n_categories * max(1.75, n_series * 0.72)))
    fig, ax = _prepare_plot((fig_width, 9.4))
    total_width = 0.82
    bar_width = total_width / n_series
    x_centers = [float(index) for index in range(n_categories)]

    all_values = [row_map.get((category, series_name), 0.0) for category in categories for series_name in series_names]
    used_symlog = _maybe_apply_symlog_y(ax, all_values)

    containers_and_values = []
    for series_index, series_name in enumerate(series_names):
        offset = (series_index - (n_series - 1) / 2.0) * bar_width
        x_values = [center + offset for center in x_centers]
        y_values = [row_map.get((category, series_name), 0.0) for category in categories]
        container = ax.bar(
            x_values,
            y_values,
            width=bar_width * 0.92,
            label=_display_name(series_name),
            color=PASTEL_COLORS.get(series_name, "#D9D9D9"),
            edgecolor="#2f2f2f",
            linewidth=0.92,
            hatch=ALGORITHM_HATCHES.get(series_name, "//"),
            zorder=3,
        )
        shown_values = [annotation_map.get((category, series_name), value) for category, value in zip(categories, y_values)]
        containers_and_values.append((container, shown_values))

    tick_labels = [category_label_formatter(category) if category_label_formatter else category for category in categories]
    ax.set_xticks(x_centers)
    ax.set_xticklabels(tick_labels, rotation=10 if len(tick_labels) > 4 else 0, ha="right" if len(tick_labels) > 4 else "center")
    if _should_zoom_near_equal_axis(all_values):
        used_symlog = False
        _apply_zoomed_near_equal_axis(ax, all_values)
    else:
        _set_post_scale_limits(ax, all_values, used_symlog)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(f"{ylabel} (symlog)" if used_symlog else ylabel, fontsize=12)
    ax.legend(ncol=min(3, len(series_names)), fontsize=9, frameon=True)
    if annotate:
        for container, shown_values in containers_and_values:
            _annotate_bars(ax, container, shown_values, font_size=GROUPED_ANNOTATION_FONT_SIZE)
    _save_chart_data([dict(row) for row in rows], output_path)
    _finalize_plot(fig, output_path)

def _line_chart(
    rows: Sequence[Dict[str, float | str]],
    x_key: str,
    y_key: str,
    series_key: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    x_tick_label_key: str | None = None,
) -> None:
    """Creates an academic multi-line chart with distinct markers."""

    if not rows:
        return

    grouped: Dict[str, List[Dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[series_key])].append(row)

    fig, ax = _prepare_plot((11.0, 6.0))
    all_y_values = [_to_float(row[y_key]) for row in rows]
    used_symlog = _maybe_apply_symlog_y(ax, all_y_values)

    for series_name in _algorithm_order(rows, key=series_key):
        if series_name not in grouped:
            continue
        series_rows = sorted(grouped[series_name], key=lambda row: _sort_key(str(row[x_key])))
        x_values = [_to_float(row[x_key]) for row in series_rows]
        y_values = [_to_float(row[y_key]) for row in series_rows]
        ax.plot(
            x_values,
            y_values,
            marker=ALGORITHM_MARKERS.get(series_name, "o"),
            markersize=6,
            linewidth=2.0,
            linestyle=ALGORITHM_LINESTYLES.get(series_name, "-"),
            label=_display_name(series_name),
            color=PASTEL_COLORS.get(series_name, "#7f7f7f"),
            markerfacecolor="white",
            markeredgecolor="#1f2937",
            markeredgewidth=1.0,
            zorder=4,
        )
        if len(x_values) <= 6:
            for x_value, y_value in zip(x_values, y_values):
                ax.text(
                    x_value,
                    y_value,
                    _format_value_label(y_value),
                    fontsize=8.2,
                    ha="center",
                    va="bottom",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#9ca3af", "lw": 0.4, "alpha": 0.75},
                    zorder=5,
                )

    if _should_zoom_near_equal_axis(all_y_values):
        used_symlog = False
        _apply_zoomed_near_equal_axis(ax, all_y_values)
    else:
        _set_post_scale_limits(ax, all_y_values, used_symlog)
    if x_tick_label_key is not None:
        tick_labels_by_x: Dict[float, str] = {}
        for row in rows:
            x_value = _to_float(row[x_key])
            if math.isfinite(x_value) and x_tick_label_key in row:
                tick_labels_by_x.setdefault(x_value, str(row[x_tick_label_key]))
        if tick_labels_by_x:
            ordered_x_values = sorted(tick_labels_by_x)
            ax.set_xticks(ordered_x_values)
            ax.set_xticklabels(
                [tick_labels_by_x[x_value].replace(" + ", "\n") for x_value in ordered_x_values],
                rotation=0,
                ha="center",
            )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(f"{ylabel} (symlog)" if used_symlog else ylabel, fontsize=12)
    ax.legend(fontsize=9, frameon=True)
    _save_chart_data([dict(row) for row in rows], output_path)
    _finalize_plot(fig, output_path)




def _scatter_actual_vs_predicted(rows: Sequence[Dict[str, float | str]], output_path: Path) -> None:
    """Creates actual-vs-predicted execution-time scatter chart from real output rows."""

    if not rows:
        return
    sampled_rows = list(rows[:5000])
    fig, ax = _prepare_plot((8.0, 7.2))
    algorithms = _algorithm_order(sampled_rows, key="algorithm")
    all_values: List[float] = []
    for algorithm in algorithms:
        algorithm_rows = [row for row in sampled_rows if str(row.get("algorithm")) == algorithm]
        x_values = [_to_float(row.get("actual_execution_time")) for row in algorithm_rows]
        y_values = [_to_float(row.get("predicted_execution_time")) for row in algorithm_rows]
        all_values.extend(x_values)
        all_values.extend(y_values)
        ax.scatter(
            x_values,
            y_values,
            s=14,
            alpha=0.58,
            label=_display_name(algorithm),
            color=PASTEL_COLORS.get(algorithm, "#7f7f7f"),
            edgecolor="#1f2937",
            linewidth=0.25,
        )
    if all_values:
        upper = max(all_values) * 1.05 if max(all_values) > 0 else 1.0
        ax.plot([0, upper], [0, upper], color="#111827", linestyle="--", linewidth=1.2, label="Ideal")
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
    ax.set_title("Actual Execution Time vs Predicted Execution Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Actual Execution Time [sim time unit]", fontsize=12)
    ax.set_ylabel("Predicted / Estimated Execution Time [sim time unit]", fontsize=12)
    ax.legend(fontsize=8, frameon=True)
    _save_chart_data([dict(row) for row in sampled_rows], output_path)
    _finalize_plot(fig, output_path)


def _heatmap_per_core_utilization(rows: Sequence[Dict[str, float | str]], output_path: Path) -> None:
    """Creates a per-core utilization heatmap averaged by algorithm and core id."""

    if not rows:
        return
    algorithms = _algorithm_order(rows, key="algorithm")
    core_ids = sorted({int(_to_float(row.get("core_id"))) for row in rows})
    if not algorithms or not core_ids:
        return
    grouped: Dict[tuple[str, int], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("algorithm")), int(_to_float(row.get("core_id"))))].append(_to_float(row.get("utilization_per_core")))
    matrix: List[List[float]] = []
    for algorithm in algorithms:
        matrix.append([
            0.0 if not grouped[(algorithm, core_id)] else mean(grouped[(algorithm, core_id)])
            for core_id in core_ids
        ])
    fig, ax = plt.subplots(figsize=(max(9.0, len(core_ids) * 0.42), max(4.8, len(algorithms) * 0.55)), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_title("Per-Core Utilization Heatmap", fontsize=16, fontweight="bold")
    ax.set_xlabel("Core ID", fontsize=12)
    ax.set_ylabel("Algorithm", fontsize=12)
    ax.set_xticks(range(len(core_ids)))
    ax.set_xticklabels([str(core_id) for core_id in core_ids], rotation=45 if len(core_ids) > 16 else 0)
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels([_display_name(algorithm) for algorithm in algorithms])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Utilization [ratio]")
    chart_rows = []
    for algorithm, row_values in zip(algorithms, matrix):
        for core_id, value in zip(core_ids, row_values):
            chart_rows.append({"algorithm": algorithm, "core_id": float(core_id), "utilization_per_core": value})
    _save_chart_data(chart_rows, output_path)
    _finalize_plot(fig, output_path)


def _prediction_metric_rows(rows: Sequence[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    """Aggregates MAE/MSE/RMSE/MAPE by algorithm for report charts."""

    grouped: Dict[str, List[Dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("algorithm"))].append(row)
    output: List[Dict[str, float | str]] = []
    for algorithm, algorithm_rows in grouped.items():
        for metric in ["MAE", "MSE", "RMSE", "MAPE_epsilon", "sMAPE"]:
            output.append({
                "algorithm": algorithm,
                "metric": metric,
                "metric_display_name": _metric_label(metric),
                "value": _decimal_to_output(_decimal_mean(row.get(metric) for row in algorithm_rows)),
            })
    return output


def _before_after_metric_rows(rows: Sequence[Dict[str, float | str]], before_key: str, after_key: str, metric_name: str) -> List[Dict[str, float | str]]:
    """Aggregates before/after DVFS metrics by algorithm."""

    grouped: Dict[str, List[Dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("algorithm"))].append(row)
    output: List[Dict[str, float | str]] = []
    for algorithm, algorithm_rows in grouped.items():
        output.append({"algorithm": algorithm, "stage": f"Before DVFS - {metric_name}", "value": _decimal_to_output(_decimal_mean(row.get(before_key) for row in algorithm_rows))})
        output.append({"algorithm": algorithm, "stage": f"After DVFS - {metric_name}", "value": _decimal_to_output(_decimal_mean(row.get(after_key) for row in algorithm_rows))})
    return output


def _training_curve_chart(rows: Sequence[Dict[str, float | str]], output_path: Path) -> None:
    """Plots comparable non-negative supervised prediction-loss curves only."""

    if not rows:
        return
    # Do not mix reinforcement-learning rewards with supervised losses.  The GAT
    # optimized objective contains Gaussian NLL/CVaR terms and can be negative,
    # so the public training-loss plot uses prediction MSE for both GNN and GAT.
    chart_rows = [row for row in rows if str(row.get("curve")) == "training_loss"]
    if not chart_rows:
        return
    grouped: Dict[str, List[Dict[str, float | str]]] = defaultdict(list)
    for row in chart_rows:
        grouped[str(row.get("model"))].append(row)
    fig, ax = _prepare_plot((11.0, 6.0))
    for series_name, series_rows in sorted(grouped.items()):
        ordered = sorted(series_rows, key=lambda row: _to_float(row.get("step")))
        ax.plot(
            [_to_float(row.get("step")) for row in ordered],
            [_to_float(row.get("value")) for row in ordered],
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=f"{series_name} / prediction MSE",
        )
    ax.set_title("Training Loss Curves", fontsize=16, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Prediction MSE [sim time unit²]", fontsize=12)
    ax.legend(fontsize=8, frameon=True)
    _save_chart_data([dict(row) for row in chart_rows], output_path)
    _finalize_plot(fig, output_path)


def _policy_reward_chart(rows: Sequence[Dict[str, float | str]], output_path: Path) -> None:
    """Plots policy reward separately from supervised training loss."""

    chart_rows = [row for row in rows if str(row.get("curve")) in {"policy_reward", "policy_moving_baseline"}]
    if not chart_rows:
        return
    grouped: Dict[str, List[Dict[str, float | str]]] = defaultdict(list)
    for row in chart_rows:
        grouped[str(row.get("curve"))].append(row)
    fig, ax = _prepare_plot((11.0, 6.0))
    for series_name, series_rows in sorted(grouped.items()):
        ordered = sorted(series_rows, key=lambda row: _to_float(row.get("step")))
        ax.plot(
            [_to_float(row.get("step")) for row in ordered],
            [_to_float(row.get("value")) for row in ordered],
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=series_name,
        )
    ax.set_title("Policy Reward Curve", fontsize=16, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward [native unit]", fontsize=12)
    ax.legend(fontsize=8, frameon=True)
    _save_chart_data([dict(row) for row in chart_rows], output_path)
    _finalize_plot(fig, output_path)

def _box_plot(rows: Sequence[Dict[str, float | str]], value_key: str, series_key: str, output_path: Path, title: str, ylabel: str) -> None:
    """Creates a box plot per algorithm with patterned fills and mean markers."""

    if not rows:
        return

    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[series_key])].append(_to_float(row[value_key]))

    order = _algorithm_order(rows, key=series_key)
    data = [grouped[name] for name in order if name in grouped]
    labels = [_display_name(name) for name in order if name in grouped]

    fig, ax = _prepare_plot((12.0, 6.8))
    # Matplotlib renamed the boxplot keyword from ``labels`` to
    # ``tick_labels`` in newer releases.  Use the new spelling first and
    # fall back to the old one so the reporting code works on both the
    # project environment and newer local Python installs.
    try:
        box = ax.boxplot(data, patch_artist=True, tick_labels=labels, showmeans=True)
    except TypeError:
        box = ax.boxplot(data, patch_artist=True, labels=labels, showmeans=True)
    ordered_names = [name for name in order if name in grouped]
    for patch, algorithm_name in zip(box["boxes"], ordered_names):
        patch.set_facecolor(PASTEL_COLORS.get(algorithm_name, "#D9D9D9"))
        patch.set_edgecolor("#2f2f2f")
        patch.set_hatch(ALGORITHM_HATCHES.get(algorithm_name, "//"))
        patch.set_linewidth(0.9)
    for mean_marker, algorithm_name in zip(box["means"], ordered_names):
        mean_marker.set_marker(ALGORITHM_MARKERS.get(algorithm_name, "o"))
        mean_marker.set_markerfacecolor("white")
        mean_marker.set_markeredgecolor("#111827")
        mean_marker.set_markersize(8)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="x", rotation=15)
    _save_chart_data([dict(row) for row in rows], output_path)
    _finalize_plot(fig, output_path)


def _scatter_chart(rows: Sequence[Dict[str, float | str]], x_key: str, y_key: str, series_key: str, output_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    """Creates a scatter chart with distinct markers and explicit per-model mean markers."""

    if not rows:
        return

    grouped: Dict[str, List[Dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[series_key])].append(row)

    fig, ax = _prepare_plot((11.2, 6.6))
    for series_name in _algorithm_order(rows, key=series_key):
        if series_name not in grouped:
            continue
        series_rows = grouped[series_name]
        x_values = [_to_float(row[x_key]) for row in series_rows]
        y_values = [_to_float(row[y_key]) for row in series_rows]
        marker = ALGORITHM_MARKERS.get(series_name, "o")
        color = PASTEL_COLORS.get(series_name, "#7f7f7f")
        ax.scatter(
            x_values,
            y_values,
            alpha=0.32,
            s=42,
            marker=marker,
            label=f"{_display_name(series_name)} samples",
            color=color,
            edgecolors="#4b5563",
            linewidths=0.45,
            zorder=2,
        )
        center_x = mean(x_values)
        center_y = mean(y_values)
        ax.scatter(
            [center_x],
            [center_y],
            s=180,
            marker=marker,
            facecolors="white",
            edgecolors="#111827",
            linewidths=1.6,
            zorder=5,
        )
        ax.annotate(
            f"μ {_display_name(series_name)}",
            (center_x, center_y),
            textcoords="offset points",
            xytext=(7, 7),
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "lw": 0.8, "alpha": 0.92},
            zorder=6,
        )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=8, frameon=True, ncol=2)
    _save_chart_data([dict(row) for row in rows], output_path)
    _finalize_plot(fig, output_path)


def _rank_algorithms(baseline_overall: Sequence[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    """Builds a rank table across multiple metrics."""

    if not baseline_overall:
        return []

    metric_directions = {
        "mean_deadline_miss_ratio": "min",
        "mean_utilization": "max",
        "mean_energy": "min",
        "mean_mode_switch_probability": "min",
        "mean_average_cvar": "min",
        "mean_aging_index": "min",
        "mean_average_reliability": "max",
        "mean_completed_hi_ratio": "max",
    }
    algorithm_rows = {str(row["algorithm"]): dict(row) for row in baseline_overall}
    score_map = {name: 0.0 for name in algorithm_rows}

    for metric_name, direction in metric_directions.items():
        ordered = sorted(
            algorithm_rows.items(),
            key=lambda item: _to_float(item[1][metric_name]),
            reverse=(direction == "max"),
        )
        for rank_index, (algorithm_name, _) in enumerate(ordered, start=1):
            score_map[algorithm_name] += rank_index

    rows: List[Dict[str, float | str]] = []
    for algorithm_name, score in sorted(score_map.items(), key=lambda item: item[1]):
        rows.append({
            "algorithm": algorithm_name,
            "display_name": _display_name(algorithm_name),
            "aggregate_rank_score": score,
        })
    return rows


def _pairwise_delta_rows(
    baseline_overall: Sequence[Dict[str, float | str]],
    reference_algorithm: str,
) -> List[Dict[str, float | str]]:
    """Builds pairwise deltas against one reference algorithm."""

    rows_by_algorithm = {str(row["algorithm"]): dict(row) for row in baseline_overall}
    if reference_algorithm not in rows_by_algorithm:
        return []
    reference_row = rows_by_algorithm[reference_algorithm]
    output: List[Dict[str, float | str]] = []
    for algorithm_name, row in rows_by_algorithm.items():
        if algorithm_name == reference_algorithm:
            continue
        output.append(
            {
                "reference_algorithm": reference_algorithm,
                "algorithm": algorithm_name,
                "delta_dmr": _to_float(row["mean_deadline_miss_ratio"]) - _to_float(reference_row["mean_deadline_miss_ratio"]),
                "delta_utilization": _to_float(row["mean_utilization"]) - _to_float(reference_row["mean_utilization"]),
                "delta_energy": _to_float(row["mean_energy"]) - _to_float(reference_row["mean_energy"]),
                "delta_aging": _to_float(row["mean_aging_index"]) - _to_float(reference_row["mean_aging_index"]),
                "delta_reliability": _to_float(row["mean_average_reliability"]) - _to_float(reference_row["mean_average_reliability"]),
                "delta_cvar": _to_float(row["mean_average_cvar"]) - _to_float(reference_row["mean_average_cvar"]),
            }
        )
    return output


def _values_from_possible_keys(
    rows: Sequence[Dict[str, str | float]],
    primary_key: str,
    aliases: Sequence[str] = (),
    *,
    positive_only: bool = False,
) -> List[float]:
    """Collects numeric values from the first available key among possible aliases."""

    values: List[float] = []
    candidate_keys = (primary_key, *aliases)
    for row in rows:
        selected_value = None
        for key in candidate_keys:
            if key in row and row.get(key) not in (None, ""):
                selected_value = _to_float(row.get(key))
                break
        if selected_value is None:
            continue
        if positive_only and selected_value <= 0.0:
            continue
        values.append(selected_value)
    return values


def _mean_from_possible_keys(
    rows: Sequence[Dict[str, str | float]],
    primary_key: str,
    aliases: Sequence[str] = (),
    *,
    positive_only: bool = False,
    default: float = 0.0,
) -> float:
    """Returns the arithmetic mean over the first available metric key or alias."""

    values = _values_from_possible_keys(rows, primary_key, aliases=aliases, positive_only=positive_only)
    return default if not values else mean(values)


def _metric_available(rows: Sequence[Dict[str, str | float]], *keys: str) -> bool:
    """Returns true if at least one of the provided keys is present in any row."""

    return any(any(key in row and row.get(key) not in (None, "") for key in keys) for row in rows)


def _learned_models_comparison_rows(
    baseline_overall: Sequence[Dict[str, float | str]],
    learned_model_rows: Sequence[Dict[str, str]],
    quantization_overall_row: Dict[str, float | str] | None,
) -> List[Dict[str, float | str]]:
    """Builds a focused comparison table for learned models only."""

    learned_algorithms = {"gnn", "proposed_gat_float32", "proposed_gat_quantization_int8"}
    baseline_by_algorithm = {str(row["algorithm"]): dict(row) for row in baseline_overall if str(row["algorithm"]) in learned_algorithms}

    model_meta_grouped = _mean_by_group(learned_model_rows, group_key="algorithm", metric_keys=[
        "parameter_count",
        "predictor_parameter_count",
        "policy_parameter_count",
        "model_size_bytes",
        "predictor_size_bytes",
        "policy_size_bytes",
        "predictor_latency_ms",
        "policy_latency_ms",
        "scheduler_latency_ms",
    ])
    model_meta_by_algorithm = {str(row["algorithm"]): dict(row) for row in model_meta_grouped}
    float_metadata_reference = model_meta_by_algorithm.get("proposed_gat_float32", {})

    rows: List[Dict[str, float | str]] = []
    for algorithm_name in ["gnn", "proposed_gat_float32", "proposed_gat_quantization_int8"]:
        if algorithm_name not in baseline_by_algorithm:
            continue
        row = {
            "algorithm": algorithm_name,
            "display_name": _display_name(algorithm_name),
            **baseline_by_algorithm[algorithm_name],
        }
        row.update(model_meta_by_algorithm.get(algorithm_name, {}))
        if quantization_overall_row is not None and algorithm_name == "proposed_gat_float32":
            row.setdefault("predictor_latency_ms", quantization_overall_row.get("average_float_predictor_latency_ms", 0.0))
            row.setdefault("policy_latency_ms", quantization_overall_row.get("average_float_policy_latency_ms", 0.0))
        if quantization_overall_row is not None and algorithm_name == "proposed_gat_quantization_int8":
            row["predictor_latency_ms"] = row.get("predictor_latency_ms") or quantization_overall_row.get("average_int8_predictor_latency_ms", 0.0)
            row["policy_latency_ms"] = row.get("policy_latency_ms") or quantization_overall_row.get("average_int8_policy_latency_ms", 0.0)
            row["scheduler_latency_ms"] = row.get("scheduler_latency_ms") or quantization_overall_row.get("average_int8_scheduler_latency_ms", 0.0)
            row["predictor_size_bytes"] = row.get("predictor_size_bytes") or quantization_overall_row.get("average_int8_predictor_size_bytes", 0.0)
            row["policy_size_bytes"] = row.get("policy_size_bytes") or quantization_overall_row.get("average_int8_policy_size_bytes", 0.0)
            row["model_size_bytes"] = row.get("model_size_bytes") or quantization_overall_row.get("average_int8_scheduler_size_bytes", 0.0)
            for count_key in ("parameter_count", "predictor_parameter_count", "policy_parameter_count"):
                if _to_float(row.get(count_key)) <= 0.0 and count_key in float_metadata_reference:
                    row[count_key] = float_metadata_reference[count_key]

        if quantization_overall_row is not None and algorithm_name == "proposed_gat_float32":
            row.setdefault("scheduler_latency_ms", quantization_overall_row.get("average_float_scheduler_latency_ms", 0.0))
            row.setdefault("predictor_size_bytes", quantization_overall_row.get("average_float_predictor_size_bytes", 0.0))
            row.setdefault("policy_size_bytes", quantization_overall_row.get("average_float_policy_size_bytes", 0.0))
            row.setdefault("model_size_bytes", quantization_overall_row.get("average_float_scheduler_size_bytes", 0.0))

        predictor_parameters = _to_float(row.get("predictor_parameter_count"))
        policy_parameters = _to_float(row.get("policy_parameter_count"))
        if _to_float(row.get("parameter_count")) <= 0.0 and (predictor_parameters > 0.0 or policy_parameters > 0.0):
            row["parameter_count"] = predictor_parameters + policy_parameters

        predictor_size = _to_float(row.get("predictor_size_bytes"))
        policy_size = _to_float(row.get("policy_size_bytes"))
        if _to_float(row.get("model_size_bytes")) <= 0.0 and (predictor_size > 0.0 or policy_size > 0.0):
            row["model_size_bytes"] = predictor_size + policy_size

        for required_key in [
            "parameter_count",
            "predictor_parameter_count",
            "policy_parameter_count",
            "model_size_bytes",
            "predictor_size_bytes",
            "policy_size_bytes",
            "predictor_latency_ms",
            "policy_latency_ms",
            "scheduler_latency_ms",
        ]:
            row.setdefault(required_key, 0.0)

        rows.append(row)
    return rows


def _rows_to_metric_matrix(
    rows: Sequence[Dict[str, float | str]],
    metric_keys: Sequence[str],
    key_name: str = "algorithm",
    normalize: bool = False,
) -> List[Dict[str, float | str]]:
    """Converts wide rows to a metric matrix for grouped charting and Excel export."""

    matrix: List[Dict[str, float | str]] = []
    metric_maxima: Dict[str, float] = {}
    if normalize:
        for metric_key in metric_keys:
            metric_maxima[metric_key] = max(1e-12, max(abs(_to_float(row.get(metric_key))) for row in rows))
    for row in rows:
        row_name = str(row[key_name])
        for metric_key in metric_keys:
            value = _to_float(row.get(metric_key))
            matrix_row: Dict[str, float | str] = {
                key_name: row_name,
                "metric": metric_key,
                "metric_display_name": _metric_label(metric_key),
                "unit": _metric_unit(metric_key),
                "raw_value": _format_value_label(value),
                "value": value / metric_maxima[metric_key] if normalize else value,
            }
            matrix.append(matrix_row)
    return matrix


def _quantization_method_overall_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, float | str]]:
    """Aggregates quantization-method rows across configurations."""

    if not rows or not any("method_name" in row for row in rows):
        return []
    metric_keys = [
        "average_predictor_size_bytes",
        "average_policy_size_bytes",
        "average_scheduler_size_bytes",
        "average_relative_size_reduction_vs_fp32",
        "average_mean_absolute_prediction_error_vs_fp32",
        "average_predictor_latency_ms",
        "average_policy_latency_ms",
        "average_scheduler_latency_ms",
        "num_successful_graphs",
        "num_failed_graphs",
        "success_rate",
        "average_makespan",
        "average_energy",
        "average_utilization",
        "average_deadline_miss_ratio",
        "average_task_deadline_miss_ratio",
        "average_mode_switch_probability",
        "average_average_cvar",
        "average_aging_index",
        "average_average_reliability",
        "average_completed_hi_ratio",
        "average_completed_lo_ratio",
        "average_dropped_lo_ratio",
        "average_service_loss_ratio",
    ]
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        method_name = str(row.get("method_name", "unknown"))
        status = str(row.get("status", "success")).strip().lower()
        if method_name == "dynamic_int8_linear" or status == "skipped":
            continue
        grouped[method_name].append(row)

    output: List[Dict[str, float | str]] = []
    for method_name in [name for name in _algorithm_order([{"method_name": name} for name in grouped], key="method_name") if name in grouped]:
        method_rows = grouped[method_name]
        first = method_rows[0]
        summary: Dict[str, float | str] = {
            "method_name": method_name,
            "algorithm": method_name,
            "display_name": first.get("display_name", _display_name(method_name)),
            "reason_for_inclusion": first.get("reason_for_inclusion", ""),
            "is_pdf_method": first.get("is_pdf_method", "0"),
        }
        for metric_key in metric_keys:
            summary[metric_key] = _decimal_to_output(_decimal_mean(row.get(metric_key) for row in method_rows))
        output.append(summary)
    return output


def _quantization_normalized_rows(rows: Sequence[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    """Builds normalized rows for multi-method quantization charts."""

    metric_specs = [
        ("average_predictor_size_bytes", _metric_label("predictor_size_bytes")),
        ("average_policy_size_bytes", _metric_label("policy_size_bytes")),
        ("average_scheduler_size_bytes", _metric_label("scheduler_size_bytes")),
        ("average_relative_size_reduction_vs_fp32", "Size Reduction vs FP32 [ratio]"),
        ("average_mean_absolute_prediction_error_vs_fp32", _metric_label("prediction_mae")),
        ("average_predictor_latency_ms", _metric_label("predictor_latency_ms")),
        ("average_policy_latency_ms", _metric_label("policy_latency_ms")),
        ("average_scheduler_latency_ms", _metric_label("scheduler_latency_ms")),
        ("success_rate", "Success Rate [ratio]"),
        ("average_utilization", _metric_label("utilization")),
        ("average_deadline_miss_ratio", _metric_label("deadline_miss_ratio")),
        ("average_task_deadline_miss_ratio", _metric_label("task_deadline_miss_ratio")),
        ("average_energy", _metric_label("energy")),
        ("average_mode_switch_probability", _metric_label("mode_switch_probability")),
        ("average_completed_hi_ratio", _metric_label("completed_hi_ratio")),
        ("average_completed_lo_ratio", _metric_label("completed_lo_ratio")),
        ("average_dropped_lo_ratio", _metric_label("dropped_lo_ratio")),
        ("average_service_loss_ratio", _metric_label("service_loss_ratio")),
    ]
    matrix: List[Dict[str, float | str]] = []
    maxima: Dict[str, float] = {}
    for metric_key, _ in metric_specs:
        maxima[metric_key] = max(1e-12, max(abs(_to_float(row.get(metric_key))) for row in rows)) if rows else 1e-12
    for row in rows:
        method_name = str(row["method_name"])
        for metric_key, metric_display_name in metric_specs:
            raw_value = _to_float(row.get(metric_key))
            matrix.append({
                "method_name": method_name,
                "algorithm": method_name,
                "display_name": row.get("display_name", _display_name(method_name)),
                "metric": metric_key,
                "metric_display_name": metric_display_name,
                "raw_value": _format_value_label(raw_value),
                "value": raw_value / maxima[metric_key],
            })
    return matrix




def _filter_defense_report_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Removes invalid/skipped Dynamic INT8 rows from defense-report tables/charts."""

    filtered: List[Dict[str, str]] = []
    dynamic_tokens = (
        "dynamic_int8_linear",
        "dynamic int8 linear",
        "proposed_gat_dynamic_int8_linear",
    )
    for row in rows:
        token_text = " ".join(
            str(row.get(key, ""))
            for key in (
                "method_name",
                "algorithm",
                "display_name",
                "precision_variant",
                "legacy_algorithm_name",
                "summary_kind",
            )
        ).lower()
        if any(token in token_text for token in dynamic_tokens):
            continue
        if str(row.get("status", "")).strip().lower() == "skipped":
            continue
        filtered.append(row)
    return filtered

def generate_defense_outputs(
    all_config_output_root: str | Path,
    structural_output_root: str | Path | None = None,
    output_root: str | Path = Path("outputs") / "defense_report",
) -> Dict[str, str]:
    """Builds clean tables and charts for defense/demo use."""

    all_config_output_root = Path(all_config_output_root)
    structural_output_root = Path(structural_output_root) if structural_output_root is not None else None
    output_root = Path(output_root)
    tables_dir = output_root / "tables"
    plots_dir = output_root / "plots"
    plots_pdf_dir = output_root / "plots_pdf"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_pdf_dir.mkdir(parents=True, exist_ok=True)

    aggregate_root = all_config_output_root / "_aggregate"
    baseline_rows = _decorate_rows_with_configuration_axes(load_csv_rows(aggregate_root / "baseline_summary_aggregate.csv"))
    uncertainty_rows = load_csv_rows(aggregate_root / "uncertainty_summary_aggregate.csv")
    quantization_rows = _filter_defense_report_rows(load_csv_rows(aggregate_root / "quantization_summary_aggregate.csv"))
    learned_model_rows = _filter_defense_report_rows(load_csv_rows(aggregate_root / "learned_models_summary_aggregate.csv"))

    # Detailed artifacts can be very large.  Keep their original aggregate CSVs as
    # the canonical full-detail outputs and load only the files needed for charts.
    task_level_rows_path = aggregate_root / "task_level_results_aggregate.csv"
    core_level_rows_path = aggregate_root / "core_level_results_aggregate.csv"
    schedule_extended_rows_path = aggregate_root / "schedule_summary_extended_aggregate.csv"
    energy_power_rows_path = aggregate_root / "energy_power_summary_aggregate.csv"
    prediction_metrics_rows_path = aggregate_root / "prediction_metrics_aggregate.csv"
    prediction_detail_rows_path = aggregate_root / "prediction_detail_results_aggregate.csv"
    power_trace_rows_path = aggregate_root / "power_trace_aggregate.csv"
    training_loss_rows_path = aggregate_root / "training_loss_curves_aggregate.csv"

    task_level_sample_rows = sample_csv_rows(task_level_rows_path, limit=5000)
    core_level_rows = load_csv_rows(core_level_rows_path)
    schedule_extended_rows = load_csv_rows(schedule_extended_rows_path)
    energy_power_rows = load_csv_rows(energy_power_rows_path)
    prediction_metrics_rows = load_csv_rows(prediction_metrics_rows_path)
    prediction_detail_rows = load_csv_rows(prediction_detail_rows_path)
    power_trace_rows = load_csv_rows(power_trace_rows_path)
    training_loss_rows = load_csv_rows(training_loss_rows_path)

    baseline_overall = _mean_by_group(
        baseline_rows,
        group_key="algorithm",
        metric_keys=BASELINE_METRICS,
    )
    save_rows_as_csv(baseline_overall, tables_dir / "baseline_overall_table.csv")
    save_rows_as_csv(
        [
            {"metric": metric_name, "display_name": METRIC_DISPLAY_NAMES.get(metric_name, metric_name), "unit": _metric_unit(metric_name)}
            for metric_name in sorted(METRIC_UNITS)
        ],
        tables_dir / "metric_units_table.csv",
    )

    baseline_rank_rows = _rank_algorithms(baseline_overall)
    save_rows_as_csv(baseline_rank_rows, tables_dir / "baseline_rank_table.csv")

    pairwise_vs_float = _pairwise_delta_rows(baseline_overall, reference_algorithm="proposed_gat_float32")
    pairwise_vs_int8 = _pairwise_delta_rows(baseline_overall, reference_algorithm="proposed_gat_quantization_int8")
    save_rows_as_csv(pairwise_vs_float, tables_dir / "pairwise_vs_proposed_float32.csv")
    save_rows_as_csv(pairwise_vs_int8, tables_dir / "pairwise_vs_proposed_int8.csv")

    baseline_metric_matrix = _build_scaled_metric_matrix(
        baseline_overall,
        metric_keys=[
            "mean_deadline_miss_ratio",
            "mean_task_deadline_miss_ratio",
            "mean_utilization",
            "mean_energy",
            "mean_average_cvar",
            "mean_average_reliability",
            "mean_aging_index",
            "mean_mode_switch_probability",
            "mean_completed_hi_ratio",
        ],
        key_name="algorithm",
    )
    save_rows_as_csv(baseline_metric_matrix, tables_dir / "baseline_metric_matrix_scaled_table.csv")

    # Detailed output tables and charts requested for final defense validation.
    # Huge detailed CSVs stay in ``outputs/all_config_runs/_aggregate``.  The
    # defense-report folder stores a manifest and samples to avoid duplicating
    # millions of rows or attempting memory-heavy XLSX exports.
    detailed_manifest = detailed_output_manifest_rows(
        {
            "task_level_results_aggregate": task_level_rows_path,
            "core_level_results_aggregate": core_level_rows_path,
            "schedule_summary_extended_aggregate": schedule_extended_rows_path,
            "energy_power_summary_aggregate": energy_power_rows_path,
            "prediction_metrics_aggregate": prediction_metrics_rows_path,
            "prediction_detail_results_aggregate": prediction_detail_rows_path,
            "power_trace_aggregate": power_trace_rows_path,
            "training_loss_curves_aggregate": training_loss_rows_path,
        }
    )
    save_rows_as_csv(detailed_manifest, tables_dir / "detailed_output_manifest_table.csv")

    if task_level_sample_rows:
        save_rows_as_csv(task_level_sample_rows, tables_dir / "task_level_results_sample_table.csv", xlsx_row_limit=5000)
    if core_level_rows:
        save_rows_as_csv(core_level_rows, tables_dir / "core_level_results_table.csv")
        _heatmap_per_core_utilization(core_level_rows, plots_dir / "per_core_utilization_heatmap.png")
    if schedule_extended_rows:
        save_rows_as_csv(schedule_extended_rows, tables_dir / "schedule_summary_extended_table.csv")
    if energy_power_rows:
        save_rows_as_csv(energy_power_rows, tables_dir / "energy_power_summary_table.csv")
        energy_before_after = _before_after_metric_rows(
            energy_power_rows,
            before_key="energy_before_dvfs",
            after_key="energy_after_dvfs",
            metric_name="Energy",
        )
        power_before_after = _before_after_metric_rows(
            energy_power_rows,
            before_key="power_before_dvfs",
            after_key="power_after_dvfs",
            metric_name="Average Power",
        )
        save_rows_as_csv(energy_before_after, tables_dir / "energy_before_after_dvfs_table.csv")
        save_rows_as_csv(power_before_after, tables_dir / "power_before_after_dvfs_table.csv")
        _grouped_bar_chart(energy_before_after, "stage", "algorithm", "value", plots_dir / "energy_before_after_dvfs.png", "Energy Before vs After DVFS", "Stage", "Energy [relative energy unit]")
        _grouped_bar_chart(power_before_after, "stage", "algorithm", "value", plots_dir / "power_before_after_dvfs.png", "Average Power Before vs After DVFS", "Stage", "Power [relative power unit]")
    if prediction_metrics_rows:
        save_rows_as_csv(prediction_metrics_rows, tables_dir / "prediction_metrics_table.csv")
        prediction_metric_chart_rows = _prediction_metric_rows(prediction_metrics_rows)
        save_rows_as_csv(prediction_metric_chart_rows, tables_dir / "prediction_metrics_overall_table.csv")
        _grouped_bar_chart(prediction_metric_chart_rows, "metric_display_name", "algorithm", "value", plots_dir / "prediction_error_metrics.png", "MAE / MSE / RMSE / MAPE", "Metric", "Value")
    if prediction_detail_rows:
        save_rows_as_csv(prediction_detail_rows[:5000], tables_dir / "prediction_detail_sample_table.csv")
        _scatter_actual_vs_predicted(prediction_detail_rows, plots_dir / "actual_vs_predicted_execution_time.png")
    if power_trace_rows:
        # Plot one representative configuration/application to keep the trace readable.
        first_key = (power_trace_rows[0].get("configuration", ""), power_trace_rows[0].get("application_id", ""), power_trace_rows[0].get("graph_id", ""))
        trace_sample = [row for row in power_trace_rows if (row.get("configuration", ""), row.get("application_id", ""), row.get("graph_id", "")) == first_key]
        save_rows_as_csv(trace_sample, tables_dir / "power_trace_sample_table.csv")
        if trace_sample:
            _line_chart(trace_sample, "time", "power_after_dvfs", "algorithm", plots_dir / "power_trace_after_dvfs_sample.png", "Power Trace After DVFS", "Time [sim time unit]", "Power [relative power unit]")
            _line_chart(trace_sample, "time", "power_before_dvfs", "algorithm", plots_dir / "power_trace_before_dvfs_sample.png", "Power Trace Before DVFS", "Time [sim time unit]", "Power [relative power unit]")
    if training_loss_rows:
        save_rows_as_csv(training_loss_rows, tables_dir / "training_loss_curves_table.csv")
        _training_curve_chart(training_loss_rows, plots_dir / "training_loss_validation_loss_curves.png")
        _policy_reward_chart(training_loss_rows, plots_dir / "policy_reward_curves.png")

    quantization_overall_row = None
    quantization_methods_overall = _quantization_method_overall_rows(quantization_rows)
    quantization_methods_scaled: List[Dict[str, float | str]] = []
    if quantization_methods_overall:
        save_rows_as_csv(quantization_methods_overall, tables_dir / "quantization_methods_overall_table.csv")
        quantization_methods_scaled = _build_scaled_metric_matrix(
            quantization_methods_overall,
            metric_keys=[
                "average_predictor_size_bytes",
                "average_policy_size_bytes",
                "average_scheduler_size_bytes",
                "average_relative_size_reduction_vs_fp32",
                "average_mean_absolute_prediction_error_vs_fp32",
                "average_predictor_latency_ms",
                "average_policy_latency_ms",
                "average_scheduler_latency_ms",
                "success_rate",
                "average_utilization",
                "average_deadline_miss_ratio",
                "average_task_deadline_miss_ratio",
                "average_energy",
                "average_mode_switch_probability",
                "average_completed_hi_ratio",
                "average_completed_lo_ratio",
                "average_cvar",
                "average_reliability",
                "average_aging_index",
            ],
            key_name="method_name",
        )
        save_rows_as_csv(quantization_methods_scaled, tables_dir / "quantization_methods_scaled_table.csv")

        method_by_name = {str(row["method_name"]): row for row in quantization_methods_overall}
        fp32_row = method_by_name.get("original_fp32_reference", {})
        qat_row = method_by_name.get("qat_int8_pdf", {})
        quantization_overall_row = {
            "average_float_predictor_size_bytes": fp32_row.get("average_predictor_size_bytes", 0.0),
            "average_int8_predictor_size_bytes": qat_row.get("average_predictor_size_bytes", 0.0),
            "average_float_policy_size_bytes": fp32_row.get("average_policy_size_bytes", 0.0),
            "average_int8_policy_size_bytes": qat_row.get("average_policy_size_bytes", 0.0),
            "average_float_scheduler_size_bytes": fp32_row.get("average_scheduler_size_bytes", 0.0),
            "average_int8_scheduler_size_bytes": qat_row.get("average_scheduler_size_bytes", 0.0),
            "average_relative_size_reduction": qat_row.get("average_relative_size_reduction_vs_fp32", 0.0),
            "average_mean_absolute_prediction_error": qat_row.get("average_mean_absolute_prediction_error_vs_fp32", 0.0),
            "average_float_predictor_latency_ms": fp32_row.get("average_predictor_latency_ms", 0.0),
            "average_int8_predictor_latency_ms": qat_row.get("average_predictor_latency_ms", 0.0),
            "average_float_policy_latency_ms": fp32_row.get("average_policy_latency_ms", 0.0),
            "average_int8_policy_latency_ms": qat_row.get("average_policy_latency_ms", 0.0),
            "average_float_scheduler_latency_ms": fp32_row.get("average_scheduler_latency_ms", 0.0),
            "average_int8_scheduler_latency_ms": qat_row.get("average_scheduler_latency_ms", 0.0),
        }
        save_rows_as_csv([quantization_overall_row], tables_dir / "quantization_overall_table.csv")
    elif quantization_rows:
        quantization_overall_row = {
            "average_float_predictor_size_bytes": _mean_from_possible_keys(
                quantization_rows,
                "average_float_predictor_size_bytes",
                aliases=("average_float_size_bytes",),
            ),
            "average_int8_predictor_size_bytes": _mean_from_possible_keys(
                quantization_rows,
                "average_int8_predictor_size_bytes",
                aliases=("average_qat_size_bytes", "average_int8_size_bytes"),
            ),
            "average_float_policy_size_bytes": _mean_from_possible_keys(
                quantization_rows,
                "average_float_policy_size_bytes",
                default=0.0,
            ),
            "average_int8_policy_size_bytes": _mean_from_possible_keys(
                quantization_rows,
                "average_int8_policy_size_bytes",
                default=0.0,
            ),
            "average_float_scheduler_size_bytes": _mean_from_possible_keys(
                quantization_rows,
                "average_float_scheduler_size_bytes",
                aliases=("average_float_size_bytes",),
            ),
            "average_int8_scheduler_size_bytes": _mean_from_possible_keys(
                quantization_rows,
                "average_int8_scheduler_size_bytes",
                aliases=("average_qat_size_bytes", "average_int8_size_bytes"),
            ),
            "average_relative_size_reduction": _mean_from_possible_keys(
                quantization_rows,
                "average_relative_size_reduction",
            ),
            "average_mean_absolute_prediction_error": _mean_from_possible_keys(
                quantization_rows,
                "average_mean_absolute_prediction_error",
            ),
            "average_float_predictor_latency_ms": _mean_from_possible_keys(
                quantization_rows,
                "average_float_predictor_latency_ms",
                positive_only=True,
                default=0.0,
            ),
            "average_int8_predictor_latency_ms": _mean_from_possible_keys(
                quantization_rows,
                "average_int8_predictor_latency_ms",
                positive_only=True,
                default=0.0,
            ),
            "average_float_policy_latency_ms": _mean_from_possible_keys(
                quantization_rows,
                "average_float_policy_latency_ms",
                positive_only=True,
                default=0.0,
            ),
            "average_int8_policy_latency_ms": _mean_from_possible_keys(
                quantization_rows,
                "average_int8_policy_latency_ms",
                positive_only=True,
                default=0.0,
            ),
            "average_float_scheduler_latency_ms": _mean_from_possible_keys(
                quantization_rows,
                "average_float_scheduler_latency_ms",
                positive_only=True,
                default=0.0,
            ),
            "average_int8_scheduler_latency_ms": _mean_from_possible_keys(
                quantization_rows,
                "average_int8_scheduler_latency_ms",
                positive_only=True,
                default=0.0,
            ),
        }
        save_rows_as_csv([quantization_overall_row], tables_dir / "quantization_overall_table.csv")

    learned_models_comparison = _learned_models_comparison_rows(baseline_overall, learned_model_rows, quantization_overall_row)
    save_rows_as_csv(learned_models_comparison, tables_dir / "learned_models_comparison_table.csv")

    learned_model_metric_matrix = _build_scaled_metric_matrix(
        learned_models_comparison,
        metric_keys=[
            "parameter_count",
            "model_size_bytes",
            "scheduler_latency_ms",
            "mean_deadline_miss_ratio",
            "mean_utilization",
            "mean_energy",
            "mean_average_cvar",
            "mean_average_reliability",
            "mean_aging_index",
        ],
        key_name="algorithm",
    )
    save_rows_as_csv(learned_model_metric_matrix, tables_dir / "learned_models_metric_matrix_scaled_table.csv")

    if quantization_overall_row is not None:
        quantization_metric_specs = [
            ("proposed_gat_float32", "predictor_size_bytes", _metric_label("predictor_size_bytes"), quantization_overall_row.get("average_float_predictor_size_bytes", 0.0)),
            ("proposed_gat_quantization_int8", "predictor_size_bytes", _metric_label("predictor_size_bytes"), quantization_overall_row.get("average_int8_predictor_size_bytes", 0.0)),
            ("proposed_gat_float32", "policy_size_bytes", _metric_label("policy_size_bytes"), quantization_overall_row.get("average_float_policy_size_bytes", 0.0)),
            ("proposed_gat_quantization_int8", "policy_size_bytes", _metric_label("policy_size_bytes"), quantization_overall_row.get("average_int8_policy_size_bytes", 0.0)),
            ("proposed_gat_float32", "scheduler_size_bytes", _metric_label("scheduler_size_bytes"), quantization_overall_row.get("average_float_scheduler_size_bytes", 0.0)),
            ("proposed_gat_quantization_int8", "scheduler_size_bytes", _metric_label("scheduler_size_bytes"), quantization_overall_row.get("average_int8_scheduler_size_bytes", 0.0)),
            ("proposed_gat_float32", "predictor_latency_ms", _metric_label("predictor_latency_ms"), quantization_overall_row.get("average_float_predictor_latency_ms", 0.0)),
            ("proposed_gat_quantization_int8", "predictor_latency_ms", _metric_label("predictor_latency_ms"), quantization_overall_row.get("average_int8_predictor_latency_ms", 0.0)),
            ("proposed_gat_float32", "policy_latency_ms", _metric_label("policy_latency_ms"), quantization_overall_row.get("average_float_policy_latency_ms", 0.0)),
            ("proposed_gat_quantization_int8", "policy_latency_ms", _metric_label("policy_latency_ms"), quantization_overall_row.get("average_int8_policy_latency_ms", 0.0)),
            ("proposed_gat_float32", "scheduler_latency_ms", _metric_label("scheduler_latency_ms"), quantization_overall_row.get("average_float_scheduler_latency_ms", 0.0)),
            ("proposed_gat_quantization_int8", "scheduler_latency_ms", _metric_label("scheduler_latency_ms"), quantization_overall_row.get("average_int8_scheduler_latency_ms", 0.0)),
            ("proposed_gat_quantization_int8", "prediction_mae", _metric_label("prediction_mae"), quantization_overall_row.get("average_mean_absolute_prediction_error", 0.0)),
        ]
        quantization_fp32_int8_matrix = [
            {
                "algorithm": algorithm_name,
                "metric": metric_name,
                "metric_display_name": metric_display_name,
                "unit": _metric_unit(metric_name),
                "raw_value": _format_value_label(raw_value),
                "value": raw_value,
            }
            for algorithm_name, metric_name, metric_display_name, raw_value in quantization_metric_specs
            if _to_float(raw_value) > 0.0
        ]
        save_rows_as_csv(quantization_fp32_int8_matrix, tables_dir / "quantization_fp32_int8_overview_table.csv")

        quantization_fp32_int8_scaled_matrix: List[Dict[str, float | str]] = []
        grouped_metric_values = defaultdict(list)
        for row in quantization_fp32_int8_matrix:
            grouped_metric_values[str(row["metric"])].append(_to_float(row["raw_value"]))
        metric_exponents = {
            metric: _power10_order_exponent(max(abs(v) for v in values))
            for metric, values in grouped_metric_values.items()
            if values
        }
        for row in quantization_fp32_int8_matrix:
            metric = str(row["metric"])
            exponent = metric_exponents.get(metric, 0)
            divisor = 10.0 ** exponent if exponent != 0 else 1.0
            raw_value = _to_float(row["raw_value"])
            scaled_row = dict(row)
            scaled_row["metric_display_name"] = _scale_label(str(row["metric_display_name"]), exponent)
            scaled_row["scaled_value"] = raw_value / divisor
            scaled_row["scale_exponent"] = float(exponent)
            scaled_row["scale_note"] = f"displayed_value = raw_value / 10^{exponent}"
            quantization_fp32_int8_scaled_matrix.append(scaled_row)
        save_rows_as_csv(quantization_fp32_int8_scaled_matrix, tables_dir / "quantization_fp32_int8_overview_scaled_table.csv")

    uncertainty_metric_keys = [
        "mean_mode_switch_probability",
        "mean_deadline_miss_ratio",
        "mean_utilization",
        "mean_energy",
        "mean_average_cvar",
        "mean_aging_index",
        "mean_average_reliability",
        "mean_completed_hi_ratio",
    ]
    uncertainty_overall = _mean_by_group(
        uncertainty_rows,
        group_key="algorithm",
        metric_keys=uncertainty_metric_keys,
    )
    save_rows_as_csv(uncertainty_overall, tables_dir / "uncertainty_overall_table.csv")

    # Baseline overall bars.
    for metric_name, human_name in [
        ("mean_deadline_miss_ratio", _metric_label("mean_deadline_miss_ratio")),
        ("mean_task_deadline_miss_ratio", _metric_label("mean_task_deadline_miss_ratio")),
        ("mean_utilization", _metric_label("mean_utilization")),
        ("mean_energy", _metric_label("mean_energy")),
        ("mean_mode_switch_probability", _metric_label("mean_mode_switch_probability")),
        ("mean_average_cvar", _metric_label("mean_average_cvar")),
        ("mean_aging_index", _metric_label("mean_aging_index")),
        ("mean_average_reliability", _metric_label("mean_average_reliability")),
        ("mean_completed_hi_ratio", _metric_label("mean_completed_hi_ratio")),
    ]:
        _bar_chart(
            baseline_overall,
            label_key="algorithm",
            value_key=metric_name,
            output_path=plots_dir / f"baseline_{metric_name}.png",
            title=f"{human_name} by Algorithm",
            xlabel="Algorithm",
            ylabel=human_name,
        )
        _zoomed_delta_companion_chart(
            baseline_overall,
            label_key="algorithm",
            value_key=metric_name,
            output_path=plots_dir / f"baseline_{metric_name}_delta_from_min.png",
            title=f"{human_name} by Algorithm",
            xlabel="Algorithm",
            unit_label=_metric_unit(metric_name),
        )

    _grouped_bar_chart(
        baseline_metric_matrix,
        category_key="metric_display_name",
        series_key="algorithm",
        value_key="scaled_value",
        annotation_key="raw_value",
        output_path=plots_dir / "baseline_metrics_overview_scaled.png",
        title="Baseline Models: Multi-Metric Comparison with 10^k Scaling",
        xlabel="Metric (each category shows its 10^k scale)",
        ylabel="Displayed Value = Raw Value / 10^k",
        annotate=True,
        category_order=list(dict.fromkeys(str(row["metric_display_name"]) for row in baseline_metric_matrix)),
    )

    # Distribution views.
    for metric_name, ylabel in [
        ("mean_deadline_miss_ratio", _metric_label("mean_deadline_miss_ratio")),
        ("mean_task_deadline_miss_ratio", _metric_label("mean_task_deadline_miss_ratio")),
        ("mean_utilization", _metric_label("mean_utilization")),
        ("mean_energy", _metric_label("mean_energy")),
        ("mean_average_cvar", _metric_label("mean_average_cvar")),
    ]:
        _box_plot(
            baseline_rows,
            value_key=metric_name,
            series_key="algorithm",
            output_path=plots_dir / f"distribution_{metric_name}.png",
            title=f"Distribution of {ylabel} Across Configurations",
            ylabel=ylabel,
        )

    _scatter_chart(
        baseline_rows,
        x_key="mean_energy",
        y_key="mean_deadline_miss_ratio",
        series_key="algorithm",
        output_path=plots_dir / "tradeoff_energy_vs_dmr.png",
        title="Energy vs DMR Trade-off",
        xlabel=_metric_label("mean_energy"),
        ylabel=_metric_label("mean_deadline_miss_ratio"),
    )
    _scatter_chart(
        baseline_rows,
        x_key="mean_aging_index",
        y_key="mean_average_reliability",
        series_key="algorithm",
        output_path=plots_dir / "tradeoff_aging_vs_reliability.png",
        title="Aging vs Reliability Trade-off",
        xlabel="Aging Index",
        ylabel=_metric_label("mean_average_reliability"),
    )

    # Configuration-axis charts with more detail.
    for axis_key, axis_name in [
        ("graph_count", "Graph Count"),
        ("density", "Density"),
        ("regularity", "Regularity"),
        ("width", "Width"),
    ]:
        for metric_name, ylabel in [
            ("mean_deadline_miss_ratio", "Deadline Miss Ratio"),
            ("mean_task_deadline_miss_ratio", "Task-level DMR"),
            ("mean_utilization", _metric_label("mean_utilization")),
            ("mean_energy", _metric_label("mean_energy")),
        ]:
            grouped_rows = _mean_by_two_keys(baseline_rows, key_1="algorithm", key_2=axis_key, metric_keys=[metric_name])
            _line_chart(
                grouped_rows,
                x_key=axis_key,
                y_key=metric_name,
                series_key="algorithm",
                output_path=plots_dir / f"baseline_{metric_name}_vs_{axis_key}.png",
                title=f"{ylabel} vs {axis_name}",
                xlabel=axis_name,
                ylabel=ylabel,
            )
            _grouped_bar_chart(
                grouped_rows,
                category_key=axis_key,
                series_key="algorithm",
                value_key=metric_name,
                output_path=plots_dir / f"baseline_{metric_name}_vs_{axis_key}_grouped.png",
                title=f"{ylabel} vs {axis_name} (Grouped Comparison)",
                xlabel=axis_name,
                ylabel=ylabel,
            )

    # Uncertainty charts: merged paired bars with metric-specific 10^k scaling for readability.
    uncertainty_scaled_rows = _build_uncertainty_scaled_rows(uncertainty_overall)
    save_rows_as_csv(uncertainty_scaled_rows, tables_dir / "uncertainty_merged_scaled_table.csv")
    if uncertainty_scaled_rows:
        uncertainty_category_order = []
        seen_metrics = set()
        for row in uncertainty_scaled_rows:
            metric_name = str(row["metric"])
            if metric_name in seen_metrics:
                continue
            seen_metrics.add(metric_name)
            uncertainty_category_order.append(str(row["metric_display_name"]))
        _grouped_bar_chart(
            uncertainty_scaled_rows,
            category_key="metric_display_name",
            series_key="algorithm",
            value_key="scaled_value",
            annotation_key="raw_value",
            output_path=plots_dir / "uncertainty_merged_scaled.png",
            title="Uncertainty Ablation: Paired Comparison Across Metrics",
            xlabel="Metric (scaled by metric-specific powers of 10)",
            ylabel="Scaled Value [unit shown per metric]",
            annotate=True,
            category_order=uncertainty_category_order,
            series_order=["with_uncertainty", "without_uncertainty"],
        )
        # Normalized uncertainty chart intentionally omitted; scaled raw-value chart above is the defense chart.


    # Quantization charts.

    if quantization_methods_overall:
        quantization_metric_order = [
            ("average_predictor_size_bytes", "Quantization Methods: Predictor Footprint", _metric_label("predictor_size_bytes")),
            ("average_policy_size_bytes", "Quantization Methods: Policy Footprint", _metric_label("policy_size_bytes")),
            ("average_scheduler_size_bytes", "Quantization Methods: End-to-End Scheduler Footprint", _metric_label("scheduler_size_bytes")),
            ("average_relative_size_reduction_vs_fp32", "Quantization Methods: Size Reduction vs FP32", "Size Reduction vs FP32 [ratio]"),
            ("average_mean_absolute_prediction_error_vs_fp32", "Quantization Methods: Prediction MAE vs FP32", _metric_label("prediction_mae")),
            ("average_predictor_latency_ms", "Quantization Methods: Predictor Latency", _metric_label("predictor_latency_ms")),
            ("average_policy_latency_ms", "Quantization Methods: Policy Latency", _metric_label("policy_latency_ms")),
            ("average_scheduler_latency_ms", "Quantization Methods: End-to-End Scheduler Latency", _metric_label("scheduler_latency_ms")),
            ("success_rate", "Quantization Methods: Execution Success Rate", "Success Rate [ratio]"),
            ("average_utilization", "Quantization Methods: Scheduling Utilization", _metric_label("utilization")),
            ("average_deadline_miss_ratio", "Quantization Methods: Deadline Miss Ratio", _metric_label("deadline_miss_ratio")),
            ("average_task_deadline_miss_ratio", "Quantization Methods: Task-level DMR", _metric_label("task_deadline_miss_ratio")),
            ("average_energy", "Quantization Methods: Scheduling Energy", _metric_label("energy")),
            ("average_mode_switch_probability", "Quantization Methods: Mode-Switch Probability", _metric_label("mode_switch_probability")),
            ("average_completed_hi_ratio", "Quantization Methods: Completed HI Ratio", _metric_label("completed_hi_ratio")),
            ("average_completed_lo_ratio", "Quantization Methods: Completed LO Ratio", _metric_label("completed_lo_ratio")),
            ("average_dropped_lo_ratio", "Quantization Methods: Dropped LO Ratio", _metric_label("dropped_lo_ratio")),
            ("average_service_loss_ratio", "Quantization Methods: Service Loss Ratio", _metric_label("service_loss_ratio")),
        ]
        for value_key, title, ylabel in quantization_metric_order:
            if any(_to_float(row.get(value_key)) > 0.0 for row in quantization_methods_overall):
                _bar_chart(
                    quantization_methods_overall,
                    label_key="method_name",
                    value_key=value_key,
                    output_path=plots_dir / f"quantization_methods_{value_key}.png",
                    title=title,
                    xlabel="Quantization Method",
                    ylabel=ylabel,
                )
                _zoomed_delta_companion_chart(
                    quantization_methods_overall,
                    label_key="method_name",
                    value_key=value_key,
                    output_path=plots_dir / f"quantization_methods_{value_key}_delta_from_min.png",
                    title=title,
                    xlabel="Quantization Method",
                    unit_label=_metric_unit(value_key.replace("average_", "")),
                )
        if quantization_methods_scaled:
            category_order = list(dict.fromkeys(str(row["metric_display_name"]) for row in quantization_methods_scaled))
            _grouped_bar_chart(
                quantization_methods_scaled,
                category_key="metric_display_name",
                series_key="method_name",
                value_key="scaled_value",
                annotation_key="raw_value",
                output_path=plots_dir / "quantization_methods_complete_scaled.png",
                title="Quantization Methods: Multi-Metric Comparison with 10^k Scaling",
                xlabel="Metric (each category shows its 10^k scale)",
                ylabel="Displayed Value = Raw Value / 10^k",
                annotate=True,
                category_order=category_order,
                series_order=[
                    "original_fp32_reference",
                    "qat_int8_pdf",
                                "weight_only_int8_per_channel",
                    "fp16_mixed_precision",
                ],
            )

    if quantization_overall_row is not None:
        use_scheduler_size = _to_float(quantization_overall_row.get("average_float_scheduler_size_bytes", 0.0)) > 0.0 or _to_float(quantization_overall_row.get("average_int8_scheduler_size_bytes", 0.0)) > 0.0
        if use_scheduler_size:
            size_rows = [
                {"algorithm": "proposed_gat_float32", "value": quantization_overall_row.get("average_float_scheduler_size_bytes", 0.0)},
                {"algorithm": "proposed_gat_quantization_int8", "value": quantization_overall_row.get("average_int8_scheduler_size_bytes", 0.0)},
            ]
            size_title = "Average Scheduler Size: FP32 vs INT8"
            size_output = plots_dir / "quantization_scheduler_size.png"
        else:
            size_rows = [
                {"algorithm": "proposed_gat_float32", "value": quantization_overall_row.get("average_float_predictor_size_bytes", 0.0)},
                {"algorithm": "proposed_gat_quantization_int8", "value": quantization_overall_row.get("average_int8_predictor_size_bytes", 0.0)},
            ]
            size_title = "Average Predictor Size: FP32 vs INT8"
            size_output = plots_dir / "quantization_predictor_size.png"
        _bar_chart(
            size_rows,
            label_key="algorithm",
            value_key="value",
            output_path=size_output,
            title=size_title,
            xlabel="Variant",
            ylabel=_metric_label("predictor_size_bytes"),
        )
        if _to_float(quantization_overall_row.get("average_float_predictor_latency_ms", 0.0)) > 0.0 or _to_float(quantization_overall_row.get("average_int8_predictor_latency_ms", 0.0)) > 0.0:
            latency_rows = [
                {"algorithm": "proposed_gat_float32", "value": quantization_overall_row.get("average_float_predictor_latency_ms", 0.0)},
                {"algorithm": "proposed_gat_quantization_int8", "value": quantization_overall_row.get("average_int8_predictor_latency_ms", 0.0)},
            ]
            _bar_chart(
                latency_rows,
                label_key="algorithm",
                value_key="value",
                output_path=plots_dir / "quantization_predictor_latency.png",
                title="Average Predictor Latency: FP32 vs INT8",
                xlabel="Variant",
                ylabel=_metric_label("predictor_latency_ms"),
            )
        mae_rows = [{"algorithm": "proposed_gat_quantization_int8", "value": quantization_overall_row.get("average_mean_absolute_prediction_error", 0.0)}]
        _bar_chart(
            mae_rows,
            label_key="algorithm",
            value_key="value",
            output_path=plots_dir / "quantization_prediction_mae.png",
            title="Average Prediction MAE After INT8 Quantization",
            xlabel="Variant",
            ylabel=_metric_label("prediction_mae"),
        )
        if Path(tables_dir / "quantization_fp32_int8_overview_table.csv").exists():
            quant_raw = load_csv_rows(tables_dir / "quantization_fp32_int8_overview_table.csv")
            quant_scaled = load_csv_rows(tables_dir / "quantization_fp32_int8_overview_scaled_table.csv")
            _grouped_bar_chart(
                quant_raw,
                category_key="metric_display_name",
                series_key="algorithm",
                value_key="raw_value",
                output_path=plots_dir / "quantization_fp32_vs_int8_complete_raw.png",
                title="FP32 vs INT8: Raw Model Footprint and Latency Comparison",
                xlabel="Metric",
                ylabel="Raw Value [unit shown per metric]",
                category_order=list(dict.fromkeys(str(row["metric_display_name"]) for row in quant_raw)),
                series_order=["proposed_gat_float32", "proposed_gat_quantization_int8"],
            )
            _grouped_bar_chart(
                quant_scaled,
                category_key="metric_display_name",
                series_key="algorithm",
                value_key="scaled_value",
                annotation_key="raw_value",
                output_path=plots_dir / "quantization_fp32_vs_int8_complete_scaled.png",
                title="FP32 vs INT8: Multi-Metric Comparison with 10^k Scaling",
                xlabel="Metric (each category shows its 10^k scale)",
                ylabel="Displayed Value = Raw Value / 10^k",
                category_order=list(dict.fromkeys(str(row["metric_display_name"]) for row in quant_scaled)),
                series_order=["proposed_gat_float32", "proposed_gat_quantization_int8"],
            )

    # Learned-model focused charts.
    if learned_models_comparison:
        for value_key, title, ylabel in [
            ("parameter_count", "Learned Models: Parameter Count", _metric_label("parameter_count")),
            ("model_size_bytes", "Learned Models: Model Storage Footprint", _metric_label("model_size_bytes")),
            ("predictor_latency_ms", "Learned Models: Predictor Latency", _metric_label("predictor_latency_ms")),
            ("policy_latency_ms", "Learned Models: Policy Latency", _metric_label("policy_latency_ms")),
            ("scheduler_latency_ms", "Learned Models: End-to-End Schedule Latency", _metric_label("scheduler_latency_ms")),
            ("mean_deadline_miss_ratio", "Learned Models: DMR", _metric_label("mean_deadline_miss_ratio")),
            ("mean_energy", "Learned Models: Energy", _metric_label("mean_energy")),
            ("mean_aging_index", "Learned Models: Aging Index", _metric_label("mean_aging_index")),
            ("mean_average_reliability", "Learned Models: Reliability", _metric_label("mean_average_reliability")),
        ]:
            _bar_chart(
                learned_models_comparison,
                label_key="algorithm",
                value_key=value_key,
                output_path=plots_dir / f"learned_models_{value_key}.png",
                title=title,
                xlabel="Learned Model",
                ylabel=ylabel,
            )
            _zoomed_delta_companion_chart(
                learned_models_comparison,
                label_key="algorithm",
                value_key=value_key,
                output_path=plots_dir / f"learned_models_{value_key}_delta_from_min.png",
                title=title,
                xlabel="Learned Model",
                unit_label=_metric_unit(value_key),
            )
        _grouped_bar_chart(
            learned_model_metric_matrix,
            category_key="metric_display_name",
            series_key="algorithm",
            value_key="scaled_value",
            annotation_key="raw_value",
            output_path=plots_dir / "learned_models_overview_scaled.png",
            title="Learned Models: Multi-Metric Comparison with 10^k Scaling",
            xlabel="Metric (each category shows its 10^k scale)",
            ylabel="Displayed Value = Raw Value / 10^k",
            category_order=list(dict.fromkeys(str(row["metric_display_name"]) for row in learned_model_metric_matrix)),
        )

    # Structural charts.
    structural_summary_rows: List[Dict[str, str]] = []
    structural_by_node: List[Dict[str, float | str]] = []
    structural_by_core: List[Dict[str, float | str]] = []
    structural_by_utilization: List[Dict[str, float | str]] = []
    if structural_output_root is not None:
        structural_summary_rows = load_csv_rows(Path(structural_output_root) / "structural_sweep_summary.csv")
        if structural_summary_rows:
            structural_by_node = _mean_by_two_keys(
                structural_summary_rows,
                key_1="algorithm",
                key_2="node_count",
                metric_keys=["mean_utilization", "mean_deadline_miss_ratio", "mean_energy"],
            )
            save_rows_as_csv(structural_by_node, tables_dir / "structural_by_node_count.csv")
            _line_chart(structural_by_node, "node_count", "mean_utilization", "algorithm", plots_dir / "structural_utilization_vs_nodes.png", "Structural Sweep: Utilization vs Node Count", "Node Count", _metric_label("mean_utilization"))
            _line_chart(structural_by_node, "node_count", "mean_deadline_miss_ratio", "algorithm", plots_dir / "structural_dmr_vs_nodes.png", "Structural Sweep: DMR vs Node Count", "Node Count", "Average DMR")
            _line_chart(structural_by_node, "node_count", "mean_energy", "algorithm", plots_dir / "structural_energy_vs_nodes.png", "Structural Sweep: Energy vs Node Count", "Node Count", _metric_label("mean_energy"))
            _grouped_bar_chart(structural_by_node, "node_count", "algorithm", "mean_utilization", plots_dir / "structural_utilization_vs_nodes_grouped.png", "Structural Sweep: Utilization vs Node Count (Grouped)", "Node Count", _metric_label("mean_utilization"))
            _grouped_bar_chart(structural_by_node, "node_count", "algorithm", "mean_deadline_miss_ratio", plots_dir / "structural_dmr_vs_nodes_grouped.png", "Structural Sweep: DMR vs Node Count (Grouped)", "Node Count", "Average DMR")
            _grouped_bar_chart(structural_by_node, "node_count", "algorithm", "mean_energy", plots_dir / "structural_energy_vs_nodes_grouped.png", "Structural Sweep: Energy vs Node Count (Grouped)", "Node Count", _metric_label("mean_energy"))

            structural_by_core = _mean_by_two_keys(
                structural_summary_rows,
                key_1="algorithm",
                key_2="core_count",
                metric_keys=["mean_utilization", "mean_deadline_miss_ratio", "mean_energy"],
            )
            save_rows_as_csv(structural_by_core, tables_dir / "structural_by_core_count_all_grid_reference.csv")

            # Defense-facing B subset: do not average the full Cartesian node×core grid.
            # It keeps the proportional workload path already present in the current
            # outputs, e.g. 4/50, 8/100, 16/200, 32/400, 64/800.
            structural_proportional_subset = _add_core_task_pair_labels(_add_proportional_total_energy(_structural_proportional_subset_rows(structural_summary_rows)))
            save_rows_as_csv(structural_proportional_subset, tables_dir / "structural_proportional_workload_subset_raw.csv")
            structural_by_core_proportional = _mean_by_two_keys(
                structural_proportional_subset,
                key_1="algorithm",
                key_2="core_count",
                metric_keys=["node_count", "nodes_per_core", "mean_utilization", "mean_deadline_miss_ratio", "mean_task_deadline_miss_ratio", "mean_energy", "mean_energy_proportional_workload_total", "workload_adjustment_factor_for_energy"],
            )
            structural_by_core_proportional = _add_core_task_pair_labels(structural_by_core_proportional)
            proportional_core_task_order = _core_task_pair_order(structural_by_core_proportional)
            save_rows_as_csv(structural_by_core_proportional, tables_dir / "structural_by_core_count_proportional_workload.csv")
            _line_chart(structural_by_core_proportional, "core_count", "mean_deadline_miss_ratio", "algorithm", plots_dir / "structural_proportional_dmr_vs_cores.png", "Proportional Core/Task Scaling: DMR (4C/50T to 64C/800T)", "Core/task pair", "Deadline Miss Ratio", x_tick_label_key="core_task_pair")
            _line_chart(structural_by_core_proportional, "core_count", "mean_task_deadline_miss_ratio", "algorithm", plots_dir / "structural_proportional_task_dmr_vs_cores.png", "Proportional Core/Task Scaling: Task DMR (4C/50T to 64C/800T)", "Core/task pair", "Task-level DMR", x_tick_label_key="core_task_pair")
            _line_chart(structural_by_core_proportional, "core_count", "mean_utilization", "algorithm", plots_dir / "structural_proportional_utilization_vs_cores.png", "Proportional Core/Task Scaling: Utilization (4C/50T to 64C/800T)", "Core/task pair", _metric_label("mean_utilization"), x_tick_label_key="core_task_pair")
            _line_chart(structural_by_core_proportional, "core_count", "mean_energy_proportional_workload_total", "algorithm", plots_dir / "structural_proportional_energy_vs_cores.png", "Proportional Core/Task Scaling: Total Energy (4C/50T to 64C/800T)", "Core/task pair", _metric_label("mean_energy_proportional_workload_total"), x_tick_label_key="core_task_pair")
            _grouped_bar_chart(structural_by_core_proportional, "core_task_pair", "algorithm", "mean_deadline_miss_ratio", plots_dir / "structural_proportional_dmr_vs_cores_grouped.png", "Proportional Core/Task Scaling: DMR by Pair", "Core/task pair", "Deadline Miss Ratio", category_order=proportional_core_task_order)
            _grouped_bar_chart(structural_by_core_proportional, "core_task_pair", "algorithm", "mean_energy_proportional_workload_total", plots_dir / "structural_proportional_energy_vs_cores_grouped.png", "Proportional Core/Task Scaling: Total Energy by Pair", "Core/task pair", _metric_label("mean_energy_proportional_workload_total"), category_order=proportional_core_task_order)

            structural_by_utilization = _mean_by_two_keys(
                structural_summary_rows,
                key_1="algorithm",
                key_2="utilization_level",
                metric_keys=["mean_utilization", "mean_deadline_miss_ratio", "mean_energy"],
            )
            save_rows_as_csv(structural_by_utilization, tables_dir / "structural_by_utilization_level.csv")
            _line_chart(structural_by_utilization, "utilization_level", "mean_deadline_miss_ratio", "algorithm", plots_dir / "structural_dmr_vs_utilization_level.png", "Structural Sweep: DMR vs Utilization Level", "Utilization Level", "Average DMR")
            _line_chart(structural_by_utilization, "utilization_level", "mean_utilization", "algorithm", plots_dir / "structural_utilization_vs_utilization_level.png", "Structural Sweep: Utilization vs Utilization Level", "Utilization Level", _metric_label("mean_utilization"))
            _line_chart(structural_by_utilization, "utilization_level", "mean_energy", "algorithm", plots_dir / "structural_energy_vs_utilization_level.png", "Structural Sweep: Energy vs Utilization Level", "Utilization Level", _metric_label("mean_energy"))
            _grouped_bar_chart(structural_by_utilization, "utilization_level", "algorithm", "mean_deadline_miss_ratio", plots_dir / "structural_dmr_vs_utilization_level_grouped.png", "Structural Sweep: DMR vs Utilization Level (Grouped)", "Utilization Level", "Average DMR")
            _grouped_bar_chart(structural_by_utilization, "utilization_level", "algorithm", "mean_utilization", plots_dir / "structural_utilization_vs_utilization_level_grouped.png", "Structural Sweep: Utilization vs Utilization Level (Grouped)", "Utilization Level", _metric_label("mean_utilization"))
            _grouped_bar_chart(structural_by_utilization, "utilization_level", "algorithm", "mean_energy", plots_dir / "structural_energy_vs_utilization_level_grouped.png", "Structural Sweep: Energy vs Utilization Level (Grouped)", "Utilization Level", _metric_label("mean_energy"))

    summary_artifacts = build_defense_summary_folder(
        all_config_output_root=all_config_output_root,
        structural_output_root=structural_output_root,
        output_root=output_root,
    )

    png_pdf_audit = ensure_pdf_for_png_outputs(output_root)

    save_json(
        {
            "all_config_output_root": str(all_config_output_root),
            "structural_output_root": str(structural_output_root) if structural_output_root is not None else None,
            "baseline_rows_loaded": len(baseline_rows),
            "uncertainty_rows_loaded": len(uncertainty_rows),
            "quantization_rows_loaded": len(quantization_rows),
            "learned_model_rows_loaded": len(learned_model_rows),
            "task_level_sample_rows_loaded": len(task_level_sample_rows),
            "task_level_rows_total": count_csv_data_rows(task_level_rows_path),
            "core_level_rows_loaded": len(core_level_rows),
            "schedule_extended_rows_loaded": len(schedule_extended_rows),
            "energy_power_rows_loaded": len(energy_power_rows),
            "prediction_metrics_rows_loaded": len(prediction_metrics_rows),
            "prediction_detail_rows_loaded": len(prediction_detail_rows),
            "power_trace_rows_loaded": len(power_trace_rows),
            "training_loss_rows_loaded": len(training_loss_rows),
            "structural_rows_loaded": len(structural_summary_rows),
            "plots_pdf_dir": str(plots_pdf_dir),
            "png_pdf_audit": png_pdf_audit,
            "summary_artifacts": summary_artifacts,
        },
        output_root / "defense_report_manifest.json",
    )

    return {
        "report_root": str(output_root),
        "tables_dir": str(tables_dir),
        "plots_dir": str(plots_dir),
        "plots_pdf_dir": str(plots_pdf_dir),
        "summary_dir": str(output_root / "summary"),
    }
