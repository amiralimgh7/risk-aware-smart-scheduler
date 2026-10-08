"""Validates key output invariants such as PDF/PNG pairs, latency, reliability, 64-core coverage, and detailed tables.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

from experiments.reporting import ensure_pdf_for_png_outputs

LATENCY_COLUMNS = [
    "predictor_latency_ms",
    "policy_latency_ms",
    "scheduler_latency_ms",
    "average_predictor_latency_ms",
    "average_policy_latency_ms",
    "average_scheduler_latency_ms",
    "average_float_predictor_latency_ms",
    "average_int8_predictor_latency_ms",
    "average_float_policy_latency_ms",
    "average_int8_policy_latency_ms",
    "average_float_scheduler_latency_ms",
    "average_int8_scheduler_latency_ms",
]

RELIABILITY_COLUMNS = ["mean_average_reliability", "average_average_reliability", "average_reliability"]
DMR_COLUMNS = ["mean_task_deadline_miss_ratio", "average_task_deadline_miss_ratio", "task_deadline_miss_ratio"]


def _load_rows(path: Path) -> List[Dict[str, str]]:
    """Internal helper for load rows."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _count_csv_data_rows(path: Path) -> int:
    """Internal helper for count csv data rows."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)


def _csv_header(path: Path) -> List[str]:
    """Internal helper for csv header."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return [str(column) for column in next(reader)]
        except StopIteration:
            return []


def _to_float(value: object) -> float:
    """Internal helper for to float."""
    try:
        number = float(str(value).strip())
    except Exception:
        return math.nan
    return number if math.isfinite(number) else math.nan


def _positive_count(rows: Iterable[Dict[str, str]], columns: Iterable[str]) -> int:
    """Internal helper for positive count."""
    count = 0
    for row in rows:
        for column in columns:
            if column in row and _to_float(row.get(column)) > 0.0:
                count += 1
    return count




def _finite_rows_with_columns(rows: Iterable[Dict[str, str]], required_columns: Iterable[str]) -> List[Dict[str, str]]:
    """Return rows that contain all requested columns with finite numeric values."""

    output: List[Dict[str, str]] = []
    for row in rows:
        ok = True
        for column in required_columns:
            if column not in row or not math.isfinite(_to_float(row.get(column))):
                ok = False
                break
        if ok:
            output.append(row)
    return output

def _find_aggregate_root(all_config_output_root: Path | None) -> Path | None:
    """Internal helper for find aggregate root."""
    if all_config_output_root is None:
        return None
    if (all_config_output_root / "_aggregate").exists():
        return all_config_output_root / "_aggregate"
    return all_config_output_root


def _issue(status: str, code: str, message: str, detail: str = "") -> Dict[str, str]:
    """Internal helper for issue."""
    return {"status": status, "code": code, "message": message, "detail": detail}


def build_argument_parser() -> argparse.ArgumentParser:
    """Run the build argument parser step and return its computed result."""
    parser = argparse.ArgumentParser(description="Check generated ESRLab outputs for common suspicious/illogical cases.")
    parser.add_argument("--all-config-output-root", type=Path, default=None, help="Example: outputs/all_config_runs")
    parser.add_argument("--defense-output-root", type=Path, default=None, help="Example: outputs/defense_report")
    parser.add_argument("--structural-output-root", type=Path, default=None, help="Example: outputs/structural_sweep")
    parser.add_argument("--write-report", type=Path, default=None, help="Optional JSON/CSV report prefix or folder.")
    parser.add_argument("--minimum-plot-count", type=int, default=50, help="Fail if the defense report contains fewer PNG plots than this. This catches partially generated reports after a crash.")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser


def main() -> None:
    """Run the main step and return its computed result."""
    args = build_argument_parser().parse_args()
    issues: List[Dict[str, str]] = []

    if args.defense_output_root is not None:
        audit = ensure_pdf_for_png_outputs(args.defense_output_root)
        if audit.get("pdf_failed", 0) or audit.get("png_failed_from_pdf", 0):
            issues.append(_issue("ERROR", "PLOT_PAIR_FAILED", "Some plot PDF/PNG pairs could not be completed.", json.dumps(audit)))
        elif audit.get("pdf_files", 0) == 0:
            issues.append(_issue("ERROR", "NO_PLOT_PDFS", "No canonical plot PDFs were found; the defense report was not generated.", json.dumps(audit)))
        else:
            issues.append(_issue("OK", "PDF_FIRST_PNG_PAIRS", "Plot PDFs are canonical and PNGs are derived from PDF outputs.", json.dumps(audit)))
        if audit.get("orphan_png_files", 0):
            issues.append(_issue("WARNING", "ORPHAN_PNG_FALLBACK", "Some PNGs did not have canonical PDFs and were handled through the raster fallback.", json.dumps(audit)))
        plots = args.defense_output_root / "plots"
        plots_pdf = args.defense_output_root / "plots_pdf"
        png_count = len(list(plots.glob("*.png"))) if plots.exists() else 0
        if png_count < args.minimum_plot_count:
            issues.append(_issue("ERROR", "DEFENSE_REPORT_PARTIAL", "Defense report appears incomplete; too few PNG plots were generated.", f"png_count={png_count}, minimum={args.minimum_plot_count}"))
        else:
            issues.append(_issue("OK", "DEFENSE_REPORT_PLOT_COUNT", "Defense report plot count is large enough.", f"png_count={png_count}"))
        task_dmr_png = plots / "baseline_mean_task_deadline_miss_ratio.png"
        task_dmr_pdf = plots_pdf / "baseline_mean_task_deadline_miss_ratio.pdf"
        if not task_dmr_png.exists() or not task_dmr_pdf.exists():
            issues.append(_issue("ERROR", "TASK_DMR_CHART_MISSING", "Task-level DMR chart or its PDF counterpart is missing.", f"png={task_dmr_png.exists()}, pdf={task_dmr_pdf.exists()}"))
        else:
            issues.append(_issue("OK", "TASK_DMR_CHART", "Task-level DMR chart exists as PNG and PDF."))

    aggregate_root = _find_aggregate_root(args.all_config_output_root)
    if aggregate_root is not None:
        learned_rows = _load_rows(aggregate_root / "learned_models_summary_aggregate.csv")
        quant_rows = _load_rows(aggregate_root / "quantization_summary_aggregate.csv")
        base_rows = _load_rows(aggregate_root / "baseline_summary_aggregate.csv")
        unc_rows = _load_rows(aggregate_root / "uncertainty_summary_aggregate.csv")
        detailed_paths = [
            ("task-level detailed output", aggregate_root / "task_level_results_aggregate.csv"),
            ("core-level detailed output", aggregate_root / "core_level_results_aggregate.csv"),
            ("extended schedule output", aggregate_root / "schedule_summary_extended_aggregate.csv"),
            ("energy/power output", aggregate_root / "energy_power_summary_aggregate.csv"),
            ("prediction metrics output", aggregate_root / "prediction_metrics_aggregate.csv"),
        ]
        for detail_name, detail_path in detailed_paths:
            row_count = _count_csv_data_rows(detail_path)
            if row_count <= 0:
                issues.append(_issue("ERROR", "DETAILED_OUTPUT_MISSING", f"Missing {detail_name} CSV aggregate."))
            else:
                issues.append(_issue("OK", "DETAILED_OUTPUT_EXISTS", f"{detail_name} CSV aggregate exists.", f"rows={row_count}"))

        schedule_rows = _load_rows(aggregate_root / "schedule_summary_extended_aggregate.csv")
        energy_rows = _load_rows(aggregate_root / "energy_power_summary_aggregate.csv")
        dmr_check_rows = _finite_rows_with_columns(
            schedule_rows,
            ["number_of_total_tasks", "number_of_deadline_missed_tasks", "deadline_miss_ratio", "service_loss_ratio"],
        )
        dmr_errors = []
        for row in dmr_check_rows[:20000]:
            total_tasks = _to_float(row.get("number_of_total_tasks"))
            missed_tasks = _to_float(row.get("number_of_deadline_missed_tasks"))
            reported_dmr = _to_float(row.get("deadline_miss_ratio"))
            expected_dmr = 0.0 if total_tasks <= 0.0 else missed_tasks / total_tasks
            if abs(reported_dmr - expected_dmr) > 1e-9:
                dmr_errors.append(f"reported={reported_dmr}, expected={expected_dmr}")
                if len(dmr_errors) >= 3:
                    break
        if dmr_check_rows and not dmr_errors:
            issues.append(_issue("OK", "DMR_FORMULA", "DMR is computed as deadline_missed_tasks / total_tasks and does not include service loss."))
        elif dmr_errors:
            issues.append(_issue("ERROR", "DMR_FORMULA", "Deadline Miss Ratio formula mismatch found.", "; ".join(dmr_errors)))

        energy_check_rows = _finite_rows_with_columns(energy_rows, ["energy_after_dvfs", "energy_before_dvfs"])
        energy_errors = []
        for row in energy_check_rows[:20000]:
            after = _to_float(row.get("energy_after_dvfs"))
            before = _to_float(row.get("energy_before_dvfs"))
            if after > before + 1e-9:
                energy_errors.append(f"algorithm={row.get('algorithm')}, after={after}, before={before}")
                if len(energy_errors) >= 3:
                    break
        if energy_check_rows and not energy_errors:
            issues.append(_issue("OK", "DVFS_ENERGY_ORDER", "Energy after DVFS is not greater than the no-DVFS baseline."))
        elif energy_errors:
            issues.append(_issue("ERROR", "DVFS_ENERGY_ORDER", "energy_after_dvfs is larger than energy_before_dvfs.", "; ".join(energy_errors)))

        mode_switch_columns = ["mode_switch_time_overhead", "mode_switch_energy_overhead"]
        if schedule_rows and all(column in schedule_rows[0] for column in mode_switch_columns):
            issues.append(_issue("OK", "MODE_SWITCH_OVERHEAD_COLUMNS", "Mode-switch time and energy overhead columns are present in schedule output."))
        elif schedule_rows:
            issues.append(_issue("ERROR", "MODE_SWITCH_OVERHEAD_COLUMNS", "Mode-switch overhead columns are missing from schedule output."))

        for name, rows in [("learned models", learned_rows), ("quantization", quant_rows)]:
            present_latency_cols = [col for col in LATENCY_COLUMNS if rows and col in rows[0]]
            positives = _positive_count(rows, present_latency_cols)
            if present_latency_cols and positives == 0:
                issues.append(_issue("ERROR", "ZERO_LATENCY", f"All latency values are zero in {name} aggregate.", ", ".join(present_latency_cols)))
            elif present_latency_cols:
                issues.append(_issue("OK", "LATENCY_NONZERO", f"Latency has positive values in {name} aggregate.", f"positive_cells={positives}"))

        for name, rows in [("baseline", base_rows), ("quantization", quant_rows), ("uncertainty", unc_rows)]:
            reliability_values = []
            for row in rows:
                for col in RELIABILITY_COLUMNS:
                    if col in row:
                        value = _to_float(row.get(col))
                        if math.isfinite(value):
                            reliability_values.append(value)
            if reliability_values:
                mean_value = sum(reliability_values) / len(reliability_values)
                min_value = min(reliability_values)
                if min_value > 0.9999:
                    issues.append(_issue("WARN", "RELIABILITY_TOO_CLOSE_TO_ONE", f"Reliability in {name} is still extremely close to 1.", f"min={min_value:.12g}, mean={mean_value:.12g}"))
                else:
                    issues.append(_issue("OK", "RELIABILITY_RANGE", f"Reliability in {name} is not stuck at six-nines.", f"min={min_value:.12g}, mean={mean_value:.12g}"))

        for name, path, rows in [
            ("baseline", aggregate_root / "baseline_summary_aggregate.csv", base_rows),
            ("quantization", aggregate_root / "quantization_summary_aggregate.csv", quant_rows),
            ("uncertainty", aggregate_root / "uncertainty_summary_aggregate.csv", unc_rows),
        ]:
            header = _csv_header(path) or (list(rows[0].keys()) if rows else [])
            if header:
                has_task_dmr = any(col in header for col in DMR_COLUMNS)
                if not has_task_dmr:
                    issues.append(_issue("ERROR", "TASK_DMR_COLUMN_MISSING", f"Task-level DMR column is missing in {name} aggregate."))

        for rowset_name, rows in [("learned_models", learned_rows)]:
            absolute_count = 0
            for row in rows:
                text = str(row.get("checkpoint_dir", ""))
                if ":\\" in text or text.startswith("/"):
                    absolute_count += 1
            if absolute_count:
                issues.append(_issue("WARN", "ABSOLUTE_PATHS", f"Absolute checkpoint paths remain in {rowset_name} aggregate.", f"count={absolute_count}"))

    if args.structural_output_root is not None:
        structural_rows = _load_rows(args.structural_output_root / "structural_sweep_summary.csv")
        if not structural_rows:
            issues.append(_issue("ERROR", "STRUCTURAL_SUMMARY_MISSING", "Structural sweep summary was not found."))
        else:
            first = structural_rows[0]
            for required in ["base_node_count", "node_count", "core_count", "nodes_per_core", "scale_nodes_with_cores"]:
                if required not in first:
                    issues.append(_issue("ERROR", "STRUCTURAL_METADATA_MISSING", f"Missing structural metadata column: {required}"))
            if "scale_nodes_with_cores" in first:
                enabled = any(_to_float(row.get("scale_nodes_with_cores")) == 1.0 for row in structural_rows)
                if enabled:
                    issues.append(_issue("OK", "CORE_NODE_SCALING", "Structural sweep records node scaling with core count."))
                else:
                    issues.append(_issue("WARN", "CORE_NODE_SCALING_OFF", "Structural sweep appears to use fixed node count across core counts."))
            core_counts = sorted({int(_to_float(row.get("core_count"))) for row in structural_rows if math.isfinite(_to_float(row.get("core_count")))})
            if 64 not in core_counts:
                issues.append(_issue("ERROR", "CORE_64_MISSING", "Structural sweep does not include 64-core scenario.", f"core_counts={core_counts}"))
            else:
                issues.append(_issue("OK", "CORE_64_PRESENT", "Structural sweep includes 64-core scenario.", f"core_counts={core_counts}"))
            proportional_ok = True
            bad_examples = []
            for row in structural_rows:
                base = _to_float(row.get("base_node_count"))
                cores = _to_float(row.get("core_count"))
                nodes = _to_float(row.get("node_count"))
                if math.isfinite(base) and math.isfinite(cores) and math.isfinite(nodes) and base > 0 and cores > 0:
                    expected = round(base * cores / 4.0)
                    if abs(nodes - expected) > 0.5:
                        proportional_ok = False
                        bad_examples.append(f"base={base}, cores={cores}, nodes={nodes}, expected={expected}")
                        if len(bad_examples) >= 3:
                            break
            if proportional_ok:
                issues.append(_issue("OK", "TASKS_SCALE_WITH_CORES", "Task/node count scales proportionally with core count."))
            else:
                issues.append(_issue("ERROR", "TASKS_NOT_PROPORTIONAL", "Task/node count is not proportional to core count.", "; ".join(bad_examples)))

    if args.write_report is not None:
        report_target = args.write_report
        if report_target.suffix:
            json_path = report_target.with_suffix(".json")
            csv_path = report_target.with_suffix(".csv")
        else:
            report_target.mkdir(parents=True, exist_ok=True)
            json_path = report_target / "output_sanity_report.json"
            csv_path = report_target / "output_sanity_report.csv"
        json_path.write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["status", "code", "message", "detail"])
            writer.writeheader()
            writer.writerows(issues)
        print(f"json_report={json_path}")
        print(f"csv_report={csv_path}")

    for issue in issues:
        print(f"[{issue['status']}] {issue['code']}: {issue['message']} {issue['detail']}")

    if args.fail_on_error and any(issue["status"] == "ERROR" for issue in issues):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
