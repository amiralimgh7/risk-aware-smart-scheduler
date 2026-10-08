"""Repairs semantic output definitions for existing aggregates without retraining.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

MAPE_EPSILON = 1e-3


def _float(value, default=0.0):
    """Internal helper for float."""
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _read_csv(path: Path):
    """Internal helper for read csv."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows):
    """Internal helper for write csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    headers = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stream_csv(path: Path) -> Iterator[dict]:
    """Internal helper for stream csv."""
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _rewrite_csv_stream(path: Path, transform):
    """Internal helper for rewrite csv stream."""
    if not path.exists():
        return 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    changed = 0
    with path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        original_headers = list(reader.fieldnames or [])
        rows_buffer = []
        headers = list(original_headers)
        # Keep this function streaming in spirit, but allow dynamic new columns.  We
        # first buffer only when the file is moderate; for the large files used here
        # we pass explicit columns via transform metadata below when needed.
        for row in reader:
            new_row, did = transform(row)
            if did:
                changed += 1
            for key in new_row.keys():
                if key not in headers:
                    headers.append(key)
            rows_buffer.append(new_row)
    with tmp.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_buffer)
    tmp.replace(path)
    return changed


def _group_key(row):
    """Internal helper for group key."""
    return (
        row.get("configuration", ""),
        row.get("algorithm", ""),
        row.get("application_id", ""),
        row.get("graph_id", ""),
    )


def _infer_outputs_root(report_root: Path) -> Path:
    # report_root is normally outputs/defense_report.
    """Internal helper for infer outputs root."""
    if report_root.name == "defense_report":
        return report_root.parent
    return report_root.parent


def _default_aggregate_root(report_root: Path, all_config_output_root: Path | None) -> Path:
    """Internal helper for default aggregate root."""
    if all_config_output_root is not None:
        return all_config_output_root / "_aggregate"
    return _infer_outputs_root(report_root) / "all_config_runs" / "_aggregate"


def _first_existing(paths: Iterable[Path]) -> Path | None:
    """Internal helper for first existing."""
    for path in paths:
        if path.exists():
            return path
    return None


def _path_status(path: Path | None) -> str:
    """Internal helper for path status."""
    return str(path) if path is not None else ""


def repair_peak_power(report_root: Path, *, all_config_output_root: Path | None = None) -> dict:
    """Repairs peak-power columns from total-system power traces.

    Priority: full aggregate trace -> report sample trace.  The previous repair only
    saw the sample trace and changed a few rows; this version repairs the full
    energy aggregate when the aggregate trace exists.
    """

    tables = report_root / "tables"
    aggregate_root = _default_aggregate_root(report_root, all_config_output_root)
    energy_candidates = [
        tables / "energy_power_summary_table.csv",
        aggregate_root / "energy_power_summary_aggregate.csv",
    ]
    trace_candidates = [
        aggregate_root / "power_trace_aggregate.csv",
        tables / "power_trace_sample_table.csv",
    ]
    energy_path = _first_existing(energy_candidates)
    trace_path = _first_existing(trace_candidates)
    if energy_path is None:
        return {"status": "skipped", "reason": "energy_power_summary CSV not found", "searched": [str(p) for p in energy_candidates]}
    if trace_path is None:
        return {"status": "skipped", "reason": "power_trace CSV not found", "searched": [str(p) for p in trace_candidates]}

    peaks: Dict[Tuple[str, str, str, str], Tuple[float, float]] = {}
    trace_rows = 0
    for row in _stream_csv(trace_path):
        trace_rows += 1
        key = _group_key(row)
        after = _float(row.get("power_after_dvfs"))
        before = _float(row.get("power_before_dvfs"))
        old_after, old_before = peaks.get(key, (0.0, 0.0))
        peaks[key] = (max(old_after, after), max(old_before, before))

    rows = _read_csv(energy_path)
    changed = 0
    for row in rows:
        key = _group_key(row)
        if key in peaks:
            after, before = peaks[key]
            row["peak_total_system_power_consumption_after_dvfs"] = f"{after:.17g}"
            row["peak_total_system_power_consumption_before_dvfs"] = f"{before:.17g}"
            row["peak_power_consumption_after_dvfs"] = f"{after:.17g}"
            row["peak_power_consumption_before_dvfs"] = f"{before:.17g}"
            changed += 1
    _write_csv(energy_path, rows)
    return {
        "status": "ok",
        "rows_changed": changed,
        "energy_file": str(energy_path),
        "trace_file": str(trace_path),
        "trace_rows_loaded": trace_rows,
        "peak_keys": len(peaks),
    }


def repair_schedule_from_task_level(report_root: Path, *, all_config_output_root: Path | None = None) -> dict:
    """Recomputes schedule count definitions from task-level detailed CSV.

    Works on the aggregate detailed CSVs when the report only contains a sample.
    """

    tables = report_root / "tables"
    aggregate_root = _default_aggregate_root(report_root, all_config_output_root)
    task_path = _first_existing([
        aggregate_root / "task_level_results_aggregate.csv",
        tables / "task_level_results_table.csv",
        tables / "task_level_results_sample_table.csv",
    ])
    schedule_path = _first_existing([
        aggregate_root / "schedule_summary_extended_aggregate.csv",
        tables / "schedule_summary_extended_table.csv",
    ])
    if task_path is None:
        return {"status": "skipped", "reason": "task-level CSV not found"}
    if schedule_path is None:
        return {"status": "skipped", "reason": "schedule summary CSV not found"}

    agg = defaultdict(lambda: {
        "total": 0,
        "executed": 0,
        "met": 0,
        "missed_executed": 0,
        "dropped": 0,
    })
    task_rows = 0
    for row in _stream_csv(task_path):
        task_rows += 1
        key = _group_key(row)
        executed = _float(row.get("executed", row.get("is_executed", row.get("was_executed"))))
        # If no explicit executed flag exists, infer from start/finish/core fields.
        if "executed" not in row and "is_executed" not in row and "was_executed" not in row:
            assigned = str(row.get("assigned_core", row.get("core_id", ""))).strip()
            finish = str(row.get("finish_time", "")).strip()
            executed = 1.0 if assigned not in {"", "nan", "None", "-1"} or finish not in {"", "nan", "None"} else 0.0
        deadline_met = _float(row.get("deadline_met"))
        deadline_miss = _float(row.get("deadline_miss", row.get("deadline_missed")))
        item = agg[key]
        item["total"] += 1
        if executed >= 0.5:
            item["executed"] += 1
        if deadline_met >= 0.5:
            item["met"] += 1
        if executed >= 0.5 and deadline_miss >= 0.5:
            item["missed_executed"] += 1
        if executed < 0.5:
            item["dropped"] += 1

    rows = _read_csv(schedule_path)
    changed = 0
    for row in rows:
        key = _group_key(row)
        if key not in agg:
            continue
        item = agg[key]
        total = int(item["total"])
        executed = int(item["executed"])
        met = int(item["met"])
        missed_executed = int(item["missed_executed"])
        dropped = int(item["dropped"])
        failed = missed_executed + dropped
        mode_switches = _float(row.get("number_of_mode_switches", row.get("mode_switches")))
        row["number_of_total_tasks"] = str(total)
        row["number_of_executed_tasks"] = str(executed)
        row["number_of_deadline_met_tasks"] = str(met)
        row["number_of_deadline_missed_tasks"] = str(missed_executed)
        row["number_of_deadline_missed_executed_tasks"] = str(missed_executed)
        row["number_of_dropped_or_unexecuted_tasks"] = str(dropped)
        row["number_of_failed_or_dropped_tasks"] = str(failed)
        row["deadline_meet_ratio"] = "0" if total == 0 else f"{met / total:.17g}"
        row["deadline_miss_ratio"] = "0" if total == 0 else f"{missed_executed / total:.17g}"
        row["deadline_failure_ratio"] = "0" if total == 0 else f"{failed / total:.17g}"
        row["executed_task_deadline_miss_ratio"] = "0" if executed == 0 else f"{missed_executed / executed:.17g}"
        row["executed_miss_ratio_over_total"] = "0" if total == 0 else f"{missed_executed / total:.17g}"
        row["service_loss_ratio"] = "0" if total == 0 else f"{dropped / total:.17g}"
        row["mode_switch_rate"] = "0" if total == 0 else f"{mode_switches / total:.17g}"
        row["task_deadline_miss_ratio"] = "0" if total == 0 else f"{missed_executed / total:.17g}"
        row["definition_note"] = "deadline_missed_tasks counts only executed tasks that missed; failed_or_dropped includes dropped/unexecuted tasks"
        changed += 1
    _write_csv(schedule_path, rows)
    return {"status": "ok", "rows_changed": changed, "task_rows_loaded": task_rows, "task_file": str(task_path), "schedule_file": str(schedule_path)}


def repair_prediction_metrics(report_root: Path, *, all_config_output_root: Path | None = None) -> dict:
    """Adds stable MAPE/sMAPE and method-type fields from prediction details."""

    tables = report_root / "tables"
    aggregate_root = _default_aggregate_root(report_root, all_config_output_root)
    detail_path = _first_existing([
        aggregate_root / "prediction_detail_results_aggregate.csv",
        tables / "prediction_detail_results_table.csv",
        tables / "prediction_detail_sample_table.csv",
    ])
    summary_path = _first_existing([
        aggregate_root / "prediction_metrics_aggregate.csv",
        tables / "prediction_metrics_table.csv",
    ]) or (tables / "prediction_metrics_table.csv")
    if detail_path is None:
        return {"status": "skipped", "reason": "prediction detail CSV not found"}

    agg = defaultdict(lambda: {"n": 0, "abs": 0.0, "sq": 0.0, "mape_raw": 0.0, "mape_eps": 0.0, "smape": 0.0})
    detail_rows = 0
    for row in _stream_csv(detail_path):
        detail_rows += 1
        key = _group_key(row)
        actual = _float(row.get("actual_execution_time"))
        pred = _float(row.get("predicted_execution_time"))
        err = abs(actual - pred)
        sq = err * err
        raw = 0.0 if abs(actual) <= 1e-12 else err / abs(actual)
        eps = err / max(abs(actual), MAPE_EPSILON)
        smape = 0.0 if abs(actual) + abs(pred) <= 1e-12 else 2.0 * err / (abs(actual) + abs(pred))
        a = agg[key]
        a["n"] += 1
        a["abs"] += err
        a["sq"] += sq
        a["mape_raw"] += raw
        a["mape_eps"] += eps
        a["smape"] += smape

    rows = _read_csv(summary_path) if summary_path.exists() else []
    by_key = {_group_key(row): row for row in rows}
    learned = {"gnn", "proposed_gat_float32", "proposed_gat_dynamic_int8_linear", "proposed_gat_quantization_int8", "proposed_gat_qat_int8_pdf"}
    for key, a in agg.items():
        n = max(1, int(a["n"]))
        row = by_key.get(key)
        if row is None:
            row = {"configuration": key[0], "algorithm": key[1], "application_id": key[2], "graph_id": key[3]}
            rows.append(row)
        row["method_type"] = "learned_predictor" if key[1] in learned else "classical_scheduler_estimate"
        row["MAE"] = f"{a['abs']/n:.17g}"
        row["MSE"] = f"{a['sq']/n:.17g}"
        row["RMSE"] = f"{math.sqrt(a['sq']/n):.17g}"
        row["MAPE_raw"] = f"{a['mape_raw']/n:.17g}"
        row["MAPE_epsilon"] = f"{a['mape_eps']/n:.17g}"
        row["sMAPE"] = f"{a['smape']/n:.17g}"
        row["MAPE"] = f"{a['mape_eps']/n:.17g}"
        row["mape_epsilon"] = f"{MAPE_EPSILON:.17g}"
        row["unit_error"] = "sim_time_unit"
    _write_csv(summary_path, rows)
    return {"status": "ok", "rows_changed": len(rows), "detail_rows_loaded": detail_rows, "detail_file": str(detail_path), "summary_file": str(summary_path)}


def repair_names(report_root: Path) -> dict:
    """Clarifies INT8 and latency names in summary tables without changing values."""

    tables = report_root / "tables"
    changed = 0
    for path in list(tables.glob("*.csv")):
        rows = _read_csv(path)
        if not rows:
            continue
        did = False
        for row in rows:
            alg = row.get("algorithm")
            method = row.get("method_name") or row.get("precision_variant")
            if alg == "proposed_gat_quantization_int8":
                row["legacy_algorithm_name"] = alg
                row["algorithm"] = "proposed_gat_dynamic_int8_linear"
                row["precision_variant"] = "dynamic_int8_linear"
                did = True
            if method:
                row.setdefault("precision_variant", method)
                did = True
            if "scheduler_latency_ms" in row and "scheduler_decision_latency_ms" not in row:
                row["scheduler_decision_latency_ms"] = row["scheduler_latency_ms"]
                row["end_to_end_latency_ms"] = row["scheduler_latency_ms"]
                did = True
            if "latency_ms" in row and "inference_latency_ms" not in row:
                row["inference_latency_ms"] = row["latency_ms"]
                did = True
        if did:
            _write_csv(path, rows)
            changed += 1
    return {"status": "ok", "files_changed": changed}


def main() -> None:
    """Run the main step and return its computed result."""
    parser = argparse.ArgumentParser(description="Repair semantic definitions in existing defense_report/all_config outputs without retraining.")
    parser.add_argument("defense_output_root", type=Path)
    parser.add_argument("--all-config-output-root", type=Path, default=None)
    parser.add_argument("--write-report", type=Path, default=None)
    args = parser.parse_args()

    report = args.defense_output_root
    results = {
        "peak_power": repair_peak_power(report, all_config_output_root=args.all_config_output_root),
        "schedule_counts": repair_schedule_from_task_level(report, all_config_output_root=args.all_config_output_root),
        "prediction_metrics": repair_prediction_metrics(report, all_config_output_root=args.all_config_output_root),
        "names": repair_names(report),
    }
    report_path = args.write_report or report / "semantic_repair_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
