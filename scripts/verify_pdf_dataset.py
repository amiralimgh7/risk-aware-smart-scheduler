"""Command-line entry points for dataset generation, experiments, reporting, repair, and sanity checks.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from dataset_tools.serialization import load_json, save_json
from experiments.pipeline import PDF_DENSITIES, PDF_GRAPH_COUNTS, PDF_REGULARITIES, PDF_WIDTHS, build_pdf_configuration_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser for dataset verification."""

    parser = argparse.ArgumentParser(description="Verify that the generated phase-1 dataset matches the PDF grid.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("outputs") / "dataset_verification.json")
    return parser


def main() -> None:
    """Verifies directory presence and application counts."""

    args = build_argument_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    missing_configurations: List[str] = []
    mismatched_counts: List[Dict[str, str | int]] = []
    found_graph_counts: List[int] = []

    for graph_count in PDF_GRAPH_COUNTS:
        found_graph_counts.append(graph_count)
        for density in PDF_DENSITIES:
            for regularity in PDF_REGULARITIES:
                for width in PDF_WIDTHS:
                    configuration_dir = build_pdf_configuration_path(dataset_root, graph_count, density, regularity, width)
                    manifest_path = configuration_dir / "manifest.json"
                    applications_dir = configuration_dir / "applications"
                    application_files = sorted(applications_dir.glob("application_*.json"))
                    if not manifest_path.exists():
                        missing_configurations.append(str(configuration_dir))
                        continue
                    manifest = load_json(manifest_path)
                    if len(application_files) != graph_count:
                        mismatched_counts.append(
                            {
                                "configuration": str(configuration_dir),
                                "expected_graph_count": graph_count,
                                "actual_application_files": len(application_files),
                                "manifest_generated_files": int(manifest.get("generated_files", -1)),
                            }
                        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "dataset_root": str(dataset_root),
            "expected_graph_counts": found_graph_counts,
            "missing_configurations": missing_configurations,
            "mismatched_counts": mismatched_counts,
            "num_missing_configurations": len(missing_configurations),
            "num_mismatched_counts": len(mismatched_counts),
            "is_valid": len(missing_configurations) == 0 and len(mismatched_counts) == 0,
        },
        args.output_json,
    )

    print(f"dataset_root={dataset_root}")
    print(f"num_missing_configurations={len(missing_configurations)}")
    print(f"num_mismatched_counts={len(mismatched_counts)}")
    print(f"report_path={args.output_json.resolve()}")


if __name__ == "__main__":
    main()
