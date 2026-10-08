# Final Delivery Notes

This package contains the cleaned source code for the mixed-criticality DAG scheduling project.

## Contents

- Python implementation of the proposed scheduler and all baseline schedulers.
- Workload generation and structural-sweep scripts.
- Training, simulation, aggregation, and reporting pipelines.
- Quantization evaluation utilities.
- Tests and code-quality audit metadata.
- Portable Windows PowerShell scripts for full execution, report-only execution, and tests.

## Excluded Content

The package intentionally excludes generated heavy outputs, temporary files, caches, virtual environments, and machine-specific files.

## Recommended Review Flow

1. Create a virtual environment.
2. Install dependencies from `requirements_py313_windows.txt`.
3. Run `docs/RUN_TESTS_WINDOWS.ps1`.
4. Inspect `docs/REFERENCE_MAPPING.md` and `docs/FORMULA_AND_OUTPUT_CONVENTIONS.md`.
5. Run the full or report-only pipeline if outputs are required.

## Expected Test Result

The main regression tests are expected to report:

```text
9 passed
```
