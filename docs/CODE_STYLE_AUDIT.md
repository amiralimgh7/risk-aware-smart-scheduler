# Code Style Audit

The source package was checked for delivery-oriented code style requirements.

## Naming

- Python module and file names use `snake_case`.
- Functions and variables use `snake_case`.
- Classes use `PascalCase`.
- Command-line scripts use descriptive `snake_case` names.

## Documentation

- Python modules include module-level docstrings.
- Public functions and classes include docstrings.
- Delivery documentation is English-first.
- Persian documentation is included only as supplementary material.

## Packaging Hygiene

The following generated or temporary files are excluded from the delivery package:

- `__pycache__/`
- `.pytest_cache/`
- local virtual environments
- temporary report-generation files
- large generated output folders
- machine-specific paths

## Machine-Specific Path Check

Delivery scripts resolve the repository root from their own location or the current working directory. They do not require absolute paths tied to a specific computer.
