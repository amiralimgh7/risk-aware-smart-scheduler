$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  py -3.13 -m venv .venv
}

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
$env:PIP_DEFAULT_TIMEOUT = "300"

python -m pip install --timeout 300 --retries 20 -r requirements_py313_windows.txt
python -m compileall -q .
python -m pytest -q tests/test_phase_1.py tests/test_phase_2_and_3.py

if (Test-Path "outputs") {
  Remove-Item -Recurse -Force "outputs"
}

python -m scripts.generate_pdf_dataset --output-root datasets/pdf_phase_1 --force-regenerate

python -m scripts.run_all_pdf_baselines datasets/pdf_phase_1 `
  --output-root outputs/all_config_runs `
  --num-epochs 3 `
  --policy-episodes 8 `
  --enable-latency-benchmark `
  --force-rerun

python -m scripts.run_pdf_structural_sweep `
  --output-dir outputs/structural_sweep `
  --node-counts 50 100 200 400 800 `
  --core-counts 4 8 16 32 64 `
  --utilization-levels 0.25 0.5 0.75 `
  --graphs-per-setting 10 `
  --num-epochs 2 `
  --policy-episodes 6 `
  --core-reference-count 4 `
  --seed 0

python -m scripts.generate_defense_outputs outputs/all_config_runs `
  --structural-output-root outputs/structural_sweep `
  --output-root outputs/defense_report

python -m scripts.build_defense_summary outputs/all_config_runs `
  --structural-output-root outputs/structural_sweep `
  --defense-output-root outputs/defense_report

python -m scripts.check_output_sanity `
  --all-config-output-root outputs/all_config_runs `
  --structural-output-root outputs/structural_sweep `
  --defense-output-root outputs/defense_report `
  --write-report outputs/defense_report/sanity `
  --minimum-plot-count 50 `
  --fail-on-error
