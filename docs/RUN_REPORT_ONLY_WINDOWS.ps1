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

if (-not (Test-Path "outputs/all_config_runs")) {
  throw "outputs/all_config_runs was not found. Run the full pipeline first or copy the previous outputs."
}

if (-not (Test-Path "outputs/structural_sweep")) {
  throw "outputs/structural_sweep was not found. Run the structural sweep first or copy the previous outputs."
}

if (Test-Path "outputs/defense_report") {
  Remove-Item -Recurse -Force "outputs/defense_report"
}

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
