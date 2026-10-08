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
