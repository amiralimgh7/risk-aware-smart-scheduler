# Risk-Aware Smart Scheduler

BSc project by **Seyed Amirali Moghadasi**, Sharif University of Technology, Computer Engineering. A reproducible Python simulator for mixed-criticality DAG scheduling on a heterogeneous big.LITTLE platform.

The proposed scheduler combines a graph attention predictor with a reinforcement learning policy, uncertainty and CVaR estimates, reliability and aging models, and DVFS-aware energy accounting. Comparisons include HEFT, EDF-VD, NSGA-III and a classic GCN baseline. Quantization utilities compare FP32, QAT/INT8 and FP16 representations.

## Run a small example

Python 3.12 or 3.13; CPU execution is supported. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m examples.run_end_to_end
```

Use `python -m ...` from the root so sibling packages resolve. For script entry points, set `PYTHONPATH` to the root (`export PYTHONPATH="$PWD"` on Linux, `$env:PYTHONPATH=(Get-Location).Path` in PowerShell). `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` make small CPU runs faster.

## Structure

| Directory | Purpose |
|---|---|
| `generators`, `dataset_tools`, `core` | Workloads, serialization and task graphs |
| `baselines`, `models`, `training` | Reference schedulers, GAT/GCN and learning |
| `hardware_model`, `risk_modeling`, `simulator` | Timing, energy, reliability, risk and scheduling |
| `experiments`, `scripts` | Sweeps, aggregation and defense reports |
| `examples`, `tests`, `docs` | Small demo, regression tests and full-run guides |

## Full experiments

`docs/RUN_FULL_WINDOWS.ps1`, `RUN_REPORT_ONLY_WINDOWS.ps1` and `RUN_TESTS_WINDOWS.ps1` resolve paths relative to the checkout. Full sweeps are substantially larger than the demo. See [the original delivery guide](docs/FINAL_DELIVERY.md) and the run scripts for their exact arguments. Generated datasets, checkpoints and reports belong under `outputs/` and are ignored.

## Validation and scope

The complete 15-test suite covers DAG validity/utilization, mixed-criticality simulation, serialization, baseline and learning checkpoints, quantization size and CLI integration. The small end-to-end example is also executed during CI. These are simulator results; no physical processor energy or aging measurements are claimed. A full research sweep must be rerun to obtain experiment tables for a particular configuration.
