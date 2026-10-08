# Formula and Output Conventions

## Deadline Miss Ratio

Throughout the project, `DMR` means `Deadline Miss Ratio`:

```text
deadline_miss_ratio = number_of_deadline_missed_tasks / number_of_total_real_tasks
```

Dropped or unserved low-criticality tasks are reported separately as service loss.

## Mode Switch Accounting

A mixed-criticality mode switch is triggered when a high-criticality task exceeds its low-criticality execution budget. The simulator accounts for:

- mode-switch occurrence count,
- mode-switch time overhead,
- mode-switch power overhead,
- mode-switch energy overhead.

## DVFS Energy Comparison

Before/after DVFS comparisons use the same realized execution and schedule. The before-DVFS case evaluates the counterfactual energy of running the same execution at the maximum DVFS level. This prevents unfair comparisons caused by resampling runtime values.

## Reliability and Aging

The reliability model uses a time-dependent failure rate and integrates it over the execution interval. The aging index aggregates core failure-rate stress with core load.

## CVaR

CVaR is used to model upper-tail execution-time risk. Empirical CVaR is computed from the worst tail samples, with the tail size rounded up to avoid undercounting extreme cases.

## Quantization Reporting

The final report includes only scientifically defensible quantization variants:

- FP32 reference,
- QAT INT8,
- weight-only INT8 per channel,
- FP16 mixed precision.

Dynamic INT8 Linear is excluded from the final report because the available output rows showed incomplete execution behavior and artificial zero-valued metrics.
