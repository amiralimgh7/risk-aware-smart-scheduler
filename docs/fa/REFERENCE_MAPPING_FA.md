# نگاشت روش‌ها به منابع علمی

این فایل برای جلوگیری از ارجاع‌های بی‌ربط یا اضافی تهیه شده است. منابع زیر همان منابع اصلی یا مستقیم روش‌هایی هستند که در کد استفاده شده‌اند.

## HEFT

- H. Topcuoglu, S. Hariri, and M.-Y. Wu, “Performance-effective and low-complexity task scheduling for heterogeneous computing,” IEEE Transactions on Parallel and Distributed Systems, 2002.

کاربرد در کد:

```text
baselines/heft.py
```

## EDF-VD و مدل بحرانی-مختلط

- S. Baruah, V. Bonifaci, G. D'Angelo, H. Li, A. Marchetti-Spaccamela, S. van der Ster, and L. Stougie, “The preemptive uniprocessor scheduling of mixed-criticality implicit-deadline sporadic task systems,” ECRTS, 2012.
- D. Liu, J. Spasic, N. Guan, G. Chen, S. Liu, T. Stefanov, and W. Yi, “EDF-VD Scheduling of Mixed-Criticality Systems with Degraded Quality Guarantees,” RTSS, 2016.

کاربرد در کد:

```text
baselines/vd_edf.py
```

## NSGA-III

- K. Deb and H. Jain, “An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints,” IEEE Transactions on Evolutionary Computation, 2014.

کاربرد در کد:

```text
baselines/nsga_iii.py
```

## GCN

- T. N. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,” ICLR, 2017.

کاربرد در کد:

```text
models/gcn_model.py
baselines/classic_gcn.py
```

## GAT

- P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, “Graph Attention Networks,” ICLR, 2018.

کاربرد در کد:

```text
models/gat_risk.py
```

## UUniFast

- E. Bini and G. C. Buttazzo, “Measuring the performance of schedulability tests,” Real-Time Systems, 2005.

کاربرد در کد:

```text
generators/uunifast.py
```

## DVFS

- P. Pillai and K. G. Shin, “Real-Time Dynamic Voltage Scaling for Low-Power Embedded Operating Systems,” SOSP, 2001.

کاربرد در کد:

```text
hardware_model/hardware.py
simulator/list_scheduling.py
```

## فرسودگی و قابلیت اطمینان

- J. Srinivasan, S. V. Adve, P. Bose, and J. A. Rivers, “The Case for Lifetime Reliability-Aware Microprocessors,” ISCA, 2004.

کاربرد در کد:

```text
hardware_model/hardware.py
```

## CVaR

- R. T. Rockafellar and S. Uryasev, “Optimization of Conditional Value-at-Risk,” Journal of Risk, 2000.

کاربرد در کد:

```text
risk_modeling/risk_models.py
```

## REINFORCE

- R. J. Williams, “Simple statistical gradient-following algorithms for connectionist reinforcement learning,” Machine Learning, 1992.

کاربرد در کد:

```text
training/rl_policy.py
```

## QAT و Quantization

- B. Jacob et al., “Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference,” CVPR, 2018.

کاربرد در کد:

```text
models/quantization.py
models/quantization_eval.py
```
