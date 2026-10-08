# Reference Mapping

This document records the intended primary references for the main methods used in the project.

## Baseline Scheduling Methods

- **HEFT**: Heterogeneous Earliest Finish Time is mapped to the original work by Topcuoglu, Hariri, and Wu on performance-effective and low-complexity task scheduling for heterogeneous computing.
- **EDF-VD**: Earliest Deadline First with Virtual Deadlines is mapped to the mixed-criticality scheduling literature by Baruah and collaborators. The degraded-quality mixed-criticality extension is mapped to Liu et al.'s EDF-VD scheduling of mixed-criticality systems with degraded quality guarantees.
- **NSGA-III**: The many-objective optimization baseline is mapped to Deb and Jain's NSGA-III reference-point-based nondominated sorting algorithm.
- **Classical GCN**: The graph convolutional baseline is mapped to Kipf and Welling's semi-supervised graph convolutional network formulation.

## Proposed and Supporting Methods

- **GAT**: The proposed graph-attention encoder is mapped to Veličković et al.'s Graph Attention Networks.
- **CVaR**: Tail-risk modeling is mapped to Rockafellar and Uryasev's Conditional Value-at-Risk formulation.
- **REINFORCE**: Policy-gradient training is mapped to Williams' REINFORCE algorithm.
- **DVFS**: Dynamic Voltage and Frequency Scaling is mapped to classic low-power real-time scheduling and energy-aware systems literature.
- **QAT**: Quantization-aware training and integer inference are mapped to Jacob et al.'s work on quantization and training of neural networks for efficient integer-arithmetic-only inference.
- **UUniFast**: Utilization-vector generation is mapped to Bini and Buttazzo's UUniFast approach.

## Reporting Policy

Only references that are relevant to methods actually discussed in the report should be cited. Unused BibTeX entries are not needed in the final thesis source.
