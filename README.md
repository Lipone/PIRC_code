# PIRC: Partitionally Intervened Reservoir Computing

This repository contains the source code and datasets associated with the manuscript:

**Detecting Interventional Higher-Order Causality in Complex Systems**

Ren Cao, Haoran Sun, and Siyang Leng

## Overview

This repository provides the implementation of **Partitionally Intervened Reservoir Computing (PIRC)**, an intervention-based framework for detecting higher-order causal relationships directly from observational time-series data.

PIRC first reconstructs the underlying system dynamics using a partitioned reservoir computing architecture. Each candidate source-node set is associated with an independent reservoir partition. After dynamical reconstruction, partition-specific interventions are performed by selectively masking individual reservoir partitions. The resulting deviations from the baseline trajectory are used to evaluate the causal contributions of the corresponding candidate source-node sets.

The framework does not require explicit knowledge of the governing equations for causal inference.

## Requirements

The code is implemented in Python.

Please install the required Python packages before running the experiments.

The main dependencies include:

- Python
- NumPy
- SciPy
- PyTorch
- scikit-learn
- pandas
- matplotlib

The exact package versions used in the experiments can be found in the environment/requirements file provided in this repository, if applicable.

## Repository Structure

The repository contains the implementation and datasets required to reproduce the main experiments reported in the manuscript.

The experiments include:

- Lorenz63 system
- Coupled Lorenz63 network
- Higher-order Kuramoto networks
- UK power grid
- Resting-state EEG recordings

Each experiment applies PIRC to observational time-series data and evaluates the inferred higher-order causal relationships.

## Method

For a target node, PIRC constructs structured representations for all candidate source-node sets and maps each representation to an independent reservoir partition.

The complete reservoir state is used to reconstruct the target dynamics through a linear readout layer. After training, the reservoir partition associated with a candidate source-node set is selectively masked while all other model parameters remain unchanged.

The causal contribution of a candidate source-node set is evaluated from the deviation between the baseline and intervened trajectories.

Unless otherwise specified, the prediction horizon is set to:

```text
N_pred = 10
