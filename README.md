# HypSCA

This repository provides the core implementation of **HypSCA: A Hyperbolic Embedding Method for Enhanced Side-Channel Attack**.

HypSCA is a dual-space training objective for profiled deep-learning-based side-channel analysis. It combines:

- a hyperbolic HIER loss for hierarchy-aware representation learning;
- a Euclidean semi-hard triplet loss for local discriminability.

The repository focuses on the key method components and reproducibility helpers. Datasets, trained checkpoints, and large experiment outputs are not included.

## Files

```text
hier_loss.py                  # Hyperbolic HIER loss layer
Triplet_loss.py               # Euclidean semi-hard triplet loss
autorank_bo_objective.py      # AutoRank/key-rank validation objective used in BO
audit_bo_history.py           # Audit script for historical BO search records
run_bo_seed_replay.py         # Seed-replay script over saved BO search histories
