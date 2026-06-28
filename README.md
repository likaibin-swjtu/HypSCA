# HypSCA

This repository provides the core implementation of the loss components used in **HypSCA: A Hyperbolic Embedding Method for Enhanced Side-Channel Attack**.

HypSCA is a dual-space training objective for profiled deep-learning-based side-channel analysis. It combines:

- a hyperbolic HIER loss for relation-oriented representation learning;
- a Euclidean semi-hard triplet loss for local discriminability.

## Files

```text
hier_loss.py      # Hyperbolic HIER loss layer
Triplet_loss.py   # Euclidean semi-hard triplet loss
