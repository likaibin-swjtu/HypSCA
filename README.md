# HypSCA

Core implementation for **HypSCA: A Hyperbolic Embedding Method for Enhanced Side-Channel Attack**, published in *IEEE Transactions on Information Forensics and Security*, vol. 21, pp. 8661–8675, 2026.

**Authors** · Kaibin Li, Yihuai Liang, Zhengchun Zhou, and Shui Yu

[IEEE journal article](https://doi.org/10.1109/TIFS.2026.3734751) · [IACR ePrint preprint](https://eprint.iacr.org/2025/1199)

HypSCA learns representations of side-channel traces in two spaces. The hyperbolic HIER loss models hierarchical structure, while the Euclidean semi-hard triplet loss separates nearby embeddings from different classes.

## Repository contents

| File | Purpose |
| --- | --- |
| [`hier_loss.py`](hier_loss.py) | TensorFlow/Keras implementation of the hyperbolic HIER loss. |
| [`Triplet_loss.py`](Triplet_loss.py) | Euclidean semi-hard triplet loss. |
| [`autorank_bo_objective.py`](autorank_bo_objective.py) | AutoRank validation objective and trial logging for Bayesian optimization. |
| [`audit_bo_history.py`](audit_bo_history.py) | Summarizes saved hyperparameter search records. |
| [`run_bo_seed_replay.py`](run_bo_seed_replay.py) | Replays search on recorded trial results to compare Bayesian optimization and random search across seeds. |

The repository provides the two loss implementations and selected experiment utilities. The history audit and replay scripts require saved search records. Datasets, trained checkpoints, and experiment outputs are outside the scope of this release.

## Citation

If you use HypSCA or this code in your research, please cite the journal article.

```bibtex
@article{li2026hypsca,
  author  = {Li, Kaibin and Liang, Yihuai and Zhou, Zhengchun and Yu, Shui},
  title   = {{HypSCA}: A Hyperbolic Embedding Method for Enhanced Side-Channel Attack},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  volume  = {21},
  pages   = {8661--8675},
  doi     = {10.1109/TIFS.2026.3734751}
}
```
