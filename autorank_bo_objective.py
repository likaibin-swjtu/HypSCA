"""AutoRank validation objective used by the HypSCA BO pipeline.

This is the cleaned release version of the AutoRank logic from the internal
``auto_rank_hier_new.py`` script.  It documents the objective used during
Bayesian optimization (BO): after each training epoch, the current encoder is
evaluated by a template attack on the validation/attack split, and the first
trace index where the mean key rank becomes zero is written to
``logs["my_metric_val"]``.  KerasTuner then minimizes this scalar.

The code is intentionally limited to the reproducibility-critical pieces:

1. converting a guessing-entropy curve into the zero-based TGE0 score used by
   BO;
2. the Keras callback that computes ``my_metric_val``;
3. a compact trial logger for auditing BO histories.

Dataset loading, model construction, and the HIER/triplet losses are provided
separately in the released core code.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import tensorflow as tf
    _KerasCallbackBase = tf.keras.callbacks.Callback
except ImportError:  # Allows helper tests without a TensorFlow install.
    tf = None

    class _KerasCallbackBase:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


KERAS_TUNER_OBJECTIVE = "my_metric_val"


@dataclass(frozen=True)
class AutoRankConfig:
    """Configuration for the key-rank validation objective."""

    nb_traces_attacks: int
    correct_key: int
    dataset: str = "ASCAD"
    nb_attacks: int = 10
    predict_batch_size: int = 2048
    failure_score: int = 10000
    all_zero_rank_sentinel: int = 1000000999999


def set_reproducible_seed(seed: int) -> None:
    """Set the random seeds used by Python, NumPy, and TensorFlow."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)


def tge0_from_rank_curve(
    mean_rank_curve: Sequence[float],
    *,
    failure_score: int = 10000,
    all_zero_rank_sentinel: int = 1000000999999,
) -> int:
    """Return the zero-based TGE0 score used as the BO objective.

    The historical BO callback used a zero-based index: if the first trace is
    already sufficient, the score is 0.  Final paper tables may convert this to
    one-based trace counts depending on the reporting convention.

    If the curve never reaches key rank 0 within the evaluated attack budget,
    ``failure_score`` is returned.  The large sentinel preserves the original
    audit behavior for an all-zero curve, which normally indicates a malformed
    rank curve rather than a valid immediate success.
    """

    curve = np.asarray(mean_rank_curve, dtype=np.float64)
    if curve.size == 0:
        return int(failure_score)
    if np.all(curve == 0):
        return int(all_zero_rank_sentinel)
    if curve[-1] > 0:
        return int(failure_score)
    return int(np.argmax(curve < 1))


class AutoRankCallback(_KerasCallbackBase):
    """Keras callback that writes the AutoRank score to ``my_metric_val``.

    Parameters
    ----------
    validation:
        Five-element tuple ``(X_profile, y_profile, X_attack, y_attack,
        attack_metadata)``.  ``attack_metadata`` is the plaintext/ciphertext
        array required by the side-channel attack routine.
    config:
        Attack-evaluation settings.
    template_training:
        Function with signature ``(embeddings, labels, pool=False)`` returning
        class means, class covariance matrices, and class ids.
    template_attacking_proba:
        Function that converts attack embeddings to class probabilities.
    perform_attacks:
        Function that returns ``(mean_rank_curve, key_log_prob)``.
    """

    def __init__(
        self,
        validation: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        config: AutoRankConfig,
        *,
        template_training: Callable[..., Tuple[Any, Any, Any]],
        template_attacking_proba: Callable[..., np.ndarray],
        perform_attacks: Callable[..., Tuple[np.ndarray, Any]],
    ) -> None:
        if tf is None:
            raise ImportError("TensorFlow is required to use AutoRankCallback.")
        super().__init__()
        self.validation = validation
        self.config = config
        self.template_training = template_training
        self.template_attacking_proba = template_attacking_proba
        self.perform_attacks = perform_attacks
        self.last_tge0: Optional[int] = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs if logs is not None else {}
        logs[KERAS_TUNER_OBJECTIVE] = float("inf")

        x_profile, y_profile, x_attack, _y_attack, attack_metadata = self.validation

        profile_embeddings = self.model.predict(
            x_profile,
            batch_size=self.config.predict_batch_size,
            verbose=0,
        )
        attack_embeddings = self.model.predict(
            x_attack,
            batch_size=self.config.predict_batch_size,
            verbose=0,
        )

        means, covariances, classes = self.template_training(
            profile_embeddings,
            y_profile.astype(np.int32),
            pool=False,
        )
        class_probabilities = self.template_attacking_proba(
            means,
            covariances,
            attack_embeddings,
            classes,
        )
        mean_rank_curve, _key_log_prob = self.perform_attacks(
            self.config.nb_traces_attacks,
            class_probabilities,
            self.config.correct_key,
            attack_metadata,
            log=True,
            dataset=self.config.dataset,
            nb_attacks=self.config.nb_attacks,
        )

        tge0 = tge0_from_rank_curve(
            mean_rank_curve,
            failure_score=self.config.failure_score,
            all_zero_rank_sentinel=self.config.all_zero_rank_sentinel,
        )
        self.last_tge0 = tge0
        logs[KERAS_TUNER_OBJECTIVE] = float(tge0)


class SearchResultLogger:
    """Small JSON/text logger for BO trial scores."""

    def __init__(self, log_dir: Union[str, Path], dataset: str, leakage_model: str) -> None:
        self.log_dir = Path(log_dir)
        self.dataset = dataset
        self.leakage_model = leakage_model
        self.records: List[Dict[str, Any]] = []
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_trial_result(
        self,
        trial_number: int,
        parameters: Mapping[str, Any],
        tge0_result: Union[float, int],
    ) -> None:
        record = {
            "trial_number": int(trial_number),
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "dataset": self.dataset,
            "leakage_model": self.leakage_model,
            "parameters": _to_jsonable(dict(parameters)),
            "tge0_result": _to_jsonable(tge0_result),
        }
        self.records.append(record)
        with (self.log_dir / "autorank_trials.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_summary(self, path: Union[str, Path], top_k: int = 20) -> None:
        ranked = sorted(
            self.records,
            key=lambda item: float(item["tge0_result"]),
        )
        summary = {
            "dataset": self.dataset,
            "leakage_model": self.leakage_model,
            "objective": KERAS_TUNER_OBJECTIVE,
            "records": self.records,
            "top_results": ranked[:top_k],
        }
        Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def extract_autorank_callback(callbacks: Iterable[Any]) -> Optional[AutoRankCallback]:
    """Return the AutoRank callback from a Keras callback list, if present."""

    for callback in callbacks:
        if isinstance(callback, AutoRankCallback):
            return callback
    return None


def trial_parameters(trial: Any, names: Sequence[str]) -> Dict[str, Any]:
    """Extract selected hyperparameters from a KerasTuner trial object."""

    params: Dict[str, Any] = {}
    for name in names:
        try:
            params[name] = trial.hyperparameters.get(name)
        except Exception:
            params[name] = None
    return params


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if value == float("inf"):
        return "inf"
    return value


def example_keras_tuner_objective() -> str:
    """Return the objective name to use in KerasTuner."""

    return KERAS_TUNER_OBJECTIVE


def example_autorank_config() -> Dict[str, Any]:
    """Minimal ASCAD-style configuration example for documentation."""

    return asdict(
        AutoRankConfig(
            nb_traces_attacks=10000,
            correct_key=224,
            dataset="ASCAD",
            nb_attacks=10,
        )
    )
