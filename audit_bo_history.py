import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HYP_ROOT = ROOT / "hypsca"
TUNER_ROOT = HYP_ROOT / "root_to_save_the_searching_history"
LOG_ROOT = HYP_ROOT / "SearchLogs"
OUTPUT_ROOT = ROOT / "experiments" / "revision_round3" / "bayesian_optimization"


SCENARIOS = {
    "ASCAD_HW": {
        "project": "ASCAD_HW_BO_my_metric_val_HIER_Triplet",
        "log": "HIER_Triplet_ASCAD_HW_desync0_20251226_100612.json",
        "censor_limit": 10000,
    },
    "ASCAD_ID": {
        "project": "ASCAD_ID_BO_my_metric_val_HIER_Triplet",
        "log": "HIER_Triplet_ASCAD_ID_desync0_20251228_164125.json",
        "censor_limit": 10000,
    },
    "ASCAD_rand_HW": {
        "project": "ASCAD_rand_HW_BO_my_metric_val_HIER_Triplet",
        "log": "HIER_Triplet_ASCAD_rand_HW_desync0_20251231_101507.json",
        "censor_limit": 10000,
    },
    "ASCAD_rand_ID": {
        "project": "ASCAD_rand_ID_BO_my_metric_val_HIER_Triplet",
        "log": "HIER_Triplet_ASCAD_rand_ID_desync0_20260104_151337.json",
        "censor_limit": 10000,
    },
    "AESHD_HW": {
        "project": "AESHD_HW_BO_my_metric_val_HIER_Triplet",
        "log": "HIER_Triplet_AESHD_HW_desync0_20260112_003320.json",
        "censor_limit": 5000,
    },
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_value(value):
    if isinstance(value, float):
        return round(value, 12)
    return value


def config_key(values):
    return tuple(
        sorted((name, normalize_value(value)) for name, value in values.items())
    )


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def serialize_config(key):
    return {name: value for name, value in key}


def extract_space(oracle):
    space = []
    for item in oracle["hyperparameters"]["space"]:
        config = item["config"]
        values = config.get("values")
        if isinstance(values, str):
            values = values.split()
        space.append(
            {
                "name": config["name"],
                "class_name": item["class_name"],
                "values": values,
                "default": config.get("default"),
            }
        )
    return space


def audit_scenario(name, spec):
    project_dir = TUNER_ROOT / spec["project"]
    oracle = load_json(project_dir / "oracle.json")
    external_log = load_json(LOG_ROOT / spec["log"])
    order = oracle.get("end_order") or oracle.get("start_order") or []

    trials = []
    for order_index, trial_id in enumerate(order, start=1):
        trial_path = project_dir / f"trial_{trial_id}" / "trial.json"
        if not trial_path.exists():
            continue
        trial = load_json(trial_path)
        if trial.get("status") != "COMPLETED" or trial.get("score") is None:
            continue
        values = trial["hyperparameters"]["values"]
        trials.append(
            {
                "order": order_index,
                "trial_id": trial_id,
                "score": float(trial["score"]),
                "best_step": int(trial.get("best_step", 0)),
                "values": values,
            }
        )

    if not trials:
        raise RuntimeError(f"No completed trials for {name}")

    scores = np.asarray([trial["score"] for trial in trials], dtype=float)
    cumulative_best = np.minimum.accumulate(scores)
    best_index = int(np.argmin(scores))
    best_trial = trials[best_index]

    groups = defaultdict(list)
    for trial in trials:
        groups[config_key(trial["values"])].append(trial["score"])

    repeated_groups = [values for values in groups.values() if len(values) >= 2]
    stable_groups = [(key, values) for key, values in groups.items() if len(values) >= 3]
    within_stds = [float(np.std(values, ddof=1)) for values in repeated_groups]
    within_iqrs = [
        percentile(values, 75) - percentile(values, 25) for values in repeated_groups
    ]
    if stable_groups:
        best_median_key, best_median_values = min(
            stable_groups, key=lambda item: np.median(item[1])
        )
        best_replicated = {
            "configuration": serialize_config(best_median_key),
            "repetitions": len(best_median_values),
            "median_tge0": float(np.median(best_median_values)),
            "mean_tge0": float(np.mean(best_median_values)),
            "std_tge0": float(np.std(best_median_values, ddof=1)),
            "min_tge0": float(np.min(best_median_values)),
            "max_tge0": float(np.max(best_median_values)),
            "values": [float(value) for value in best_median_values],
        }
    else:
        best_replicated = None

    budgets = [10, 25, 50, 100, 200, 300, 400, 500, 600]
    convergence = {
        str(budget): float(cumulative_best[min(budget, len(trials)) - 1])
        for budget in budgets
        if len(trials) >= min(budget, len(trials))
    }

    log_trials = external_log.get("all_trials", [])
    log_scores = [
        float(item["tge0_result"])
        for item in log_trials
        if isinstance(item.get("tge0_result"), (int, float))
    ]
    timestamp_values = []
    for item in log_trials:
        try:
            timestamp_values.append(
                datetime.strptime(item["timestamp"], "%Y-%m-%d %H:%M:%S")
            )
        except (KeyError, ValueError):
            pass
    wall_hours = None
    if len(timestamp_values) >= 2:
        wall_hours = (
            max(timestamp_values) - min(timestamp_values)
        ).total_seconds() / 3600.0

    space = extract_space(oracle)
    default_initial_points = max(3 * len(space), 3)
    censor_limit = spec["censor_limit"]

    return {
        "scenario": name,
        "project": spec["project"],
        "bo_seed": oracle.get("seed"),
        "surrogate": "GaussianProcessRegressor(Matern nu=2.5)",
        "acquisition": "lower confidence bound: mu - beta * sigma",
        "alpha": oracle.get("alpha"),
        "beta": oracle.get("beta"),
        "num_initial_points_recorded": oracle.get("num_initial_points"),
        "default_initial_points_implied": default_initial_points,
        "search_space": space,
        "completed_trials": len(trials),
        "external_log_trials": len(log_trials),
        "unique_configurations": len(groups),
        "repeated_configurations": len(repeated_groups),
        "max_repetitions_for_one_configuration": max(len(v) for v in groups.values()),
        "wall_clock_hours_from_log": wall_hours,
        "best_oracle_tge0": float(best_trial["score"]),
        "best_oracle_trial_number": best_trial["order"],
        "best_oracle_trial_id": best_trial["trial_id"],
        "best_oracle_best_epoch_zero_based": best_trial["best_step"],
        "best_oracle_configuration": best_trial["values"],
        "median_trial_tge0": float(np.median(scores)),
        "q25_trial_tge0": percentile(scores, 25),
        "q75_trial_tge0": percentile(scores, 75),
        "censored_trial_fraction": float(np.mean(scores >= censor_limit)),
        "extreme_sentinel_count_oracle": int(np.sum(scores > censor_limit)),
        "external_log_best_tge0": float(np.min(log_scores)) if log_scores else None,
        "external_log_extreme_sentinel_count": int(
            np.sum(np.asarray(log_scores) > censor_limit)
        )
        if log_scores
        else 0,
        "median_within_configuration_std": float(np.median(within_stds))
        if within_stds
        else None,
        "median_within_configuration_iqr": float(np.median(within_iqrs))
        if within_iqrs
        else None,
        "best_replicated_configuration": best_replicated,
        "cumulative_best_by_budget": convergence,
    }


def write_markdown(results):
    lines = [
        "# Historical Bayesian-Optimization Audit",
        "",
        "The values below come from KerasTuner's `trial.json` files, which are the",
        "scores actually consumed by the Bayesian oracle.",
        "",
        "| Scenario | BO seed | Trials | Unique configs | Best TGE0 | Best trial | Median within-config SD | Wall time (h) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        within_sd = result["median_within_configuration_std"]
        wall_hours = result["wall_clock_hours_from_log"]
        lines.append(
            "| {scenario} | {seed} | {trials} | {unique} | {best:.1f} | {best_trial} | {sd} | {wall} |".format(
                scenario=result["scenario"],
                seed=result["bo_seed"],
                trials=result["completed_trials"],
                unique=result["unique_configurations"],
                best=result["best_oracle_tge0"],
                best_trial=result["best_oracle_trial_number"],
                sd=f"{within_sd:.1f}" if within_sd is not None else "NA",
                wall=f"{wall_hours:.1f}" if wall_hours is not None else "NA",
            )
        )

    lines.extend(
        [
            "",
            "## Reproducibility Findings",
            "",
            "1. The historical implementation used one monolithic BO search per scenario, not three separately seeded BO phases.",
            "2. The oracle used a Gaussian process with a Matern-2.5 kernel and the lower-confidence-bound objective `mu - 2.6 sigma`.",
            "3. `num_initial_points` was not set explicitly; KerasTuner therefore used three times the number of registered hyperparameters.",
            "4. Each scenario has only one BO-oracle seed. Per-trial model seeds do not constitute independent BO runs.",
            "5. Repeated configurations provide direct evidence of training/objective noise and should be summarized instead of quoting a single best trial.",
            "6. The callback returns zero-based TGE0 during search, whereas final attack tables use one-based trace counts.",
            "7. The callback contains an extreme sentinel for an all-zero rank curve; sentinel counts are included in the JSON audit.",
            "",
            "## Reviewer-Facing Assessment",
            "",
            "The existing history is sufficient to disclose the real budget, surrogate, acquisition rule, initialization default, and within-configuration variance. It is not sufficient to claim robustness across independent BO seeds. A new lightweight seeded replay or new seeded searches are required for that narrow claim.",
            "",
        ]
    )
    (OUTPUT_ROOT / "historical_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [
        audit_scenario(name, spec) for name, spec in SCENARIOS.items()
    ]
    (OUTPUT_ROOT / "historical_audit.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    with (OUTPUT_ROOT / "historical_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "bo_seed",
                "completed_trials",
                "unique_configurations",
                "repeated_configurations",
                "best_oracle_tge0",
                "best_oracle_trial_number",
                "median_trial_tge0",
                "censored_trial_fraction",
                "median_within_configuration_std",
                "median_within_configuration_iqr",
                "wall_clock_hours_from_log",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in writer.fieldnames})

    write_markdown(results)
    print(json.dumps(results, indent=2))
    print(f"Saved audit to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
