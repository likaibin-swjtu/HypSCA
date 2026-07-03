import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = (
    ROOT
    / "hypsca"
    / "root_to_save_the_searching_history"
    / "ASCAD_HW_BO_my_metric_val_HIER_Triplet"
)
OUTPUT_DIR = (
    ROOT / "experiments" / "revision_round3" / "bayesian_optimization" / "seed_replay"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20260614, 20260634)))
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--initial-points", type=int, default=21)
    parser.add_argument("--beta", type=float, default=2.6)
    parser.add_argument("--alpha", type=float, default=1e-4)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(value):
    return round(value, 12) if isinstance(value, float) else value


def key_from_values(values):
    return tuple(sorted((name, normalize(value)) for name, value in values.items()))


def values_from_key(key):
    return {name: value for name, value in key}


def load_historical_surface():
    oracle = load_json(PROJECT_DIR / "oracle.json")
    order = oracle["end_order"]
    distributions = defaultdict(list)
    for trial_id in order:
        path = PROJECT_DIR / f"trial_{trial_id}" / "trial.json"
        if not path.exists():
            continue
        trial = load_json(path)
        if trial.get("status") != "COMPLETED" or trial.get("score") is None:
            continue
        distributions[key_from_values(trial["hyperparameters"]["values"])].append(
            float(trial["score"])
        )

    space = {}
    ordered_names = []
    for item in oracle["hyperparameters"]["space"]:
        config = item["config"]
        ordered_names.append(config["name"])
        values = config["values"]
        if isinstance(values, str):
            values = values.split()
        space[config["name"]] = [normalize(value) for value in values]

    keys = sorted(distributions, key=str)
    encoded = []
    for key in keys:
        values = values_from_key(key)
        vector = []
        for name in ordered_names:
            choices = space[name]
            index = choices.index(normalize(values[name]))
            vector.append((index + 0.5) / len(choices))
        encoded.append(vector)

    return keys, np.asarray(encoded, dtype=float), distributions, ordered_names


def observe(distributions, key, rng):
    values = distributions[key]
    return float(values[rng.integers(0, len(values))])


def run_bo(keys, encoded, distributions, seed, budget, initial_points, beta, alpha):
    rng = np.random.default_rng(seed)
    budget = min(budget, len(keys))
    initial_points = min(initial_points, budget)
    selected = [
        int(index)
        for index in rng.choice(len(keys), size=initial_points, replace=False)
    ]
    observations = [observe(distributions, keys[index], rng) for index in selected]

    while len(selected) < budget:
        model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=seed + len(selected),
        )
        model.fit(encoded[selected], np.asarray(observations))
        remaining = np.asarray(
            [index for index in range(len(keys)) if index not in set(selected)],
            dtype=int,
        )
        mean, std = model.predict(encoded[remaining], return_std=True)
        acquisition = mean - beta * std
        next_index = int(remaining[int(np.argmin(acquisition))])
        selected.append(next_index)
        observations.append(observe(distributions, keys[next_index], rng))

    best_position = int(np.argmin(observations))
    best_index = selected[best_position]
    robust_medians = [
        float(np.median(distributions[keys[index]])) for index in selected
    ]
    robust_position = int(np.argmin(robust_medians))
    robust_index = selected[robust_position]
    cumulative_best = np.minimum.accumulate(np.asarray(observations))

    return {
        "seed": seed,
        "best_observed_tge0": float(observations[best_position]),
        "best_observed_configuration": values_from_key(keys[best_index]),
        "best_observed_configuration_historical_median": float(
            np.median(distributions[keys[best_index]])
        ),
        "best_robust_historical_median_tge0": robust_medians[robust_position],
        "best_robust_configuration": values_from_key(keys[robust_index]),
        "best_robust_configuration_repetitions": len(distributions[keys[robust_index]]),
        "cumulative_best_at_25": float(cumulative_best[min(25, len(cumulative_best)) - 1]),
        "cumulative_best_at_50": float(cumulative_best[-1]),
        "selected_indices": selected,
        "observations": observations,
    }


def run_random(keys, distributions, seed, budget):
    rng = np.random.default_rng(seed)
    budget = min(budget, len(keys))
    selected = [
        int(index) for index in rng.choice(len(keys), size=budget, replace=False)
    ]
    observations = [observe(distributions, keys[index], rng) for index in selected]
    robust_medians = [
        float(np.median(distributions[keys[index]])) for index in selected
    ]
    return {
        "seed": seed,
        "best_observed_tge0": float(np.min(observations)),
        "best_robust_historical_median_tge0": float(np.min(robust_medians)),
    }


def summarize(results):
    observed = np.asarray([item["best_observed_tge0"] for item in results])
    robust = np.asarray(
        [item["best_robust_historical_median_tge0"] for item in results]
    )
    return {
        "runs": len(results),
        "best_observed_mean": float(np.mean(observed)),
        "best_observed_std": float(np.std(observed, ddof=1)),
        "best_observed_min": float(np.min(observed)),
        "best_observed_max": float(np.max(observed)),
        "robust_selected_median_mean": float(np.mean(robust)),
        "robust_selected_median_std": float(np.std(robust, ddof=1)),
        "robust_selected_median_min": float(np.min(robust)),
        "robust_selected_median_max": float(np.max(robust)),
    }


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keys, encoded, distributions, ordered_names = load_historical_surface()
    bo_results = [
        run_bo(
            keys,
            encoded,
            distributions,
            seed,
            args.budget,
            args.initial_points,
            args.beta,
            args.alpha,
        )
        for seed in args.seeds
    ]
    random_results = [
        run_random(keys, distributions, seed, args.budget) for seed in args.seeds
    ]
    output = {
        "experiment_type": "offline finite-space seeded replay",
        "scope": "ASCAD-F/HW historical objective surface",
        "candidate_configurations": len(keys),
        "ordered_hyperparameters": ordered_names,
        "budget": min(args.budget, len(keys)),
        "initial_points": args.initial_points,
        "surrogate": "GaussianProcessRegressor(Matern nu=2.5)",
        "acquisition": f"mu - {args.beta} * sigma",
        "alpha": args.alpha,
        "objective_noise": (
            "One historical trial score is sampled from each selected "
            "configuration's empirical repetition distribution."
        ),
        "limitation": (
            "This replay tests BO initialization and observed objective noise. "
            "It is not a replacement for independently retraining the model."
        ),
        "bo_summary": summarize(bo_results),
        "random_summary": summarize(random_results),
        "bo_runs": bo_results,
        "random_runs": random_results,
    }
    alpha_tag = str(args.alpha).replace(".", "p")
    result_path = OUTPUT_DIR / f"results_alpha_{alpha_tag}.json"
    result_path.write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: output[key] for key in [
        "experiment_type",
        "candidate_configurations",
        "budget",
        "initial_points",
        "bo_summary",
        "random_summary",
    ]}, indent=2))
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
