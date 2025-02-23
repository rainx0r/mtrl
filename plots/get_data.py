import json
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.stats
import wandb

CACHE_DIR = Path(__file__).parent.parent / "cache"


def get_metric(
    entity: str,
    project: str,
    run_name: str,
    metric: str,
    source: Literal["summary", "config"] = "summary",
) -> list[float]:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{entity}_{project}_{run_name}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            data = json.load(f)
    else:
        print(f"No cache hit for {entity}/{project}/{run_name}, downloading...")
        api = wandb.Api(overrides={"entity": entity})
        runs = api.runs(
            project, filters={"config.exp_name": run_name, "state": "finished"}
        )
        data = [
            {"summary": dict(run.summary._json_dict), "config": dict(run.config)}
            for run in runs
        ]
        with open(cache_file, "w") as f:
            json.dump(data, f)

    return [run[source][metric] for run in data]


def get_metric_history(
    entity: str,
    project: str,
    run_name: str,
    metric: str,
    samples: int = 500,
) -> list[dict[int, float]]:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = (
        CACHE_DIR
        / f"{entity}_{project}_{run_name}_{metric.replace('/', '_')}_history.json"
    )
    if cache_file.exists():
        with open(cache_file, "r") as f:
            data = json.load(f)
    else:
        print(
            f"No cache hit for {entity}/{project}/{run_name} - {metric}, downloading..."
        )
        api = wandb.Api(overrides={"entity": entity})
        runs = api.runs(
            project, filters={"config.exp_name": run_name, "state": "finished"}
        )
        data = []
        for run in runs:
            history = run.history(samples=samples, keys=[metric], pandas=False)
            history = {int(item["_step"]): item[metric] for item in history}
            data.append(history)
        with open(cache_file, "w") as f:
            json.dump(data, f)

    return data

def iqm(scores: list[float]) -> float:
    return scipy.stats.trim_mean(scores, proportiontocut=0.25)

def compute_ci(scores: list[float]) -> tuple[float, float]:
    n = len(scores)
    mean = np.mean(scores)
    std_err = scipy.stats.sem(scores)
    ci = scipy.stats.t.interval(0.95, df=n-1, loc=mean, scale=std_err)
    return ci

