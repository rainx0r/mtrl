# from wandb.apis.public.runs import Run
# import json
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

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
    cache_file = CACHE_DIR / f"{entity}_{project}_{run_name}_{metric.replace('/', '_')}_history.json"
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


def main():
    api = wandb.Api()
    runs = api.runs("reggies-phd-research/mtrl-mt50-results")

    # run_names = ['mt10_mtmhsac_task_weights_false', 'mt10_softmodules_task_weights_false', 'mt10_moore_fix', 'mt10_mtmhsac_sm_params_3_layers_v2', 'mt10_mtmhsac_v2_moore_params_3_layers',
    #     'mt10_mtmhsac_paco_params_v2_3_layers', 'mt10_paco', 'mt10_mtmhsac_v2_width_1024', 'mt10_mtmhsac_v2_width_2048', 'mt10_mtmhsac_v2_width_4096']

    run_names = [
        "mt50_mtmhsac_v2",
        "mt50_softmodules_task_weights_false",
        "mt50_moore",
        "mt50_mtmhsac_moore_params_v2",
        "mtmhsac_moore_params_log_std-10_clipped_q",
        "mt50_paco",
        "mt50_mtmhsac_sm_params_v2",
        "mt50_mtmhsac_v2_paco_params_3_layers",
        "mt50_mtmhsac_v2_2048_width",
        "mt50_mtmhsac_v2_4096_width",
    ]

    # run_names = ['mt10_paco', 'mt10_softmodules_task_weights_false', 'mt10_moore_fix', 'mt10_mtmhsac_task_weights_false', 'mt10_care']

    mt10_results = defaultdict(list)

    # Get success rate
    for run in runs:
        if run.name in run_names and run.state != "running":
            mt10_results[run.name].append(run.summary["charts/mean_success_rate"])

    print(mt10_results)


if __name__ == "__main__":
    main()
