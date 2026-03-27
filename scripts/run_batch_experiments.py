import json
import gc
import time

from inspect_ai import eval
from itertools import product

from collect_baseline import collect_reviews
from collect_sectioned import sectioned_reviews
from collect_council import council_reviews


REVIEW_SYSTEMS = {
    "baseline": collect_reviews,
    "sectioned": sectioned_reviews,
    "council": council_reviews,
}


def cleanup_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    time.sleep(2)


def expand_configs(configs):
    experiments = []

    for config in configs:
        list_params = {k: v for k, v in config.items() if isinstance(v, list)}

        if list_params:
            keys = list(list_params.keys())
            values = [list_params[k] for k in keys]

            for combination in product(*values):
                exp = {
                    "model": config["model"],
                    "model_args": config.get("model_args", {}).copy(),
                    "generation": config.get("generation", {}).copy(),
                }
                for k, v in zip(keys, combination):
                    if k not in ["model", "model_args", "generation"]:
                        exp["generation"][k] = v
                    else:
                        exp[k] = v
                experiments.append(exp)
        else:
            experiments.append({
                "model": config["model"],
                "model_args": config.get("model_args", {}).copy(),
                "generation": config.get("generation", {}).copy(),
            })

    return experiments


def run_experiments(experiments, task_fns, max_tasks, max_connections, **task_kwargs):
    if not isinstance(task_fns, list):
        task_fns = [task_fns]

    for idx, exp in enumerate(experiments):
        model_args = exp["model_args"].copy()
        generation_kwargs = exp["generation"].copy()

        tasks = [
            task_fn(**task_kwargs, **generation_kwargs)
            for task_fn in task_fns
        ]

        eval(
            tasks=tasks,
            model=exp["model"],
            model_args=model_args,
            max_tasks=max_tasks,
            max_connections=max_connections,
        )
        cleanup_gpu()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-connections", type=int, default=10)
    parser.add_argument("--config-path", type=str, default="configs/gpt-oss-20b.json")
    parser.add_argument("--system", type=str, nargs="+", default=["baseline"],
                        choices=list(REVIEW_SYSTEMS.keys()), help="Review system(s) to use")
    parser.add_argument("--no-perturbed", action="store_false", dest="include_perturbed", default=True)
    parser.add_argument("--no-original", action="store_false", dest="include_original", default=True)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        model_configs = json.load(f)

    experiments = expand_configs(model_configs)
    task_fns = [REVIEW_SYSTEMS[system_name] for system_name in args.system]

    run_experiments(
        experiments,
        task_fns=task_fns,
        epochs=args.epochs,
        limit=args.limit,
        shuffle=args.shuffle,
        max_tasks=args.max_tasks,
        max_connections=args.max_connections,
        include_original=args.include_original,
        include_perturbed=args.include_perturbed,
    )