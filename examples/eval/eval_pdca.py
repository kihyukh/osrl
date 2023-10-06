from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import dsrl
import numpy as np
import os
import pyrallis
import torch
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from pyrallis import field

from osrl.algorithms import PDCA, PDCATrainer
from osrl.common.exp_util import load_config_and_all_models, seed_all


@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"
    root: str = "logs"
    noise_scale: List[float] = None
    eval_episodes: int = 20
    best: bool = False
    num_models: int = 40
    eval_result_filename: str = "eval.txt"
    device: str = "cpu"
    threads: int = 4

@pyrallis.wrap()
def get_experiment_list(args: EvalConfig):
    def get_model_count(path):
        checkpoint_path = os.path.join(path, "checkpoint")
        if not os.path.exists(checkpoint_path):
            return 0
        model_files = sorted([
            f for f in os.listdir(checkpoint_path)
            if (
                os.path.isfile(os.path.join(checkpoint_path, f)) and
                f.endswith(".pt") and
                f.split('_')[-1][:-3].isnumeric()
            )
        ])
        return len(model_files)

    experiments = []
    for dirpath, dirnames, filenames in os.walk(args.root):
        level = len(dirpath.split('/'))
        if level != 3:
            continue
        experiments.extend([f'{dirpath}/{dirname}' for dirname in dirnames])

    exp_counts = [(exp, get_model_count(exp)) for exp in experiments]
    return [(exp, count) for exp, count in exp_counts if count > 0]

@pyrallis.wrap()
def print_experiments_to_eval(args: EvalConfig):
    def has_eval(path):
        result_path = os.path.join(path, args.eval_result_filename)
        return os.path.exists(result_path)

    print(' '.join ([exp
        for exp, count in get_experiment_list()
        if count == args.num_models and not has_eval(exp)
    ]))


@pyrallis.wrap()
def eval(args: EvalConfig):

    cfg, models = load_config_and_all_models(args.path)
    print(len(models))
    seed_all(cfg["seed"])
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    if "Metadrive" in cfg["task"]:
        import gym
    else:
        import gymnasium as gym  # noqa

    env = wrap_env(
        env=gym.make(cfg["task"]),
        reward_scale=cfg["reward_scale"],
    )
    env = OfflineEnvWrapper(env)
    env.set_target_cost(cfg["cost_threshold"])

    rewards = np.zeros(len(models))
    costs = np.zeros(len(models))
    for i, model in enumerate(models):
        # setup model
        pdca_model = PDCA(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0],
            a_hidden_sizes=cfg["a_hidden_sizes"],
            c_hidden_sizes=cfg["c_hidden_sizes"],
            gamma=cfg["gamma"],
            B=cfg["B"],
            cost_threshold=cfg["cost_threshold"],
            episode_len=cfg["episode_len"],
            device=args.device,
        )
        pdca_model.load_state_dict(model["model_state"])
        pdca_model.to(args.device)
        trainer = PDCATrainer(pdca_model,
                env,
                reward_scale=cfg["reward_scale"],
                cost_scale=cfg["cost_scale"],
                device=args.device)

        ret, cost, length = trainer.evaluate(args.eval_episodes)
        normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
        rewards[i] = normalized_ret
        costs[i] = normalized_cost

    return rewards.mean(), costs.mean(), len(models)

if __name__ == "__main__":
    #r, c, l = eval()
    print_experiments_to_eval()

