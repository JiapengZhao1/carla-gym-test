"""
Example of a custom gym environment. Run this example for a demo.

This example shows the usage of:
  - a custom environment
  - Ray Tune for grid search to try different learning rates

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
import argparse
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import os
import random

import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

from carla_gym.multi_env import MultiActorCarlaEnv, MultiActorCarlaEnvPZ, DISCRETE_ACTIONS

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
    argparser.add_argument("--xml_config_path", default="configs.xml", help="Path to the xml config file")
    argparser.add_argument("--maps_path", default="/Game/Carla/Maps/", help="Path to the CARLA maps")
    argparser.add_argument("--render_mode", default="human", help="Path to the CARLA maps")

    args = vars(argparser.parse_args())
    args["discrete_action_space"] = True
    # The scenario xml config should have "enable_planner" flag
    #env = MultiActorCarlaEnvPZ(**args)
    # otherwise for PZ AEC: env = carla_gym.env(**args)

    print(f"Running with following CLI options: {args}")

    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))

    config = (
        get_trainable_cls("PPO")
        .get_default_config()
        # or "corridor" if registered above
        .environment(MultiActorCarlaEnvPZ)
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    #stop = {
        #"training_iteration": args.stop_iters,
        #"timesteps_total": args.stop_timesteps,
        #"episode_reward_mean": args.stop_reward,
    #}

    if False:
        # manual training with train loop using PPO and fixed learning rate
        #if args.run != "PPO":
            #raise ValueError("Only support --run PPO with --no-tune.")
        #print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        config.lr = 1e-3
        algo = config.build()
        # run manual training loop and print results after each iteration
        for _ in range(100):
            result = algo.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            #if (
                #result["timesteps_total"] >= args.stop_timesteps
                #or result["episode_reward_mean"] >= args.stop_reward
            #):
                #break
        algo.stop()
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=air.RunConfig(),
        )
        results = tuner.fit()

        #if args.as_test:
            #print("Checking if learning goals were achieved")
            #check_learning_achieved(results, args.stop_reward)

    ray.shutdown()