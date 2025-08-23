import sys,os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
import torch
import cloudpickle as cpkl

SEED = 42
DEBUG = True
FAST = False

if FAST:
    N_RL_TRAIN_STEPS = 100_000
else:
    N_RL_TRAIN_STEPS = 2_000_000

venv = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=np.random.default_rng(SEED),
    n_envs=8,
    post_wrappers=[
        lambda env, _: RolloutInfoWrapper(env)
    ],  # needed for computing rollouts later
)
expert = load_policy(
    "ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="seals/CartPole-v0",
    venv=venv,
)
if DEBUG:
    venv.seed(SEED)
    expert_rewards, _ = evaluate_policy(expert, venv, 100, return_episode_rewards=True)
    print(f"expert rewards: {np.mean(expert_rewards)}")
   
if os.path.exists("./expert_trajs.cpkl") and DEBUG:
    with open("./expert_trajs.cpkl",'rb') as f:
        rollouts = cpkl.load(f)
    print("!!!!load expert trajs from disk")
else: 
    rollouts = rollout.rollout(
        expert,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=np.random.default_rng(SEED),
    )
    with open("./expert_trajs.cpkl",'wb') as f:
        cpkl.dump(rollouts,f)

learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0005,
    gamma=0.95,
    clip_range=0.1,
    vf_coef=0.1,
    n_epochs=5,
    seed=SEED,
)

reward_net = BasicShapedRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    normalize_input_layer=RunningNorm,
)

#learner.save("./learner.zip")

#torch.save(reward_net.state_dict(),"./reward_net.pth")

airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=2048,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=16,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

if DEBUG:
    learner_rewards_before_training = [0]
else:
    venv.seed(SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )

if os.path.exists("./learner.zip") and os.path.exists("./reward_net.pth") and DEBUG:
    learner = learner.load("./learner.zip")
    reward_net.load_state_dict(torch.load("./reward_net.pth"))
    print("!!!! LOAD learner from disk")
else:
    airl_trainer.train(N_RL_TRAIN_STEPS) # Train for 2_000_000 steps to match expert.
    learner.save("./learner.zip")
    torch.save(reward_net.state_dict(),"./reward_net.pth")

if DEBUG: 
    learner_rewards_after_training = [0]
else:
    venv.seed(SEED)
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )
print(
    "Rewards before training:",
    np.mean(learner_rewards_before_training),
    "+/-",
    np.std(learner_rewards_before_training),
)
print(
    "Rewards after training:",
    np.mean(learner_rewards_after_training),
    "+/-",
    np.std(learner_rewards_after_training),
)

if 1:
    # 采集 learner 的轨迹
    learner_rollouts = rollout.rollout(
        learner,
        venv,
        rollout.make_sample_until(min_episodes=100),
        rng=np.random.default_rng(SEED),
    )
    #learner_rollouts = rollout.flatten_trajectories_with_rew(learner_rollouts)
    # 用 reward_net 计算每一步奖励
    episode_rewards = []
    for traj in learner_rollouts:
        dones = np.zeros(len(traj.acts),dtype=np.bool)
        dones[-1] = traj.terminal
        rewards = reward_net.predict(
            state=traj.obs[0:-1],
            action=traj.acts,
            next_state=traj.obs[1:],
            done=dones,
        )
        episode_rewards.append(np.sum(rewards))

    print(
        "Rewards based reward_net:",
        np.mean(episode_rewards),
        "+/-",
        np.std(episode_rewards),
    )