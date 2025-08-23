import sys,os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.util.logger import HierarchicalLogger
from stable_baselines3.common.logger import Logger as sb3_logger
from stable_baselines3.common.logger import make_output_format
import torch
import pandas as pd
import cloudpickle as cpkl
from loguru import logger
SEED = 42

DEBUG = False

local_dataset_file = "./dataset/data/train-00000-of-00001-88766a2266cab2bb.parquet"

logger.add("./log.txt")

env = make_vec_env(
    #"seals:seals/CartPole-v0",
    "seals:seals/Ant-v1",
    rng=np.random.default_rng(SEED),
    n_envs=8,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
)
if os.path.exists(local_dataset_file):
    df = pd.read_parquet(local_dataset_file)
    rollouts = [] 
    for row in range(len(df)):
        obs = df.loc[row,'obs']
        acts = df.loc[row,'acts']
        terminal = df.loc[row,'terminal']
        rews = df.loc[row,'rews']
        infos = df.loc[row,'infos']
        rollouts.append(TrajectoryWithRew(obs=obs,acts=acts,infos=infos,terminal=terminal,rews=rews))        
    logger.warning(f"load from local dataset {local_dataset_file}")
else:
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        #env_name="seals-CartPole-v0",
        env_name="seals-Ant-v1",
        venv=env,
    )
    if DEBUG:
        env.seed(SEED)
        expert_rewards, _ = evaluate_policy(expert, env, 100, return_episode_rewards=True)
        logger.info(f"expert rewards: {np.mean(expert_rewards)}")
    
    if os.path.exists("./expert_trajs.cpkl") and DEBUG:
        with open("./expert_trajs.cpkl",'rb') as f:
            rollouts = cpkl.load(f)
        logger.warning("!!!!load expert trajs from disk")
    else: 
        rollouts = rollout.rollout(
            expert,
            env,
            rollout.make_sample_until(min_episodes=6000),
            rng=np.random.default_rng(SEED),
        )
        with open("./expert_trajs.cpkl",'wb') as f:
            cpkl.dump(rollouts,f)

learner = PPO(
    env=env,
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
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
) 

#learner.save("./learner.zip")

#torch.save(reward_net.state_dict(),"./reward_net.pth")
csv_output = make_output_format("csv","./airl_log")
airl_logger = HierarchicalLogger(sb3_logger(folder="./airl_log",output_formats=[csv_output]))
airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=2048,
    gen_replay_buffer_capacity=5000 * 3, #!!!!!!
    n_disc_updates_per_round=16*4,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
    custom_logger = airl_logger
)

if DEBUG:
    learner_rewards_before_training = [0]
else:
    env.seed(SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True,
    )

os.makedirs("./saved",exist_ok=True)
def callback_save(r, learner, reward_net):
    if r % 100 == 0:
        learner.save(f"./saved/{r:0>9d}.zip")
        torch.save(reward_net,f"./saved/{r:0>9d}.pth")
if os.path.exists("./learner.zip") and os.path.exists("./reward_net.pth") and DEBUG:
    learner = learner.load("./learner.zip")
    reward_net.load_state_dict(torch.load("./reward_net.pth"))
    logger.warning("!!!! LOAD learner from disk")
else:
    airl_trainer.train(200_000_00,callback=lambda r: callback_save(r,learner,reward_net))  # Train for 2_000_000 steps to match expert.
    learner.save("./learner.zip")
    torch.save(reward_net.state_dict(),"./reward_net.pth")

if DEBUG:
    learner_rewards_after_training = [0]
else:
    env.seed(SEED)
    learner_rewards_after_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True,
    )

logger.info("mean reward after training: {} +/- {}".format(np.mean(learner_rewards_after_training), np.std(learner_rewards_after_training)))
logger.info("mean reward before training: {} +/- {}".format(np.mean(learner_rewards_before_training), np.std(learner_rewards_before_training)))


if 1:
    # 采集 learner 的轨迹
    learner_rollouts = rollout.rollout(
        learner,
        env,
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

    logger.info("mean AIRL reward for learner: {} +/- {}".format(np.mean(episode_rewards), np.std(episode_rewards)))