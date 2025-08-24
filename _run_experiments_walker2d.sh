export HF_ENDPOINT=https://hf-mirror.com
python -m imitation.scripts.eval_policy with seals_walker logging.log_root=./output rollout_save_path=./output/expert_models/seals_walker_0/rollouts/final.npz eval_n_episodes=40 eval_n_timesteps=None
bash experiments/imit_benchmark.sh
python -m imitation.scripts.analyze analyze_imitation with csv_output_path=./analyze_output.csv
