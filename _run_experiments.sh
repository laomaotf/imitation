export HF_ENDPOINT=https://hf-mirror.com
bash experiments/rollouts_from_policies.sh
bash experiments/imit_benchmark.sh
python -m imitation.scripts.analyze analyze_imitation with csv_output_path=./analyze_output.csv
