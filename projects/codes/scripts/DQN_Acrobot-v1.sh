# run DQN on Acrobot-v1, not the best tuned parameters
codes_dir=$(dirname $(dirname $(readlink -f "$0"))) # "codes" path
python $codes_dir/DQN/main.py --env_name Acrobot-v1 --train_eps 100 --epsilon_decay 1500 --lr 0.002 --memory_capacity 200000 --batch_size 128 --device cuda