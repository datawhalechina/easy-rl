# run DQN on CartPole-v1, not finished yet
codes_dir=$(dirname $(dirname $(readlink -f "$0"))) # "codes" path
python $codes_dir/DQN/main.py --env_name CartPole-v1 --train_eps 2000 --gamma 0.99 --epsilon_decay 6000 --lr 0.00001 --memory_capacity 200000 --batch_size 64 --device cuda