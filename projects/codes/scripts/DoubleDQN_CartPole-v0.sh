# run Double DQN on CartPole-v0
codes_dir=$(dirname $(dirname $(readlink -f "$0"))) # "codes" path
python $codes_dir/DoubleDQN/main.py --device cuda