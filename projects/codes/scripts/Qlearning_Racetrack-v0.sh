codes_dir=$(dirname $(dirname $(readlink -f "$0"))) # "codes" path
python $codes_dir/QLearning/main.py --env_name Racetrack-v0 --device cpu