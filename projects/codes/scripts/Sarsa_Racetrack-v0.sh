codes_dir=$(dirname $(dirname $(readlink -f "$0"))) # "codes" path
python $codes_dir/Sarsa/main.py --env_name Racetrack-v0