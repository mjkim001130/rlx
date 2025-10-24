# D4RL
# # D4RL antmaze-umaze-v2
# python main.py --env_name=antmaze-umaze-v2 --offline_steps=500000 --eval_interval=50000
# # D4RL antmaze-umaze-diverse-v2
# python main.py --env_name=antmaze-umaze-diverse-v2 --offline_steps=500000 --eval_interval=50000
# # D4RL antmaze-medium-play-v2
# python main.py --env_name=antmaze-medium-play-v2 --offline_steps=500000 --eval_interval=50000
# # D4RL antmaze-medium-diverse-v2
# python main.py --env_name=antmaze-medium-diverse-v2 --offline_steps=500000 --eval_interval=50000
# # D4RL antmaze-large-play-v2
# python main.py --env_name=antmaze-large-play-v2 --offline_steps=500000 --eval_interval=50000
# # D4RL antmaze-large-diverse-v2
# python main.py --env_name=antmaze-large-diverse-v2 --offline_steps=500000 --eval_interval=50000


SEEDS=(0 1 2 3 4 5)
ENVS=(
    antmaze-umaze-v2
    antmaze-umaze-diverse-v2
    antmaze-medium-play-v2
    antmaze-medium-diverse-v2
    antmaze-large-play-v2
    antmaze-large-diverse-v2
)
OFFLINE_STEPS=500000

for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "[RUN] env=${ENV} seed=${SEED}"
    python main.py \
      --env_name="${ENV}" \
      --offline_steps="${OFFLINE_STEPS}" \
      --seed="${SEED}"
    echo "[DONE] env=${ENV} seed=${SEED}"
  done
done

echo "All jobs finished."