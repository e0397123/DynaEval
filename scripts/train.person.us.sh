#!/bin/bash

#SBATCH --job-name=train
#SBATCH -n 1
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH --output=log_dir/train.persona.us.234567.log

export dataset=persona
export dataset_dir=data/${dataset}
export task=us
export seed=135792

python -u train.py \
        --data=${dataset_dir}/${dataset}_${task}.pkl \
        --from_begin \
        --device=cuda \
        --epochs=20 \
        --batch_size=512 \
        --seed=${seed} \
        --wf=2 \
        --wp=2 \
        --model_name_or_path roberta-base-nli-stsb-mean-tokens \
        --model_save_path output/${dataset}-${task}-${seed} > log_dir/train.persona.us.135792.log 2>&1 &

