#!/bin/bash

#SBATCH --job-name=train
#SBATCH -n 1
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH --output=log_dir/train.empathetic.hup.567890.log

export dataset=empathetic
export dataset_dir=data/${dataset}
export task=hup
export seed=567890

python -u train.py \
        --data=${dataset_dir}/${dataset}_${task}.pkl \
        --from_begin \
        --device=cuda \
        --epochs=20 \
        --batch_size=512 \
        --seed=${seed} \
        --wf=-1 \
        --wp=-1 \
        --model_name_or_path roberta-base-nli-stsb-mean-tokens \
        --model_save_path output/${dataset}-${task}-${seed}

