#!/bin/bash

#SBATCH --job-name=eval
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=log_dir/eval.fed.log

export model_save_path=output/empathetic-us-roberta-base-nli-mean-10000-zc/
export checkpoint_name=best.pt
export dataset=fedturn
export dataset_dir=data/${dataset}

python -u score.py \
	--data=${dataset_dir}/${dataset}_eval.pkl \
    --device=cuda \
    --batch_size=1 \
    --model_name_or_path roberta-base-nli-stsb-mean-tokens \
    --wp -1 \
    --wf -1 \
	--model_save_path ${model_save_path} \
	--oot_model ${checkpoint_name}
	
