#!/bin/bash

#SBATCH --job-name=score
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p new
#SBATCH -w hlt06
#SBATCH --output=score.log

export dataset=feddial
export dataset_dir=data/${dataset}
export checkpoint_number=best.pt
export lm_path=SRoBERTa
export model_save_path=output/empathetic-us-345678/
export checkpoint_name=best.pt

/home/chen/anaconda3/envs/gcn/bin/python3 -u score.py \
	--data=${dataset_dir}/${dataset}_eval.pkl \
    --device=cuda \
    --batch_size=1 \
    --model_name_or_path ${lm_path} \
    --wp 4 \
    --wf 4 \
	--model_save_path ${model_save_path} \
	--oot_model ${checkpoint_name}
	
