#!/bin/bash
 
#SBATCH --job-name=eval
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=log_dir/eval_persona-us-best.log

export dataset=persona
export dataset_dir=data/${dataset}
export task=us
export seed=10000
export model_path=best_ckpt/persona/us/
export checkpoint_number=best.pt

echo "evaluate ${dataset}-${task}"

python -u eval.py \
	--data=${dataset_dir}/${dataset}_${task}.pkl \
	--device=cuda \
	--batch_size=256 \
	--model_name_or_path roberta-base-nli-stsb-mean-tokens \
    --wp 10 \
	--wf 10 \
	--model_save_path ${model_path} \
	--oot_model ${checkpoint_number}
