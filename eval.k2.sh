#!/bin/bash
 
#SBATCH --job-name=eval
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa3
#SBATCH --output=log_dir/empathetic-us-roberta-base-nli-mean-345678-k2.log

export dataset=empathetic
export dataset_dir=data/${dataset}
export task=us
export seed=345678
export model_path=output/empathetic-us-roberta-base-nli-mean-345678-k2
export checkpoint_number=best.pt

echo "evaluate ${dataset}-${task}"

python -u eval.py \
	--data=${dataset_dir}/${dataset}_${task}.pkl \
	--device=cuda \
	--batch_size=512 \
	--model_name_or_path roberta-base-nli-stsb-mean-tokens \
    --wp -1 \
	--wf -1 \
	--model_save_path ${model_path} \
	--oot_model ${checkpoint_number}
