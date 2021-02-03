# DynaEval
Code associated with the DynaEval


## Installation

```bash
conda env create -f environment.yml
conda activate gcn
```

## Example commands

### Preprocess training data
The following command will preprocess data for `empathetic` corpus for `us` task.

```bash
export dataset=empathetic
export dataset_dir=data/${dataset}
export task=us

python -u preprocess.py \
        --data_path=${dataset_dir} \
        --dataset=${dataset} --perturb_type ${task}

```

### Train DynaEval model
The following command will train a model on `empathetic` corpus for `us` task.

```bash
export dataset=empathetic
export dataset_dir=data/${dataset}
export task=us

python -u train.py \ 
        --data=${dataset_dir}/${dataset}_${task}.pkl \
        --from_begin \
        --device=cuda \
        --model_name_or_path roberta-base-nli-stsb-mean-tokens \
        --model_save_path output/${dataset}-${task}-roberta-base-nli-mean
```

### Evaluate DynaEval model
The following command will evaluate a trained model on `empathetic` corpus for `us` task.

```bash
export dataset=empathetic
export dataset_dir=data/${dataset}
export task=us
export model_path=your_model_path
export checkpoint_path=your_checkpoint_path

python -u eval.py \
        --data=${dataset_dir}/${dataset}_${task}.pkl \
        --device=cuda \
        --model_name_or_path roberta-base-nli-stsb-mean-tokens \
        --model_save_path ${model_path} \
        --oot_model ${checkpoint_path}
```

### Score 
The following command provides metric scores based on a trained model

#### Preprocess evaluation data
The following command will preprocess evaluation data for dialogue evaluation task.

```bash
export dataset=fed
export dataset_dir=data/${dataset}

python -u create_eval_data.py \
        --data_path=${dataset_dir} \
        --dataset=${dataset}

```

#### Generate score file

```bash
export model_path=your_model_path
export checkpoint_path=your_checkpoint_path

python -u score.py \
        --data=data/evaluation/fed_eval.pkl \
        --device=cuda \
        --model_name_or_path roberta-base-nli-stsb-mean-tokens \
        --loss_type=coh \
        --model_save_path ${model_path} \
        --oot_model ${checkpoint_path}

```

