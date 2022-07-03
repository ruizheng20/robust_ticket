#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

log_file='run_glue.log'  # training log
result_file='run_glue.csv'  # attack results
cpkt='./save_models/fine-tune/' # Model saving path
seed=42
lr=2e-5
epoch=3
model='bert-base-uncased'

# IMDB
python run_glue.py \
--model_name $model \
--ckpt_dir $cpkt \
--dataset_name imdb \
--num_labels 2 \
--bsz 32 \
--epochs $epoch \
--lr $lr \
--seed $seed \
--max_seq_length 256 \
--result_file $result_file \
--num_examples 100 \
--force_overwrite 1 >> ${log_file}


# AGNEWS
python run_glue.py \
--model_name $model \
--ckpt_dir $cpkt \
--dataset_name ag_news \
--num_labels 4 \
--bsz 32 \
--epochs $epoch \
--lr $lr \
--seed $seed \
--max_seq_length 256 \
--result_file $result_file \
--num_examples 200 \
--force_overwrite 1 >> ${log_file}



# SST-2
python run_glue.py \
--model_name $model \
--ckpt_dir $cpkt \
--dataset_name glue \
--task_name sst2 \
--epochs $epoch \
--result_file ${result_file} \
--num_examples 872 \
--lr $lr \
--seed $seed \
--max_seq_length 128 \
--force_overwrite 1 >> ${log_file}
