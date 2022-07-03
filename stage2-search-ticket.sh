#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cpkt='./save_models/search_ticket/'  # search tickets saving path
log_file='search_ticket.log'
epoch=20

# IMDB
model_path='./your_fine-tune_path/'
lr=0.1
amp=0.5
step=5
python search_ticket.py \
--model_name ${model_path} \
--lr $lr \
--dataset_name imdb \
--num_labels 2 \
--bsz 32 \
--max_seq_length 256 \
--lambda_amp $amp \
--ckpt_dir ${cpkt} \
--weight_decay 1e-6 \
--adv_steps $step \
--epochs $epoch >> ${log_file}

# AGNEWS
model_path='./your_fine-tune_path/'
lr=0.05
amp=0.5
step=5
python search_ticket.py \
--model_name ${model_path} \
--lr $lr \
--dataset_name ag_news \
--num_labels 4 \
--bsz 32 \
--max_seq_length 256 \
--lambda_amp $amp \
--ckpt_dir ${cpkt} \
--weight_decay 1e-6 \
--adv_steps $step \
--epochs $epoch >> ${log_file}

# SST-2
model_path='./your_fine-tune_path/'
lr=0.1
amp=0.5
step=5
python search_ticket.py \
--model_name ${model_path} \
--lr $lr \
--dataset_name glue \
--task_name sst2 \
--num_labels 2 \
--bsz 32 \
--max_seq_length 128 \
--lambda_amp $amp \
--ckpt_dir ${cpkt} \
--weight_decay 1e-6 \
--adv_steps $step \
--epochs $epoch >> ${log_file}