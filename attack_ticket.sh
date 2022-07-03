#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

all_attack_type=("textfooler" "bertattack" "textbugger")
log_file="robust-ticket.log"

# IMDB
model_log=imdb_robust-ticket
file='./your_draw-retrain-ticket_path' # model for attack

for attack_method in "${all_attack_type[@]}"
do
python attack_ticket_more_attackers.py \
--model_name_or_path $file \
--num_examples 1000 \
--dataset_name imdb \
--neighbour_vocab_size 50 \
--modify_ratio 0.9 \
--sentence_similarity 0.2 \
--attack_method ${attack_method} \
--attack_log ${model_log}_${attack_method}.csv \
--official_log ${model_log}_${attack_method}_official.csv \
--perturbed_file ${model_log}_${attack_method}_perturbed_sentences.csv >> ${log_file}
done

# AGNEWS
model_log=agnews_robust-ticket
file='./your_draw-retrain-ticket_path' # model for attack

for attack_method in "${all_attack_type[@]}"
do
python attack_ticket_more_attackers.py \
--model_name_or_path $file \
--num_examples 1000 \
--dataset_name ag_news \
--neighbour_vocab_size 50 \
--modify_ratio 0.9 \
--sentence_similarity 0.2 \
--attack_method ${attack_method} \
--attack_log ${model_log}_${attack_method}.csv \
--official_log ${model_log}_${attack_method}_official.csv \
--perturbed_file ${model_log}_${attack_method}_perturbed_sentences.csv >> ${log_file}
done

# SST-2
model_log=sst2_robust-ticket
file='./your_draw-retrain-ticket_path' # model for attack

for attack_method in "${all_attack_type[@]}"
do
python attack_ticket_more_attackers.py \
--model_name_or_path $file \
--num_examples 1000 \
--dataset_name glue \
--task_name sst2 \
--neighbour_vocab_size 50 \
--modify_ratio 0.9 \
--sentence_similarity 0.2 \
--attack_method ${attack_method} \
--attack_log ${model_log}_${attack_method}.csv \
--official_log ${model_log}_${attack_method}_official.csv \
--perturbed_file ${model_log}_${attack_method}_perturbed_sentences.csv >> ${log_file}
done