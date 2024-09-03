#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Haolin Chen <haolin.chen@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
seed_list=(2021 2022 2023 2024 2025)

for SEED in ${seed_list[@]}; do

export WANDB_NAME=vilora_i6t4_0.4

OUTPUT_DIR=./output/vilora/mnli/${SEED}/deberta-v3-base_${WANDB_NAME}

if [ ! -f $OUTPUT_DIR/all_results.json ]
then

COMMAND="#!/bin/bash
python run_glue_no_trainer.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name mnli \
--max_length 256 \
--per_device_train_batch_size 32 \
--learning_rate 0.4 \
--num_train_epochs 7 \
--eval_steps 2000 \
--lr_scheduler_type linear \
--num_warmup_steps 500 \
--weight_decay 0 \
--apply_lora \
--lora_type vilora \
--lora_alpha 16 \
--adalora_target_r 4 \
--adalora_init_r 6 \
--adalora_orth_reg_weight 0.1 \
--adalora_tinit 8000 \
--adalora_tfinal 50000 \
--adalora_delta_t 100 \
--seed ${SEED} \
--output_dir ${OUTPUT_DIR} \
--with_tracking \
--report_to wandb \
--use_ivon"

echo $COMMAND > /tmp/temp_sbatch.sh

sbatch -A $ACCOUNT -p gpu --gpus 1 --constraint="h100|rtx3090" --time 08:00:00 --cpus-per-task 12 --mem-per-gpu 24G --nodes 1 --ntasks 1 --job-name ${WANDB_NAME}_${SEED} --output logs/mnli_${WANDB_NAME}_${SEED}.out --error logs/mnli_${WANDB_NAME}_${SEED}.err /tmp/temp_sbatch.sh

fi

done