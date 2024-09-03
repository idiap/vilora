#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Haolin Chen <haolin.chen@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
seed_list=(2021 2022 2023 2024 2025)

for SEED in ${seed_list[@]}; do

export WANDB_NAME=adalora_i6t4_8e-4

OUTPUT_DIR=./output/adalora/cola/${SEED}/deberta-v3-base_${WANDB_NAME}

if [ ! -f $OUTPUT_DIR/all_results.json ]
then

COMMAND="#!/bin/bash
python run_glue_no_trainer.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name cola \
--max_length 64 \
--per_device_train_batch_size 32 \
--learning_rate 8e-4 \
--num_train_epochs 25 \
--eval_steps 100 \
--lr_scheduler_type linear \
--num_warmup_steps 100 \
--weight_decay 0 \
--apply_lora \
--lora_type adalora \
--lora_alpha 32 \
--adalora_target_r 4 \
--adalora_init_r 6 \
--adalora_orth_reg_weight 0.1 \
--adalora_tinit 800 \
--adalora_tfinal 3500 \
--adalora_delta_t 10 \
--seed ${SEED} \
--output_dir ${OUTPUT_DIR} \
--with_tracking \
--report_to wandb"

echo $COMMAND > /tmp/temp_sbatch.sh

sbatch -A $ACCOUNT -p gpu --gpus 1 --constraint="h100|rtx3090" --time 01:00:00 --cpus-per-task 12 --mem-per-gpu 24G --nodes 1 --ntasks 1 --job-name ${WANDB_NAME}_${SEED} --output logs/cola_${WANDB_NAME}_${SEED}.out --error logs/cola_${WANDB_NAME}_${SEED}.err /tmp/temp_sbatch.sh

fi

done