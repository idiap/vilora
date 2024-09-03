#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Haolin Chen <haolin.chen@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
seed_list=(2021 2022 2023 2024 2025)

for SEED in ${seed_list[@]}; do

export WANDB_NAME=full_ivon_0.1_wd1e-5

OUTPUT_DIR=./output/full_ivon/stsb/${SEED}/deberta-v3-base_${WANDB_NAME}

if [ ! -f $OUTPUT_DIR/all_results.json ]
then

COMMAND="#!/bin/bash
python run_glue_no_trainer.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name stsb \
--max_length 128 \
--per_device_train_batch_size 32 \
--learning_rate 0.1 \
--num_train_epochs 25 \
--eval_steps 100 \
--lr_scheduler_type linear \
--num_warmup_steps 100 \
--weight_decay 0 \
--seed ${SEED} \
--output_dir ${OUTPUT_DIR} \
--with_tracking \
--report_to wandb \
--use_ivon \
--ivon_weight_decay 1e-5"

echo $COMMAND > /tmp/temp_sbatch.sh

sbatch -A $ACCOUNT -p gpu --gpus 1 --constraint="h100|rtx3090" --time 01:00:00 --cpus-per-task 12 --mem-per-gpu 24G --nodes 1 --ntasks 1 --job-name ${WANDB_NAME}_${SEED} --output logs/stsb_${WANDB_NAME}_${SEED}.out --error logs/stsb_${WANDB_NAME}_${SEED}.err /tmp/temp_sbatch.sh

fi

done