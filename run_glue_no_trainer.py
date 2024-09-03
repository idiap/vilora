# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Haolin Chen <haolin.chen@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import ivon


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.40.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=0,
        help="Whether to run evaluation at the end of every n steps.",
    )
    parser.add_argument(
        "--eval_after_epochs",
        type=int,
        default=1,
        help="Begin evaluation after n epochs.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to enable mixed precision training or not.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to enable gradient checkpointing or not.",
    )
    parser.add_argument(
        "--apply_lora",
        action="store_true",
        help="Whether to apply LoRA or not.",
    )
    parser.add_argument(
        "--lora_type",
        type=str,
        default="lora",
        help="LoRA type.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="The file path of LoRA parameters.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--lora_modules",
        type=str,
        default="",
        help="LoRA target modules.",
    )
    parser.add_argument(
        "--adalora_target_r",
        type=int,
        default=8,
        help="Target AdaLoRA rank.",
    )
    parser.add_argument(
        "--adalora_init_r",
        type=int,
        default=12,
        help="Initial AdaLoRA rank",
    )
    parser.add_argument(
        "--adalora_tinit",
        type=int,
        default=0,
        help="Number of warmup steps for AdaLoRA wherein no pruning is performed.",
    )
    parser.add_argument(
        "--adalora_tfinal",
        type=int,
        default=0,
        help="Fix the resulting budget distribution and fine-tune the model for tfinal steps when using AdaLoRA.",
    )
    parser.add_argument(
        "--adalora_delta_t",
        type=int,
        default=100,
        help="Step interval of rank allocation.",
    )
    parser.add_argument(
        "--adalora_orth_reg_weight",
        type=float,
        default=0.5,
        help="The orthogonal regularization coefficient.",
    )
    parser.add_argument(
        "--use_ivon",
        action="store_true",
        help="Whether to use IVON or not.",
    )
    parser.add_argument(
        "--ivon_ess",
        type=int,
        default=0,
        help="IVON effective sample size.",
    )
    parser.add_argument(
        "--ivon_hess_init",
        type=float,
        default=1.0,
        help="IVON Hessian initial value.",
    )
    parser.add_argument(
        "--ivon_beta1",
        type=float,
        default=0.9,
        help="IVON beta1.",
    )
    parser.add_argument(
        "--ivon_beta2",
        type=float,
        default=0.99999,
        help="IVON beta2.",
    )
    parser.add_argument(
        "--ivon_weight_decay",
        type=float,
        default=1e-8,
        help="IVON weight decay.",
    )
    parser.add_argument(
        "--ivon_train_mc_samples",
        type=int,
        default=1,
        help="IVON training MC samples.",
    )
    parser.add_argument(
        "--ivon_infer_mc_samples",
        type=int,
        default=0,
        help="IVON inference MC samples.",
    )
    parser.add_argument(
        "--ivon_clip_radius",
        type=float,
        default=1e-3,
        help="IVON clip radius.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Evaluation only.",
    )
    parser.add_argument(
        "--project_prefix",
        type=str,
        default="glue_no_trainer_vilora_",
        help="Tracker project prefix."
    )
    parser.add_argument(
        "--use_custom_lr_schedule",
        action="store_true",
        help="Use custom learning rate schedule.",
    )
    parser.add_argument(
        "--lr_schedule_first_lr",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--lr_schedule_const_epochs",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lr_schedule_second_lr",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--lr_schedule_end_lr",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--vilora_criterion",
        type=str,
        default="SNR_mean_abs",
        help="VILoRA importance score criterion.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_glue_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir, mixed_precision="fp16" if args.fp16 else "no") if args.with_tracking else Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("nyu-mll/glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=args.trust_remote_code,
    )
    config.gradient_checkpointing = args.gradient_checkpointing
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
    )

    if args.apply_lora:
        from peft import LoraConfig, AdaLoraConfig, VILoraConfig, TaskType, get_peft_model, PeftConfig

        if args.lora_type == "lora":
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_modules.split(",") if args.lora_modules else None,
            )
        elif args.lora_type == "adalora":
            lora_config = AdaLoraConfig(
                task_type=TaskType.SEQ_CLS,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_r=args.adalora_target_r,
                init_r=args.adalora_init_r,
                tinit=args.adalora_tinit,
                tfinal=args.adalora_tfinal,
                deltaT=args.adalora_delta_t,
                orth_reg_weight=args.adalora_orth_reg_weight,
            )
        elif args.lora_type == "vilora":
            lora_config = VILoraConfig(
                task_type=TaskType.SEQ_CLS,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_r=args.adalora_target_r,
                init_r=args.adalora_init_r,
                tinit=args.adalora_tinit,
                tfinal=args.adalora_tfinal,
                deltaT=args.adalora_delta_t,
                orth_reg_weight=args.adalora_orth_reg_weight,
                criterion=args.vilora_criterion,
            )
            args.use_ivon = True

        if args.lora_path:
            lora_config = PeftConfig.from_pretrained(args.lora_path)
            lora_config.inference_mode = False
            model = get_peft_model(model, lora_config)
            model.load_adapter(args.lora_path, "default", is_trainable=True)
        else:
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.mixed_precision == 'fp16' else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    if args.use_custom_lr_schedule:
        args.learning_rate = 1.0

    if args.use_ivon:
        optimizer = ivon.IVON([p for p in model.parameters() if p.requires_grad], 
                              lr=args.learning_rate, 
                              ess=len(train_dataset) if args.ivon_ess == 0 else args.ivon_ess, 
                              hess_init=args.ivon_hess_init, 
                              beta1=args.ivon_beta1,
                              beta2=args.ivon_beta2,
                              weight_decay=args.ivon_weight_decay, 
                              mc_samples=args.ivon_train_mc_samples, 
                              clip_radius=args.ivon_clip_radius, 
                              device=accelerator.device)
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if not args.use_custom_lr_schedule:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        def get_custom_lr_scheduler(optimizer, num_warmup_steps, first_lr, const_steps, second_lr, end_lr, num_training_steps):
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return (first_lr / num_warmup_steps) * current_step
                elif current_step < const_steps:
                    return first_lr
                else:
                    # Linearly decay from second_lr to end_lr
                    remaining_steps = num_training_steps - const_steps
                    decay_factor = (second_lr - end_lr) / remaining_steps
                    return max(second_lr - decay_factor * (current_step - const_steps), end_lr)

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        const_steps = args.lr_schedule_const_epochs if args.lr_schedule_const_epochs > 100 else args.lr_schedule_const_epochs * num_update_steps_per_epoch
        lr_scheduler = get_custom_lr_scheduler(optimizer=optimizer, 
                                               num_warmup_steps=args.num_warmup_steps, 
                                               first_lr=args.lr_schedule_first_lr, 
                                               const_steps=const_steps,
                                               second_lr=args.lr_schedule_second_lr,
                                               end_lr=args.lr_schedule_end_lr,
                                               num_training_steps=args.max_train_steps)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    if args.apply_lora and args.lora_type == "vilora":
        model.base_model.set_optimizer(optimizer)

    def save_ivon_hessian_state(resize=False):
        def resize_state_dict_by_rank_pattern(rank_pattern, state_dict, adapter_name):
            for name, rank_idx in rank_pattern.items():
                rank = sum(rank_idx)
                prefix = ".".join(name.split(".")[0:-2]) if adapter_name in name else ".".join(name.split(".")[0:-1])
                for layer in ["lora_E", "lora_A", "lora_B"]:
                    key = f"base_model.model.{prefix}.{layer}.{adapter_name}"
                    if layer != "lora_B":
                        state_dict[key] = (
                            state_dict[key][rank_idx] if rank != state_dict[key].shape[0] else state_dict[key]
                        )
                    else:
                        state_dict[key] = (
                            state_dict[key][:, rank_idx] if rank != state_dict[key].shape[1] else state_dict[key]
                        )
            return state_dict
        
        if resize:
            opt_params = optimizer.param_groups[0]['params']
            opt_hess = optimizer.param_groups[0]['hess']
            named_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            named_hess = {}
            offset = 0
            with torch.no_grad():
                for i, (n, p) in enumerate(named_params.items()):
                    assert p.shape == opt_params[i].shape, n
                    p = p.detach()
                    numel = p.numel()
                    named_hess[n] = opt_hess[offset : offset + numel].view(*p.shape)
                    offset += numel

            named_hess_resized = resize_state_dict_by_rank_pattern(model.base_model.peft_config["default"].rank_pattern, named_hess, "default")
            hess_resized = torch.cat([p_hess.view(-1) for p_hess in named_hess_resized.values()])

            torch.save({
                'hess': hess_resized,
                'ess': optimizer.state_dict()['param_groups'][0]['ess'],
                'weight_decay': optimizer.state_dict()['param_groups'][0]['weight_decay'],
                'numel': len(hess_resized)
            }, os.path.join(args.output_dir, "ivon_hessian_resized.state"))
        else:
            torch.save({
                'hess': optimizer.state_dict()['param_groups'][0]['hess'],
                'ess': optimizer.state_dict()['param_groups'][0]['ess'],
                'weight_decay': optimizer.state_dict()['param_groups'][0]['weight_decay'],
                'numel': optimizer.state_dict()['param_groups'][0]['numel']
            }, os.path.join(args.output_dir, "ivon_hessian.state"))
    
    def load_ivon_hessian_state(path):
        state_dict = torch.load(path, map_location=accelerator.device)
        assert optimizer.state_dict()['param_groups'][0]['ess'] == state_dict['ess']
        assert optimizer.state_dict()['param_groups'][0]['weight_decay'] == state_dict['weight_decay']
        assert optimizer.state_dict()['param_groups'][0]['numel'] == state_dict['numel']
        optimizer.param_groups[0]['hess'] = state_dict['hess']
        print(f"IVON Hessian state loaded from {path}.")

    if args.lora_path and args.use_ivon:
        if lora_config.peft_type in ["ADALORA", "VILORA"]:
            ivon_hessian_path = os.path.join(args.lora_path, "ivon_hessian_resized.state")
        else:
            ivon_hessian_path = os.path.join(args.lora_path, "ivon_hessian.state")
        if os.path.exists(ivon_hessian_path):
            load_ivon_hessian_state(ivon_hessian_path)
        else:
            print(f"IVON Hessian state does not exist at {ivon_hessian_path}!")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.apply_lora and args.lora_type in ["adalora", "vilora"]:
        model.base_model.peft_config["default"].total_step = args.max_train_steps

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(args.project_prefix + args.task_name, experiment_config)

    # Get the metric function
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    def do_eval(epoch_end=False):
        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            sampled_logits = []
            if args.use_ivon and args.ivon_infer_mc_samples > 0:
                for _ in range(args.ivon_infer_mc_samples):
                    with optimizer.optimizer.sampled_params():
                        with torch.no_grad():
                            outputs = model(**batch)
                        sampled_logits.append(outputs.logits)
                logits = torch.mean(torch.stack(sampled_logits), dim=0)
                predictions = logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                predictions, references = accelerator.gather((predictions, batch["labels"]))
            else:
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch} step {completed_steps}: {eval_metric}")

        if args.with_tracking:
            if epoch_end:
                accelerator.log(
                    {
                        "accuracy" if args.task_name is not None else "glue": eval_metric,
                        "train_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )
            else:
                 accelerator.log(
                    {
                        "accuracy" if args.task_name is not None else "glue": eval_metric,
                    },
                    step=completed_steps,
                )
                 
        return eval_metric
    
    if args.eval_only:
        epoch = 0
        eval_metric = do_eval()
        exit()

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            if args.use_ivon:
                mc_loss = 0
                for _ in range(args.ivon_train_mc_samples):
                    with optimizer.optimizer.sampled_params(train=True):
                        outputs = model(**batch)
                        loss = outputs.loss
                        mc_loss += loss.detach().float()
                        loss = loss / args.gradient_accumulation_steps
                        accelerator.backward(loss)
                mc_loss /= args.ivon_train_mc_samples
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += mc_loss
                progress_bar.set_description(f"Loss: {mc_loss:.4f}, LR: {lr_scheduler.get_lr()[0]:.2E}")
            else:
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                progress_bar.set_description(f"Loss: {loss.item():.4f}, LR: {lr_scheduler.get_lr()[0]:.2E}")
            if args.with_tracking:
                accelerator.log(
                    {
                        "loss": loss.item(),
                        "lr": lr_scheduler.get_lr()[0]
                    },
                    step=completed_steps,
                )
                if args.use_ivon:
                    accelerator.log(optimizer.optimizer.hess_stats()[0], step=completed_steps)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                if args.apply_lora and args.lora_type in ["adalora", "vilora"]:
                    model.base_model.update_and_allocate(completed_steps)
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

            if args.eval_steps > 0 and epoch >= args.eval_after_epochs and completed_steps % args.eval_steps == 0:
                eval_metric = do_eval()
                model.train()

        eval_metric = do_eval(epoch_end=True)

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        if args.use_ivon:
            if args.apply_lora and args.lora_type in ["adalora", "vilora"]:
                save_ivon_hessian_state(resize=True)
            else:
                save_ivon_hessian_state()
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    main()
