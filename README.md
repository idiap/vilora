# A Bayesian Interpretation of Adaptive Low-Rank Adaptation

This repository contains the source code for the paper "A Bayesian Interpretation of Adaptive Low-Rank Adaptation" by Haolin Chen and Philip N. Garner.

It comprises three components:

1. `run_glue_no_trainer.py`: the main Python script which is adapted from the Hugging Face [Transformers](https://github.com/huggingface/transformers/tree/v4.40.0/examples/pytorch) version 4.40.0.
2. peft: a customized Python package based on Hugging Face [PEFT](https://github.com/huggingface/peft/tree/v0.11.0) version 0.11.0. It includes the implementation of importance scores for AdaLoRA.
3. ivon: a slightly modified implementation of Improved Variational Online Newton ([IVON](https://github.com/team-approx-bayes/ivon)).

Licenses: 1 and 2 are licensed under Apache-2.0, 3 are licensed under GPL-3.0.

# Setup

1. Follow instructions from [Transformers](https://github.com/huggingface/transformers) to setup the python envrionment.
2. Install the customized peft and ivon packages.

## Fine-tuning

Scripts for fine-tuning are in `scripts`.

| File name                 | Model            | Optimizer | Criterion                                                       |
| ------------------------- | ---------------- | --------- | --------------------------------------------------------------- |
| full.sh                   | Full fine-tuning | Adam      | N/A                                                             |
| lora_all.sh               | LoRA             | Adam      | $r=2/4$                                                       |
| adalora.sh                | AdaLoRA          | Adam      | Sensitivity                                                     |
| adalora_ivon{_clr}.sh     | AdaLoRA          | IVON      | Sensitivity                                                     |
| vilora{_clr}.sh           | AdaLoRA          | IVON      | $\mathrm{SNR}(\|\theta\|)$                                    |
| vilora{_clr}_criterion.sh | AdaLoRA          | IVON      | $\mathrm{SNR}(\|\theta\|), \|\mu\|/\sigma, \|\mu\|, 1/\sigma$ |

* `clr` stands for customized learning rate schedule, which is used with IVON on COLA, STS-B, MRPC, and RTE.

## Evaluation

Evaluation is conducted automatically after fine-tuning.

<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>
