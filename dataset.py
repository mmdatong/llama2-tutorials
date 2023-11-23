# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets


def get_preprocessed_samsum(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("/root/.cache/huggingface/datasets/samsum", split=split)

    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset


def get_preprocessed_arithmetic(dataset_config, tokenizer, split):
    if split=="train":
        data_path = "arithmetic_data/arithmetic_train.csv"
    elif split=="validation":
        data_path = "arithmetic_data/arithmetic_validation.csv"
    elif split=="test":
        data_path = "arithmetic_data/arithmetic_test.csv"

    dataset = datasets.load_dataset(
        "csv", 
        data_files={split: "arithmetic_data/arithmetic_train.csv"}
        )[split]


    prompt = (
        f"Calculate the following expression:\n{{instruction}}\n---\nAnswer:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(instruction=sample["instruction"]),
            "output": sample["output"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(sample["output"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + answer,
            "attention_mask" : [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
