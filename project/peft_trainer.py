import pandas as pd
import time
import torch
import argparse

# import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GenerationConfig, TrainingArguments, Trainer
import constants as const

# from transformers import TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

train_file = const.TRAIN_DATASET
validation_file = const.DEV_DATASET
df_train = pd.read_csv(train_file)
df_val = pd.read_csv(validation_file)
src_col = "prompt_cs_generation"
trg_col = "counterspeech"

df_train = df_train[[src_col, trg_col]].dropna()
df_val = df_val[[src_col, trg_col]].dropna()

print(f"Train size: {df_train.shape}", f"Test size: {df_val.shape}")


def main(num_epochs, learning_rate, batch_size, max_length, model_name, output_folder):
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")
    else:
        print(f"Folder already exists: {output_folder}")
    
    # Print the training configuration
    print(f"Training configuration:")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}")
    print(f"Model name: {model_name}")
    print(f"Output folder: {output_folder}")

    sample_data = df_train.sample()
    dash_line = "_".join(" " for x in range(100))

    print(f"Input Text:\n{sample_data[src_col].iloc[0]}")
    print(dash_line, "\n")
    print(f"Groundtruth Counterspeech:\n{sample_data[trg_col].iloc[0]}")


    data_dict_train = {
        "input_prompt": df_train[src_col].values.tolist(),
        "output_prompt": df_train[trg_col].values.tolist(),
    }

    data_dict_eval = {
        "input_prompt": df_val[src_col].values.tolist(),
        "output_prompt": df_val[trg_col].values.tolist(),
    }

    # Create a Dataset object
    dataset_train = Dataset.from_dict(data_dict_train)
    dataset_eval = Dataset.from_dict(data_dict_eval)

    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


    print(print_number_of_trainable_model_parameters(original_model))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    max_source_length = max_length
    min_source_length = 64
    max_target_length = 512
    min_target_length = 32


    def tokenize_function(sample):
        sample["input_ids"] = tokenizer(
            sample["input_prompt"],
            max_length=max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        sample["labels"] = tokenizer(
            sample["output_prompt"],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return sample


    dataset_train_tokenized = dataset_train.map(tokenize_function, batched=True)
    dataset_eval_tokenized = dataset_eval.map(tokenize_function, batched=True)


    print(f"Input:")
    print(dataset_train_tokenized["input_ids"][0])
    print(dash_line, "\n")
    print(f"Output:")
    print(dataset_train_tokenized["labels"][0])


    lora_config = LoraConfig(
        r=768,
        lora_alpha=4096,
        target_modules=["q", "v"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    peft_model = get_peft_model(original_model, lora_config)
    print(print_number_of_trainable_model_parameters(peft_model))

    output_dir = f"/home/amey/coarl-counterspeech/checkpoints{model_name}-peft-finetune-{str(int(time.time()))}"

    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=50,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        save_total_limit=1,
        do_predict=True,  # generation in evaluation
    )

    torch.cuda.empty_cache()
    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=dataset_train_tokenized,
        eval_dataset=dataset_eval_tokenized,
    )

    print(dash_line)
    print(f"Starting PEFT training")
    peft_trainer.train()
    print(dash_line)

    print(f"Saving finetuned model to {output_folder}")
    peft_trainer.model.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Configuration")

    # Define arguments with default values from TRAINING_CONFIG
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum input length')
    parser.add_argument('--model_name', type=str, default='google/flan-t5-small', help='Model name to use for training')
    parser.add_argument('--output_folder', type=str, default='output', help='Folder to save training results')

    # Parse arguments from command line
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.num_epochs, args.learning_rate, args.batch_size, args.max_length, args.model_name, args.output_folder)