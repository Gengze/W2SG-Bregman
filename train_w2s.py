import json
import os
import random
import subprocess
from typing import Dict, List, Optional
import datasets
import fire
import numpy as np
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
import time
import sys
from datetime import datetime

import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset, load_reward_dataset, load_helpful_dataset,
                                     load_w2s_dataset, tokenize_dataset, load_multi_label_dataset, load_multi_label_w2s_dataset)
from weak_to_strong.loss import (logconf_loss_fn, product_loss_fn, xent_loss, bce_loss, logconf_bce_loss_fn, kl_loss, reverse_kl_loss, reverse_bce_loss,
                                CACE_loss, aux_conf_loss, SL_loss)
from weak_to_strong.train import ModelConfig, train_and_save_model, train_and_save_reward_model
import re
from ruptures import Binseg


MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        path="./Models/gpt2",
        default_lr=1e-5,
        eval_batch_size=64,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        )
    ),
    ModelConfig(
        name="gpt2-medium",
        path="./Models/gpt2-medium",
        default_lr=1e-5,
        eval_batch_size=64,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        )
    ),
    ModelConfig(
        name="gpt2-large",
        path="./Models/gpt2-large",
        default_lr=1e-5,
        # eval_batch_size=64,
        eval_batch_size=32,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        )
    ),
    ModelConfig(
        name="gpt2-xl",
        path="./Models/gpt2-xl",
        default_lr=1e-5,
        eval_batch_size=64,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        )
    ),
]
MODELS_DICT: Dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
}


loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
    "bce": bce_loss(),
    "reverse_bce": reverse_bce_loss(),
    "kl": kl_loss(),
    "reverse_kl": reverse_kl_loss(),
    "CACE": CACE_loss(),
    "aux": aux_conf_loss(),
    "SL": SL_loss(),
}

VALID_LOSSES: List[str] = list(loss_dict.keys())


def get_config_foldername(config: dict) -> str:
    def shorten_key(key: str) -> str:
        return "".join(word[0] for word in key.split("_"))

    def shorten_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, str):
            value = value.split("/")[-1]
            if "_" in value:
                return "_".join(word[:4] for word in value.split("_"))
            else:
                return value
        else:
            return str(value)

    return "-".join(f"{shorten_key(k)}={shorten_value(v)}" for k, v in sorted(config.items()))


def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "cai",
    loss: str = "bce",
    w2s_loss: Optional[str] = None,
    n_docs: int = 20000, # the number of docs for ground-truth fine-tuning
    n_w2s_docs: Optional[int] = 0, # the number of docs for weak-to-stong fine-tuning
    n_test_docs: int = 10000,
    use_mixed_data: bool = False, # if use mixed data, you should double the n_docs, as training weak model and w2s will both use mixture data
    use_human_data: bool = False,
    use_reward_mechanism: bool = False, # if use reward mechanism, the extra data will be the same as w2s data, but the model will be given extra reward when it produce harmful content
    n_extra_docs: Optional[int] = 0,
    model_size: str = "gpt2",
    # model_path: str = "gpt2", # load local model path
    lr: Optional[float] = None,
    optim: Optional[str] = None,
    epochs: int = 2,
    force_retrain: bool = False,
    seed: int = 42,
    minibatch_size_per_device: Optional[int] = None,
    train_with_dropout: bool = False,
    results_folder: str = "results",
    weak_labels_folder: Optional[str] = None,
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    # Note: you can pass either weak_model_size or weak_labels_path. If you pass
    # weak_model_size, we will guess the path to the weak labels based on the weak
    # model. If you pass weak_labels_path, we will use that path instead.
    # If you pass neither, we will train on ground truth.
    weak_model_size: Optional[str] = None,
    # weak_model_path: Optional[str] = None, # local local weak model path
    weak_labels_path: Optional[str] = None,
    sweep_subfolder: str = "default_gpt",
    # Set to a very large value so that by default we don't do any intermediate evals but
    # still do final evals (which requires eval_every to be set to a non-zero, non-None value)
    eval_every: int = 1000000,
    sync_command: Optional[str] = None,
    # whethe freeze base LM when fine-tuning
    freeze_lm: bool = False,
    conf_threshold: Optional[float] = 0.75,
    reward_conf: Optional[float] = 0.2,
    reward_alpha: Optional[float] = 0.5,
    epsilon: Optional[float] = 0.0,
    noisy_rate: Optional[float] = 0.0,
    noisy_alpha: Optional[float] = 1.0,
    save_hard_label: Optional[bool] = False,
    CACE_c: Optional[float] = 0.0,
):
    # for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"train_config: ds: {ds_name}, weak_model_size: {weak_model_size}, model_size: {model_size}, seed: {seed}, loss: {w2s_loss}, na: {noisy_alpha}, nr: {noisy_rate}, epochs: {epochs}, save_hard_label: {save_hard_label}")
    
    if w2s_loss == 'bce':
        epsilon = 0 
    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1
    assert ds_name in VALID_DATASETS, f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
    assert (
        weak_model_size is None or weak_labels_path is None
    ), "Can't pass both weak_model_size and weak_labels_path"
    model_config = MODELS_DICT[model_size]
    use_default_lr = False
    if lr is None:
        lr = model_config.default_lr
        use_default_lr = True

    if optim is None:
        optim = model_config.default_optimizer

    # The commented out terms are the ones that should not change final results
    config = {
        "batch_size": batch_size,
        "max_ctx": max_ctx,
        "ds_name": ds_name,
        "loss": w2s_loss if w2s_loss is not None else loss,
        "n_docs": n_docs,
        "n_test_docs": n_test_docs,
        "model_size": model_size,
        "lr": lr,
        "optim": optim,
        "epochs": epochs,
        "seed": seed,
        "train_with_dropout": train_with_dropout,
        "linear_probe": linear_probe,
        "lr_schedule": lr_schedule,
        "eval_every": eval_every,
        "use_mixed_data": use_mixed_data,
        "use_human_data": use_human_data,
        "use_reward_mechanism": use_reward_mechanism,
        "n_extra_docs": n_extra_docs if use_human_data else 0,
    }

    if weak_model_size is not None:
        weak_model_config = config.copy()
        weak_model_config["model_size"] = weak_model_size
        weak_model_config["loss"] = loss
        weak_model_config["use_human_data"] = False
        weak_model_config["use_reward_mechanism"] = False
        weak_model_config["n_extra_docs"] = 0
        weak_model_config["epochs"] = 1
        weak_model_config["seed"] = config['seed']
        config["epsilon"] = epsilon

        if use_default_lr:
            weak_model_config["lr"] = MODELS_DICT[weak_model_size].default_lr

        weak_model_config_name = get_config_foldername(weak_model_config)

        weak_labels_path = (
            results_folder + "/" + sweep_subfolder + "/" + weak_model_config_name + "/weak_labels"
        )
    
    eval_batch_size = model_config.eval_batch_size

    
    # Load reward dataset
    rejected_dataset, chosen_dataset = load_reward_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))

    # load extra helpful dataset
    if use_human_data:
        extra_rejected_dataset, extra_chosen_dataset = load_helpful_dataset(ds_name, seed=seed, split_sizes=dict(train=n_extra_docs, test=0))
        extra_rejected_dataset, extra_chosen_dataset = extra_rejected_dataset["train"], extra_chosen_dataset["train"]
        extra_rejected_dataset = extra_rejected_dataset.remove_columns([col for col in extra_rejected_dataset.column_names if col in ['chosen', 'rejected']])
        extra_chosen_dataset = extra_chosen_dataset.remove_columns([col for col in extra_chosen_dataset.column_names if col in ['chosen', 'rejected']])
        print("len(extra train):", len(extra_rejected_dataset))
    
    # Split the training dataset in half
    train_dataset_rejected, test_ds_rejected = rejected_dataset["train"], rejected_dataset["test"]
    train_dataset_chosen, test_ds_chosen = chosen_dataset["train"], chosen_dataset["test"]
    
    if weak_labels_path is None:
        train1_ds_rejected, train1_ds_chosen = train_dataset_rejected, train_dataset_chosen
        train2_ds_rejected, train2_ds_chosen = load_w2s_dataset(ds_name, seed=seed, split_sizes=dict(train=n_w2s_docs))
        train2_ds_rejected, train2_ds_chosen = train2_ds_rejected["train"], train2_ds_chosen["train"]
        train1_ds_rejected = train1_ds_rejected.shuffle(seed=seed)
        train1_ds_chosen = train1_ds_chosen.shuffle(seed=seed)
        print("len(train1):", len(train1_ds_rejected), "len(train2):", len(train2_ds_rejected))
        config_name = get_config_foldername(config)
    else:
        if not weak_labels_path.endswith("weak_labels"):
            weak_labels_path = weak_labels_path + "/weak_labels"
        
        train1_ds_rejected = load_from_disk(weak_labels_path + "/rejected")
        train1_ds_chosen = load_from_disk(weak_labels_path + "/chosen")
        
        train2_ds_rejected = None
        train2_ds_chosen = None
        
        if use_human_data:
            train1_ds_rejected = train1_ds_rejected.remove_columns([col for col in train1_ds_rejected.column_names if col not in extra_rejected_dataset.column_names])
            train1_ds_chosen = train1_ds_chosen.remove_columns([col for col in train1_ds_chosen.column_names if col not in extra_chosen_dataset.column_names])
            train1_ds_rejected = concatenate_datasets([train1_ds_rejected, extra_rejected_dataset])
            train1_ds_chosen = concatenate_datasets([train1_ds_chosen, extra_chosen_dataset])
        if use_reward_mechanism:
            config["reward_alpha"] = reward_alpha

        
        train1_ds_rejected = train1_ds_rejected.shuffle(seed)
        train1_ds_chosen = train1_ds_chosen.shuffle(seed)

        weak_model_config = json.load(open(weak_labels_path.replace("weak_labels", "config.json")))
        config["weak_model_size"] = weak_model_config["model_size"]
        config_name = get_config_foldername(config)
        config["weak_model"] = weak_model_config

    save_path = os.path.join(results_folder, sweep_subfolder, config_name)
    logger.configure(
        name="{sweep_subfolder}_{config_name}_{datetime_now}",
        save_path=save_path,
        sweep_subfolder=sweep_subfolder,
        config_name=config_name,
    )
    # Tokenize datasets
    print(model_config)

    tokenizer = get_tokenizer(model_config.path)
    
    train1_ds_rejected = tokenize_dataset(train1_ds_rejected, tokenizer, max_ctx)
    train1_ds_chosen = tokenize_dataset(train1_ds_chosen, tokenizer, max_ctx)

    test_ds_rejected = tokenize_dataset(test_ds_rejected, tokenizer, max_ctx)
    test_ds_chosen = tokenize_dataset(test_ds_chosen, tokenizer, max_ctx)

    def get_noisy_index(n_w2s_docs, noisy_rate):
        n_noisy = int(n_w2s_docs * noisy_rate)
        
        all_indices = list(range(n_w2s_docs))
        noisy_index = random.sample(all_indices, n_noisy)
        return noisy_index

    def modify_soft_labels_map(dataset_rejected, dataset_chosen, noisy_index: List[int], noise_alpha):
        noisy_set = set(noisy_index)
        
        def modify_rejected(example, idx, alpha = noise_alpha):
            if idx in noisy_set:
                example['soft_label'] = 0.5 + alpha* (example['soft_label'] - 0.5)
            return example
        
        def modify_chosen(example, idx, alpha = noise_alpha):
            if idx in noisy_set:
                example['soft_label'] = 0.5 + alpha* (example['soft_label'] - 0.5)
            return example
        
        modified_rejected = dataset_rejected.map(
            modify_rejected,
            with_indices=True
        )
        modified_chosen = dataset_chosen.map(
            modify_chosen,
            with_indices=True
        )    
        return modified_rejected, modified_chosen

    if noisy_rate > 0: 
        noisy_index = get_noisy_index(n_w2s_docs, noisy_rate)
        train1_ds_rejected, train1_ds_chosen = modify_soft_labels_map(train1_ds_rejected, train1_ds_chosen, noisy_index, noisy_alpha)

    if train2_ds_rejected:
        train2_ds_rejected = tokenize_dataset(train2_ds_rejected, tokenizer, max_ctx)
    if train2_ds_chosen:
        train2_ds_chosen = tokenize_dataset(train2_ds_chosen, tokenizer, max_ctx)
    

    auto_threshold = 0
    
    if w2s_loss.startswith("Quan_CACE"):
        num_bin = 10
        confidence_scores = np.abs((np.array(train1_ds_rejected['soft_label']) -0.5))
        quantile_positions = np.linspace(0, 1, num_bin + 1)
        quantile_values = np.quantile(confidence_scores, quantile_positions)
        quan_idx = int(w2s_loss.split('_')[-1])
        auto_threshold = quantile_values[quan_idx]
    


    def get_loss_function(loss, loss_dict, auto_threshold):
        if loss.startswith("CACE_"):
            try:
                confidence_threshold = float(loss.split("_")[1])
                loss_fn = CACE_loss(confidence_threshold=confidence_threshold)
                return loss_fn
            except (IndexError, ValueError):
                raise ValueError(f"Invalid CACE loss format: {loss}. Expected 'CACE_<number>'")

        
        if loss.startswith("Quan_CACE"):
            loss_fn = CACE_loss(confidence_threshold = auto_threshold)
            return loss_fn
        
        if loss.startswith("SL_"):
            match = re.search(r"a([\d.]+)_b([\d.]+)", loss)
            if match:
                a_value = float(match.group(1))
                b_value = float(match.group(2))
            loss_fn = SL_loss(alpha = a_value, beta = b_value)
            return loss_fn
        
        logconf_pattern = r'^logconf_c(\d+\.?\d*)_w(\d+\.?\d*)$'
        match = re.match(logconf_pattern, loss)
        if match:
            try:
                aux_coef = float(match.group(1))
                warmup_frac = float(match.group(2))
                loss_fn = aux_conf_loss(aux_coef=aux_coef, warmup_frac=warmup_frac)
                return loss_fn
            except ValueError:
                raise ValueError(f"Invalid logconf loss format: {loss}. Expected 'logconf_c<number>_w<number>'")
        
        if loss in loss_dict:
            return loss_dict[loss]
        else:
            raise KeyError(f"Loss function '{loss}' not found in loss_dict and doesn't match any special pattern")

    loss_fn = get_loss_function(w2s_loss, loss_dict, auto_threshold)
    print(f"Training model, size {model_size}")

    test_results_rejected, test_results_chosen, weak_ds_rejected, weak_ds_chosen, init_acc, distance_to_init, final_eval_loss = train_and_save_reward_model(
        model_config,
        train1_ds_rejected,
        train1_ds_chosen,
        test_ds_rejected,
        test_ds_chosen,
        inference_ds_rejected=train2_ds_rejected,
        inference_ds_chosen=train2_ds_chosen,
        ds_name=ds_name,
        batch_size=batch_size,
        save_path=save_path,
        loss_fn=loss_fn,
        lr=lr,
        epochs=epochs,
        force_retrain=force_retrain,
        eval_batch_size=eval_batch_size,
        minibatch_size_per_device=minibatch_size_per_device,
        train_with_dropout=train_with_dropout,
        linear_probe=linear_probe,
        lr_schedule=lr_schedule,
        optimizer_name=optim,
        eval_every=eval_every,
        freeze_lm=freeze_lm,
        use_reward_mechanism=use_reward_mechanism,
        reward_conf=reward_conf,
        reward_alpha=reward_alpha,
        epsilon=epsilon,
    )
    
    
    def label_hard(ds):
        def map_soft_label(example):
            soft_label = example['soft_label']
            if soft_label < 0.5:
                example['soft_label'] = 0.0
            else:
                example['soft_label'] = 1.0
            return example
        
        ds = ds.map(map_soft_label)
        return ds

    


    if weak_ds_rejected is not None:
        if save_hard_label:
            weak_ds_rejected = label_hard(weak_ds_rejected)
            

        weak_ds_rejected.save_to_disk(save_path + "/" + "weak_labels" + "/" + "rejected")

    if weak_ds_chosen is not None:
        if save_hard_label:
            weak_ds_chosen = label_hard(weak_ds_chosen)

        weak_ds_chosen.save_to_disk(save_path + "/" + "weak_labels" + "/" + "chosen")

    acc = np.mean([x["acc"] for x in test_results_rejected])
    res_dict = {"accuracy": acc}
    print("accuracy:", acc)

    log_message = f"ds: {ds_name}, weak_model_size: {weak_model_size}, model_size: {model_size}, seed: {seed}, loss: {w2s_loss}, alpha: {noisy_alpha}, noisy_rate: {noisy_rate}, distance_to_init: {distance_to_init}, init_acc: {init_acc}, eval_acc: {acc}, eval_loss: {final_eval_loss}, hard_label: {save_hard_label}"
    with open("my_result_log.log", "a") as log_file:
        log_file.write(log_message + "\n")


    with open(os.path.join(save_path, f"config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(save_path, f"results_summary.json"), "w") as f:
        json.dump(res_dict, f, indent=2)

    if sync_command is not None:
        print("Syncing results to remote storage...")
        try:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(["upload", save_path, results_folder])
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(f"Sync command failed with return code {result.returncode}")
        except Exception as e:
            raise RuntimeError("Failed to sync results to remote storage.") from e


if __name__ == "__main__":

    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nStart: {start_datetime}")
    

    fire.Fire(main)
    torch.cuda.empty_cache()

    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%-%m-%d %H:%M:%S")
    elapsed_time = end_time - start_time
    
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    
    print(f"End: {end_datetime}")
    print(f"Time: {time_str} (HH:MM:SS)\n")