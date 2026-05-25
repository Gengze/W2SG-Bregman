import itertools
import os
import pickle
import time
from dataclasses import dataclass
from typing import Callable, Optional, List

import datasets
import numpy as np
import torch
import torch_optimizer as toptim
from transformers.modeling_utils import load_sharded_checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
import torch.nn.functional as F
import weak_to_strong.logger as logger
from weak_to_strong.common import clear_mem
from weak_to_strong.eval import eval_model_acc, eval_reward_model_acc, eval_model_acc_single_head, eval_reward_model_acc_loss
from weak_to_strong.loss import xent_loss, bce_loss
from weak_to_strong.model import TransformerWithHead, TransformerWithSingleHead, TransformerWithMultiLabelHead
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.distributed.fsdp import (
  FullyShardedDataParallel as FSDP,
  CPUOffload,
  MixedPrecision
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from functools import partial
import copy


@dataclass
class ModelConfig:
    name: str
    path: str # local model path
    default_lr: float
    eval_batch_size: int
    custom_kwargs: Optional[dict] = None
    gradient_checkpointing: bool = False
    model_parallel: bool = False
    default_optimizer: str = "adam"


def train_model(
    model: torch.nn.Module,
    ds: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-5,
    loss_fn: Callable = xent_loss,
    log_every: int = 10,
    eval_every: int = 100,
    eval_batch_size: int = 256,
    minibatch_size: int = 8,
    eval_ds: Optional[datasets.Dataset] = None,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
    epochs: int = 1,
    lr_schedule: str = "cosine_anneal",
    optimizer_name: str = "adam",
    epsilon: Optional[float] = 0.0,
):
    print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"

    if train_with_dropout:
        model.train()
    else:
        model.eval()
    if gradient_checkpointing:
        (
            model if hasattr(model, "gradient_checkpointing_enable") else model.module
        ).gradient_checkpointing_enable()

    nsteps = len(ds) * epochs // batch_size

    def lr_schedule_fn(step):
        if lr_schedule == "constant":
            return 1
        else:
            assert False, f"invalid lr schedule, {lr_schedule}, must be constant or cosine_anneal"

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adafactor":
        optimizer = toptim.Adafactor(model.parameters(), lr=lr)
    else:
        assert False, f"invalid optimizer {optimizer_name}, must be adam or adafactor"
    if lr_schedule == "cosine_anneal":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
    step = 0
    it = itertools.chain.from_iterable(itertools.repeat(ds, epochs))
    losses = []
    accuracies = []
    eval_acc_dict = {}
    grad_norms = []


    io_device = model.device if hasattr(model, "device") else 0


    init_acc = 0
    init_model = copy.deepcopy(model)
    init_eval_results = eval_model_acc(model, eval_ds, eval_batch_size)
    init_acc = np.mean([r["acc"] for r in init_eval_results ])

    def get_grad_norm(model):
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += torch.sum(param.grad ** 2).item()
        grad_norm = grad_norm ** 0.5  
        return grad_norm 

    while step < nsteps:
        loss_tot = 0
        if eval_every and (step + 1) % eval_every == 0:
            eval_results = eval_model_acc(model, eval_ds, eval_batch_size)
            if gradient_checkpointing:
                (
                    model if hasattr(model, "gradient_checkpointing_enable") else model.module
                ).gradient_checkpointing_enable()
            if train_with_dropout:
                model.train()
            eval_accs = np.mean([r["acc"] for r in eval_results])
            eval_acc_dict[step] = eval_accs
            logger.logkv("eval_accuracy", eval_accs)
        all_logits = []
        all_labels = []
        for i in range(batch_size // minibatch_size):
            try:
                mbatch = [next(it) for _ in range(minibatch_size)]
            except StopIteration:
                break
            input_ids = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["input_ids"]) for ex in mbatch])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            labels = torch.tensor([ex["soft_label"] for ex in mbatch]).to(io_device)
            logits = model(input_ids)
            all_logits.extend(logits.to(io_device))
            all_labels.extend(labels)
        all_logits = torch.stack(all_logits)
        all_labels = torch.stack(all_labels)
        if epsilon > 0.0:
            loss = loss_fn(all_logits, all_labels, step_frac=step / nsteps, epsilon = epsilon)
        else:
            loss = loss_fn(all_logits, all_labels, step_frac=step / nsteps)
        loss_tot += loss.item()
        loss.backward()
        losses.append(loss_tot)

        current_grad_norm = get_grad_norm(model)
        grad_norms.append(current_grad_norm)

        accuracies.append(
            torch.mean(
                (torch.argmax(all_logits, dim=1) == torch.argmax(all_labels, dim=1)).to(
                    torch.float32
                )
            ).item()
        )
        logger.logkvs(
            {
                "step": step,
                "progress": step / nsteps,
                "loss": loss_tot,
                "grad_norm": grad_norms[-1],
                "train_accuracy": accuracies[-1],
                "lr": lr_scheduler.get_last_lr()[0],
            }
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        if log_every and step % log_every == 0:
            print(
                f"Step: {step}/{nsteps} Recent losses: {np.mean(losses)} grad_norm:{np.mean(grad_norms)} Avg. Acc: {np.mean(accuracies)} Total Num. Losses: {len(losses)}"
            )
            losses = []
            accuracies = []
        step += 1
        logger.dumpkvs()
    final_eval_results = None

    def get_distance_to_init(model1, model2):
        l2_distance = 0.0
        
        with torch.no_grad():
            compute_device = torch.device('cpu')
            
            for param1, param2 in zip(model1.parameters(), model2.parameters()):
                param1 = param1.to(compute_device)
                param2 = param2.to(compute_device)
                
                l2_distance += torch.sum((param1 - param2) ** 2).item()
        
        return l2_distance ** 0.5

    distance_to_init = get_distance_to_init(init_model, model)

    if eval_every:
        print("Final evaluation:")
        final_eval_results = eval_model_acc(model, eval_ds, eval_batch_size)
        logger.logkv("eval_accuracy", np.mean([r["acc"] for r in final_eval_results]))
        logger.dumpkvs()
    return final_eval_results, init_acc, distance_to_init


def train_reward_model(
    model: torch.nn.Module,
    ds_rejected: datasets.Dataset,
    ds_chosen: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-5,
    loss_fn: Callable = bce_loss,
    log_every: int = 10,
    eval_every: int = 100,
    eval_batch_size: int = 256,
    minibatch_size: int = 8,
    eval_ds_rejected: Optional[datasets.Dataset] = None,
    eval_ds_chosen: Optional[datasets.Dataset] = None,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
    epochs: int = 1,
    lr_schedule: str = "cosine_anneal",
    optimizer_name: str = "adam",
    epsilon: float = 0.0,
):
    assert len(ds_rejected) == len(ds_chosen)
    assert len(eval_ds_rejected) == len(eval_ds_chosen)
    print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"


    if train_with_dropout:
        model.train()
    else:
        model.eval()
    if gradient_checkpointing:
        (
            model if hasattr(model, "gradient_checkpointing_enable") else model.module
        ).gradient_checkpointing_enable()

    nsteps = len(ds_rejected) * epochs // batch_size

    def lr_schedule_fn(step):
        if lr_schedule == "constant":
            return 1
        else:
            assert False, f"invalid lr schedule, {lr_schedule}, must be constant or cosine_anneal"

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adafactor":
        optimizer = toptim.Adafactor(model.parameters(), lr=lr)
    else:
        assert False, f"invalid optimizer {optimizer_name}, must be adam or adafactor"
    if lr_schedule == "cosine_anneal":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
    step = 0
    it_rejected = itertools.chain.from_iterable(itertools.repeat(ds_rejected, epochs))
    it_chosen = itertools.chain.from_iterable(itertools.repeat(ds_chosen, epochs))
    losses = []
    accuracies = []
    eval_acc_dict = {}
    grad_norms = [] 
    mini_grad_list = []


    io_device = model.device if hasattr(model, "device") else 0
    
    
    init_acc = 0
    init_model = copy.deepcopy(model)
    init_eval_results_rejected, init_eval_results_chosen, init_eval_loss = eval_reward_model_acc_loss(model, eval_ds_rejected, eval_ds_chosen, eval_batch_size, loss_fn, epsilon)
    
    

    init_acc = np.mean([r["acc"] for r in init_eval_results_rejected])

    
    torch.cuda.empty_cache()


    def get_grad_norm(model):
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += torch.sum(param.grad ** 2).item() 
        grad_norm = grad_norm ** 0.5
        return grad_norm
    

    while step < nsteps:
        loss_tot = 0
        if eval_every and (step + 1) % eval_every == 0:
            eval_results_rejected, eval_results_chosen, eval_loss = eval_reward_model_acc_loss(model, eval_ds_rejected, eval_ds_chosen, eval_batch_size, loss_fn, epsilon)
            if gradient_checkpointing:
                (
                    model if hasattr(model, "gradient_checkpointing_enable") else model.module
                ).gradient_checkpointing_enable()
            if train_with_dropout:
                model.train()
            eval_accs = np.mean([r["acc"] for r in eval_results_rejected])
            eval_acc_dict[step] = eval_accs
            logger.logkv("eval_accuracy", eval_accs)
            
        all_logits_rejected = []
        all_logits_chosen = []
        all_logits = []
        all_labels = []
        all_gt_labels = []
        
        for i in range(batch_size // minibatch_size):
            torch.cuda.empty_cache()

            try:
                mbatch_rejected = [next(it_rejected) for _ in range(minibatch_size)]
                mbatch_chosen = [next(it_chosen) for _ in range(minibatch_size)]
            except StopIteration:
                break
            input_ids_rejected = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["input_ids"]) for ex in mbatch_rejected])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            input_ids_chosen = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["input_ids"]) for ex in mbatch_chosen])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            # soft_labels are the same in both two parts
            labels = torch.tensor([ex["soft_label"] for ex in mbatch_rejected]).to(io_device)
            # if gt_label exists, during weak-to-strong generalization, use gt_label as reference
            gt_labels = torch.tensor([ex["gt_label"] if "gt_label" in ex else ex["soft_label"] for ex in mbatch_rejected]).to(io_device)
            logits_rejected = model(input_ids_rejected)
            logits_chosen = model(input_ids_chosen)
            #####
            
            del mbatch_rejected, mbatch_chosen
            torch.cuda.empty_cache()

            logits = logits_chosen - logits_rejected
            logits = logits.to(io_device)
            labels = labels.unsqueeze(1)
            if epsilon > 0.0:
                current_loss = loss_fn(logits, labels, step_frac=step / nsteps, epsilon=epsilon)
            else:
                current_loss = loss_fn(logits, labels, step_frac=step / nsteps)
            current_loss /= (batch_size // minibatch_size)
            loss_tot += current_loss.item()
            current_loss.backward()
            
        
            all_logits.extend(logits.detach())
            all_labels.extend(labels.detach())
            all_gt_labels.extend(gt_labels.detach())
            #######

        all_logits = torch.stack(all_logits)
        all_labels = torch.stack(all_labels)
        all_gt_labels = torch.stack(all_gt_labels)
        all_logits = all_logits.squeeze()
        losses.append(loss_tot)

        current_grad_norm = get_grad_norm(model)
        grad_norms.append(current_grad_norm)
        

        accuracies.append(
            torch.mean(
                ((all_logits >= 0.0) == (all_gt_labels == 1.0)).to(
                    torch.float32
                )
            ).item()
        )
        logger.logkvs(
            {
                "step": step,
                "progress": step / nsteps,
                "loss": loss_tot,
                "grad_norm": grad_norms[-1],
                "train_accuracy": accuracies[-1],
                "lr": lr_scheduler.get_last_lr()[0],
            }
        )

            
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        if log_every and step % log_every == 0:
            print(
                f"Step: {step}/{nsteps} Recent losses:{np.mean(losses)} Acc:{np.mean(accuracies)} grad_norm:{np.mean(grad_norms)} Total Num. Losses: {len(losses)}"
            )


            losses = []
            accuracies = []
            grad_norms = []

        step += 1

        
        
        logger.dumpkvs()

    
    final_eval_results_rejected = None
    final_eval_results_chosen = None
    
    def get_distance_to_init(model1, model2):

        distance_total = 0.0
        distance_linear = 0.0
        distance_LLM = 0.0
        
        with torch.no_grad():
            compute_device = torch.device('cpu')
            
            for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
                
                param1 = param1.to(compute_device)
                param2 = param2.to(compute_device)
                
                diff_sq = torch.sum((param1 - param2) ** 2).item()
                
                distance_total += diff_sq
                
                if "score" in name1:
                    distance_linear += diff_sq
                else:
                    distance_LLM += diff_sq
        
        return distance_total ** 0.5, distance_linear ** 0.5, distance_LLM ** 0.5

    distance_to_init, distance_linear, distance_LLM = get_distance_to_init(init_model, model)

    if eval_every:
        print("Final evaluation:")
        final_eval_results_rejected, final_eval_results_chosen, final_eval_loss = eval_reward_model_acc_loss(model, eval_ds_rejected, eval_ds_chosen, eval_batch_size, loss_fn, epsilon)
        logger.logkv("eval_accuracy", np.mean([r["acc"] for r in final_eval_results_rejected]))
        logger.dumpkvs()
        final_eval_acc = np.mean([r["acc"] for r in final_eval_results_rejected])

    return final_eval_results_rejected, final_eval_results_chosen, init_acc, distance_to_init, final_eval_loss



def train_and_save_model(
    model_config: ModelConfig,
    train_ds: datasets.Dataset,
    test_ds: datasets.Dataset,
    inference_ds: Optional[datasets.Dataset] = None,
    *,
    ds_name: str = "sciq",
    batch_size: int,
    lr: float,
    epochs: int,
    eval_batch_size: Optional[int] = None,
    minibatch_size_per_device: Optional[int] = None,
    save_path: Optional[str] = None,
    loss_fn: Callable = xent_loss,
    label: str = "default",
    force_retrain: bool = False,
    train_with_dropout: bool = False,
    linear_probe: bool = False,
    lr_schedule: str = "constant",
    optimizer_name: str = "adam",
    eval_every: Optional[int] = None,
    epsilon: Optional[float] = 0.0,
):
    
    if eval_batch_size is None:
        eval_batch_size = batch_size

    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1

    gradient_checkpointing = model_config.gradient_checkpointing
    custom_kwargs = model_config.custom_kwargs or {}


    def maybe_load_model(model):
        if os.path.exists(os.path.join(save_path, "results.pkl")) and not force_retrain:
            print("loading from", save_path)
            checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
            if not os.path.exists(checkpoint_path):
                # Assume this means we have a sharded checkpoint, and load it appropriately
                load_sharded_checkpoint(model, checkpoint_path)
            else:
                state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"))
                state_dict = {
                    k.replace("transformer.module", "transformer"): v
                    for (k, v) in state_dict.items()
                }
                custom_kwargs["state_dict"] = state_dict
            return True
        return False

    already_trained = False
    # Load the model
    if model_config.model_parallel:
        assert torch.cuda.device_count() > 1, f"you might want more gpus for {model_config.name}"
        if ds_name == 'anthropic_hh':
            model = TransformerWithSingleHead.from_pretrained(
                model_config.path,
                num_labels=1,
                device_map="auto",
                linear_probe=linear_probe,
                **custom_kwargs,
                )
        else:
            model = TransformerWithHead.from_pretrained(
                model_config.path,
                num_labels=2,
                device_map="auto",
                linear_probe=linear_probe,
                **custom_kwargs,
                )
        already_trained = maybe_load_model(model)
        # slight misnomer, more like minibatch_size_per_dp_replica
        minibatch_size = minibatch_size_per_device
    else:
        if ds_name == 'anthropic_hh':
            model = TransformerWithSingleHead.from_pretrained(
                model_config.path, num_labels=1, linear_probe=linear_probe, **custom_kwargs
            ).to("cuda")
        else:
            model = TransformerWithHead.from_pretrained(
                model_config.path, num_labels=2, linear_probe=linear_probe, **custom_kwargs
            ).to("cuda")
        already_trained = maybe_load_model(model)
        # data parallel:  currently not supported with model parallel

        minibatch_size = min(minibatch_size_per_device * torch.cuda.device_count(), batch_size)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, output_device=0)
            print(
                "Using",
                torch.cuda.device_count(),
                "GPUs, setting minibatch_size to",
                minibatch_size,
            )
        else:
            minibatch_size = minibatch_size_per_device

    if already_trained:
        test_results = eval_model_acc(model, test_ds, eval_batch_size)
    else:
        start = time.time()
        if ds_name == 'anthropic_hh':
            test_results, init_acc, distance_to_init = train_reward_model(
                model,
                train_ds,
                batch_size,
                lr=lr,
                epochs=epochs,
                eval_ds=test_ds,
                gradient_checkpointing=gradient_checkpointing,
                loss_fn=loss_fn,
                eval_batch_size=eval_batch_size,
                eval_every=eval_every,
                minibatch_size=minibatch_size,
                train_with_dropout=train_with_dropout,
                lr_schedule=lr_schedule,
                optimizer_name=optimizer_name,
                epsilon=epsilon,
            )
        else:
            test_results, init_acc, distance_to_init = train_model(
                model,
                train_ds,
                batch_size,
                lr=lr,
                epochs=epochs,
                eval_ds=test_ds,
                gradient_checkpointing=gradient_checkpointing,
                loss_fn=loss_fn,
                eval_batch_size=eval_batch_size,
                eval_every=eval_every,
                minibatch_size=minibatch_size,
                train_with_dropout=train_with_dropout,
                lr_schedule=lr_schedule,
                optimizer_name=optimizer_name,
                epsilon=epsilon,
            )
        print("Model training took", time.time() - start, "seconds")
        if save_path:
            # Note: If the model is wrapped by DataParallel, we need to unwrap it before saving
            (model if hasattr(model, "save_pretrained") else model.module).save_pretrained(
                save_path
            )
            print("saved", save_path)

    inference_results = None
    if inference_ds:
        inference_results = eval_model_acc(model, inference_ds, eval_batch_size)
        logger.logkv("inference_accuracy", np.mean([r["acc"] for r in inference_results]))

        print("inference_accuracy", np.mean([r["acc"] for r in inference_results]))

    if save_path:
        with open(os.path.join(save_path, "results.pkl"), "wb") as f:
            pickle.dump(
                {
                    "avg_acc_test": float(np.mean([r["acc"] for r in test_results])),
                    "avg_acc_inference": float(
                        np.mean([r["acc"] for r in inference_results] if inference_results else [])
                    ),
                    "test_results": test_results,
                    "inference_results": inference_results if inference_results else [],
                },
                f,
            )
    clear_mem()
    logger.shutdown()

    return test_results, inference_results, init_acc, distance_to_init

def train_and_save_reward_model(
    model_config: ModelConfig,
    train_ds_rejected: datasets.Dataset,
    train_ds_chosen: datasets.Dataset,
    test_ds_rejected: datasets.Dataset,
    test_ds_chosen: datasets.Dataset,
    inference_ds_rejected: Optional[datasets.Dataset] = None,
    inference_ds_chosen: Optional[datasets.Dataset] = None,
    *,
    ds_name: str = "sciq",
    batch_size: int,
    lr: float,
    epochs: int,
    eval_batch_size: Optional[int] = None,
    minibatch_size_per_device: Optional[int] = None,
    save_path: Optional[str] = None,
    loss_fn: Callable = bce_loss,
    label: str = "default",
    force_retrain: bool = False,
    train_with_dropout: bool = False,
    linear_probe: bool = False,
    lr_schedule: str = "constant",
    optimizer_name: str = "adam",
    eval_every: Optional[int] = None,
    freeze_lm: bool = False,
    use_reward_mechanism: bool = False,
    reward_conf: Optional[float] = 0.2,
    reward_alpha: Optional[float] = 0.5,
    epsilon: Optional[float] = 0.0,
):
    if eval_batch_size is None:
        eval_batch_size = batch_size

    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1

    gradient_checkpointing = model_config.gradient_checkpointing
    custom_kwargs = model_config.custom_kwargs or {}

    def maybe_load_model(model):
        if os.path.exists(os.path.join(save_path, "results.pkl")) and not force_retrain:
            print("loading from", save_path)
            checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
            if not os.path.exists(checkpoint_path):
                # Assume this means we have a sharded checkpoint, and load it appropriately
                load_sharded_checkpoint(model, checkpoint_path)
            else:
                state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"))
                state_dict = {
                    k.replace("transformer.module", "transformer"): v
                    for (k, v) in state_dict.items()
                }
                custom_kwargs["state_dict"] = state_dict
            return True
        return False

    already_trained = False
    # Load the model
    if model_config.model_parallel:
        assert torch.cuda.device_count() > 1, f"you might want more gpus for {model_config.name}"
        model = TransformerWithSingleHead.from_pretrained(
            model_config.path,
            num_labels=1,
            device_map="auto",
            linear_probe=linear_probe,
            **custom_kwargs,
            )
        already_trained = maybe_load_model(model)
        if already_trained:
            model.load_state_dict(custom_kwargs["state_dict"])
    
        # slight misnomer, more like minibatch_size_per_dp_replica
        minibatch_size = minibatch_size_per_device
        if freeze_lm:
            for name, param in model.named_parameters():
                if "score" not in name:
                    param.requires_grad = False
    else:
        model = TransformerWithSingleHead.from_pretrained(
            model_config.path, num_labels=1, linear_probe=linear_probe, **custom_kwargs
        ).to("cuda")
        already_trained = maybe_load_model(model)
        if already_trained:
            model.load_state_dict(custom_kwargs["state_dict"])

        # data parallel:  currently not supported with model parallel

        minibatch_size = min(minibatch_size_per_device * torch.cuda.device_count(), batch_size)
        if freeze_lm:
            for name, param in model.named_parameters():
                if "score" not in name:
                    param.requires_grad = False

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, output_device=0)
            print(
                "Using",
                torch.cuda.device_count(),
                "GPUs, setting minibatch_size to",
                minibatch_size,
            )
        else:
            minibatch_size = minibatch_size_per_device
    
    

    if already_trained:
        test_results_rejected, test_results_chosen = eval_reward_model_acc(model, test_ds_rejected, test_ds_chosen, eval_batch_size)
    else:
        start = time.time()
        if not use_reward_mechanism:
            test_results_rejected, test_results_chosen, init_acc, distance_to_init, final_eval_loss = train_reward_model(
                model,
                train_ds_rejected,
                train_ds_chosen,
                batch_size,
                lr=lr,
                epochs=epochs,
                eval_ds_rejected=test_ds_rejected,
                eval_ds_chosen=test_ds_chosen,
                gradient_checkpointing=gradient_checkpointing,
                loss_fn=loss_fn,
                eval_batch_size=eval_batch_size,
                eval_every=eval_every,
                minibatch_size=minibatch_size,
                train_with_dropout=train_with_dropout,
                lr_schedule=lr_schedule,
                optimizer_name=optimizer_name,
                epsilon=epsilon,
            )
        else:
            test_results_rejected, test_results_chosen = train_reward_model_v2(
                model,
                train_ds_rejected,
                train_ds_chosen,
                batch_size,
                lr=lr,
                epochs=epochs,
                eval_ds_rejected=test_ds_rejected,
                eval_ds_chosen=test_ds_chosen,
                gradient_checkpointing=gradient_checkpointing,
                loss_fn=loss_fn,
                eval_batch_size=eval_batch_size,
                eval_every=eval_every,
                minibatch_size=minibatch_size,
                train_with_dropout=train_with_dropout,
                lr_schedule=lr_schedule,
                optimizer_name=optimizer_name,
                reward_conf=reward_conf,
                reward_alpha=reward_alpha,
            )
        print("Model training took", time.time() - start, "seconds")
        if save_path:
            # Note: If the model is wrapped by DataParallel, we need to unwrap it before saving
            (model if hasattr(model, "save_pretrained") else model.module).save_pretrained(
                save_path
            )
            print("saved", save_path)

    inference_results_rejected = None
    inference_results_chosen = None
    if inference_ds_rejected and inference_ds_chosen:
        inference_results_rejected, inference_results_chosen = eval_reward_model_acc(model, inference_ds_rejected, inference_ds_chosen, eval_batch_size)
        logger.logkv("inference_accuracy", np.mean([r["acc"] for r in inference_results_rejected]))

        print("inference_accuracy", np.mean([r["acc"] for r in inference_results_rejected]))

    if save_path:
        with open(os.path.join(save_path, "results.pkl"), "wb") as f:
            pickle.dump(
                {
                    "avg_acc_test": float(np.mean([r["acc"] for r in test_results_rejected])),
                    "avg_acc_inference": float(
                        np.mean([r["acc"] for r in inference_results_rejected] if inference_results_rejected else [])
                    ),
                    "test_results_rejected": test_results_rejected,
                    "test_results_chosen": test_results_chosen,
                    "inference_results_rejected": inference_results_rejected if inference_results_rejected else [],
                    "inference_results_chosen": inference_results_chosen if inference_results_chosen else [],
                },
                f,
            )
    clear_mem()
    logger.shutdown()

    return test_results_rejected, test_results_chosen, inference_results_rejected, inference_results_chosen, init_acc, distance_to_init, final_eval_loss

