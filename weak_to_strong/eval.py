import datasets
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import copy
from weak_to_strong.loss import (logconf_loss_fn, product_loss_fn, xent_loss, bce_loss, logconf_bce_loss_fn, kl_loss, reverse_kl_loss, reverse_bce_loss,
                                reverse_logconf_bce_loss_fn, logconf_kl_loss_fn, reverse_logconf_kl_loss_fn, 
                                jeffreys_divergence, hellinger_distance, jensen_divergence, chi_square_divergence,
                                logconf_jeffreys_divergence, logconf_hellinger_distance,
                                logconf_jensen_divergence, logconf_chi_square_divergence)
from typing import Callable, Optional
from tqdm import tqdm


def to_batch(x, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]

def to_batch_2(x1, x2, batch_size):
    assert len(x1) == len(x2)
    for i in range(0, len(x1), batch_size):
        yield (x1[i : i + batch_size], x2[i : i + batch_size])


def unpack(x):
    assert isinstance(x, torch.Tensor), type(x)
    return x.detach().float().cpu().numpy().tolist()

def get_log_ps(logits, idxs, loss_mask):
        """
        args:
        logits: A tensor of shape (batch_size, seq_len, vocab_size)
        idxs: A torch.long tensor of shape (batch_size, seq_len)
        loss_mask: A torch.float tensor of shape (batch_size, seq_len)

        returns:
        A tensor of shape (batch_size, seq_len), the log probabilities of each sequence in the batch
        """

        idxs = idxs[:, 1:].unsqueeze(2)
        loss_mask = loss_mask[:, 1:]
        log_p_distributions = F.log_softmax(logits, dim=-1)[:, :-1]
        log_ps = torch.gather(log_p_distributions, dim=2, index=idxs).squeeze(2)
        return (log_ps * loss_mask)#.sum(dim=-1)
    

def eval_model_acc(model: nn.Module, ds: datasets.Dataset, eval_batch_size: int = 16) -> None:
    """
    This function evaluates the accuracy of a given model on a given dataset.

    Parameters:
    model (nn.Module): The model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): A list of dictionaries containing the input_ids, ground truth label, predicted label,
                    accuracy of prediction, logits and soft label for each example in the dataset.
    """

    model.eval()

    with torch.no_grad():
        results = []
        # for ex in ds:
        for batch in to_batch(ds, eval_batch_size):
            # pad input_ids to common length
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            labels = batch["soft_label"]
            # run forward pass
            raw_logits = model(input_ids)

            probs = unpack(torch.nn.functional.softmax(raw_logits, dim=-1))
            logits = unpack(raw_logits)

            preds = np.argmax(probs, axis=-1)
            labels = np.argmax(labels, axis=-1)

            results.extend(
                [
                    dict(
                        txt=txt,
                        input_ids=input_id,
                        gt_label=label,
                        hard_label=pred,
                        acc=label == pred,
                        logits=logit,
                        soft_label=prob,
                    )
                    for input_id, txt, label, pred, prob, logit in zip(
                        batch["input_ids"], batch["txt"], labels, preds, probs, logits
                    )
                ]
            )
        accs = [r["acc"] for r in results]
        print("Accuracy:", np.mean(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))

        return datasets.Dataset.from_list(results)



def eval_reward_model_acc(model: nn.Module, ds_rejected: datasets.Dataset, ds_chosen: datasets.Dataset, eval_batch_size: int = 16) -> None:
    """
    This function evaluates the accuracy of a given reward model on a given dataset.

    Parameters:
    model (nn.Module): The reward model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): A list of dictionaries containing the input_ids, ground truth label, predicted label,
                    accuracy of prediction, logits and soft label for each example in the dataset.
    """

    model.eval()

    with torch.no_grad():
        results_rejected = []
        results_chosen = []
        # for ex in ds:
        for batch_rejected, batch_chosen in to_batch_2(ds_rejected, ds_chosen, eval_batch_size):
            # pad input_ids to common length
            input_ids_rejected = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_rejected["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            input_ids_chosen = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_chosen["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            labels = batch_rejected["soft_label"]
            # run forward pass
            raw_logits_rejected = model(input_ids_rejected)
            raw_logits_chosen = model(input_ids_chosen)
            raw_logits = raw_logits_chosen - raw_logits_rejected
            raw_logits = raw_logits.squeeze()
            probs = unpack(torch.sigmoid(raw_logits))
            logits = unpack(raw_logits)
 
            preds = np.array([int(a >= 0.5) for a in probs])
            labels = np.array([int(a >= 0.5) for a in labels])
            results_rejected.extend(
                [
                    dict(
                        txt=txt,
                        input_ids=input_id,
                        gt_label=label,
                        hard_label=pred,
                        acc=label == pred,
                        logits=logit,
                        soft_label=prob,
                    )
                    for input_id, txt, label, pred, prob, logit in zip(
                        batch_rejected["input_ids"], batch_rejected["txt"], labels, preds, probs, logits
                    )
                ]
            )
            results_chosen.extend(
                [
                    dict(
                        txt=txt,
                        input_ids=input_id,
                        gt_label=label,
                        hard_label=pred,
                        acc=label == pred,
                        logits=logit,
                        soft_label=prob,
                    )
                    for input_id, txt, label, pred, prob, logit in zip(
                        batch_chosen["input_ids"], batch_chosen["txt"], labels, preds, probs, logits
                    )
                ]
            )
        accs = [r["acc"] for r in results_rejected]
        print("Accuracy:", np.mean(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))

        return datasets.Dataset.from_list(results_rejected), datasets.Dataset.from_list(results_chosen)


def eval_reward_model_acc_loss(
    model: nn.Module, 
    ds_rejected: datasets.Dataset, 
    ds_chosen: datasets.Dataset, 
    eval_batch_size: int = 16,
    loss_fn: Callable = bce_loss,
    epsilon: float = 0.0
) -> None:
    """
    This function evaluates the accuracy and loss of a given reward model on a given dataset.

    Parameters:
    model (nn.Module): The reward model to be evaluated.
    ds_rejected (datasets.Dataset): Dataset containing rejected examples.
    ds_chosen (datasets.Dataset): Dataset containing chosen examples.
    eval_batch_size (int): Batch size for evaluation.
    loss_fn (Callable): Loss function to use for evaluation.
    epsilon (float): Epsilon parameter for the loss function.

    Returns:
    Tuple containing:
        - Dataset with rejected examples' predictions
        - Dataset with chosen examples' predictions
        - Mean evaluation loss across all batches
    """
    model.eval()

    total_batches = len(ds_rejected) // eval_batch_size
    if len(ds_rejected) % eval_batch_size != 0:
        total_batches += 1

    with torch.no_grad():
        results_rejected = []
        results_chosen = []
        batch_eval_loss_list = []
        

        for batch_rejected, batch_chosen in to_batch_2(ds_rejected, ds_chosen, eval_batch_size):
            # pad input_ids to common length
            input_ids_rejected = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_rejected["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            input_ids_chosen = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_chosen["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            
            # Convert labels to tensor
            labels = torch.tensor(batch_rejected["soft_label"]).float().to(input_ids_rejected.device)
            
            # run forward pass
            raw_logits_rejected = model(input_ids_rejected)
            raw_logits_chosen = model(input_ids_chosen)
            raw_logits = raw_logits_chosen - raw_logits_rejected
            raw_logits = raw_logits.squeeze()
            
            # Calculate and store batch loss
            if epsilon > 0.0:
                batch_loss = loss_fn(raw_logits, labels, step_frac= -1,epsilon=epsilon)
            else:
                batch_loss = loss_fn(raw_logits, labels, step_frac= -1)
            batch_eval_loss_list.append(batch_loss.item())

            # Convert to numpy for metrics
            probs = unpack(torch.sigmoid(raw_logits))
            logits = unpack(raw_logits)
            preds = np.array([int(a >= 0.5) for a in probs])
            labels_np = np.array([int(a >= 0.5) for a in batch_rejected["soft_label"]])
            
            # Store results
            results_rejected.extend(
                [
                    dict(
                        txt=txt,
                        input_ids=input_id,
                        gt_label=label,
                        hard_label=pred,
                        acc=label == pred,
                        logits=logit,
                        soft_label=prob,
                    )
                    for input_id, txt, label, pred, prob, logit in zip(
                        batch_rejected["input_ids"], batch_rejected["txt"], labels_np, preds, probs, logits
                    )
                ]
            )
            results_chosen.extend(
                [
                    dict(
                        txt=txt,
                        input_ids=input_id,
                        gt_label=label,
                        hard_label=pred,
                        acc=label == pred,
                        logits=logit,
                        soft_label=prob,
                    )
                    for input_id, txt, label, pred, prob, logit in zip(
                        batch_chosen["input_ids"], batch_chosen["txt"], labels_np, preds, probs, logits
                    )
                ]
            )

        # Calculate metrics
        accs = [r["acc"] for r in results_rejected]
        mean_acc = np.mean(accs)
        acc_std = np.std(accs) / np.sqrt(len(accs))
        eval_loss = np.mean(batch_eval_loss_list)

        print(f"Eval_Accuracy: {mean_acc:.4f} +/- {acc_std:.4f}")
        print(f"Eval_Loss: {eval_loss:.4f}")

        return datasets.Dataset.from_list(results_rejected), datasets.Dataset.from_list(results_chosen), eval_loss

