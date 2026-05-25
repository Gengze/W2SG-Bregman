import torch
from typing import Callable, Optional

class LossFnBase:
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        This function calculates the loss between logits and labels.
        """
        raise NotImplementedError

class aux_conf_loss(LossFnBase):
    """
    Auxiliary confidence loss for binary classification with scalar labels.
    Uses sigmoid activation and supports evaluation mode.

    Attributes:
        aux_coef: Maximum mixing coefficient (α_max in paper)
        warmup_frac: Fraction of training steps for warmup (default 0.1)
    """

    def __init__(
        self,
        aux_coef: float = 0.5,
        warmup_frac: float = 0.2,  # in terms of fraction of total training steps
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,      # shape: (batch_size,)
        labels: torch.Tensor,      # shape: (batch_size,) scalar labels
        step_frac: float,          # current step fraction [0,1]
        eval: bool = False,        # evaluation mode flag
        epsilon: Optional[float] = 0.0,
    ) -> torch.Tensor:
        # Ensure tensors are on same device
        labels = labels.to(logits.device).float()
        logits = logits.float()
        
        # Handle coefficient calculation
        if eval or (step_frac == -1):
            coef = 1.0  # full weight to strong predictions in eval mode
        else:
            # Linear warmup: from 0 to aux_coef over warmup_frac steps
            coef = min(1.0, step_frac / self.warmup_frac) * self.aux_coef
        
        # Get predictions (sigmoid instead of softmax)
        preds = torch.sigmoid(logits)  # shape: (batch_size,)
        
        # Calculate adaptive threshold
        mean_weak = torch.mean(labels)  # scalar value
        threshold = torch.quantile(preds, 1 - mean_weak)  # for binary case
        
        # Create hard predictions (binary)
        strong_preds = (preds >= threshold).float()  # shape: (batch_size,)
        
        # Mix targets (labels: weak, strong_preds: hard)
        target = labels * (1 - coef) + strong_preds.detach() * coef
        
        # Binary cross entropy loss
        loss = torch.nn.functional.binary_cross_entropy(
            preds,
            target,
            reduction='none'
        )
        
        return loss.mean()



# Custom loss function
class xent_loss(LossFnBase):
    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, step_frac: float
    ) -> torch.Tensor:
        """
        This function calculates the cross entropy loss between logits and labels.

        Parameters:
        logits: The predicted values.
        labels: The actual values.
        step_frac: The fraction of total training steps completed.

        Returns:
        The mean of the cross entropy loss.
        """
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss.mean()

class reverse_xent_loss(LossFnBase):
    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, step_frac: float, epsilon: float = 0.0,
    ) -> torch.Tensor:
        """
        This function calculates the cross entropy loss between logits and labels.

        Parameters:
        logits: The predicted values.
        labels: The actual values.
        step_frac: The fraction of total training steps completed.

        Returns:
        The mean of the cross entropy loss.
        """
        labels = labels * (1 - epsilon) + epsilon / 2
        loss = torch.nn.functional.cross_entropy(labels, logits)
        
        return loss.mean()

class CACE_loss(LossFnBase):
    def __init__(
        self, 
        confidence_threshold: float = 0.01, 
    ):

        super().__init__()
        self.c = confidence_threshold
        
    def __call__(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        step_frac: Optional[float] = None,
        epsilon: Optional[float] = 0.0,
    ) -> torch.Tensor:

        
        labels = labels.to(logits.device)
        high_confidence_mask = (labels - 0.5).abs() >= self.c
        
        BCE_loss = bce_loss()
        RBCE_loss = reverse_bce_loss()

        ce_loss = BCE_loss(logits, labels, raw_loss_tensor = True)
        
        rce_loss = RBCE_loss(logits, labels, epsilon = epsilon, raw_loss_tensor = True)
        
        loss = torch.where(high_confidence_mask, ce_loss, rce_loss)
        
        return loss.mean()

class SL_loss(LossFnBase):
    def __init__(
        self, 
        alpha: float = 1,
        beta: float = 0.1,  
    ):
        """
        Symmetric Learning Loss: 
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def __call__(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        step_frac: Optional[float] = None,
        epsilon: Optional[float] = 0.0,
    ) -> torch.Tensor:
        
        
        BCE_loss = bce_loss()
        RBCE_loss = reverse_bce_loss()

        ce_loss = BCE_loss(logits, labels)
        
        rce_loss = RBCE_loss(logits, labels, epsilon = epsilon)
        
        loss = self.alpha * ce_loss + self.beta * rce_loss
        
        return loss.mean()

class product_loss_fn(LossFnBase):
    """
    This class defines a custom loss function for product of predictions and labels.

    Attributes:
    alpha: A float indicating how much to weigh the weak model.
    beta: A float indicating how much to weigh the strong model.
    warmup_frac: A float indicating the fraction of total training steps for warmup.
    """

    def __init__(
        self,
        alpha: float = 1.0,  # how much to weigh the weak model
        beta: float = 1.0,  # how much to weigh the strong model
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
    ):
        self.alpha = alpha
        self.beta = beta
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
    ) -> torch.Tensor:
        preds = torch.softmax(logits, dim=-1)
        target = torch.pow(preds, self.beta) * torch.pow(labels, self.alpha)
        target /= target.sum(dim=-1, keepdim=True)
        target = target.detach()
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        return loss.mean()


class logconf_loss_fn(LossFnBase):
    """
    This class defines a custom loss function for log confidence.

    Attributes:
    aux_coef: A float indicating the auxiliary coefficient.
    warmup_frac: A float indicating the fraction of total training steps for warmup.
    """

    def __init__(
        self,
        aux_coef: float = 0.5,
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
    ) -> torch.Tensor:
        logits = logits.float()
        labels = labels.float()
        coef = 1.0 if step_frac > self.warmup_frac else step_frac
        coef = coef * self.aux_coef
        preds = torch.softmax(logits, dim=-1)
        mean_weak = torch.mean(labels, dim=0)
        assert mean_weak.shape == (2,)
        threshold = torch.quantile(preds[:, 0], mean_weak[1])
        strong_preds = torch.cat(
            [(preds[:, 0] >= threshold)[:, None], (preds[:, 0] < threshold)[:, None]],
            dim=1,
        )
        target = labels * (1 - coef) + strong_preds.detach() * coef
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        return loss.mean()


class kl_loss(LossFnBase):
    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, step_frac: Optional[float] = None
    ) -> torch.Tensor:
        """
        This function calculates the cross entropy loss between logits and labels.

        Parameters:
        logits: The predicted values.
        labels: The actual values.
        step_frac: The fraction of total training steps completed.

        Returns:
        The mean of the binary ce loss.
        """
        preds = torch.cat((torch.zeros_like(logits), logits), dim=1)
        gt_labels = torch.cat(((torch.ones_like(labels) - labels), labels), dim=1)
        # print(gt_labels)
        # print(preds)
        # loss = torch.nn.functional.kl_div(torch.log(torch.cat((1 - preds, preds), dim=1)), torch.cat((1 - labels, labels), dim=1))
        # loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(preds, dim=1), torch.softmax(gt_labels, dim=1))
        loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(preds, dim=1), gt_labels.to(preds.device))
        return loss.mean()

class reverse_kl_loss(LossFnBase):
    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, step_frac: Optional[float] = None, epsilon: float = 0.0,
    ) -> torch.Tensor:
        """
        This function calculates the cross entropy loss between logits and labels.

        Parameters:
        logits: The predicted values.
        labels: The actual values.
        step_frac: The fraction of total training steps completed.

        Returns:
        The mean of the binary ce loss.
        """
        preds = torch.cat((torch.zeros_like(logits), logits), dim=1)
        labels = labels * (1 - epsilon) + epsilon / 2
        gt_labels = torch.cat(((torch.ones_like(labels) - labels), labels), dim=1)
        # print(preds)
        # loss = torch.nn.functional.kl_div(torch.log(torch.cat((1 - preds, preds), dim=1)), torch.cat((1 - labels, labels), dim=1))
        # loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(gt_labels, dim=1), torch.softmax(preds, dim=1))
        loss = torch.nn.functional.kl_div(torch.log(gt_labels.to(preds.device)), torch.softmax(preds, dim=1))
        return loss.mean()


class bce_loss(LossFnBase):
    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, step_frac: Optional[float] = None,
        raw_loss_tensor: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        This function calculates the cross entropy loss between logits and labels.

        Parameters:
        logits: The predicted values.
        labels: The actual values.
        step_frac: The fraction of total training steps completed.

        Returns:
        The mean of the binary ce loss.
        """
        if raw_loss_tensor:
            loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits), labels.to(logits.device), reduction = 'none')
            return loss

        loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits), labels.to(logits.device))
        
            

        return loss.mean()

class reverse_bce_loss(LossFnBase):
    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, step_frac: Optional[float] = None, epsilon: float = 0.0,
        raw_loss_tensor: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        This function calculates the cross entropy loss between logits and labels.

        Parameters:
        logits: The predicted values.
        labels: The actual values.
        step_frac: The fraction of total training steps completed.

        Returns:
        The mean of the binary ce loss.
        """
        labels = labels * (1 - epsilon) + epsilon / 2
        if raw_loss_tensor:
            loss = torch.nn.functional.binary_cross_entropy(labels.to(logits.device), torch.sigmoid(logits), reduction='none')
            return loss

        loss = torch.nn.functional.binary_cross_entropy(labels.to(logits.device), torch.sigmoid(logits))

        

        return loss.mean()
    


