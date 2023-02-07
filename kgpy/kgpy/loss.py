"""
Provide interface here to better manage loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


def get_loss_fn(loss_fn_name, loss_margin=1):
    """
    Determine loss function based on user input (self.loss_fn_name). 

    Throw exception when invalid.

    Parameters:
    -----------
        loss_fn_name: str
            Name of loss function fed to constructor
        loss_margin: int
            Optional margin or hinge style losses

    Returns:
    --------
    loss.Loss
        Specific loss function
    """
    if loss_fn_name == "ranking":
        return MarginRankingLoss(margin=loss_margin)
    elif loss_fn_name == "bce":
        return BCELoss()
    elif loss_fn_name == "softplus":
        return SoftPlusLoss()
  
    raise ValueError(f"Invalid loss function type - {loss_fn_name}")



class Loss(ABC, nn.Module):
    """
    Base loss class specification
    """
    def __init__(self):
        super(Loss, self).__init__()


    @abstractmethod
    def forward(self):
        """
        Compute loss on sample.
        To be implemented by specific loss function.
        """
        pass


class MarginRankingLoss(Loss):
    """
    Wrapper for Margin Ranking Loss
    """

    def __init__(self, margin):
        """
        Constructor

        Parameters:
        ----------
            margin: int
                loss margin
        """
        super().__init__()
        self.margin = margin


    def forward(self, **kwargs):
        """
        Compute loss on sample.


        Parameters:
        -----------
            kwargs: dict
                Contents depend on training method. See above in docstring

        Returns:
        --------
        float
            loss
        """
        device = kwargs.get("device", "cpu")
        reduction = kwargs.get("reduction", "mean")

        positive_scores = kwargs['positive_scores']
        negative_scores = kwargs['negative_scores']

        target = torch.ones_like(positive_scores, device=device)

        return F.margin_ranking_loss(positive_scores, negative_scores, target, margin=self.margin, reduction=reduction)



class BCELoss(Loss):
    """
    Wrapper for Binary cross entropy loss (includes logits)
    """
    def __init__(self):
        super().__init__()


    def forward(self, **kwargs):
        """
        Compute loss on sample.

        Can either contain:
            1. Arrays for positive and negative scores separately
            2. Array of all the scores and all the targets (0 or 1)

        Parameters:
        -----------
            kwargs: dict
                Contents depend on training method. See above in docstring

        Returns:
        --------
        float
            loss
        """
        device = kwargs.get("device", "cpu")
        reduction = kwargs.get("reduction", "mean")

        if 'positive_scores' in kwargs:
            all_scores = torch.cat((kwargs['positive_scores'], kwargs['negative_scores']))

            target_positives = torch.ones_like(kwargs['positive_scores'], device=device)
            target_negatives = torch.zeros_like(kwargs['negative_scores'], device=device)
            all_targets = torch.cat((target_positives, target_negatives))
        else:
            all_scores  = kwargs['all_scores']
            all_targets = kwargs['all_targets']
        

        return F.binary_cross_entropy_with_logits(all_scores, all_targets, reduction=reduction)



class SoftPlusLoss(Loss):
    """
    Wrapper for Softplus loss (used in original ComplEx paper)
    """
    def __init__(self):
        super().__init__()


    def forward(self, **kwargs):
        """
        Compute loss on sample.

        Parameters:
        -----------
            kwargs: dict
                Contents depend on training method. See above in docstrings
                
        Returns:
        --------
        float
            loss
        """
        device = kwargs.get("device", "cpu")
        positive_scores = kwargs['positive_scores']
        negative_scores = kwargs['negative_scores']

        positive_scores *= -1
        all_scores = torch.cat((positive_scores, negative_scores))

        return F.softplus(all_scores, beta=1).mean() 


# def NegativeSamplingLoss(Loss):
#     """
#     See RotatE paper
#     """
#     def __init__(self, margin):
#         super().__init__()
#         self.margin = margin


#     def forward(self, positive_scores, negative_scores, negative_weights, device="cpu"):
#         """
#         Compute loss on sample.

#         Parameters:
#         -----------
#             positive_scores: Tensor
#                 Scores for true triplets
#             negative_scores: Tensor
#                 Scores for corrupted triplets
#             negative_weights: Tensor
#                 Weights for negative_scores
#             device: str
#                 device being used. defaults to "cpu"

#         Returns:
#         --------
#         float
#             loss
#         """
#         # TODO: Make sure work on batch level
#         pos_score = - F.logsigmoid(self.margin - positive_scores)

#         # TODO: Sum - batch?
#         neg_score = negative_weights * F.logsigmoid(negative_scores - self.margin)

#         #return (pos_score - neg_score).mean()
