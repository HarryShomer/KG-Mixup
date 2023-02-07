import torch 
import torch.nn as nn
from abc import ABC, abstractmethod


class Regularizer(nn.Module, ABC):
    """
    Base module
    """
    @abstractmethod
    def forward(self, factors):
        pass



class Complex_N3(nn.Module):
    """
    Nuclear 3 norm introduced here - https://arxiv.org/abs/1806.07297

    NOTE: For the time being this only works on ComplEX
    """
    def __init__(self):
        super(Complex_N3).__init__()


    def get_factors(self, head_embs, rel_embs, tail_embs):
        """
        Idk why this is needed
        """
        return (      
            torch.sqrt(head_embs[0] ** 2 + head_embs[1] ** 2),
            torch.sqrt(rel_embs[0] ** 2 + rel_embs[1] ** 2),
            torch.sqrt(tail_embs[0] ** 2 + tail_embs[1] ** 2)
        )


    def forward(self, head_embs, rel_embs, tail_embs):
        """
        Compute regularization
        """
        factors = self.get_factors(head_embs, rel_embs, tail_embs)

        norm = 0
        for f in factors:
            norm +=  torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]


class DURA_W(Regularizer):
    """
    DURA Regularization introduced here - https://arxiv.org/pdf/2011.05816.pdf
    """
    def __init__(self, weight: float):
        super(DURA_W, self).__init__()
        self.weight = weight


    def forward(self, factors):
        norm = 0
        for factor in factors:
            h, r, t = factor

            norm += 0.5 * torch.sum(t**2 + h**2)
            norm += 1.5 * torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]