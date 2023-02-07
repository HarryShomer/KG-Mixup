"""
Implementation of DistMult. 

See paper for more details - https://arxiv.org/pdf/1412.6575.pdf.
"""
import torch

from .base.single_emb_model import SingleEmbeddingModel


class DistMult(SingleEmbeddingModel):
    def __init__(
        self, 
        num_entities, 
        num_relations, 
        emb_dim=100, 
        margin=1, 
        regularization = 'l3',
        reg_weight = 1e-6,
        weight_init=None,
        loss_fn="ranking",
        device='cpu',
        **kwargs
    ):
        super().__init__(
            type(self).__name__,
            num_entities, 
            num_relations, 
            emb_dim, 
            emb_dim, 
            margin, 
            regularization,
            reg_weight,
            weight_init, 
            loss_fn,
            True,
            device
        )


    def score_hrt(self, triplets):
        """
        Score function is -> h^T * diag(M) * t. We have r = diag(M).

        Parameters:
        -----------
            triplets: list
                List of triplets of form [sub, rel, obj]

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        h = self.ent_embs(triplets[:, 0])
        r = self.rel_embs(triplets[:, 1])
        t = self.ent_embs(triplets[:, 2])

        return torch.sum(h * r * t, dim=-1)


    def score_head(self, triplets):
        """
        Get the score for a given set of triplets against *all possible* heads.
        
        Parameters:
        -----------
            triplets: list
                List of triplets of form [rel, object]

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        h = self.ent_embs(torch.arange(self.num_entities, device=self._cur_device()).long())
        r = self.rel_embs(triplets[:, 0])
        t = self.ent_embs(triplets[:, 1])

        return torch.sum(h[None, :, :] * r[:, None, :] * t[:, None, :], dim=-1)


    def score_tail(self, triplets):
        """
        Get the score for a given set of triplets against *all possible* tails.

        Parameters:
        -----------
            triplets: list
                List of triplets of form [rel, subject]

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        h = self.ent_embs(triplets[:, 1])
        r = self.rel_embs(triplets[:, 0])
        t = self.ent_embs(torch.arange(self.num_entities, device=self._cur_device()).long())

        return torch.sum(h[:, None, :] * r[:, None, :] * t[None, :, :], dim=-1)
