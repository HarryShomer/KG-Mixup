"""
Implementation of RotatE. 

See paper for more details - https://arxiv.org/abs/1902.10197.
"""
import torch
import torch.nn as nn

# from .base.complex_emb_model import ComplexEmbeddingModel
from .base.single_emb_model import SingleEmbeddingModel


class RotatE(SingleEmbeddingModel):
    """
    Implementation of RotatE
    """

    def __init__(
        self, 
        num_entities, 
        num_relations, 
        emb_dim=100, 
        margin=9, 
        regularization = None,
        reg_weight = 0,
        weight_init="uniform",
        loss_fn= "bce",
        device='cpu',
        epsilon=2, # It's hardcoded as 2 in the author's code
        **kwargs
    ):
        self.gamma = margin
        self.epsilon = epsilon 
        self.ent_emb_range = (self.gamma - self.epsilon) / (emb_dim * 2)
        self.rel_emb_range = (self.gamma - self.epsilon) / emb_dim

        super().__init__(
            type(self).__name__, 
            num_entities, 
            num_relations, 
            emb_dim * 2,  # Ent embs are 2x size 
            emb_dim,
            margin, 
            regularization,
            reg_weight,
            weight_init, 
            loss_fn,
            False,  # TODO: True,
            device
        )
    

    def _init_embs(self):
        """
        Override base implementation to use RotatE init range
        """
        # Init embedding in range [-{}_emb_range, {}_emb_range]
        nn.init.uniform_(
            tensor = self.ent_embs.weight.data, 
            a=-self.ent_emb_range, 
            b=self.ent_emb_range
        )
        nn.init.uniform_(
            tensor = self.rel_embs.weight.data, 
            a=-self.rel_emb_range, 
            b=self.rel_emb_range
        )


    def _process_batch_embs(self, h, r, t):
        """
        """
        # Was in author's source code...I'm assuming they had a reason
        pi = 3.14159265358979323846

        # Entity embeddings are 2 times size of relation embs
        h_re, h_im = torch.chunk(h, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = r / (self.ent_emb_range / pi)
        r_re = torch.cos(phase_relation)
        r_im = torch.sin(phase_relation)

        h_re = h_re.view(-1, r_re.shape[0], h_re.shape[-1]).permute(1, 0, 2)
        t_re = t_re.view(-1, r_re.shape[0], t_re.shape[-1]).permute(1, 0, 2)
        h_im = h_im.view(-1, r_re.shape[0], h_im.shape[-1]).permute(1, 0, 2)
        t_im = t_im.view(-1, r_re.shape[0], t_im.shape[-1]).permute(1, 0, 2)
        r_im = r_im.view(-1, r_re.shape[0], r_im.shape[-1]).permute(1, 0, 2)
        r_re = r_re.view(-1, r_re.shape[0], r_re.shape[-1]).permute(1, 0, 2)

        return h_re, h_im, r_re, r_im, t_re, t_im


    def score_hrt(self, triplets):
        """        
        Score = || h * r - t || in complex space

        They use L1 norm.

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
        
        h_re, h_im, r_re, r_im, t_re, t_im  = self._process_batch_embs(h, r, t)

        re_score = (h_re * r_re - h_im * r_im) - t_re
        im_score = (h_re * r_im + h_im * r_re) - t_im

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0).sum(dim = -1)

        return self.gamma - score.permute(1, 0).flatten()


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

        h_re, h_im, r_re, r_im, t_re, t_im  = self._process_batch_embs(h, r, t)

        re_score = (r_re * t_re + r_im * t_im) - h_re
        im_score = (r_re * t_im - r_im * t_re) - h_im

        scores = torch.stack([re_score, im_score], dim = 0)
        scores = scores.norm(dim = 0).sum(dim = 1)

        return scores

         
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

        h_re, h_im, r_re, r_im, t_re, t_im  = self._process_batch_embs(h, r, t)

        re_score = (h_re * r_re - h_im * r_im) - t_re
        im_score = (h_re * r_im + h_im * r_re) - t_im

        scores = torch.stack([re_score, im_score], dim = 0)
        scores = scores.norm(dim = 0).sum(dim = 1)

        return scores