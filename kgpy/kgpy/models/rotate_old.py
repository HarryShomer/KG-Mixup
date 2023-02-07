"""
Implementation of RotatE. 

See paper for more details - https://arxiv.org/abs/1902.10197.
"""
import torch

from .base.complex_emb_model import ComplexEmbeddingModel


class RotatE(ComplexEmbeddingModel):
    """
    Implementation of RotatE
    """

    def __init__(
        self, 
        num_entities, 
        num_relations, 
        emb_dim=100, 
        margin=1, 
        regularization = None,
        reg_weight = 0,
        weight_init="normal",
        loss_fn= "ranking",
        device='cpu'
    ):
        super().__init__(
            type(self).__name__, 
            num_entities, 
            num_relations, 
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
        h_re = self.ent_emb_re(triplets[:, 0])
        h_im = self.ent_emb_im(triplets[:, 0])
        t_re = self.ent_emb_re(triplets[:, 2])
        t_im = self.ent_emb_im(triplets[:, 2])
        r_re = self.rel_emb_re(triplets[:, 1])
        r_im = self.rel_emb_im(triplets[:, 1])

        # Vector product - complex space
        re_score = (h_re * r_re - h_im * r_im) - t_re
        im_score = (h_re * r_im + h_im * r_re) - t_im

        scores = torch.stack([re_score, im_score], dim = 0)
        scores = scores.norm(dim = 0).sum(dim = 1)

        return scores

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    # TODO
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
        h_re = self.ent_emb_re(torch.arange(self.num_entities, device=self._cur_device()).long())
        h_im = self.ent_emb_im(torch.arange(self.num_entities, device=self._cur_device()).long())
        t_re = self.ent_emb_re(triplets[:, 1])
        t_im = self.ent_emb_im(triplets[:, 1])
        r_re = self.rel_emb_re(triplets[:, 0])
        r_im = self.rel_emb_im(triplets[:, 0])

        re_score = (h_re * r_re - h_im * r_im) - t_re
        im_score = (h_re * r_im + h_im * r_re) - t_im

        scores = torch.stack([re_score, im_score], dim = 0)
        scores = scores.norm(dim = 0).sum(dim = 1)

        return scores

         
    # TODO
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
        t_re = self.ent_emb_re(torch.arange(self.num_entities, device=self._cur_device()).long())
        t_im = self.ent_emb_im(torch.arange(self.num_entities, device=self._cur_device()).long())
        h_re = self.ent_emb_re(triplets[:, 1])
        h_im = self.ent_emb_im(triplets[:, 1])
        r_re = self.rel_emb_re(triplets[:, 0])
        r_im = self.rel_emb_im(triplets[:, 0])

        re_score = (h_re * r_re - h_im * r_im) - t_re
        im_score = (h_re * r_im + h_im * r_re) - t_im

        scores = torch.stack([re_score, im_score], dim = 0)
        scores = scores.norm(dim = 0).sum(dim = 1)

        return scores