"""
Base complex embedding model class
"""
import torch
import numpy as np
import torch.nn as nn
from collections.abc import Iterable

from .base_emb_model import EmbeddingModel
from kgpy.regularizers import *


class ComplexEmbeddingModel(EmbeddingModel):
    """
    Embeddings are in complex space
    """
    def __init__(
        self, 
        model_name, 
        num_entities, 
        num_relations, 
        ent_emb_dim, 
        rel_emb_dim, 
        loss_margin, 
        regularization, 
        reg_weight,
        weight_init, 
        loss_fn,
        norm_constraint,
        device
    ):
        super().__init__(
            model_name, 
            num_entities, 
            num_relations, 
            ent_emb_dim, 
            rel_emb_dim,
            loss_margin, 
            regularization, 
            reg_weight,
            weight_init, 
            loss_fn,
            norm_constraint,
            device
        )
        
        self.ent_emb_re, self.ent_emb_im, self.rel_emb_re, self.rel_emb_im = self._create_embeddings()
        self._init_embs()
        
        if self.norm_constraint:
           self._normalize_relations(2)

        if self.regularization not in ["", None, "N3"]:
            raise ValueError("For complex-space embedding methods. Only N3 regularization is available")


    def _create_embeddings(self):
        """
        Create the complex embeddings.

        Returns:
        --------
        tuple
            entity_emb_re, entity_emb_im, relation_emb_re, relation_emb_im
        """
        entity_emb_re = nn.Embedding(self.num_entities, self.ent_emb_dim)
        relation_emb_re = nn.Embedding(self.num_relations, self.rel_emb_dim)
        entity_emb_im = nn.Embedding(self.num_entities, self.ent_emb_dim)
        relation_emb_im = nn.Embedding(self.num_relations, self.rel_emb_dim)

        return entity_emb_re, entity_emb_im, relation_emb_re, relation_emb_im

    
    def _init_embs(self):
        """
        Initialize the embeddings

        Returns:
        --------
        None
        """
        weight_init_method = self._get_weight_init_method()

        weight_init_method(self.ent_emb_re.weight)
        weight_init_method(self.rel_emb_re.weight)
        weight_init_method(self.ent_emb_im.weight)
        weight_init_method(self.rel_emb_im.weight)


    def _normalize_entities(self, p):
        """
        Normalize entity embeddings by some p-norm. Does so in-place.

        Parameters:
        -----------
            p: int
                p-norm value

        Returns:
        --------
            None
        """
        e_re_sum = self.ent_emb_re.weight.pow(p).sum(dim=-1)
        e_im_sum = self.ent_emb_im.weight.pow(p).sum(dim=-1)
        
        e_norm = torch.sqrt(e_re_sum + e_im_sum)

        self.ent_emb_re.weight.data = self.ent_emb_re.weight.data / e_norm.reshape(-1, 1)
        self.ent_emb_im.weight.data = self.ent_emb_im.weight.data / e_norm.reshape(-1, 1)


    def _normalize_relations(self, p):
        """
        Normalize relations embeddings by some p-norm.  Does so in-place

        Parameters:
        -----------
            p: int
                p-norm value

        Returns:
        --------
            None
        """
        r_re_sum = self.rel_emb_re.weight.pow(p).sum(dim=-1)
        r_im_sum = self.rel_emb_im.weight.pow(p).sum(dim=-1)
        
        r_norm = torch.sqrt(r_re_sum + r_im_sum)

        self.rel_emb_re.weight.data = self.rel_emb_re.weight.data / r_norm.reshape(-1, 1)
        self.rel_emb_im.weight.data = self.rel_emb_im.weight.data / r_norm.reshape(-1, 1)


    def loss(self, *args, **kwargs):
        """
        Get Loss for given scores

        Parameters:
        -----------
            kwargs: dict
                Contents depend on training method and type of loss.

        Returns:
        --------
        float
            loss for samples
        """
        return self.loss_fn(device=self._cur_device(), **kwargs) + self.regularize(kwargs.get("trips"))



    # def regularize(self):
    #     """
    #     Apply regularization if specified.

    #     Returns:
    #     --------
    #     float
    #         Regularization term for loss
    #     """
    #     if self.regularization is None or self.regularization == 0:
    #         return 0

    #     lp = int(self.regularization[1])

    #     entity_re = self._norm(self.ent_emb_re, lp)
    #     entity_im = self._norm(self.ent_emb_im, lp)
    #     relation_re = self._norm(self.rel_emb_re, lp)
    #     relation_im = self._norm(self.rel_emb_im, lp)

    #     if isinstance(self.reg_weight, Iterable):
    #         return self.reg_weight[0] * (entity_re**lp + entity_im**lp) + self.reg_weight[1] * (relation_re**lp + relation_im**lp)
        
    #     return self.reg_weight / lp * (entity_re**lp + entity_im**lp + relation_re**lp + relation_im**lp) 


    def regularize(self, triplets):
        """
        Apply regularization if specified.

        TODO: Only works for N3!!!

        Returns:
        --------
        float
            Regularization term for loss
        """
        if triplets is not None and self.regularization == "N3":
            h_embs = (self.ent_emb_re(triplets[:, 0]), self.ent_emb_im(triplets[:, 0]))
            r_embs = (self.rel_emb_re(triplets[:, 1]), self.rel_emb_im(triplets[:, 1]))
            t_embs = (self.ent_emb_re(triplets[:, 2]), self.ent_emb_im(triplets[:, 2]))

            factors = (      
                torch.sqrt(h_embs[0] ** 2 + h_embs[1] ** 2),
                torch.sqrt(r_embs[0] ** 2 + r_embs[1] ** 2),
                torch.sqrt(t_embs[0] ** 2 + t_embs[1] ** 2)
            )

            norm = 0
            for f in factors:
                norm +=  torch.sum(
                    torch.abs(f) ** 3
                )
            return self.reg_weight * (norm / factors[0].shape[0])
        else:
            return 0


    def _cur_device(self):
        """
        Get the current device being used

        Returns:
        --------
        str
            device name
        """
        return self.device


