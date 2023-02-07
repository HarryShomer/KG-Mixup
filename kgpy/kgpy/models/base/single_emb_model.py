"""
Base single embedding model class
"""
import torch
import numpy as np
import torch.nn as nn
from collections.abc import Iterable

from .base_emb_model import EmbeddingModel



class SingleEmbeddingModel(EmbeddingModel):
    """
    Each entity / relation gets one embedding
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
        self.ent_embs, self.rel_embs = self._create_embeddings()
        self._init_embs()

        if self.norm_constraint:
           self._normalize_relations(2)


    def _create_embeddings(self):
        """
        Create the embeddings.

        Parameters:
        -----------
            complex_emb: bool
                True if complex

        Returns:
        --------
        tuple
            entity_embs, relation_embs
        """
        entity_emb = nn.Embedding(self.num_entities, self.ent_emb_dim)
        relation_emb = nn.Embedding(self.num_relations, self.rel_emb_dim)

        return entity_emb, relation_emb


    def _init_embs(self):
        """
        Initialize the embeddings

        Returns:
        --------
        None
        """
        weight_init_method = self._get_weight_init_method()

        weight_init_method(self.ent_embs.weight)
        weight_init_method(self.rel_embs.weight)


    def _normalize_entities(self, p):
        """
        Normalize entity embeddings by some p-norm. Does so in-place

        Parameters:
        -----------
            p: int
                p-norm value

        Returns:
        --------
            None
        """
        self.ent_embs = self._normalize(self.ent_embs, p)


    
    def _normalize_relations(self, p):
        """
        Normalize relations embeddings by some p-norm.  Does so in-place

        Parameters:
        -----------
            p: int
                p-norm value

        Returns:
        --------
            Norne
        """
        self.rel_embs = self._normalize(self.rel_embs, p)


    def regularize(self, *args):
        """
        Apply regularization if specified.

        Returns:
        --------
        float
            Regularization term for loss
        """
        if self.regularization is None:
            return 0

        lp = int(self.regularization[1])
        entity_norm = self._norm(self.ent_embs, lp)
        relation_norm = self._norm(self.rel_embs, lp)

        if isinstance(self.reg_weight, Iterable):
            return self.reg_weight[0] * entity_norm**lp + self.reg_weight[1] * relation_norm**lp
        
        return self.reg_weight * (entity_norm**lp + relation_norm**lp) 


    def _cur_device(self):
        """
        Get the current device being used

        Returns:
        --------
        str
            device name
        """
        return self.device