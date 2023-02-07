"""
Base embedding model class
"""
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections.abc import Iterable

from kgpy import loss


class EmbeddingModel(ABC, nn.Module):
    """
    Base Embedding Model Class

    Attributes:
    -----------
    name: str
        Name of model
    num_entities: int
        number of entities 
    num_relations: int
        number of relations
    ent_emb_dim: int
        hidden dimension for entity embeddings
    rel_emb_dim: int
        hidden dimension for relation embeddings
    regularization: str 
        Type of regularization. One of [None, 'l1', 'l2', 'l3']
    reg_weight: list/float
        Regularization weights. When list 1st entry is weight for entities and 2nd for relation embeddings.
    weight_init: str
        weight_init method to use
    loss_fn: loss.Loss
        Loss function object
    norm_constraint: bool
        Whether Take norm of entities after each gradient and relations at beginning   
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
        norm_constraint,   # TODO: Split by relation and entitiy? Also allow specfication of norm?
        device
    ):
        super(EmbeddingModel, self).__init__()
        
        self.name = model_name
        self.ent_emb_dim = ent_emb_dim
        self.rel_emb_dim = rel_emb_dim
        self.weight_init = "uniform" if weight_init is None else weight_init.lower()
        self.norm_constraint = norm_constraint
        self.device = device

        self.num_entities = num_entities
        self.num_relations = num_relations

        # When not same reg weight for relation/entities you need to have only supplied 2
        if (isinstance(reg_weight, Iterable) and len(reg_weight) != 2):
            raise ValueError(f"`reg_weight` parameter must be either a constant or an iterable of length 2. You passed {reg_weight}")

        if regularization not in [None, 'l1', 'l2', 'l3', 'N3']:
            raise ValueError(f"`regularization` parameter must be one of [None, 'l1', 'l2', 'l3', 'N3']. You passed {regularization}")

        self.regularization = regularization
        self.reg_weight = reg_weight

        self.loss_fn = loss.get_loss_fn(loss_fn.lower(), loss_margin)


    @abstractmethod
    def score_hrt(self, triplets):
        """
        Get the score for a given set of triplets.

        To be implemented by the specific model.

        Parameters:
        -----------
            triplets: List of triplets

        Returns:
        --------
            List of scores
        """
        pass

    @abstractmethod
    def score_head(self, triplets):
        """
        Get the score for a given set of (relation, tails) against all heads

        To be implemented by the specific model.

        Parameters:
        -----------
            triplets: List of (relation, tail) samples

        Returns:
        --------
            List of scores
        """
        pass

    @abstractmethod
    def score_tail(self, triplets):
        """
        Get the score for a given set of (head, relation) against all tails

        To be implemented by the specific model.

        Parameters:
        -----------
            triplets: List of (head, relation) samples

        Returns:
        --------
            List of scores
        """
        pass


    @abstractmethod
    def _create_embeddings(self):
        """
        Create the embeddings.

        To be implemented by the specific type of model.

        Returns:
        --------
            Embeddings
        """
        pass


    @abstractmethod
    def _init_embs(self):
        """
        Initialize the embeddings.

        To be implemented by the specific type of model.

        Returns:
        --------
        None
        """
        pass


    @abstractmethod
    def _normalize_entities(self):
        """
        Normalize entity embeddings by some p-norm.

        To be implemented by the specific type of model.
        """
        pass


    @abstractmethod
    def _normalize_relations(self):
        """
        Normalize relations embeddings by some p-norm.

        To be implemented by the specific type of model.
        """
        pass


    @abstractmethod
    def regularize(self):
        """
        Apply specific type of regularization if specified.

        To be implemented by the specific type of model.

        Returns:
        --------
            Regularization term for loss
        """
        pass


    @abstractmethod
    def _cur_device(self):
        """
        Get the current device being used

        To be implemented by the specific type of model.
        
        Returns:
        --------
        str
            device name
        """
        pass


    def forward(self, triplets, mode=None, negative_ents=None, **kwargs):
        """
        Forward pass for our model.
        1. Normalizes entity embeddings to unit length if specified
        2. Computes score for tpe of triplets
        3. Return scores for each triplet

        Parameters:
        -----------
            triplets: list
                List of triplets to train on
            mode: str
                None, head, tail
            negative_ents: torch.Tensor
                Negative entities to score against. Only applicable for self.score_hrt

        Returns:
        --------
        list
            score for each triplet in batch
        """
        if self.norm_constraint:
            self._normalize_entities(2)

        if mode is None:
            scores = self.score_hrt(triplets, negative_ents=negative_ents)
        elif mode == "head":
            scores = self.score_head(triplets)
        elif mode == "tail":
            scores = self.score_tail(triplets)
        else:
            raise ValueError("Invalid value for `mode` passed to Model.forward(). Must be one of [None, 'head', 'tail']")

        return scores

    
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
        return self.loss_fn(device=self._cur_device(), **kwargs) + self.regularize(*args)


    def _get_weight_init_method(self):
        """
        Determine the correct weight initializer method and init weights

        Parameters:
        -----------
            weight_init_method: str
                Type of weight init method. Currently only works with "uniform" and "normal"

        Returns:
        --------
            Correct nn.init function
        """
        if self.weight_init == "normal":
            return nn.init.xavier_normal_
        elif self.weight_init == "uniform":
            return nn.init.xavier_uniform_
        else:
            raise ValueError(f"Invalid weight initializer passed {self.weight_init}. Must be either 'uniform' or 'normal'.")


    def _normalize(self, emb, p):
        """
        Normalize an embedding by some p-norm.

        Parameters:
        -----------
            emb: nn.Embedding
            p: p-norm value

        Returns:
        --------
            Embedding
        """
        emb.weight.data = torch.nn.functional.normalize(emb.weight.data, p=p, dim=-1)

        return emb


    def _norm(self, emb, p, **kwargs):
        """
        Return norm of the embeddings

        Parameters:
        -----------
            emb: nn.Embedding
            p: p-norm value

        Returns:
        --------
            Norm value
        """
        return emb.weight.data.norm(p=p, **kwargs)
