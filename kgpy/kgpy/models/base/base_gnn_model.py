"""
Base GNN model class
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections.abc import Iterable

from torch.nn.init import xavier_normal_

from kgpy import loss


class BaseGNNModel(ABC, nn.Module):
    """
    Base GNN Model Class

    Attributes:
    -----------
    name: str
        Name of model
    edge_index: torch.Tensor
        2xN matrix of vertices for each edge 
    edge_type: torch.Tensor
        1xN matrix of edge types for each edge
    num_layers: int
        Number of layers in GNN
    gcn_dim: int
        Dimension of gcn
    num_entities: int
        number of entities 
    num_relations: int
        number of relations
    emb_dim: int
        hidden dimension
    weight_init: str
        weight_init method to use
    loss_fn: loss.Loss
        Loss function object
    device: str
        Device we are working on 
    """

    def __init__(
            self, 
            model_name, 
            edge_index, 
            edge_type,
            num_layers,
            gcn_dim,
            num_entities, 
            num_relations, 
            emb_dim, 
            weight_init, 
            loss_fn,
            device
        ):
        super(BaseGNNModel, self).__init__()

        self.name = model_name
        self.emb_dim = emb_dim
        self.weight_init = "uniform" if weight_init is None else weight_init.lower()
        self.device = device

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.ent_embs, self.rel_embs = self._create_embeddings()

        self.loss_fn = loss.get_loss_fn(loss_fn.lower())

        self.edge_index	= edge_index
        self.edge_type = edge_type
        self.gcn_dim = self.emb_dim if num_layers == 1 else gcn_dim
        self.num_layers = num_layers
    

    ### TODO: Hack for torch.save that expects different dims based off of base_emb_model
    @property
    def ent_emb_dim(self):
        return self.emb_dim
        
    @property
    def rel_emb_dim(self):
        return self.emb_dim


    @abstractmethod
    def forward(self, triplets, **kwargs):
        """
        Specific to each aggregation

        Returns:
        --------
        list
            score for each triplet in batch
        """
        pass


    def _create_embeddings(self):
        """
        Create and initialize the parameters.

        Returns:
        --------
        tuple of nn.Parameters
            entities and relations
        """
        weight_init_method = self._get_weight_init_method()

        entity_emb = nn.Parameter(torch.Tensor(self.num_entities, self.emb_dim).to(self.device))
        relation_emb = nn.Parameter(torch.Tensor(self.num_relations, self.emb_dim).to(self.device))

        weight_init_method(entity_emb.data)
        weight_init_method(relation_emb.data)

        return entity_emb, relation_emb
        


    def loss(self, **kwargs):
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

        return self.loss_fn(device=self.device, **kwargs) #+ self.regularize()


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







# TODO: Old version that relied on SingleEmbeddingModel
# class BaseGNNModel(SingleEmbeddingModel):
#     """
#     Base GNN Model Class

#     Attributes:
#     -----------
#     name: str
#         Name of model
#     edge_index: torch.Tensor
#         2xN matrix of vertices for each edge 
#     edge_type: torch.Tensor
#         1xN matrix of edge types for each edge
#     num_layers: int
#         Number of layers in GNN
#     gcn_dim: int
#         Dimension of gcn
#     num_entities: int
#         number of entities 
#     num_relations: int
#         number of relations
#     emb_dim: int
#         hidden dimension
#     regularization: str 
#         Type of regularization. One of [None, 'l1', 'l2', 'l3']
#     reg_weight: list/float
#         Regularization weights. When list 1st entry is weight for entities and 2nd for relation embeddings.
#     weight_init: str
#         weight_init method to use
#     loss_fn: loss.Loss
#         Loss function object
#     device: str
#         Device we are working on 
#     """

#     def __init__(
#             self, 
#             model_name, 
            
#             edge_index, 
#             edge_type,
#             num_layers,
#             gcn_dim,

#             num_entities, 
#             num_relations, 
#             emb_dim, 
#             regularization, 
#             reg_weight,
#             weight_init, 
#             loss_fn,
#             device
#         ):
#         super().__init__(
#             model_name=model_name, 
#             num_entities=num_entities, 
#             num_relations=num_relations, 
#             emb_dim=emb_dim, 
#             loss_margin=None, 
#             regularization=regularization, 
#             reg_weight=reg_weight,
#             weight_init=weight_init, 
#             loss_fn=loss_fn,
#             norm_constraint=False,
#             device=device
#         )
        
#         self.edge_index	= edge_index
#         self.edge_type = edge_type
#         self.gcn_dim = self.emb_dim if num_layers == 1 else gcn_dim
#         self.num_layers = num_layers
        


#     ### TODO: Clean this up later

#     def score_hrt(self, triplets):
#         raise NotImplementedError("Method `score_hrt` not valid for GNN model")

#     def score_head(self, triplets):
#         raise NotImplementedError("Method `score_head` not valid for GNN model")

#     def score_tail(self, triplets):
#         raise NotImplementedError("Method `score_tail` not valid for GNN model")
