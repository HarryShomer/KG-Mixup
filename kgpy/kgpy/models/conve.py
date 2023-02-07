"""
Implementation of ConvE. 

See paper for more details - https://arxiv.org/abs/1707.01476.
"""
import torch
import torch.nn.functional as F

from .base.single_emb_model import SingleEmbeddingModel


class ConvE(SingleEmbeddingModel):
    def __init__(self, 
        num_entities, 
        num_relations, 
        emb_dim=200, 
        filters=32,
        ker_sz=3,
        k_h=20,
        # Code itself is hidden_drop=.3 but he mentions a higher regularization rate (0.5) for FB15K-237 here
        # https://github.com/TimDettmers/ConvE/issues/52#issuecomment-537231786
        hidden_drop=.5,
        input_drop=.2,
        feat_drop=.2,
        margin=1, 
        regularization='l2',
        reg_weight=0,
        weight_init="normal",
        loss_fn="bce",
        device='cuda',
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
        self.inp_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feat_drop)

        # NOTE: Ensure that emb_dim = kernel_h * kernel_w
        self.k_h = k_h
        self.k_w = emb_dim // k_h
        self.filters = filters
        self.ker_sz = ker_sz

        flat_sz_h = int(2*self.k_h) - self.ker_sz + 1
        flat_sz_w = self.k_w - self.ker_sz + 1
        self.hidden_size = flat_sz_h*flat_sz_w*filters

        self.conv1 = torch.nn.Conv2d(1, filters, kernel_size=(ker_sz, ker_sz), stride=1, padding=0)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.filters)
        self.bn2 = torch.nn.BatchNorm1d(emb_dim)

        self.b = torch.nn.Parameter(torch.zeros(num_entities))  # bias

        self.fc = torch.nn.Linear(self.hidden_size, emb_dim)


    def score_function(self, e1, rel):
        """
        Scoring process of triplets

        Parameters:
        -----------
            e1: torch.Tensor
                entities passed through ConvE
            e2: torch.Tensor
                entities scored against for link prediction
            rel: torch.Tensor
                relaitons passed through ConvE
        
        Returns:
        --------
        torch.Tensor
            Raw scores to be multipled by entities (e.g. dot product)
        """
        triplets = torch.cat([e1, rel], 2)

        stacked_inputs = self.bn0(triplets)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)

        # Fails when only one initial channel (e.g. only one embedding is passed in batch)
        if x.shape[0] != 1:
            x = self.bn2(x)

        x = F.relu(x)

        return x


    def score_hrt(self, triplets, negative_ents=None):
        """
        Pass through ConvE.

        Parameters:
        -----------
            triplets: torch.Tensor
                List of triplets of form (sub, rel, obj)
            negative_ents: torch.Tensor
                Negative entities to score against for each triple

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        e1_embedded  = self.ent_embs(triplets[:, 0]).view(-1, 1, self.k_h, self.k_w)
        rel_embedded = self.rel_embs(triplets[:, 1]).view(-1, 1, self.k_h, self.k_w)

        x = self.score_function(e1_embedded, rel_embedded)

        if negative_ents is not None:
            # Each must only be multiplied by negative entities belonging to triple
            e2_embedded  = self.ent_embs(negative_ents)
            x = x.reshape(x.shape[0], 1, x.shape[1])
            x = (x * e2_embedded).sum(dim=2)
            x = (x + self.b[negative_ents]).reshape(-1, 1)
        else:
            # Each must only be multiplied by entity belong to *own* triplet!!!
            e2_embedded  = self.ent_embs(triplets[:, 2])
            x = (x * e2_embedded).sum(dim=1).reshape(-1, 1)  # This is the diagonal of the matrix product in 1-N
            x += self.b[triplets[:, 2]].unsqueeze(1)         # Bias terms associated with specific tails only
        
        return x.squeeze(1)


    def score_head(self, triplets):
        """
        Get the score for a given set of triplets against *all possible* heads.
        
        Parameters:
        -----------
            triplets: list
                List of triplets of form (rel, obj)

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        e1_embedded  = self.ent_embs(triplets[:, 1]).view(-1, 1, self.k_h, self.k_w)
        rel_embedded = self.rel_embs(triplets[:, 0]).view(-1, 1, self.k_h, self.k_w)

        x = self.score_function(e1_embedded, rel_embedded)
        x = torch.mm(x, self.ent_embs.weight.transpose(1,0))
        x += self.b.expand_as(x)

        return x

        
    def score_tail(self, triplets):
        """
        Get the score for a given set of triplets against *all possible* tails.

        Parameters:
        -----------
            triplets: list
                List of triplets of form (rel, sub)

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        return self.score_head(triplets)


