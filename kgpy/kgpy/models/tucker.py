"""
Implementation of TuckER. 

See paper for more details - https://arxiv.org/pdf/1901.09590.pdf.
"""
import torch
from .base.single_emb_model import SingleEmbeddingModel


class TuckER(SingleEmbeddingModel):
    def __init__(
        self, 
        num_entities, 
        num_relations, 
        ent_dim=200, 
        rel_dim=200,
        input_drop=0.3,
        hid_drop1=0.4,
        hid_drop2=0.5,
        bias=True,
        margin=1, 
        regularization = None,
        reg_weight = 0,
        weight_init="normal",
        loss_fn="bce",
        device='cpu',
        **kwargs
    ):
        super().__init__(
            type(self).__name__,
            num_entities, 
            num_relations, 
            ent_dim, 
            rel_dim, 
            margin, 
            regularization,
            reg_weight,
            weight_init, 
            loss_fn,
            True,
            device
        )
        self.W  = torch.nn.Parameter(torch.Tensor(rel_dim, ent_dim, ent_dim).to(device))
        torch.nn.init.xavier_normal_(self.W.data)

        self.input_drop = torch.nn.Dropout(input_drop)
        self.hid_drop1 = torch.nn.Dropout(hid_drop1)
        self.hid_drop2 = torch.nn.Dropout(hid_drop2)

        self.bn0 = torch.nn.BatchNorm1d(ent_dim)
        self.bn1 = torch.nn.BatchNorm1d(ent_dim)

        self.bias = bias 

        if bias:
            self.b = torch.nn.Parameter(torch.zeros(num_entities))


    def score_function(self, e1, r):
        """
        Scoring process of triplets

        Parameters:
        -----------
            e1: torch.Tensor
                entities passed through TuckER
            r: torch.Tensor
                relaitons passed through TuckER
        
        Returns:
        --------
        torch.Tensor
            Raw scores to be multipled by entities (e.g. dot product)
        """
        x = self.bn0(e1)
        x = self.input_drop(x)
        x = x.view(-1, 1, e1.size(1))

        # W x1 e_1 x2 w_r
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hid_drop1(W_mat)
        x = torch.bmm(x, W_mat) 
        
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hid_drop2(x)

        return x


    def score_hrt(self, triplets, negative_ents=None):
        """
        Scores for specific tails (e.g. against certain entities).

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
        e1 = self.ent_embs(triplets[:, 0])
        r  = self.rel_embs(triplets[:, 1])

        x = self.score_function(e1, r)

        if negative_ents is not None:
            # Each must only be multiplied by negative entities belonging to triple
            e2 = self.ent_embs(negative_ents)
            x = x.reshape(x.shape[0], 1, x.shape[1])
            x = (x * e2).sum(dim=2)
            x = (x + self.b[negative_ents]).reshape(-1, 1) if self.bias else x.reshape(-1, 1)
        else:
            # Each must only be multiplied by entity belong to *own* triplet!!!
            e2 = self.ent_embs(triplets[:, 2])
            x = (x * e2).sum(dim=1).reshape(-1, 1)    # This is the diagonal of the matrix product in 1-N
            x += self.b[triplets[:, 2]].unsqueeze(1)  if self.bias else x # Bias terms associated with specific tails only

        return x.squeeze(1)


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

        x = self.score_function(h, r)
        x = torch.mm(x, self.ent_embs.weight.transpose(1,0))

        if self.bias:
            x += self.b.expand_as(x)

        return x


    # TODO: For now just pass to score_tail since the same
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
        return self.score_tail(triplets)
