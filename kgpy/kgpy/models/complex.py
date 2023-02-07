"""
Implementation of Complex. 

See paper for more details - http://proceedings.mlr.press/v48/trouillon16.pdf.
"""
import torch

from .base.complex_emb_model import ComplexEmbeddingModel


class ComplEx(ComplexEmbeddingModel):

    def __init__(
        self, 
        num_entities, 
        num_relations, 
        emb_dim=100, 
        margin=1, 
        regularization = 'N3',
        reg_weight = 1e-1,
        weight_init="normal",
        loss_fn= "bce",
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


    def score_hrt(self, triplets, negative_ents=None):
        """        
        Score =  <Re(h), Re(r), Re(t)>
               + <Im(h), Re(r), Im(t)>
               + <Re(h), Im(r), Im(t)>
               - <Im(h), Im(r), Re(t)>

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
        r_re = self.rel_emb_re(triplets[:, 1])
        r_im = self.rel_emb_im(triplets[:, 1])

        if negative_ents is not None:
            # All negative tails
            t_re = self.ent_emb_re(negative_ents)
            t_im = self.ent_emb_im(negative_ents)

            return torch.sum(
                      (h_re.unsqueeze(-2) * r_re.unsqueeze(-2) * t_re) 
                    + (h_im.unsqueeze(-2) * r_re.unsqueeze(-2) * t_im)
                    + (h_re.unsqueeze(-2) * r_im.unsqueeze(-2) * t_im)
                    - (h_im.unsqueeze(-2) * r_im.unsqueeze(-2) * t_re)
                    , dim=-1
                ).flatten()
        else:
            t_re = self.ent_emb_re(triplets[:, 2])
            t_im = self.ent_emb_im(triplets[:, 2])
            return torch.sum(
                      (h_re * r_re * t_re) 
                    + (h_im * r_re * t_im)
                    + (h_re * r_im * t_im)
                    - (h_im * r_im * t_re)
                    , dim=-1
                ) 


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

        return torch.sum(
                  (h_re[None, :, :] * r_re[:, None, :] * t_re[:, None, :]) 
                + (h_im[None, :, :] * r_re[:, None, :] * t_im[:, None, :])
                + (h_re[None, :, :] * r_im[:, None, :] * t_im[:, None, :])
                - (h_im[None, :, :] * r_im[:, None, :] * t_re[:, None, :])
                , dim=-1
            ) 


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

        return torch.sum(
                  (h_re[:, None, :] * r_re[:, None, :] * t_re[None, :, :]) 
                + (h_im[:, None, :] * r_re[:, None, :] * t_im[None, :, :])
                + (h_re[:, None, :] * r_im[:, None, :] * t_im[None, :, :])
                - (h_im[:, None, :] * r_im[:, None, :] * t_re[None, :, :])
                , dim=-1
            ) 
