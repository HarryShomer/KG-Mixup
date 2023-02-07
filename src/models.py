"""
Extend standard KGE models to support scoring of synthetic triples
"""
import kgpy


class ConvE_Synthetic(kgpy.models.ConvE):
    """
    ConvE extended to score synthetic triples
    """
    def __init__(self, num_entities, num_relations, **kwargs):
        super().__init__(num_entities, num_relations, **kwargs)


    def score_synthetic(self, synth_rels, synth_heads, synth_tails):
        """
        Score synthetic samples

        Parameters:
        -----------
            synth_rels: torch.Tensor
                Synthetic relation embeddings
            synth_heads: torch.Tensor
                Synthetic head entity embeddings
            synth_tails: torch.Tensor
                Indices for tails to score against

        Returns:
        --------
        torch.Tensor
            Raw scores of triples
        """
        synth_heads  = synth_heads.view(-1, 1, self.k_h, self.k_w)
        synth_rels = synth_rels.view(-1, 1, self.k_h, self.k_w)

        x = self.score_function(synth_heads, synth_rels)
        synth_tails = self.ent_embs(synth_tails.long())

        # NOTE: No bias bec. tail is synthetic
        x = (x * synth_tails).sum(dim=1).reshape(-1, 1)  # This is the diagonal of the matrix product in 1-N
    
        return x


class TuckER_Synthetic(kgpy.models.TuckER):
    """
    TuckER extended to score synthetic triples
    """
    def __init__(self, num_entities, num_relations, **kwargs):
        super().__init__(num_entities, num_relations, **kwargs)


    def score_synthetic(self, synth_rels, synth_heads, synth_tails):
        """
        Score synthetic samples

        Parameters:
        -----------
            synth_rels: torch.Tensor
                Synthetic relation embeddings
            synth_heads: torch.Tensor
                Synthetic head entity embeddings
            synth_tails: torch.Tensor
                Indices for tails to score against

        Returns:
        --------
        torch.Tensor
            Raw scores of triples
        """
        x = self.score_function(synth_heads, synth_rels)
        synth_tails = self.ent_embs(synth_tails.long())

        # NOTE: No bias bec. tail is synthetic
        x = (x * synth_tails).sum(dim=1).reshape(-1, 1)  # This is the diagonal of the matrix product in 1-N
    
        return x