import torch
import random
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.distributions import Beta

from utils import calc_ent_rel_degree



class SyntheticGenerator:
    """
    Sample synthetic tails for minority entity-relation pairs
    """
    def __init__(
            self, 
            data, 
            alpha,
            threshold, 
            max_generate,
            neg_samples=5,
            device='cuda',
            dims=[]
        ):
        self.data = data
        self.device = device
        self.threshold = threshold
        self.neg_samples = neg_samples
        self.max_generate = max_generate
        self.beta_dist = Beta(alpha, alpha)

        self.tail_to_pairs = self._calc_tails_to_pairs()

        self.ent_rel_degree = calc_ent_rel_degree(data)
        self.trips_to_aug = self._get_trips_to_augment()  # NOTE: Is a set



    def _calc_tails_to_pairs(self):
        """
        For each tail, calculate the unique (head, relation) pairs
        """
        tail_to_pairs = defaultdict(list)

        for t in self.data['train']:
            tail_to_pairs[t[2]].append((t[0], t[1]))
        
        return tail_to_pairs


    def _calc_rels_to_pairs(self):
        """
        r -> (h, t) pairs
        """
        rel_to_pairs = defaultdict(list)

        for t in self.data['train']:
            rel_to_pairs[t[1]].append((t[0], t[2]))

        return rel_to_pairs


    def _get_inv_rel(self, rel):
        """
        Get the inverse relation.

        If > num_non_inv_rels then we convert to a regular relation.  Otherwise we convert a regular relation to an inverse

        Parameters:
        -----------
            rel: int
                relation
        
        Returns:
        --------
        int
            inverse relation
        """
        if rel < self.data.num_non_inv_rels:
            return rel + self.data.num_non_inv_rels
        
        return rel - self.data.num_non_inv_rels



    def _get_trips_to_augment(self):
        """
        Get trips to augment based on self.ent_rel_degree < threshold
        """
        trips = set()

        for t in tqdm(self.data["train"], "Trips to Augment"):

            if self.ent_rel_degree[(t[1], t[2])] < self.threshold:
                trips.add(t)

        return trips


    def generate_batch(self, triples, ent_embs, rel_embs):
        """
        For each pair in 'batch_pairs' we generate if # of tails is less than self.threshold

        Parameters:
        -----------
            triples: list
                List of triples (h, r, t) in batch
            ent_embs: torch.nn.Embedding
                All possible entity embeddings
            rel_embs: torch.nn.Embedding
                All possible relation embeddings

        Returns:
        --------
        list
            list of synthetic samples
        """
        all_repeat_trips = []
        all_sampled_edges = []

        for trip in triples.tolist():
            trip = tuple(trip)
            
            if trip in self.trips_to_aug:
                repeat_triples, sampled_edges = self.oversample_triple(trip)
                
                if repeat_triples is not None:
                    all_repeat_trips.append(repeat_triples)
                    all_sampled_edges.append(sampled_edges)

        all_repeat_trips  = torch.cat(all_repeat_trips)
        all_sampled_edges = torch.cat(all_sampled_edges)
        synth_heads, synth_rels = self.mixup_triples(all_repeat_trips, all_sampled_edges, ent_embs, rel_embs)

        return synth_rels, synth_heads, all_repeat_trips[:, 2], all_sampled_edges


    def oversample_triple(self, triple):
        """
        Oversample for a given triple
        """
        tail = triple[2]
        head_rel_pair = (triple[0], triple[1])

        num_to_sample = self.max_generate

        # Remove edge from possible mixup edges
        possible_edges = [e for e in self.tail_to_pairs[tail] if e != head_rel_pair]

        # If not enough we just sample max amount, assuming we have at least 1 edge to sample
        if num_to_sample > len(possible_edges):
            # return None, None
            if len(possible_edges) == 0:
                return None, None
            else:
                num_to_sample = len(possible_edges)
        
        sampled_edges = random.sample(possible_edges, k=num_to_sample)
        
        # Want num_to_sample copies of original triple. Makes combining easier later on
        repeat_triple = [triple for _ in range(num_to_sample)]

        repeat_triple = torch.Tensor(repeat_triple).to(self.device).long()
        sampled_edges = torch.Tensor(sampled_edges).to(self.device).long()

        return repeat_triple, sampled_edges


    def mixup_triples(self, trips, sampled_edges, ent_embs, rel_embs):
        """
        Do the actual mixing of triples
        """
        # Mixed edges - Head / Relation embs
        mix_rel_embs = rel_embs(sampled_edges[:, 1])
        mix_head_embs = ent_embs(sampled_edges[:, 0])

        # Base edge Head / Relation embs
        rel_ent_embs = rel_embs(trips[:, 1])
        head_ent_embs = ent_embs(trips[:, 0])

        alphas = self.beta_dist.sample_n(head_ent_embs.shape[0]).reshape(-1, 1).to(self.device)

        # Perform mixup
        synth_rels = alphas * rel_ent_embs + (1 - alphas) * mix_rel_embs
        synth_heads = alphas * head_ent_embs + (1 - alphas) * mix_head_embs

        return synth_heads, synth_rels



    def mix_negative_samples(self, heads_to_combine, ent_embs):
        """
        Do the mixing for the negative samples
        """
        heads_to_combine = torch.Tensor(heads_to_combine).to(self.device).long() 

        h1_embs = ent_embs(heads_to_combine[:, 0])   
        h2_embs = ent_embs(heads_to_combine[:, 1])          

        alphas = torch.rand((h1_embs.shape[0], 1)).to(self.device)
        synth_heads = alphas * h1_embs + (1 - alphas) * h2_embs

        return synth_heads
