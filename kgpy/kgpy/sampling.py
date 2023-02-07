"""
Sampling strategies for training
"""
import torch
import numpy as np 
from tqdm import tqdm
from array import array
from collections import defaultdict
from abc import ABC, abstractmethod

from time import time


class Sampler(ABC):
    """
    Abstract base class for implementing samplers
    """
    def __init__(self, triplets, batch_size, num_ents, num_rels, device, inverse):
        self.bs = batch_size
        self.triplets = triplets
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.inverse = inverse
        self.device = device

        self._build_index()
        self.keys = list(self.index.keys())

        self.trip_iter = 0


    def __iter__(self):
        """
        Number of samples so far in epoch
        """
        self.trip_iter = 0
        return self
    

    def reset(self):
        """
        Reset iter to 0 at beginning of epoch
        """
        self.trip_iter = 0
        self._shuffle()

        return self


    @abstractmethod
    def _shuffle(self):
        """
        Shuffle training samples
        """
        pass


    @abstractmethod
    def _increment_iter(self):
        """
        Increment the iterator by batch size. Constrain to be len(_) at max
        """
        pass


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
        if rel < int(self.num_rels / 2):
            return rel + int(self.num_rels / 2)
        
        return rel - int(self.num_rels / 2)


    def _build_index(self):
        """
        Create self.index mapping.

        self.index contains 2 types of mappings:
            - All possible head entities for statement (_, relation, tail)
            - All possible tail entities for statement (head, relation, _)
        
        These are stored in self.index in form of:
            - For head mapping -> ("head", relation, tail)
            - For tail mapping -> ("tail", relation, head)
        
        The value for each key is a list of possible entities (e.g. [1, 67, 32]) 
        
        Returns:
        --------
        None
        """
        self.index = defaultdict(list)  # TODO: Change to array? Might be some code that relies on this being a list

        for t in self.triplets:
            if self.inverse:
                self.index[(t[1], t[0])].append(t[2])
            else:
                self.index[("head", t[1], t[2])].append(t[0])
                self.index[("tail", t[1], t[0])].append(t[2])

        # Remove duplicates
        for k, v in self.index.items():
            self.index[k] = list(set(v)) 

            
    def _get_labels(self, samples):
        """
        Get the label arrays for the corresponding batch of samples

        Parameters:
        -----------
            samples: Tensor
                2D Tensor of batches

        Returns:
        --------
        Tensor
            Size of (samples, num_ents). 
            Entry = 1 when possible head/tail else 0
        """
        # Numpy tends to be faster
        # y = torch.zeros(samples.shape[0], self.num_ents, dtype=torch.float16, device=self.device)
        y = np.zeros((samples.shape[0], self.num_ents), dtype=np.float16)

        for i, x in enumerate(samples):
            lbls = self.index[tuple(x)]
            y[i, lbls] = 1

        return y
    



#################################################################################
#
# Different Samplers
#
#################################################################################


class One_to_K(Sampler):
    """
    Standard sampler that produces k corrupted samples for each training sample.

    Does so by randomly corupting either the head or the tail of the sample

    Parameters:
    -----------
        triplets: list
            List of triplets. Each entry is a tuple of form (head, relation, tail)
        batch_size: int
            Train batch size
        num_ents: int
            Total number of entities in dataset
        num_negative: int
            Number of corrupted samples to produce per training procedure
    """
    def __init__(self, triplets, batch_size, num_ents, num_rels, device, num_negative=1, inverse=False, filtered=True):
        super(One_to_K, self).__init__(triplets, batch_size, num_ents, num_rels, device, inverse)

        self.filtered = filtered
        self.num_negative = num_negative        
        self.reset()

        if filtered:
            all_entities = set(range(self.num_ents))
            self.non_index = defaultdict(lambda: array("i", [])) # Hold tails *not* corresponding to (h, r, ?)

            print(">>> Creating additional sampling index for more efficient filtered 1-K training")
            for p, vals in tqdm(self.index.items(), desc="Creating sampling index"):
                self.non_index[p] = array("i", all_entities - set(vals)) 



    def __len__(self):
        """
        Total Number of batches
        """
        return len(self.triplets) // self.bs


    def _increment_iter(self):
        """
        Increment the iterator by batch size. Constrain to be len(keys) at max
        """
        self.trip_iter = min(self.trip_iter + self.bs, len(self.triplets))


    def _shuffle(self):
        """
        Shuffle samples
        """
        np.random.shuffle(self.triplets)



    # def _sample_negative_from_lbl(self, samples):
    #     """
    #     Samples negative samples from the self._get_lbls() method as 0 entries == negative samples

    #     Parameters:
    #     -----------
    #         samples: list/numpy.array
    #             Samples in form of (h, r, t)
        
    #     Returns:
    #     --------
    #     list
    #         self.num_negative negative samples for each sample
    #     """
    #     all_neg_ents = []

    #     samples =  np.array([(s[1], s[0]) for s in samples])
    #     samples_lbls = self._get_labels(samples)

    #     for sample_lbl in samples_lbls:
    #         all_neg_samples = (sample_lbl == 0).nonzero()[0]
    #         neg_samples = np.random.choice(all_neg_samples, self.num_negative, replace=False)

    #         all_neg_ents.append(neg_samples)
        
    #     return all_neg_ents

    
    def _sample_negative(self, samples):
        """
        Samples `self.num_negative` triplets for each training sample in batch

        Do so by randomly replacing either the head or the tail with another entity.

        We exclude possible corruptions that would result in a real triple (see utils.randint_exclude)

        Parameters:
        -----------
            samples: list of tuples
                triplets to corrupt 

        Returns:
        --------
        torch.Tensor
            Corrupted Triplets
        """
        random_ents = []

        for t in samples:
            if self.inverse:

                if self.filtered:
                    ents_to_sample = self.non_index[(t[1], t[0])]
                    sampled_ents = np.random.choice(ents_to_sample, self.num_negative)
                else:
                    sampled_ents = np.random.randint(0, self.num_ents, self.num_negative)
            else:    
                # TODO
                raise NotImplementedError("TODO: 1-K Training without inverse triples")
                # head_ents_to_exclude = set(self.index[("head", t[1], t[2])])
                # tail_ents_to_exclude = set(self.index[("tail", t[1], t[0])])
                # head_or_tail = random.choices([0, 2], k=self.num_negative)   # Just do it all at once
                # for i, h_t in zip(range(self.num_negative), head_or_tail):
                #     if h_t == 0:
                #         rand_ent = utils.randint_exclude(0, self.num_ents, head_ents_to_exclude)
                #         new_sample = (rand_ent, t[1], t[2])
                #     else:
                #         rand_ent = utils.randint_exclude(0, self.num_ents, tail_ents_to_exclude)
                #         new_sample = (t[0], t[1], rand_ent)

            random_ents.append(sampled_ents)


        return torch.Tensor(random_ents).to(self.device).long()


    def __next__(self):
        """
        Grab next batch of samples

        Returns:
        -------
        tuple (Tensor, Tensor)
            triplets in batch, corrupted samples for batch
        """
        if self.trip_iter >= len(self.triplets)-1:
            raise StopIteration

        # Collect next self.bs samples & labels
        batch_samples = self.triplets[self.trip_iter: min(self.trip_iter + self.bs, len(self.triplets))]

        neg_samples = self._sample_negative(batch_samples)  

        batch_samples = torch.Tensor([list(x) for x in batch_samples]).to(self.device).long()

        self._increment_iter()

        return batch_samples, neg_samples



class One_to_N(Sampler):
    """
    For each of (?, r, t) and (h, r, ?) we sample each possible ent

    Parameters:
    -----------
        triplets: list
            List of triplets. Each entry is a tuple of form (head, relation, tail)
        batch_size: int
            Train batch size
        num_ents: int
            Total number of entities in dataset
    """
    def __init__(self, triplets, batch_size, num_ents, num_rels, device, inverse=False):
        super(One_to_N, self).__init__(triplets, batch_size, num_ents, num_rels, device, inverse)
        self._shuffle()


    def __len__(self):
        return len(self.index) // self.bs


    def _increment_iter(self):
        """
        Increment the iterator by batch size. Constrain to be len(keys) at max
        """
        self.trip_iter = min(self.trip_iter + self.bs, len(self.keys))


    def _shuffle(self):
        """
        Shuffle keys for both indices
        """
        np.random.shuffle(self.keys)


    def __next__(self):
        """
        Grab next batch of samples

        Returns:
        -------
        tuple
            indices, lbls, trip type - head/tail (optional)
        """
        if self.trip_iter >= len(self.keys)-1:
            raise StopIteration

        # Collect next self.bs samples
        batch_samples = self.keys[self.trip_iter: min(self.trip_iter + self.bs, len(self.keys))]
        batch_samples = np.array(batch_samples)

        self._increment_iter()

        if self.inverse:
            batch_ix  = torch.from_numpy(batch_samples).to(self.device).long()
            batch_lbls = self._get_labels(batch_samples)
            batch_lbls = torch.from_numpy(batch_lbls).type(torch.float32).to(self.device)

            return batch_ix, batch_lbls 
        else:
            # Split by type of trip and ent/rel indices
            trip_type = batch_samples[:, 0]
            batch_ix  = torch.Tensor(batch_samples[:, 1:].astype(np.float)).to(self.device).long()

            batch_lbls = self._get_labels(batch_samples) 
            batch_lbls = torch.Tensor(batch_lbls).type(torch.float32).to(self.device) 

            return batch_ix, batch_lbls, trip_type

