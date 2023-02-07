import torch
import random
import numpy as np 
from collections import defaultdict

import kgpy


class Random_Over_Sampler(kgpy.sampling.One_to_N):
    """
    Extend kgpy.sampling.One_to_N to include oversampling

    Parameters:
    -----------
        triplets: list
            List of triplets. Each entry is a tuple of form (head, relation, tail)
        batch_size: int
            Train batch size
        num_ents: int
            Total number of entities in dataset
        device: str
            cpu or gpu
        threshold: int
            Oversample to this number of samples
    """
    def __init__(self, triplets, batch_size, num_ents, num_rels, device, threshold=10, inverse=True):
        super().__init__(triplets, batch_size, num_ents, num_rels, device, inverse)
        
        self.threshold = threshold
        self.reset()
    
    def __iter__(self):
        """
        Number of samples so far in epoch
        """
        self.trip_iter = self.oversample_iter = 0
        return self


    def reset(self):
        """
        Reset iter to 0 at beginning of epoch
        """
        self.trip_iter = self.oversample_iter = 0
        self._over_sample()
        self._shuffle()

        return self

    def _increment_iter(self):
        """
        Increment the iterator by batch size. Constrain to be len(keys) at max
        """
        self.trip_iter = min(self.trip_iter + self.bs, len(self.keys))
        self.oversample_iter = min(self.oversample_iter + self.oversample_bs, len(self.oversample_trips))


    def _shuffle(self):
        """
        Shuffle keys for both indices
        """
        np.random.shuffle(self.keys)

        try:
            np.random.shuffle(self.oversample_trips)
        except AttributeError:
            print("Attribute self.oversample_trips doesn't exist yet. Will be created soon")


    def _over_sample(self):
        """
        When Tail(h, r) < self.threshold, we add self.threshold - | Tail(h, r) | samples of form (t, r^-1, h) 

        Store in self.oversample_trips, which is reset at the start of each call.
        """
        self.oversample_trips = []

        for e in self.index:
            if len(self.index[e]) < self.threshold:
                inv_rel = self._get_inv_rel(e[0])
                num_samples = self.threshold - len(self.index[e])

                rand_pair_tails = random.choices(self.index[e], k=num_samples)
                rand_pair_trips = [(t, inv_rel, e[1]) for t in rand_pair_tails]  # stored as (t, r^-1, h)

                self.oversample_trips.extend(rand_pair_trips)

        ## Needs to have the same number of batches as (h, r) pairs which is stored in __len__
        self.oversample_bs = len(self.oversample_trips) // self.__len__()
        

    def _build_index(self):
        """
        Create self.index mapping.

        Only works for inverse!
                
        Returns:
        --------
        None
        """
        self.index = defaultdict(list)

        for t in self.triplets:
            self.index[(t[1], t[0])].append(t[2])

        # Remove duplicates
        for k, v in self.index.items():
            self.index[k] = list(set(v))


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

        # Same for oversampling pairs
        oversample_batch = np.array(self.oversample_trips[self.oversample_iter: min(self.oversample_iter + self.oversample_bs, len(self.oversample_trips))])
        oversample_batch = torch.Tensor(oversample_batch).to(self.device).long()
        
        self._increment_iter()

        batch_ix  = torch.Tensor(batch_samples.astype(np.float)).to(self.device).long()
        
        batch_lbls = self._get_labels(batch_samples)
        batch_lbls = torch.Tensor(batch_lbls).type(torch.float32).to(self.device) 

        return batch_ix, batch_lbls, oversample_batch 


