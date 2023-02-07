import os
import json
import torch
import random
import numpy as np
from collections import defaultdict
from torch_geometric.utils import to_dense_adj

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "datasets")



class TestDataset(torch.utils.data.Dataset):
    """
    Dataset object for test data
    """
    def __init__(self, triplets, all_triplets, num_entities, inverse=True, only_sample=False, device='cpu'):
        self.device = device
        self.inverse = inverse
        self.triplets = triplets
        self.num_entities = num_entities
        self.only_sample = only_sample

        self._build_index(all_triplets)


    def __len__(self):
        """
        Length of dataset
        """
        return len(self.triplets)
    

    def _build_index(self, triplets):
        """
        Mapping of triplets for testing
        """
        self.index = defaultdict(list)

        for t in triplets:
            if self.inverse:
                self.index[(t[1], t[0])].append(t[2])
            else:
                self.index[("head", t[1], t[2])].append(t[0])
                self.index[("tail", t[1], t[0])].append(t[2])

        # Remove duplicates
        for k, v in self.index.items():
            self.index[k] = list(set(v))


    def __getitem__(self, index):
        """
        For inverse just returns info for the tail/subject
        For non-inverse we return for both the head and tail

        Parameters:
        -----------
            index: int
                index for specific triplet

        Returns:
        -------
        tuple
            - Tensor containing subject and relation 
            - object ix
            - Tensor versus all possible objects - whether a true fact
        """        
        triple = torch.LongTensor(self.triplets[index])
        rel_sub = torch.LongTensor([triple[1].item(), triple[0].item()])
        rel_obj = torch.LongTensor([triple[1].item(), triple[2].item()])

        # Labels for all possible objects for triplet (s, r, ?)
        if self.inverse:
            # NOTE: self.only_sample is a hack to only assign the current triple as the true label!
            # This should not be used when training! 
            # This was only added to aid me in some analysis
            possible_obj = [triple[2]] if self.only_sample else np.int32(self.index[(triple[1].item(), triple[0].item())])

            obj_label  = self.get_label(possible_obj)

            return rel_sub, triple[2], obj_label, triple
        
        # For both (s, r, ?) and (?, r, o)
        possible_obj  = np.int32(self.index[("tail", triple[1].item(), triple[0].item())])
        possible_sub  = np.int32(self.index[("head", triple[1].item(), triple[2].item())])
        obj_label  = self.get_label(possible_obj)
        sub_label  = self.get_label(possible_sub)

        return rel_sub, triple[2], obj_label, rel_obj, triple[0], sub_label, triple 


        
    def get_label(self, possible_obj):
        y = np.zeros([self.num_entities], dtype=np.float32)
        
        for o in possible_obj: 
            y[o] = 1.0
        
        return torch.FloatTensor(y)




class AllDataSet():
    """
    Base class for all possible datasets
    """
    def __init__(
        self, 
        dataset_name, 
        inverse=False, 
        relation_pos="middle",
    ):
        self.inverse = inverse
        self.dataset_name = dataset_name
        self.relation_pos = relation_pos.lower()

        self.entity2idx, self.relation2idx = self._load_mapping()

        self.num_entities = len(self.entity2idx)
        self.num_relations = len(self.relation2idx)

        self.triplets = {
            "train": self._load_triplets("train"),
            "valid": self._load_triplets("valid"),
            "test":  self._load_triplets("test")
        }

        if self.inverse:
            self.num_relations *= 2
        

    @property
    def all_triplets(self):
        return list(set(self.triplets['train'] + self.triplets['valid'] + self.triplets['test']))

    @property
    def num_non_inv_rels(self):
        if self.inverse:
            return int(self.num_relations / 2)
        else:
            return self.num_relations

    @property
    def adjacency(self):
        """
        Construct the adjacency matrix. 1 where (u, v) in G else 0
        
        Notes:
            - This does not include relation info!
            - Stores on cpu
        """
        edge_index, _ = self.get_edge_tensors()

        # Construct adjacency where 1 = Neighbor otherwise 0
        adj = to_dense_adj(edge_index).squeeze(0)
        adj = torch.where(adj > 0, 1, 0)

        return adj


    def __getitem__(self, key):
        """
        Get specific dataset split
        """
        if key == 'train':
            return self.triplets['train']
        if key == 'valid':
            return self.triplets['valid']
        if key == "test":
            return self.triplets['test']
        
        raise ValueError("No key with name", key)


    def _load_mapping(self):
        """
        Load the mappings for the relations and entities from a file

        Returns:
        --------
        tuple
            dictionaries mapping an entity or relation to it's ID
        """
        entity2idx, relation2idx = {}, {}

        with open(os.path.join(DATA_DIR, self.dataset_name, "entity2id.txt"), "r") as f:
            for line in f:
                line_components = [l.strip() for l in line.split()]
                entity2idx[line_components[0]] = int(line_components[1])

        with open(os.path.join(DATA_DIR, self.dataset_name, "relation2id.txt"), "r") as f:
            for line in f:
                line_components = [l.strip() for l in line.split()]
                relation2idx[line_components[0]] = int(line_components[1])

        return entity2idx, relation2idx


    def _load_triplets(self, data_split):
        """
        Load the triplets for a given dataset and data split.

        Use mapping IDs to represent triplet components

        Parameters:
        -----------
            data_split: str 
                Which split of the data to load (train/test/validation)

        Returns:
        --------
        list
            contain tuples representing triplets
        """
        triplets = []

        with open(os.path.join(DATA_DIR, self.dataset_name, f"{data_split}.txt"), "r") as file:
            for line in file:
                fields = [l.strip() for l in line.split()]

                # Stored in file as "s, o, r" instead of "s, r, o"
                if self.relation_pos.lower() == "end":
                    fields[1], fields[2] = fields[2], fields[1]
                
                s = self.entity2idx[fields[0]]
                r = self.relation2idx[fields[1]]
                o = self.entity2idx[fields[2]]

                triplets.append((s, r, o))
                
                if self.inverse:
                    triplets.append((o, r + self.num_relations, s))

        return triplets


    def get_edge_tensors(self, device='cpu'):
        """
        Create the edge_index and edge_type from the training data 

        Create random edges by (if specified):
            - generate non inv edge
            - Create inv to go along with it (if needed) 

        Parameters:
        ----------
            rand_edge_perc: float
                Percentage of random edges to add. E.g. when .5 add .5m new edges (where m = # of edges)
            device: str
                device to put edge tensors on

        Returns:
        --------
        tuple of torch.Tensor
            edge_index, edge_type    
        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.triplets['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)
    
        edge_index	= torch.LongTensor(edge_index).to(device)
        edge_type = torch.LongTensor(edge_type).to(device)

        return edge_index.transpose(0, 1), edge_type


    def neighbor_rels_for_entity(self):
        """
        Unique relations for entity 

        For heads, we add non-inverse
        For tails, we add inverse 
        """
        r_adj = {e: set() for e in range(self.num_entities)}

        for t in self.triplets['train']:
            if t[1] < self.num_non_inv_rels:
                r_adj[t[0]].add(t[1])
                r_adj[t[2]].add(t[1] + self.num_non_inv_rels)

        return r_adj

    
    def neighbor_ents_for_entity(self):
        """
        Neighboring entities for entity...those connected by some relation
        """
        e_adj = {e: set() for e in range(self.num_entities)}

        for t in self.triplets['train']:
            e_adj[t[0]].add(t[2])
            e_adj[t[2]].add(t[0])

        return e_adj  


    def neighbor_ent_rels_for_entity(self):
        """
        Neighboring (e, r) pairs for a given entity
        
        """
        er_adj = {e: set() for e in range(self.num_entities)}

        for t in self.triplets['train']:
            if t[1] < self.num_non_inv_rels:
                er_adj[t[0]].add((t[1], t[2]))
                er_adj[t[2]].add((t[1] + self.num_non_inv_rels, t[0]))

        return er_adj


#######################################################
#######################################################
#######################################################


class FB15K_237(AllDataSet):
    """
    Load the FB15k-237 dataset
    """
    def __init__(self, **kwargs):
        super().__init__("FB15K-237", **kwargs)


class WN18RR(AllDataSet):
    """
    Load the WN18RR dataset
    """
    def __init__(self, **kwargs):
        super().__init__("WN18RR", **kwargs)


class FB15K(AllDataSet):
    """
    Load the FB15k dataset
    """
    def __init__(self, **kwargs):
        super().__init__("FB15K", relation_pos="end", **kwargs)


class WN18(AllDataSet):
    """
    Load the WN18 dataset
    """
    def __init__(self, **kwargs):
        super().__init__("WN18", relation_pos="end", **kwargs)


class YAGO3_10(AllDataSet):
    """
    Load the YAGO3-10 dataset
    """
    def __init__(self, **kwargs):
        super().__init__("YAGO3-10", **kwargs)


class CODEX_S(AllDataSet):
    """
    Load the Codex-M dataset
    """
    def __init__(self, **kwargs):
        super().__init__("CODEX-S", **kwargs)  


class CODEX_M(AllDataSet):
    """
    Load the Codex-M dataset
    """
    def __init__(self, **kwargs):
        super().__init__("CODEX-M", **kwargs)  


class CODEX_L(AllDataSet):
    """
    Load the Codex-M dataset
    """
    def __init__(self, **kwargs):
        super().__init__("CODEX-L", **kwargs)  


class NELL_995(AllDataSet):
    """
    Load the NELL-995 dataset
    """
    def __init__(self, **kwargs):
        super().__init__("NELL-995", **kwargs)  