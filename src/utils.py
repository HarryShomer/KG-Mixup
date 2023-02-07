import os
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.optim.swa_utils import AveragedModel

import kgpy
import models
from kgpy.datasets import TestDataset





class EvaluationBySample_Original(kgpy.Evaluation):
    """
    Override of kgpy.Evaluation
    """
    def __init__(self, triplets, all_data, inverse, eval_method='filtered', bs=128, device='cpu'):
        super().__init__(triplets, all_data, inverse, eval_method=eval_method, bs=bs, device=device)
        

    def evaluate(self, model, metric_type, only_true=True, scores=False):
        """
        Override of original implementation. 
        Only returns whether each sample was a top 10 hit or not 
        NOTE: Only works for inverse!
        Parameters:
        -----------
            metric_type: str 
                one of ['mrr', 'hits10']
            only_true: bool
                Whether top 10 or just in ref to true lbl. Only applies when metric_type = "hits10". Defaults to True.
            scores: bool
                Return raw scores. Defaults to False
        """
        # Holds whether sample was in top 10 or not
        metric = []

        dataloader = torch.utils.data.DataLoader(
                        TestDataset(self.triplets, self.data.all_triplets, self.data.num_entities, inverse=self.inverse, device=self.device, only_sample=(scores and only_true)), 
                        batch_size=self.bs,
                        num_workers=8,
                        shuffle=False
                    )

        model.eval()
        with torch.no_grad():

            prog_bar = tqdm(dataloader, "Evaluating model")

            for batch in prog_bar:
                tail_trips, obj, tail_lbls = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                
                preds = model(tail_trips, mode="tail")

                if only_true:
                    ranks = self.calc_metrics(preds, obj, tail_lbls)
                else:
                    ranks = self.raw_ranks(preds, obj, tail_lbls)

                if scores and only_true:
                    preds = preds.gather(1, obj.view(-1,1)).squeeze()
                    preds = [preds.item()] if len(preds.squeeze().shape) == 0 else preds.tolist()   # When only one test sample...
                    metric.extend(preds)
                elif scores and not only_true:
                    #preds = torch.sigmoid(preds)
                    metric.extend(preds.tolist())           
                elif metric_type == "mrr":
                    metric_val = (1.0 / ranks).tolist()
                    metric.extend(metric_val)
                elif metric_type == "hits10" and only_true:
                    metric_val = torch.where(ranks <= 10, 1, 0).tolist()
                    metric.extend(metric_val)
                elif metric_type == "hits10" and not only_true:
                    metric.extend(ranks.tolist())
                else:
                    raise NotImplementedError(f"Invalid value for parameter 'metric_type' = {metric_type}")
                
        
        return metric



    def calc_metrics(self, preds, ent, lbls):
        """
        Override of original implementation.
        
        Now returns `ranks` tensor and skips calculating metrics
        """
        # [0, 1, 2, ..., BS]
        b_range	= torch.arange(preds.size()[0], device=self.device)

        # Extract scores for correct object for each sample in batch
        target_pred	= preds[b_range, ent]

        # self.byte() is equivalent to self.to(torch.uint8)
        # This makes true triplets not in the batch are equal to -1000000 by first setting **all** true triplets to -1000000
        # and then pluggin the original preds for the batch samples back in
        preds = torch.where(lbls.byte(), -torch.ones_like(preds) * 10000000, preds)
        preds[b_range, ent] = target_pred

        # Holds rank of correct score (e.g. When 1 that means the score for the correct triplet had the highest score)
        ranks = 1 + torch.argsort(torch.argsort(preds, dim=1, descending=True), dim=1, descending=False)[b_range, ent]
        ranks = ranks.float()
        
        return ranks


    def raw_ranks(self, preds, ent, lbls):
        """
        Same as "calc_metrics" but we just return the top 10 by hit10
        """
        # [0, 1, 2, ..., BS]
        b_range	= torch.arange(preds.size()[0], device=self.device)

        # Extract scores for correct object for each sample in batch
        target_pred	= preds[b_range, ent]

        # self.byte() is equivalent to self.to(torch.uint8)
        # This makes true triplets not in the batch are equal to -1000000 by first setting **all** true triplets to -1000000
        # and then pluggin the original preds for the batch samples back in
        preds = torch.where(lbls.byte(), -torch.ones_like(preds) * 10000000, preds)
        preds[b_range, ent] = target_pred

        # This just gets top 10 indices by scores
        ranks = torch.argsort(preds, dim=1, descending=True)[b_range, :10]
        ranks = ranks.float()
        
        return ranks




class BasicDataset(torch.utils.data.Dataset):
    """
    Basic dataset to iterate over samples
    """
    def __init__(self, triples, device='cpu'):
        self.device = device
        self.triples = triples


    def __len__(self):
        """
        Length of dataset
        """
        return len(self.triples)
    

    def __getitem__(self, ix):
        """
        Get specific triple

        Parameters:
        -----------
            ix: int
                index for specific triplw

        Returns:
        -------
        torch.Tensor
            triple indices
        """        
        return torch.LongTensor(self.triples[ix]).to(self.device)
        


def calc_ent_rel_degree(data):
    """
    Get # samples a given entity have with a relation r.

    Only in terms of (r, t)

    Key = (rel, tail)
    """
    ent_rel_degs = defaultdict(int)

    for t in data['train']:
        ent_rel_degs[(t[1], t[2])] += 1

    return ent_rel_degs



def get_saved_model(model_run, data, device, checkpoint_dir, model_params={}, swa=False, synthetic=True):
    """
    As the name says. Varies based on if rand sample model or original
    """
    if 'conve' in model_run.lower():
        model = models.ConvE_Synthetic if synthetic else kgpy.models.ConvE
    else:
        model  = models.TuckER_Synthetic if synthetic else kgpy.models.TuckER

    model = model(data.num_entities, data.num_relations, **model_params).to(device)

    if swa:
        model = AveragedModel(model)

    checkpoint = torch.load(os.path.join(checkpoint_dir, data.dataset_name.replace("_", "-"), f"{model_run}.tar"),  map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)

    return model



class EvaluationBySample(kgpy.Evaluation):
    """
    Override of kgpy.Evaluation to get performance by specific sample
    """
    def __init__(self, triplets, all_data, inverse, eval_method='filtered', bs=128, device='cpu'):
        super().__init__(triplets, all_data, inverse, eval_method=eval_method, bs=bs, device=device)
        

    def evaluate(self, model):
        """
        Override of original implementation. 
        
        Can now return predictions made

        NOTE: Only works for inverse!
        """
        dataloader = torch.utils.data.DataLoader(
                        TestDataset(self.triplets, self.data.all_triplets, self.data.num_entities, inverse=self.inverse, device=self.device), 
                        batch_size=self.bs,
                        num_workers=1,
                        shuffle=False
                    )


        true_scores, hits_10 = [], []

        model.eval()
        with torch.no_grad():

            prog_bar = tqdm(dataloader, "Evaluating model")

            for batch in prog_bar:
                # obj, lbls, trips = batch[1].to(self.device), batch[2].to(self.device), batch[3].to(self.device)
                tail_trips, obj, lbls = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                preds = model(tail_trips, mode="tail")

                ranks = self.calc_metrics(preds, obj, lbls)

                # Get score for true obj
                preds = preds.gather(1, obj.view(-1,1)).squeeze()
                preds = torch.sigmoid(preds)
                preds = [preds.item()] if len(preds.squeeze().shape) == 0 else preds.tolist()   # When only one test sample...
                true_scores.extend(preds)

                # Hits
                metric_val = torch.where(ranks <= 10, 1, 0).tolist()
                hits_10.extend(metric_val)

        
        return true_scores, hits_10


    def calc_metrics(self, preds, ent, lbls):
        """
        Override of original implementation.
        
        Now returns `ranks` tensor and skips calculating metrics
        """
        # [0, 1, 2, ..., BS]
        b_range	= torch.arange(preds.size()[0], device=self.device)

        # Extract scores for correct object for each sample in batch
        target_pred	= preds[b_range, ent]

        # self.byte() is equivalent to self.to(torch.uint8)
        # This makes true triplets not in the batch are equal to -1000000 by first setting **all** true triplets to -1000000
        # and then pluggin the original preds for the batch samples back in
        preds = torch.where(lbls.byte(), -torch.ones_like(preds) * 10000000, preds)
        preds[b_range, ent] = target_pred

        # Holds rank of correct score (e.g. When 1 that means the score for the correct triplet had the highest score)
        ranks = 1 + torch.argsort(torch.argsort(preds, dim=1, descending=True), dim=1, descending=False)[b_range, ent]
        ranks = ranks.float()
        
        return ranks