import sys
import torch
import numpy as np
from tqdm import tqdm

from kgpy.datasets import TestDataset


class Evaluation:
    """
    Class for predicting on a set of triplets and calculating the evaluation metrics

    Parameters:
    -----------
        triplets: list
            Triplets used for eval
        all_data: AllDataset
            AllDataset object
        inverse: bool
            Whether inverse edges
        eval_method: str
            filtered or raw
        bs: int
            Batch size
        device: str
            cpu or gpu
    """
    def __init__(self, triplets, all_data, inverse, eval_method='filtered', bs=128, device='cpu'):
        self.bs = bs
        self.data = all_data
        self.device = device
        self.inverse = inverse
        self.triplets = triplets
        self.eval_method = eval_method
        self.hits_k_vals = [1, 3, 10]

        if self.eval_method != "filtered":
            raise NotImplementedError("TODO: Implement raw evaluation metrics")
        


    def print_results(self, results, metrics=None):
        """
        Print the results of a given evaluation in a clean fashion

        Parameters:
        ----------
            results: dict
                Should be output of `self.evaluate`
            metrics: list
                Only print results for these metrics
        
        Returns:
        --------
        None
        """
        if metrics is None:
            metrics = ["samples", "mr", "mrr", "hits@1", "hits@3", "hits@10"]

        for k in metrics:
            print(f"  {k}: {round(results[k], 4)}", flush=True)


    def evaluate(self, model, raw_scores=False):
        """
        Evaluate the model on the valid/test set

        Parameters:
        -----------
            model: EmbeddingModel
                model we are fitting
            raw_scores: bool
                Whether to return raw probabilities for each triple

        Returns:
        --------
        dict
            eval metrics
        """
        metrics = ["steps", "samples", "mr", "mrr", "hits@1", "hits@3", "hits@10"]
        results = {m : 0 for m in metrics}

        dataloader = torch.utils.data.DataLoader(
                        TestDataset(self.triplets, self.data.all_triplets, self.data.num_entities, inverse=self.inverse, device=self.device), 
                        batch_size=self.bs,
                        num_workers=1
                    )

        model.eval()
        with torch.no_grad():            
            all_raw_scores = []

            for batch in tqdm(dataloader, desc="Evaluating model"):
                tail_trips, obj, tail_lbls = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)

                if not self.inverse:
                    head_trips, sub, head_lbls = batch[3].to(self.device), batch[4].to(self.device), batch[5].to(self.device)

                    head_preds = model(head_trips, mode="head")
                    tail_preds = model(tail_trips, mode="tail")
                
                    self.calc_metrics(head_preds, sub, head_lbls, results)
                    self.calc_metrics(tail_preds, obj, tail_lbls, results)
                else:
                    preds = model(tail_trips, mode="tail")
                    self.calc_metrics(preds, obj, tail_lbls, results)
                
                if raw_scores:
                    preds = preds.gather(1, obj.view(-1,1)).squeeze()
                    preds = [preds.item()] if len(preds.squeeze().shape) == 0 else preds.tolist()   # When only one test sample...
                    all_raw_scores.extend(preds)
        
        # Don't care about metrics here
        if raw_scores:
            return all_raw_scores

        ### Average out results
        results['mr']  = results['mr']  / results['steps'] 
        results['mrr'] = results['mrr'] / results['steps']  * 100

        for k in self.hits_k_vals:
            results[f'hits@{k}'] = results[f'hits@{k}'] / results['samples'] * 100

        return results



    def calc_metrics(self, preds, ent, lbls, results):
        """
        Calculate the metrics for a number of samples.

        NOTE: `results` dict is modified inplace

        Parameters:
        -----------
            preds: Tensor
                Score for triplets
            ent: Tensor
                Correct index for missing entity for triplet
            lbls: Tensor
                Tensor holding whether triplet tested is true or not
            results: dict
                Holds results for eval run so far
        
        Returns:
        --------
        None
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

        results['steps']   += 1
        results['samples'] += torch.numel(ranks) 
        results['mr']      += torch.mean(ranks).item() 
        results['mrr']     += torch.mean(1.0/ranks).item()

        for k in self.hits_k_vals:
            results[f'hits@{k}'] += torch.numel(ranks[ranks <= k])

