import os
import torch
import optuna 
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.nn.functional import binary_cross_entropy_with_logits

import kgpy
import utils
from sampling import Random_Over_Sampler


class BaseTrainer(kgpy.Trainer):
    """
    Override of training procedure
    """
    def __init__(
        self, 
        model, 
        optimizer, 
        data, 
        checkpoint_dir, 
        swa=False,
        swa_start=10,
        swa_every=10,
        swa_lr=5e-4,
        swa_anneal=10,
        model_name=None,
        **kwargs
    ):
        super().__init__(model, optimizer, data, checkpoint_dir, model_name=model_name)

        self.swa = swa 
        self.loss = torch.nn.BCEWithLogitsLoss()

        # TODO: May need to adjust anneal_epochs and swa_start
        if swa:
            self.swa_start = swa_start
            self.swa_every = swa_every
            self.swa_model = AveragedModel(self.model)     
            self.swa_scheduler = SWALR(self.optimizer, swa_lr, anneal_strategy="cos", anneal_epochs=swa_anneal)


    @property
    def model_name(self):
        """
        Either use use-supplied or default to this
        """
        if self._model_name:
            return self._model_name
        
        return self.model.name


    def fit(
            self, 
            epochs, 
            synth_generator,
            bs=128,
            train_type="1-K",
            validate_every=5, 
            non_train_batch_size=128, 
            early_stopping=5, 
            save_every=25,
            eval_method="filtered",
            negative_samples=100,
            label_smooth=0,
            decay=None,
            test_model=True,
            optuna_trial=None
        ):
        """
        Train, validate, and test the model

        Parameters:
        -----------
            epochs: int
                Number of epochs to train for
            synth_generator: SyntheticSampler
                Synthetic sampler object 
            bs: int
                Batch size
            validate_every: int
                Validate every "n" epochs. Defaults to 5
            non_train_batch_size: int 
                Batch size for non-training data. Defaults to 16
            early_stopping: int
                Stop training if the mean rank hasn't improved in last "n" validation scores. Defaults to 5
            save_every: int
                Save model every "n" epochs. Defaults to 25 When None only no epoch specific versions are saved
            negative_samples: int
                Number of negative samples to generate for each training sample. Defaults to 1
            eval_method: str
                How to evaluate data. Filtered vs raw. Defaults to filtered
            label_smooth: float
                Label smoothing when training
            decay: float
                LR decay of form of decay^epoch.
            test_model: bool
                Test the model on the test set. Set to false when fine-tuning
            optuna_trial: optuna.trial.Trial
                Optuna trial if passed. Default is None

        Returns:
        --------
        float
            Test MRR when test_model=True else Validation MRR
        """
        val_mrr = []
        
        lr_scheduler = self._get_lr_scheduler(decay)
        sampler = self._get_sampler(train_type, bs, num_negative=negative_samples)
        model_eval = kgpy.Evaluation(self.data['valid'], self.data, self.inverse, eval_method=eval_method, bs=non_train_batch_size, device=self.device)
        
        for epoch in range(1, epochs+1):
            self.model.train()
            epoch_loss = 0

            for batch in tqdm(sampler, f"Epoch {epoch}"):
                loss = self._train_batch(batch, train_type, label_smooth)
                
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                epoch_loss += loss

            # print("\nBatch Loss:", round(epoch_loss, 4), flush=True)

            # Update done b4 evaluation! Otherwise set to eval mode
            if self.swa and epoch >= self.swa_start and epoch % self.swa_every == 0:
                self.swa_model.train()
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()    

            if epoch % validate_every == 0:
                val_mrr.append(self._validate_model(model_eval, epoch))

                if optuna_trial is not None:
                    optuna_trial.report(val_mrr[-1], len(val_mrr))
        
                    if optuna_trial.should_prune(): 
                        raise optuna.exceptions.TrialPruned()

                if self._early_stop_criteria(val_mrr, early_stopping):
                    print(f"Validation loss hasn't improved in the last {early_stopping} Valid MRR scores. Stopping training now!", flush=True)
                    break

                self.save_model(self.model_name)

            if save_every is not None and epoch % save_every == 0:
                self.save_model(f"{self.model_name}_epoch-{epoch}")
            
            sampler.reset()

            # TODO: Account for SWA?
            if decay: 
                lr_scheduler.step()    

        if test_model:
            return self._test_model(eval_method, non_train_batch_size)
        
        return val_mrr[-1]



    def _update_swa_batch(self):
        """
        Compute batch norm statistics on training data for SWA model
        """
        dataloader = DataLoader(utils.BasicDataset(self.data['train'], device=self.device), batch_size=128)
        torch.optim.swa_utils.update_bn(dataloader, self.swa_model)



    def _validate_model(self, model_eval, epoch):
        """
        Evaluate model on val set

        Parameters:
        -----------
            model_eval: Evaluation
                Evaluation object
            epoch: int
                epoch number

        Returns:
        --------
        float
            MRR
        """
        if self.swa and self.swa_start >= epoch:
            self._update_swa_batch()
            model = self.swa_model
        else:
            model = self.model 

        results = model_eval.evaluate(model)
        
        print(f"\nEpoch {epoch} validation:", flush=True)
        model_eval.print_results(results)

        return results['mrr']
    

    def _test_model(self, eval_method, bs):
        """
        Evaluate model on the test set

        Parameters:
        -----------
            eval_method: str
                filtered or not
            bs: int
                batch size
        
        Returns:
        --------
        float
            MRR
        """
        if self.swa:
            self._update_swa_batch()
            model = self.swa_model
        else:
            model = self.model 

        model_eval = kgpy.Evaluation(self.data['test'], self.data, self.inverse, eval_method=eval_method, bs=bs, device=self.device)
        test_results = model_eval.evaluate(model)
        
        print("\nTest Results:", flush=True)
        model_eval.print_results(test_results)

        return test_results['mrr']


    def save_model(self, model_name):
        """
        Save the given model's state
        """
        if not os.path.isdir(os.path.join(self.checkpoint_dir, self.data.dataset_name)):
            os.makedirs(os.path.join(self.checkpoint_dir, self.data.dataset_name), exist_ok=True)
        
        if self.swa:
            model_name = f"SWA-{model_name}"

        model_obj = self.swa_model if self.swa else self.model

        torch.save({
            "model_state_dict": model_obj.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            }, 
            os.path.join(self.checkpoint_dir, self.data.dataset_name, f"{model_name}.tar")
        )

    def _get_sampler(self, train_method, bs, num_negative=None):
        """
        Retrieve a sampler object for the type of train method

        Parameters:
        -----------
            train_method: str
                1-N or 1-K
            bs: int
                batch_size
            num_negative: int
                number onf negative samples. Only applicable for 1-K training
        
        Returns:
        --------
        kgpy.Sampling.Sampler
            sampler for training
        """
        train_method = train_method.upper()

        if train_method == "1-K":
            sampler = kgpy.sampling.One_to_K(
                        self.data['train'], 
                        bs, 
                        self.data.num_entities,
                        self.data.num_relations, 
                        self.device,
                        num_negative=num_negative,
                        inverse=self.data.inverse,
                        filtered = not "nell" in self.data.dataset_name.lower()
                    )
        elif train_method == "1-N":
            sampler = kgpy.sampling.One_to_N(
                        self.data['train'], 
                        bs, 
                        self.data.num_entities, 
                        self.data.num_relations,
                        self.device,
                        inverse=self.data.inverse,
                    )
        else:
            raise ValueError(f"Invalid train method `{train_method}`")
        
        return sampler

########################################################################
########################################################################
########################################################################


class MixupTrainer(BaseTrainer):
    """
    Trainer for performing Mixup
    """
    def __init__(
        self, 
        model, 
        optimizer, 
        data, 
        checkpoint_dir, 
        synth_weight=1,
        swa=False,
        swa_every=10,
        swa_lr=5e-4,
        swa_anneal=10,
        model_name=None,
        **kwargs
    ):
        super().__init__(model, optimizer, data, checkpoint_dir, swa=swa, swa_every=swa_every, swa_lr=swa_lr, swa_anneal=swa_anneal, model_name=model_name)
        self.synth_weight = synth_weight


    def fit(
            self, 
            epochs, 
            synth_generator,
            bs=128,
            validate_every=5, 
            non_train_batch_size=128, 
            early_stopping=5, 
            save_every=25,
            eval_method="filtered",
            negative_samples=100,
            label_smooth=0,
            decay=None,
            test_model=True,
            optuna_trial=None,
            sampler=None,
            **kwargs
        ):
        """
        Train, validate, and test the model

        Parameters:
        -----------
            epochs: int
                Number of epochs to train for
            synth_generator: SyntheticSampler
                Synthetic sampler object 
            bs: int
                Batch size
            validate_every: int
                Validate every "n" epochs. Defaults to 5
            non_train_batch_size: int 
                Batch size for non-training data. Defaults to 16
            early_stopping: int
                Stop training if the mean rank hasn't improved in last "n" validation scores. Defaults to 5
            save_every: int
                Save model every "n" epochs. Defaults to 25 When None only no epoch specific versions are saved
            negative_samples: int
                Number of negative samples to generate for each training sample. Defaults to 1
            eval_method: str
                How to evaluate data. Filtered vs raw. Defaults to filtered
            label_smooth: float
                Label smoothing when training
            decay: float
                LR decay of form of decay^epoch.
            test_model: bool
                Test the model on the test set. Set to false when fine-tuning
            optuna_trial: optuna.trial.Trial
                Optuna trial if passed. Default is None

        Returns:
        --------
        float
            Test MRR when test_model=True else Validation MRR
        """
        val_mrr = []
        
        lr_scheduler = self._get_lr_scheduler(decay)
        model_eval = kgpy.Evaluation(self.data['valid'], self.data, self.inverse, eval_method=eval_method, bs=non_train_batch_size, device=self.device)
        
        if sampler is None:
            sampler = self._get_sampler(negative_samples, bs)
        
        for epoch in range(1, epochs+1):
            self.model.train()
            epoch_real_loss = epoch_synth_loss = 0

            for batch in tqdm(sampler, f"Epoch {epoch}"):
                synth_batch = synth_generator.generate_batch(batch[0], self.model.ent_embs, self.model.rel_embs)
                real_loss, synth_loss = self._train_batch(batch, synth_batch, label_smooth)
                    
                epoch_real_loss += real_loss
                epoch_synth_loss += synth_loss

            print("\nBatch Loss:", round(epoch_real_loss, 4), flush=True)
            print("Synth Loss:", round(epoch_synth_loss, 4), flush=True)
            print("Total Loss:", round(epoch_real_loss + self.synth_weight * epoch_synth_loss, 4), flush=True)

            # Update done b4 evaluation! Others set to eval mode
            if self.swa and epoch >= self.swa_start and epoch % self.swa_every == 0:
                self.swa_model.train()
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()    

            if epoch % validate_every == 0:
                val_mrr.append(self._validate_model(model_eval, epoch))

                if optuna_trial is not None:
                    optuna_trial.report(val_mrr[-1], len(val_mrr))
        
                    if optuna_trial.should_prune(): 
                        raise optuna.exceptions.TrialPruned()

                if self._early_stop_criteria(val_mrr, early_stopping):
                    print(f"Validation loss hasn't improved in the last {early_stopping} Valid MRR scores. Stopping training now!", flush=True)
                    break

                self.save_model(self.model_name)

            if save_every is not None and epoch % save_every == 0:
                self.save_model(f"{self.model_name}_epoch-{epoch}")
            
            sampler.reset()

            # TODO: Account for SWA?
            if decay: 
                lr_scheduler.step()    

        if test_model:
            return self._test_model(eval_method, non_train_batch_size)
        
        return val_mrr[-1]


    def _train_batch(self, real_batch, synth_batch, label_smooth):
        """
        Train all batches

        Parameters:
        -----------
            real_batch: tuple of Tensors
                Tuple of (positive_samples, negative_samples)
            synth_batch: tuple
                Tuple of synthetic samples
            label_smooth: float
                Label smoothing
        
        Returns:
        -------
        tuple
            batch losses for both real and synth batches
        """        
        self.optimizer.zero_grad()

        real_loss = self._train_batch_1_to_k(real_batch, label_smooth, reduction="sum")
        mixup_loss = self._train_mixup_batch(synth_batch, label_smooth, reduction="sum")
        total_loss = real_loss + self.synth_weight * mixup_loss

        total_loss = total_loss.mean()
        total_loss.backward()

        self.optimizer.step()

        return real_loss.mean().item(), mixup_loss.mean().item()


    def _train_mixup_batch(self, batch, label_smooth, reduction="mean"):
        """
        Train mixup batch
        Only predict tail, which is one with low degree
        """
        lbls = torch.ones(len(batch[0])).to(self.device)
        
        scores = self.model.score_synthetic(batch[0], batch[1], batch[2])
        scores = scores.squeeze()

        if label_smooth != 0.0:
            lbls = (1.0 - label_smooth)*lbls + (1.0 / self.data.num_entities)

        return self.model.loss(all_scores=scores, all_targets=lbls, reduction=reduction)



    def _get_sampler(self, neg_samples, bs):
        """
        Get sampler object
        
        Parameters:
        -----------
            neg_samples: int
                Only relevant when performing mixup and using 1-K sampling
            bs: int
                Batch size
        
        Returns:
        --------
        kgpy.sampling.Sampler
            sampler object
        """
        filtered = not ("nell" in self.data.dataset_name.lower() or "wn18rr" in self.data.dataset_name.lower())
        return kgpy.sampling.One_to_K(
            self.data['train'],
            bs,
            self.data.num_entities, 
            self.data.num_relations, 
            self.device,
            inverse=True,
            num_negative=neg_samples,
            filtered = filtered
        )




class OversampleTrainer(BaseTrainer):
    """
    Trainer for performing oversampling
    """
    def __init__(
        self, 
        model, 
        optimizer, 
        data, 
        checkpoint_dir, 
        threshold=10,
        swa=False,
        swa_every=10,
        swa_lr=5e-4,
        swa_anneal=10,
        model_name=None,
        **kwargs
    ):
        super().__init__(model, optimizer, data, checkpoint_dir, swa=swa, swa_every=swa_every, swa_lr=swa_lr, swa_anneal=swa_anneal, model_name=model_name)
        self.threshold = threshold


    def _train_batch(self, batch, train_type, label_smooth):
        """
        Train all batches

        Parameters:
        -----------
            batch: tuple of Tensors
                Tuple of (samples, labels, oversample triples)
            label_smooth: float
                Label smoothing
        
        Returns:
        -------
        tuple
            batch losses for total loss
        """        
        self.optimizer.zero_grad()

        real_loss = self._train_batch_1_to_n(batch, label_smooth, reduction="sum")
        over_loss = self._train_oversample_batch(batch, label_smooth, reduction="sum")

        total_loss = real_loss + over_loss

        total_loss = total_loss.mean()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()


    def _train_oversample_batch(self, batch, label_smooth, reduction):
        """
        Train for oversampled trips
        """
        triples = batch[2]
        all_scores = self.model.score_hrt(triples)

        if label_smooth != 0.0:
            all_lbls = (1.0 - label_smooth)*torch.ones(len(all_scores)).to(self.device) + (1.0 / len(all_scores))
        else:
            all_lbls = torch.ones(len(all_scores)).to(self.device)
        
        return self.model.loss(all_scores=all_scores, all_targets=all_lbls, reduction=reduction)



    def _get_sampler(self, train_type, bs, **kwargs):
        """
        Get sampler object
        
        Parameters:
        -----------
            bs: int
                Batch size
        
        Returns:
        --------
        kgpy.sampling.Sampler
            sampler object
        """
        return Random_Over_Sampler(
            self.data['train'],
            bs,
            self.data.num_entities, 
            self.data.num_relations, 
            self.device,
            inverse=True,
            threshold=self.threshold
        )



class LossTrainer(BaseTrainer):
    """
    Trainer for performing focal loss or standard reweighting
    """
    def __init__(
        self, 
        model, 
        optimizer, 
        data, 
        checkpoint_dir, 
        strategy_name,
        swa=False,
        swa_every=10,
        swa_lr=5e-4,
        swa_anneal=10,
        model_name=None,
        reweight_up=1,
        threshold=5,
        **kwargs
    ):
        super().__init__(model, optimizer, data, checkpoint_dir, swa=swa, swa_every=swa_every, swa_lr=swa_lr, swa_anneal=swa_anneal, model_name=model_name)
        self.strategy_name = strategy_name
        self.reweight_up = reweight_up
        self.threshold = threshold
        
        if self.strategy_name != "focal":
            self.pair_to_tails = self._calc_pair_to_tails()


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
        if rel < int(self.data.num_relations / 2):
            return rel + int(self.data.num_relations / 2)
        
        return rel - int(self.data.num_relations / 2)



    def _calc_pair_to_tails(self):
        """
        Calculate (h, r) pairs to all possible tails. Already includes inverse
        
        Returns:
        --------
        None
        """
        pair_to_tails = defaultdict(list)

        for t in self.data['train']:
            pair_to_tails[(t[1], t[0])].append(t[2])
        
        return pair_to_tails



    def _calc_trip_weights(self, batch_pairs):
        """
        For all real triples, we calculate the normalized loss weight.

        weight_pair = 1 + self.reweight_up

        Parameters:
        ---------- 
            batch_pairs: list of tuples
                Each is (rel, head) pair
        
        Returns:
        --------
        torch.Tensor
            Loss weights for each possible sample
        """
        all_pairs_degree = []

        for (r, h) in batch_pairs:
            h, r = h.item(), r.item()
            inv_rel = self._get_inv_rel(r)
            possible_tails = self.pair_to_tails.get((r, h), [])

            # Assume all 1. Add `self.reweight_up` to those below thresh
            degree_all_ents = [1] * self.data.num_entities 

            # Now fill in all we know are true
            for tail in possible_tails:
                if len(self.pair_to_tails.get((inv_rel, tail), [])) < self.threshold:
                    degree_all_ents[tail] = self.reweight_up + 1  # Since default is one
                
            all_pairs_degree.append(degree_all_ents)
        
        all_pairs_degree = torch.Tensor(all_pairs_degree).to(self.device)
        all_pairs_degree = all_pairs_degree / all_pairs_degree.sum()  # Normalize

        return all_pairs_degree.detach()



    def _train_batch(self, batch, train_type, label_smooth):
        """
        Train all batches

        Parameters:
        -----------
            batch: tuple of Tensors
                Tuple of (samples, labels, oversample triples)
            label_smooth: float
                Label smoothing
        
        Returns:
        -------
        tuple
            batch losses for total loss
        """        
        self.optimizer.zero_grad()

        trips, all_lbls = batch[0], batch[1]
        all_scores = self.model(trips, mode="tail")

        if label_smooth != 0.0:
            smooth_lbls = (1.0 - label_smooth)*all_lbls + (1.0 / self.data.num_entities)
        else:
            smooth_lbls = all_lbls

        if self.strategy_name == "focal":
            loss_weights = torch.where(all_lbls == 1, torch.sigmoid(all_scores), 1 - torch.sigmoid(all_scores))
            loss_weights = (1 - loss_weights)  # This equivalent to \gamma=1
            loss = binary_cross_entropy_with_logits(all_scores, smooth_lbls, weight=loss_weights.detach(), reduction="mean")
        else:
            loss_weights = self._calc_trip_weights(trips)
            loss = binary_cross_entropy_with_logits(all_scores, smooth_lbls, weight=loss_weights, reduction="sum")

        loss.backward()
        self.optimizer.step()

        return loss.item()


    def _get_sampler(self, train_type, bs, **kwargs):
        """
        Get sampler object
        
        Parameters:
        -----------
            bs: int
                Batch size
        
        Returns:
        --------
        kgpy.sampling.Sampler
            sampler object
        """
        return kgpy.sampling.One_to_N(
            self.data['train'],
            bs,
            self.data.num_entities, 
            self.data.num_relations, 
            self.device,
            inverse=True,
        )