import os
import sys
import time
import torch
import optuna
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from kgpy import utils
from kgpy import sampling
from kgpy.evaluation import Evaluation


TENSORBOARD_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "runs")



class Trainer:
    """
    Control training of model on a particular dataset
    """
    def __init__(
        self, 
        model, 
        optimizer, 
        data, 
        checkpoint_dir, 
        tensorboard=False,
        model_name=None,
        **kwargs
    ):
        self.data = data 
        self.model = model
        self.inverse = data.inverse
        self.optimizer = optimizer
        self.tensorboard = tensorboard
        self.device = model.device
        self.checkpoint_dir = checkpoint_dir
        self.start_time = utils.get_time()
        self._model_name = model_name

        if tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, model.name, data.dataset_name), flush_secs=3)


    @property
    def model_name(self):
        """
        Either use use-supplied or default to this
        """
        if self._model_name:
            return f"{self._model_name}_{self.start_time}"
        
        return f"{self.model.name}_{self.start_time}"


    def fit(
            self, 
            epochs, 
            train_batch_size, 
            train_method,
            validate_every=5, 
            non_train_batch_size=128, 
            early_stopping=5, 
            save_every=25,
            log_every_n_steps=100,
            negative_samples=1,
            eval_method="filtered",
            label_smooth=0,
            sampler=None,
            decay=None,
            test_model=True,
            optuna_trial=None,
            neg_filter=False
        ):
        """
        Train, validate, and test the model

        Parameters:
        -----------
            epochs: int
                Number of epochs to train for
            train_batch_size: int
                Batch size to use for training
            train_method: str
                1-K or 1-N
            validate_every: int
                Validate every "n" epochs. Defaults to 5
            non_train_batch_size: int 
                Batch size for non-training data. Defaults to 16
            early_stopping: int
                Stop training if the mean rank hasn't improved in last "n" validation scores. Defaults to 5
            save_every: int
                Save model every "n" epochs. Defaults to 25 When None only no epoch specific versions are saved
            log_every_n_steps: int 
                Log training loss to tensorboard every "n" steps. Defaults to 25
            negative_samples: int
                Number of negative samples to generate for each training sample. Defaults to 1
            eval_method: str
                How to evaluate data. Filtered vs raw. Defaults to filtered
            label_smooth: float
                Label smoothing when training
            sampler: kgpy.Sampler
                Sampler object. Use this if provided otherwise create based on `train_method` arg
            decay: float
                LR decay of form of decay^epoch.
            test_model: bool
                Test the model on the test set. Set to false when fine-tuning
            optuna_trial: optuna.trial.Trial
                Optuna trial if passed. Default is None
            filtered: bool
                Whether to filter false negatives

        Returns:
        --------
        float
            Test MRR when test_model=True else Validation MRR
        """
        step = 1
        val_mrr = []
        
        lr_scheduler = self._get_lr_scheduler(decay)
        sampler = self._get_sampler(sampler, train_method, train_batch_size, negative_samples, neg_filter)
        model_eval = Evaluation(self.data['valid'], self.data, self.inverse, eval_method=eval_method, bs=non_train_batch_size, device=self.device)
        
        for epoch in range(1, epochs+1):
            epoch_loss = torch.Tensor([0]).to(self.device)
            
            self.model.train()
            
            for batch in tqdm(sampler, f"Epoch {epoch}"):
                step += 1
                batch_loss = self._train_batch(batch, train_method, label_smooth)
                epoch_loss += batch_loss

                if step % log_every_n_steps == 0 and self.tensorboard:
                    self.writer.add_scalar(f'training_loss', batch_loss.item(), global_step=step)
                
            print(f"Epoch {epoch} loss:", round(epoch_loss.item(), 4))

            if epoch % validate_every == 0:
                val_mrr.append(self._validate_model(model_eval, epoch))

                if optuna_trial is not None:
                    optuna_trial.report(val_mrr[-1], len(val_mrr))
        
                    if optuna_trial.should_prune(): 
                        raise optuna.exceptions.TrialPruned()

                if self._early_stop_criteria(val_mrr, early_stopping):
                    print(f"Validation loss hasn't improved in the last {early_stopping} Valid MRR scores. Stopping training now!", flush=True)
                    break

                # Only save when we know the model performs better
                utils.save_model(self.model, self.optimizer, epoch, self.data, self.checkpoint_dir, self.model_name)

            if save_every is not None and epoch % save_every == 0:
                utils.save_model(self.model, self.optimizer, epoch, self.data, self.checkpoint_dir, f"{self.model_name}_epoch-{epoch}")
            
            sampler.reset()
            if decay: lr_scheduler.step()

        if test_model:
            return self._test_model(eval_method, non_train_batch_size)
        
        return val_mrr[-1]

   
    def _train_batch(self, batch, train_method, label_smooth):
        """
        Train model on single batch

        Parameters:
        -----------
            batch: tuple
                Tuple of head, relations, and tails for each sample in batch
            train_method: str
                Either 1-N or None
            label_smooth: float
                Label smoothing
        
        Returns:
        -------
        float
            batch loss
        """
        self.optimizer.zero_grad()

        if train_method.upper() == "1-K":
            batch_loss = self._train_batch_1_to_k(batch, label_smooth)
        elif train_method.upper() == "1-N":
            batch_loss = self._train_batch_1_to_n(batch, label_smooth)
 
        batch_loss = batch_loss.mean()
        batch_loss.backward()

        self.optimizer.step()

        return batch_loss


    def _train_batch_1_to_k(self, batch, label_smooth, reduction="mean"): 
        """
        Train model on single batch using 1-K training method

        Parameters:
        -----------
            batch: tuple of Tensors
                First tuple is positive samples and the second negative. Each ontains head, relations, and tails.
            label_smooth: float
                Amount of label smoothing to use

        Returns:
        -------
        loss
            batch loss
        """
        pos_trips, neg_ents = batch[0], batch[1]
        pos_head_rel = torch.stack((pos_trips[:, 0], pos_trips[:, 1])).T

        pos_lbls = torch.ones(len(pos_trips)).to(self.device)
        neg_lbls = torch.zeros(neg_ents.numel()).to(self.device)
        all_lbls = torch.cat((pos_lbls, neg_lbls))

        pos_scores = self.model(pos_trips)
        neg_scores = self.model(pos_head_rel, negative_ents=neg_ents)
        all_scores = torch.cat((pos_scores, neg_scores))

        if label_smooth != 0.0:
            all_lbls = (1.0 - label_smooth)*all_lbls + (1.0 / self.data.num_entities)

        return self.model.loss(all_scores=all_scores, all_targets=all_lbls, reduction=reduction, trips=pos_trips)


    def _train_batch_1_to_n(self, batch, label_smooth, reduction="mean"): 
        """
        Train model on single batch

        Parameters:
        -----------
            batch: tuple of tuples
                First tuple is positive samples and the second negative. Each ontains head, relations, and tails.
            label_smooth: float
                Amount of label smoothing to use

        Returns:
        -------
        loss
            batch loss
        """
        if 'bce' not in self.model.loss_fn.__class__.__name__.lower():
            raise ValueError("1-N training can only be used with BCE loss!")

        if not self.inverse:
            trips, lbls, trip_type = batch[0], batch[1], batch[2]

            head_trips = trips[trip_type == "head"]
            tail_trips = trips[trip_type == "tail"]

            head_lbls = lbls[trip_type == "head"]
            tail_lbls = lbls[trip_type == "tail"]

            head_scores = self.model(head_trips, mode="head")
            tail_scores = self.model(tail_trips, mode="tail")
        
            all_scores = torch.flatten(torch.cat((head_scores, tail_scores)))
            all_lbls = torch.flatten(torch.cat((head_lbls, tail_lbls)))
        else:
            trips, all_lbls = batch[0], batch[1]
            all_scores = self.model(trips, mode="tail")

        if label_smooth != 0.0:
            all_lbls = (1.0 - label_smooth)*all_lbls + (1.0 / self.data.num_entities)
        
        return self.model.loss(all_scores=all_scores, all_targets=all_lbls, reduction=reduction)



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
        results = model_eval.evaluate(self.model)

        if self.tensorboard:
            self.writer.add_scalar('Hits@1%' , results['hits@1'], epoch)
            self.writer.add_scalar('Hits@3%' , results['hits@3'], epoch)
            self.writer.add_scalar('Hits@10%', results['hits@10'], epoch)
            self.writer.add_scalar('MR'      , results['mr'], epoch)
            self.writer.add_scalar('MRR'     , results['mrr'], epoch)
        
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
        model_eval = Evaluation(self.data['test'], self.data, self.data.inverse, eval_method=eval_method, bs=bs, device=self.device)
        test_results = model_eval.evaluate(self.model)
        
        print("\nTest Results:", flush=True)
        model_eval.print_results(test_results)

        return test_results['mrr']


    def _early_stop_criteria(self, val_mrrs, patience):
        """
        Criteria for early stopping.

        Must be that the validation hasn't improved in the last `patience` validation steps. This means that if the patience=5, then the last 5
        validation scores must all be less than 6th to last one.

        Example:
        --------
            patience = 3
            val_mrrs = [10, 12, 14, 15, 15, 14, 15]

            The last 3  - [15, 14, 15] are not better than the 4th to last value (15).
        
        Parameters:
        -----------
            val_mrrs: list
                List of validation mrrs for entire training sequence
            patience: int
                Validation steps patience b4 terminating. Score has this many steps to improve
        
        Returns:
        --------
        bool
            True if should stop otherwise False
        """
        if patience is None:
            return False

        # Need at least this many scores to evaluate early stopping        
        if len(val_mrrs) < patience + 1:
            return False
        
        # If max value present multiple times, will return 1st index
        # Also -1 to consider last patience+1 scores!
        return np.argmax(val_mrrs[-patience-1:]) == 0


    def _get_lr_scheduler(self, decay):
        """
        If not None return a lr.optim.lr_scheduler object. Otherwise return None

        Parameters:
        -----------
            decay: float
                Decay to use. Can be None
        
        Returns:
        --------
        lr.optim.lr_scheduler
            Scheduler object or None
        """
        if not decay:
            return None 
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda e: decay ** e)


    def _get_sampler(self, user_sampler, train_method, bs, num_negative=None, filtered=False):
        """
        Retrieve a sampler object for the type of train method

        If user supplied something we use that

        Parameters:
        -----------
            user_sampler: kgpy.Sampling.Sampler or None
                pre-defined sampler or none
            train_method: str
                1-N or 1-K
            bs: int
                batch_size
            num_negative: int
                number onf negative samples. Only applicable for 1-K training
            filtered: bool
                Filter false negatives
        
        Returns:
        --------
        kgpy.Sampling.Sampler
            sampler for training
        """
        if isinstance(user_sampler, sampling.Sampler):
            return user_sampler

        train_method = train_method.upper()

        if train_method == "1-K":
            sampler = sampling.One_to_K(
                        self.data['train'], 
                        bs, 
                        self.data.num_entities,
                        self.data.num_relations, 
                        self.device,
                        num_negative=num_negative,
                        inverse=self.data.inverse,
                        filtered=filtered
                    )
        elif train_method == "1-N":
            sampler = sampling.One_to_N(
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

