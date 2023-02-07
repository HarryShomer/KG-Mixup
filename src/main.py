import os
import torch
import random
import argparse
import numpy as np

import kgpy
from models import ConvE_Synthetic, TuckER_Synthetic

from synth_generator import SyntheticGenerator
from training import MixupTrainer, LossTrainer, OversampleTrainer, BaseTrainer


parser = argparse.ArgumentParser(description='KG model and params to run')
parser.add_argument("--dataset", help="Dataset to run it on")
parser.add_argument("--model", help="ConvE or TuckER", type=str, default="Conve")
parser.add_argument("--strategy", help="Either 'mixup', 'focal', 'oversample', 'reweight', or 'None'", type=str, default="mixup")

parser.add_argument("--epochs", help="Number of epochs to run", default=400, type=int)
parser.add_argument("--bs", help="Batch size to use for training", default=128, type=int)
parser.add_argument("--test-batch-size", help="Batch size to use for training", default=128, type=int)
parser.add_argument("--lr", help="Learning rate to use while training", default=1e-4, type=float)
parser.add_argument("--inverse", help="Include inverse edges", action='store_true', default=True)
parser.add_argument("--label-smooth", help="label smoothing", default=0, type=float)
parser.add_argument("--decay", help="Decay function for LR of form C^epoch", type=float, default=None)
parser.add_argument("--neg-samples", help="Number of negative samples to using 1-K training", default=100, type=int)
parser.add_argument("--train-type", help="Train sampling strategy", type=str, default="1-K")

parser.add_argument("--run-pretrain", help="pretrained model to init with")
parser.add_argument("--device", help="Device to run on", type=str, default="cuda")
parser.add_argument("--validation", help="Test on validation set every n epochs", type=int, default=5)
parser.add_argument("--early-stop", help="Number of validation scores to wait for an increase before stopping", default=8, type=int)
parser.add_argument("--checkpoint-dir", help="Directory to store model checkpoints", default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "checkpoints"))
parser.add_argument("--save-every", help="Save model every n epochs", default=50, type=int)
parser.add_argument("--save-as", help="Model to save model as", default=None, type=str)
parser.add_argument("--seed", help="Random Seed", default=1, type=int)

parser.add_argument("--alpha", help="For beta distribution", type=float, default=1)

parser.add_argument("--threshold", help="Degree cutoff to decide if need augmenting", type=int, default=5)
parser.add_argument("--synth-weight", help="Weight for synth loss", type=float, default=1)
parser.add_argument("--max-generate", help="Max number of samples to augment", type=int, default=10)
parser.add_argument("--swa", help="Stochastic Weight Averaging", action='store_true', default=False)
parser.add_argument("--reweight-up", help="Weight to add for low samples in 'reweight' strategy", type=float, default=1)

parser.add_argument("--swa-lr", help="", type=float, default=5e-4)
parser.add_argument("--swa-anneal", help="", type=int, default=10)
parser.add_argument("--swa-start", help="", type=int, default=10)
parser.add_argument("--swa-every", help="", type=int, default=20)

# Same for conve and tucker
parser.add_argument("--input-drop", help="", default=0.2, type=float)

## ConvE params
parser.add_argument("--feat-drop", help="", default=0.2, type=float)
parser.add_argument("--hid-drop", help="", default=0.5, type=float)
parser.add_argument("--filters", help="conv filters", type=int, default=32)

## TuckER params
parser.add_argument("--hid-drop1", help="", default=0.4, type=float)
parser.add_argument("--hid-drop2", help="", default=0.5, type=float)
parser.add_argument("--rel-dim", help="Latent dimension of relations", type=int, default=200)
parser.add_argument("--no-bias", action='store_true', default=False)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)



def get_model(data):
    """
    Create and return the model. Use pre-train weights if specified
    """
    model_name = args.model.lower()

    if model_name == "conve":
        model_type = ConvE_Synthetic
        model_params = {"filters": args.filters, "input_drop": args.input_drop, "feat_drop": args.feat_drop, 
                        "hidden_drop": args.hid_drop, "device": args.device}

    elif model_name == "tucker":
        model_type = TuckER_Synthetic
        model_params = {"rel_dim": args.rel_dim, "bias": not args.no_bias, "input_drop": args.input_drop, 
                        "hid_drop1": args.hid_drop1, "hid_drop2": args.hid_drop2, "device": args.device}
    else:
        raise ValueError(f"No model with name {model_name}")

    model = model_type(data.num_entities, data.num_relations, **model_params)
    model = model.to(args.device)

    if args.run_pretrain is not None:
        if model_name == "conve":
            pre_trained_model = ConvE_Synthetic(data.num_entities, data.num_relations, filters=args.filters, device=args.device)
        else:
            pre_trained_model = TuckER_Synthetic(data.num_entities, data.num_relations, rel_dim=args.rel_dim, bias=not args.no_bias, device=args.device)

        pre_trained_model.to(args.device)

        checkpoint = torch.load(os.path.join(args.checkpoint_dir, data.dataset_name, f"{args.run_pretrain}.tar"),  map_location=args.device)
        pre_trained_model.load_state_dict(checkpoint['model_state_dict'])
                
        model.ent_embs = pre_trained_model.ent_embs
        model.rel_embs = pre_trained_model.rel_embs

    return model
 

def run_model(model, optimizer, data):
    """
    Wrapper for training and testing the model

    Parameters:
    ----------
        model: kgpy.models.*
            pytorch model
        optimizer: torch.optim
            pytorch optimizer
        data: datasets.AllDataset
            AllDataset object

    Returns:
    -------
        None
    """
    train_strategy = args.strategy.lower()

    train_init_keywords = {
        'model_name': args.save_as,
        "synth_weight": args.synth_weight,
        "threshold": args.threshold,
        "swa": args.swa,
        "swa_lr": args.swa_lr,
        "swa_anneal": args.swa_anneal,
        "swa_start": args.swa_start,
        "swa_every": args.swa_every,
        "strategy_name": train_strategy.lower(),
        "reweight_up": args.reweight_up
    }
    train_fit_keywords = {
        "validate_every": args.validation, 
        "non_train_batch_size": args.test_batch_size, 
        "early_stopping": args.early_stop, 
        "save_every": args.save_every,
        "label_smooth": args.label_smooth,
        "negative_samples": args.neg_samples,
        "decay": args.decay,
        "bs": args.bs,
        "train_type": args.train_type
    }    

    synth_gen = SyntheticGenerator(data, args.alpha, args.threshold, args.max_generate, device=args.device, dims=[model.ent_emb_dim, model.rel_emb_dim])

    if train_strategy in ['reweight', 'focal']:
        trainer_type = LossTrainer
    elif train_strategy == "oversample":
        trainer_type = OversampleTrainer
    elif train_strategy == "mixup":
        trainer_type = MixupTrainer
    elif train_strategy == "none":
        trainer_type = BaseTrainer
    else:
        raise ValueError("Training strategy must be one of ['mixup', 'focal', 'oversample', 'reweight', or 'none']")

    model_trainer = trainer_type(model, optimizer, data, args.checkpoint_dir, **train_init_keywords)
    model_trainer.fit(args.epochs, synth_gen, **train_fit_keywords)



def main():
    data = getattr(kgpy.datasets, args.dataset.upper())(inverse=args.inverse)

    model = get_model(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_model(model, optimizer, data)
    

if __name__ == "__main__":
    main()
