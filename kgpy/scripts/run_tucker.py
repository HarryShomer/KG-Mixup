import os
import torch
import random 
import argparse
import numpy as np

from kgpy.training import Trainer
from kgpy import models, datasets, sampling


parser = argparse.ArgumentParser(description='KG model and params to run')

parser.add_argument("--dataset", help="Dataset to run it on", default='fb15k_237')
parser.add_argument("--optimizer", help='Optimizer to use when training', type=str, default="Adam")
parser.add_argument("--epochs", help="Number of epochs to run", default=250, type=int)
parser.add_argument("--bs", help="Batch size to use for training", default=128, type=int)
parser.add_argument("--test-batch-size", help="Batch size to use for testing and validation", default=256, type=int)
parser.add_argument("--lr", help="Learning rate to use while training", default=1e-4, type=float)
parser.add_argument("--train-type", help="Type of training method to use", type=str, default="1-N")
parser.add_argument("--inverse", help="Include inverse edges", action='store_true', default=False)
parser.add_argument("--decay", help="Decay function for LR of form C^epoch", type=float, default=None)

parser.add_argument("--label-smooth", help="label smoothing", default=0.1, type=float)
parser.add_argument("--lp", help="LP regularization penalty to add to loss", type=int, default=None)
parser.add_argument("--lp-weights", help="LP regularization weights. Can give one or two.", nargs='+', default=None)
parser.add_argument("--loss", help="Loss function to use.", default="bce")
parser.add_argument("--negative-samples", help="Number of negative samples to using 1-K training", default=100, type=int)
parser.add_argument("--no-filter", action='store_true', default=False)

parser.add_argument("--device", help="Device to run on", type=str, default="cuda")
parser.add_argument("--parallel", help="Whether to train on multiple GPUs in parallel", action='store_true', default=False)
parser.add_argument("--validation", help="Test on validation set every n epochs", type=int, default=3)
parser.add_argument("--early-stop", help="Number of validation scores to wait for an increase before stopping", default=10, type=int)
parser.add_argument("--checkpoint-dir", default=os.path.join(os.path.expanduser("~"), "checkpoints"))
parser.add_argument("--tensorboard", help="Whether to log to tensorboard", action='store_true', default=False)
parser.add_argument("--save-as", help="Model to save model as", default=None, type=str)
parser.add_argument("--save-every", help="Save model every n epochs", default=50, type=int)
parser.add_argument("--seed", help="Random seed", default=42, type=int)

parser.add_argument("--ent-dim", help="Latent dimension of entities", type=int, default=200)
parser.add_argument("--rel-dim", help="Latent dimension of relations", type=int, default=200)
parser.add_argument("--input-drop", help="", default=0.3, type=float)
parser.add_argument("--hid-drop1", help="", default=0.4, type=float)
parser.add_argument("--hid-drop2", help="", default=0.5, type=float)
parser.add_argument("--bias", action='store_true', default=False)



args = parser.parse_args()

DEVICE  = args.device
DATASET = args.dataset.upper()
CHECKPOINT_DIR = "/mnt/home/shomerha/kgpy/checkpoints"

# Randomness!!!
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


data = getattr(datasets, DATASET)(inverse=args.inverse)

model_args = {
    "ent_dim": args.ent_dim,
    "rel_dim": args.rel_dim,
    "input_drop": args.input_drop,
    "hid_drop1": args.hid_drop1,
    "hid_drop2": args.hid_drop2,
    "bias": args.bias,
    "device": DEVICE
}

model = models.TuckER(data.num_entities, data.num_relations, **model_args)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


if args.train_type.lower() == "1-k":
    sampler = sampling.One_to_K(
                data['train'], 
                args.bs, 
                data.num_entities,
                data.num_relations, 
                args.device,
                num_negative=args.negative_samples,
                inverse=True,
                filtered = not args.no_filter
            )
else:
    sampler = sampling.One_to_N(
                data['train'], 
                args.bs, 
                data.num_entities,
                data.num_relations, 
                args.device,
                inverse=True,
            ) 


train_keywords = {
    "validate_every": args.validation, 
    "non_train_batch_size": args.test_batch_size, 
    "early_stopping": args.early_stop, 
    "negative_samples": args.negative_samples,
    "label_smooth": args.label_smooth,
    "save_every": args.save_every,
    "decay": args.decay,
    "sampler": sampler,
}

model_trainer = Trainer(model, optimizer, data, CHECKPOINT_DIR, tensorboard=args.tensorboard, model_name=args.save_as)
model_trainer.fit(args.epochs, args.bs, args.train_type, **train_keywords)
