import os
import torch
import random 
import argparse
import numpy as np

from kgpy import models, datasets, Evaluation, Trainer, sampling


# Randomness!!!
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


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
parser.add_argument("--validation", help="Test on validation set every n epochs", type=int, default=5)
parser.add_argument("--early-stop", help="Number of validation scores to wait for an increase before stopping", default=8, type=int)
parser.add_argument("--checkpoint-dir", default=os.path.join(os.path.expanduser("~"), "checkpoints"))
parser.add_argument("--tensorboard", help="Whether to log to tensorboard", action='store_true', default=False)
parser.add_argument("--save-as", help="Model to save model as", default=None, type=str)
parser.add_argument("--save-every", help="Save model every n epochs", default=50, type=int)

parser.add_argument("--emb-dim", help="Latent dimension of embeddings", type=int, default=200)
parser.add_argument("--input-drop", help="", default=0.2, type=float)
parser.add_argument("--feat-drop", help="", default=0.2, type=float)
parser.add_argument("--hid-drop", help="", default=0.5, type=float)
parser.add_argument("--filters", help="conv filters", type=int, default=32)

## Eval params
parser.add_argument("--test", help="Whether to just eval on test set", action='store_true', default=False)
parser.add_argument("--run", help="Checkpoint file when set test flag", type=str)

args = parser.parse_args()

DEVICE  = args.device
DATASET = args.dataset.upper()
CHECKPOINT_DIR = args.checkpoint_dir


def get_model(data):
    """
    Create model instance
    """
    model_args = {
        "emb_dim": args.emb_dim,
        "input_drop": args.input_drop,
        "feat_drop": args.feat_drop,
        "hidden_drop": args.hid_drop,
        "filters": args.filters,
        "device": DEVICE
    }

    model = models.ConvE(data.num_entities, data.num_relations, **model_args)
    model = model.to(DEVICE)

    return model


def run_train(model, data):
    """
    Train model
    """
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_trainer = Trainer(model, optimizer, data, CHECKPOINT_DIR, tensorboard=args.tensorboard, model_name=args.save_as)
    model_trainer.fit(args.epochs, args.bs, args.train_type, **train_keywords)


def run_test(model, data):
    """
    Eval saved model on test set
    """
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, data.dataset_name.replace("_", "-"), f"{args.run}.tar"),  map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)

    model_eval = Evaluation(data['train'], data, data.inverse, bs=1024, device=args.device)
    test_results = model_eval.evaluate(model)
    
    print("\nTest Results:", flush=True)
    model_eval.print_results(test_results)


def main():
    data = getattr(datasets, DATASET)(inverse=args.inverse)
    model = get_model(data)

    if not args.test:
        run_train(model, data)
    else:
        run_test(model, data)


if __name__ == "__main__":
    main()
