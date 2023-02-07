import os
import torch
import random
import argparse
import numpy as np

from kgpy import utils
from kgpy import models
from kgpy import datasets
from kgpy.training import Trainer


parser = argparse.ArgumentParser(description='KG model and params to run')
parser.add_argument("--model", help="Model to run")
parser.add_argument("--dataset", help="Dataset to run it on")
parser.add_argument("--optimizer", help='Optimizer to use when training', type=str, default="Adam")
parser.add_argument("--epochs", help="Number of epochs to run", default=250, type=int)
parser.add_argument("--batch-size", help="Batch size to use for training", default=128, type=int)
parser.add_argument("--test-batch-size", help="Batch size to use for testing and validation", default=128, type=int)
parser.add_argument("--lr", help="Learning rate to use while training", default=1e-4, type=float)
parser.add_argument("--train-type", help="Type of training method to use", type=str, default="1-N")
parser.add_argument("--inverse", help="Include inverse edges", action='store_true', default=False)
parser.add_argument("--decay", help="Decay function for LR of form C^epoch", type=float, default=None)

parser.add_argument("--label-smooth", help="label smoothing", default=0, type=float)
parser.add_argument("--lp", help="LP regularization penalty to add to loss", type=int, default=None)
parser.add_argument("--lp-weights", help="LP regularization weights. Can give one or two.", nargs='+', default=None)
parser.add_argument("--dim", help="Latent dimension of entities and relations", type=int, default=None)
parser.add_argument("--loss", help="Loss function to use.", default="bce")
parser.add_argument("--neg-samples", help="Number of negative samples to using 1-K training", default=1, type=int)
parser.add_argument("--neg-filter", help="Only sample true negative", action='store_true', default=False)

parser.add_argument("--margin", help="If ranking is loss a margin can be sepcified", default=None, type=int)
parser.add_argument("--transe-norm", help="Norm used for distance function on TransE", default=2, type=int)

parser.add_argument("--device", help="Device to run on", type=str, default="cuda")
parser.add_argument("--parallel", help="Whether to train on multiple GPUs in parallel", action='store_true', default=False)
parser.add_argument("--validation", help="Test on validation set every n epochs", type=int, default=5)
parser.add_argument("--early-stop", help="Number of validation scores to wait for an increase before stopping", default=10, type=int)
parser.add_argument("--checkpoint-dir", help="Directory to store model checkpoints", default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "checkpoints"))
parser.add_argument("--tensorboard", help="Whether to log to tensorboard", action='store_true', default=False)
parser.add_argument("--log-training-loss", help="Log training loss every n steps", default=25, type=int)
parser.add_argument("--save-every", help="Save model every n epochs", default=50, type=int)
parser.add_argument("--save-as", help="Model to save model as", default=None, type=str)
parser.add_argument("--seed", help="Random Seed", default=None, type=int)
parser.add_argument("--evaluation-method", help="Either 'raw' or 'filtered' metrics", type=str, default="filtered")

parser.add_argument('--rgcn-num-bases',	 dest='rgcn_num_bases',  default=None,   type=int, 	help='Number of basis relation vectors to use in rgcn')
parser.add_argument('--rgcn-num-blocks', dest='rgcn_num_blocks', default=None,   type=int, 	help='Number of block relation vectors to use in rgcn')

args = parser.parse_args()


if args.loss.lower() == "ranking" and args.neg_samples > 1:
    raise NotImplementedError("TODO: Ranking loss with > 1 negative samples")

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)



def run_model(model, optimizer, data):
    """
    Wrapper for training and testing the model

    Parameters:
    ----------
    model: kgpy.models.Model
        pytorch model
    optimizer: torch.optim
        pytorch optimizer
    data: datasets.AllDataset
        AllDataset object

    Returns:
    -------
        None
    """
    train_keywords = {
        "validate_every": args.validation, 
        "non_train_batch_size": args.test_batch_size, 
        "early_stopping": args.early_stop, 
        "negative_samples": args.neg_samples,
        "log_every_n_steps": args.log_training_loss,
        "save_every": args.save_every,
        "eval_method": args.evaluation_method,
        "label_smooth": args.label_smooth,
        "decay": args.decay,
        "neg_filter": args.neg_filter,
    }

    model_trainer = Trainer(model, optimizer, data, args.checkpoint_dir, tensorboard=args.tensorboard, model_name=args.save_as)
    model_trainer.fit(args.epochs, args.batch_size, args.train_type, **train_keywords)



def parse_model_args():
    """
    Parse cmd line args to create the model.

    They are only added when passed (aka not None)

    Returns:
    -------- 
    dict
        Keyword arguments for model 
    """
    model_params = {"device": args.device}

    if args.lp is not None:
        model_params['regularization'] = f"l{args.lp}"

    if isinstance(args.lp_weights, list) and len(args.lp_weights) > 1:
        model_params['reg_weight'] = [float(r) for r in args.lp_weights]
    elif isinstance(args.lp_weights, list) and len(args.lp_weights) == 1:
        model_params['reg_weight'] = float(args.lp_weights[0])

    if args.dim is not None:
        model_params['emb_dim'] = args.dim

    if args.loss is not None:
        model_params['loss_fn'] = args.loss
       
    if args.margin is not None:
        model_params['margin'] = args.margin
    
    if args.model.lower() == "transe":
        model_params['norm'] = args.transe_norm

    return model_params


def get_optimizer(model):
    """
    Get the toch.optim specificed

    Parameters:
    -----------
        model: kgpy.models.Model
            Model

    Returns:
    -------
    torch.optim
        optimizer
    """
    optimizer_name = args.optimizer.lower()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)   
    else:
        raise ValueError(f"Optimizer `{optimizer_name}` is not available. Must be one of ['Adam', 'SGD', 'Adagrad']")

    return optimizer 


def get_model(data):
    """
    Get the model specificed

    Parameters:
    -----------
        data: datasets.AllDataset
            dataset model is running on

    Returns:
    -------
    kgpy.models.Model
        model
    """
    model_name = args.model.lower()
    model_params = parse_model_args()

    # TODO: needs to be case insesitive
    # model = getattr(models, model_name)(data.num_entities, data.num_relations, **model_params)

    if model_name == "transe":
        model = models.TransE(data.num_entities, data.num_relations,  **model_params)
    elif model_name == "distmult":
        model = models.DistMult(data.num_entities, data.num_relations, **model_params)
    elif model_name == "complex":  
        model = models.ComplEx(data.num_entities, data.num_relations, **model_params)
    elif model_name == "rotate":
        model = models.RotatE(data.num_entities, data.num_relations, **model_params)
    elif model_name == "conve":
        model = models.ConvE(data.num_entities, data.num_relations, **model_params)
    elif model_name == "tucker":
        model = models.TuckER(data.num_entities, data.num_relations, **model_params)
    elif model_name == "rgcn":
        edge_index, edge_type = data.get_edge_tensors(device=args.device)
        model = models.RGCN(data.num_entities, data.num_relations, edge_index, edge_type, rgcn_num_bases=args.rgcn_num_bases, rgcn_num_blocks=args.rgcn_num_blocks, device=args.device)
    elif model_name == "compgcn":
        edge_index, edge_type = data.get_edge_tensors(device=args.device)
        model = models.CompGCN(data.num_entities, data.num_relations, edge_index, edge_type, device=args.device)
    else:
        raise ValueError(f"Model `{model_name}` is not available. See kgpy/models for possible models.")

    return model



def main():
    data = getattr(datasets, args.dataset.upper())(inverse=args.inverse)

    model = get_model(data)
    model = utils.DataParallel(model).to(args.device) if args.parallel else model.to(args.device)
    optimizer = get_optimizer(model)

    run_model(model, optimizer, data)


if __name__ == "__main__":
    main()
