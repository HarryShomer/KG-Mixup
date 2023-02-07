import os
import gc
import sys
import torch
import warnings
from random import randint
from datetime import datetime


class DataParallel(torch.nn.DataParallel):
    """
    Extend DataParallel class to access model level attributes/methods
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)



def get_time():
    return datetime.strftime(datetime.now(), "%Y-%m-%dT%H%M%S")


def get_mem():
    """
    Print all params and memory usage

    **Used for debugging purposes**
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(obj.__class__.__name__, obj.shape, type(obj), sys.getsizeof(obj.storage()), obj.device)
        except: pass



def save_model(model, optimizer, epoch, data, checkpoint_dir, model_name):
    """
    Save the given model's state
    """
    if not os.path.isdir(os.path.join(checkpoint_dir, data.dataset_name)):
        os.makedirs(os.path.join(checkpoint_dir, data.dataset_name), exist_ok=True)

    # If wrapped in DataParallel object this is how we access the underlying model
    if isinstance(model, DataParallel):
        model_obj = model.module
    else:
        model_obj = model

    torch.save({
        "model_state_dict": model_obj.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        # TODO: Remove 'latent_dim' and replace with two commented lines
        "latent_dim": model_obj.ent_emb_dim,
        # "ent_dim": model_obj.ent_emb_dim,
        # "reldim": model_obj.rel_emb_dim,
        
        "loss_fn": model_obj.loss_fn.__class__.__name__,
        "epoch": epoch,
        "inverse": data.inverse
        }, 
        os.path.join(checkpoint_dir, data.dataset_name, f"{model_name}.tar")
    )


def load_model(model, optimizer, dataset_name, checkpoint_dir, suffix=None):
    """
    Load the saved model
    """
    if suffix is None:
        file_path = os.path.join(checkpoint_dir, dataset_name, f"{model.name}.tar")
    else:
        file_path = os.path.join(checkpoint_dir, dataset_name, f"{model.name}_{suffix}.tar")

    if not os.path.isfile(file_path):
        print(f"The file {file_path} doesn't exist")
        return None, None

    # If wrapped in DataParallel object this is how we access the underlying model
    if isinstance(model, DataParallel):
        model_obj = model.module
    else:
        model_obj = model

    checkpoint = torch.load(file_path)
    model_obj.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model_obj, optimizer


def checkpoint_exists(model_name, dataset_name, checkpoint_dir, epoch=None):
    """
    Check if a given checkpoint was ever saved
    """
    if epoch is None:
        file_path = os.path.join(checkpoint_dir, dataset_name, f"{model_name}.tar")
    else:
        file_path = os.path.join(checkpoint_dir, dataset_name, f"{model_name}_epoch_{epoch}.tar")

    return os.path.isfile(file_path)


def randint_exclude(begin, end, exclude):
    """
    Randint but exclude a list of numbers

    Parameters:
    -----------
        begin: int 
            begin of range
        end: int 
            end of range (exclusive)
        exclude: Sequence 
            numbers to exclude

    Returns:
    --------
    int
        randint not in exclude
    """
    while True:
        x = randint(begin, end-1)

        if x not in exclude:
            return x


def generate_rand_edges(num_edges, num_ents, num_rels, inverse=False):
    """
    Generate `num_edges` random edge. 
    
    When inverse == True we first generate a non-inverse edge and then create the inverse edge.

    Parameters:
    -----------
        num_edges: int
            Number of edges to generate. When inverse == True we cut in half
        num_ents: int
            Number of entities in dataset
        num_rels: int
            Number of relation in dataset
        inverse: bool
            Whether there are inverse edges in dataset
        
    Returns:
    --------
    list
        Random edges of shape (s, r, o)
    """
    rand_edges = []
    num_edges = num_edges // 2 if inverse else num_edges
    num_rels  = num_rels // 2 if inverse else num_rels

    for _ in range(int(num_edges)):
        new_edge = (randint(0, num_ents-1), randint(0, num_rels-1), randint(0, num_ents-1))
        rand_edges.append(new_edge)

        if inverse:
            new_inv_edge = (new_edge[2], new_edge[1] + num_rels, new_edge[0])
            rand_edges.append(new_inv_edge)
    
    return rand_edges
