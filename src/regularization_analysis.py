import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import kgpy
from utils import *


def get_trips_below_degree(data, threshold):
    """
    Get all trips < tail-relation degree of threshold
    """
    all_trips = []
    ent_rel_deg = calc_ent_rel_degree(data)

    for t in tqdm(data["train"], "Runnning..."):
        rel_deg = ent_rel_deg[(t[1], t[2])]

        if rel_deg < threshold:
            all_trips.append(t)

    return all_trips


def get_trips_by_tail(data):
    """
    Map tail ->  all (*, *, t)
    """
    tail_2_trips = defaultdict(list)

    for t in tqdm(data["train"], "Tail to trips"):
        tail_2_trips[t[2]].append(t)
    
    return tail_2_trips



def compare_embs(ent_embs, rel_embs, data, threshold):
    """
    """
    tail_2_trips = get_trips_by_tail(data)
    low_trips = get_trips_below_degree(data, threshold)

    all_rel_dist, all_head_dist = [], []

    # Only look at low degree trips
    for trip in tqdm(low_trips, "Comparing Embs"):
        rel_emb = rel_embs[trip[1]]
        head_emb = ent_embs[trip[0]]
        tail_trips = tail_2_trips.get(trip[2], [])

        # Greater than 1 bec. itself will always be in there
        if len(tail_trips) > 1:
            # Exclude own head and rel
            rel_ix = [x[1] for x in tail_trips if x[1] != trip[1]]
            head_ix = [x[0] for x in tail_trips if x[0] != trip[0]]
            
            other_rel_embs = ent_embs[rel_ix]
            other_head_embs = ent_embs[head_ix]

            # Compare. Want dist to each point individually
            if len(rel_ix) > 0:
                rel_dist = np.linalg.norm(rel_emb - other_rel_embs, axis=1)
                all_rel_dist.append(np.mean(rel_dist))

            if len(head_ix) > 0:
                head_dist = np.linalg.norm(head_emb - other_head_embs, axis=1)
                all_head_dist.append(np.mean(head_dist))

    
    print(f"\nMean Head Distance: {np.mean(all_head_dist):.2f}")
    print(f"Mean Rel Distance: {np.mean(all_rel_dist):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fb15k_237")
    parser.add_argument("--model", help="ConvE or TuckER", type=str, default="conve")
    parser.add_argument("--device", help="Device to run on", type=str, default="cpu")
    parser.add_argument("--threshold", help="Degree threshold", type=int, default=5)
    parser.add_argument("--run", help="Name of saved model file", type=str, default=None)
    parser.add_argument("--checkpoint-dir", default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "checkpoints"))
    parser.add_argument("--filters", help="conv filters", type=int, default=32)
    parser.add_argument("--rel-dim", help="Latent dimension of relations", type=int, default=200)
    parser.add_argument("--no-bias", action='store_true', default=False)
    args = parser.parse_args()

    data = getattr(kgpy.datasets, args.dataset.upper())(inverse=True)
    
    model_params = {"rel_dim": args.rel_dim, "bias": not args.no_bias, "filters": args.filters, "device": args.device}
    model = get_saved_model(args.run, data, args.device, args.checkpoint_dir, model_params, synthetic=True)  

    # Easier with numpy
    ent_embs = model.ent_embs.weight.detach().numpy()
    rel_embs = model.rel_embs.weight.detach().numpy()

    compare_embs(ent_embs, rel_embs, data, args.threshold)


if __name__ == "__main__":
    main()
