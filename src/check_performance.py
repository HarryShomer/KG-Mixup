import os
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import kgpy
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="fb15k_237")
parser.add_argument("--model", help="ConvE or TuckER", type=str, default="conve")
parser.add_argument("--seed", help="Random Seed", default=42, type=int)
parser.add_argument("--device", help="Device to run on", type=str, default="cuda")
parser.add_argument("--run", help="Name of saved model file", type=str, default=None)

parser.add_argument("--split", help="Data split to eval on", type=str, default="test")

parser.add_argument("--plot", action='store_true', default=False)
parser.add_argument("--checkpoint-dir", default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "checkpoints"))
parser.add_argument("--swa", help="SWA model", action='store_true', default=False)

## ConvE params
parser.add_argument("--filters", help="conv filters", type=int, default=32)

## TuckER params
parser.add_argument("--rel-dim", help="Latent dimension of relations", type=int, default=200)
parser.add_argument("--no-bias", action='store_true', default=False)

args = parser.parse_args()



def get_trips_by_cat(data):
    """
    Divide args.split triples by rel-specific degree into:
        - zero = [0, 1) 
        - low  = [1, 10)
        - med  = [10, 50)
        - high = [50, inf)
    """
    ent_rel_deg = calc_ent_rel_degree(data)
    degree_bin_trips = {d: [] for d in [(0, 1), (1, 10), (10, 50), (50, 100000)]}

    for t in tqdm(data[args.split], "Oy vey!"):
        rel_deg = ent_rel_deg[(t[1], t[2])]

        for d in degree_bin_trips:
            if d[0] <= rel_deg < d[1]:
                degree_bin_trips[d].append(t)

    return degree_bin_trips



def degree_stats(data):
    """
    Mean and median degree
    """
    all_deg = []
    ent_rel_deg = calc_ent_rel_degree(data)

    for t in tqdm(data[args.split], "Clipped your wings"):
        rel_deg = ent_rel_deg[(t[1], t[2])]
        all_deg.append(rel_deg)

    print(f"Mean: {np.mean(all_deg):.2f}")
    print(f"Median: {np.median(all_deg):.2f}")
    print(f"Stdev: {np.std(all_deg):.2f}")
    print(f"Min: {np.min(all_deg):.2f}")
    print(f"Max: {np.max(all_deg):.2f}")



def model_perf_by_degree(data):
    """
    Overall performance and by degree bin

    Parameters:
    -----------
        data: kgpy.AllDataset
            data
        
    Returns:
    --------
    None
    """
    model_params = {"rel_dim": args.rel_dim, "bias": not args.no_bias, "filters": args.filters, "device": args.device}
    model = get_saved_model(args.run, data, args.device, args.checkpoint_dir, model_params, swa=args.swa, synthetic=True)  

    if args.swa:
        dataloader = DataLoader(BasicDataset(data['train'], device=args.device), batch_size=128)
        print("Updating batch statistics for the SWA model...")
        torch.optim.swa_utils.update_bn(dataloader, model)
    
    model_eval = kgpy.Evaluation(data[args.split], data, True, bs=1024, device=args.device)
    test_results = model_eval.evaluate(model)
    
    print(f"\nOverall Performance:", flush=True)
    model_eval.print_results(test_results)

    ###############################################
    #
    # Performance by Degree Cat for model
    #
    ###############################################
    degree_by_cat = get_trips_by_cat(data)

    print("\n=======================================\n")
    for deg, deg_trips in degree_by_cat.items():
        model_eval = kgpy.Evaluation(deg_trips, data, True, bs=1024, device=args.device)
        test_results = model_eval.evaluate(model)
        
        print(f"\nPerformance for Degree Cat {deg}", flush=True)
        model_eval.print_results(test_results)
    
    del model



def acc_conf_by_degree(data):
    """
    Get true/neg scores for triples in each degree bin
    """
    model_params = {"rel_dim": args.rel_dim, "bias": not args.no_bias, "filters": args.filters, "device": args.device}
    model = get_saved_model(args.run, data, args.device, args.checkpoint_dir, model_params, swa=args.swa, synthetic=True)  

    if args.swa:
        dataloader = DataLoader(BasicDataset(data['train'], device=args.device), batch_size=128)
        print("Updating batch statistics for the SWA model...")
        torch.optim.swa_utils.update_bn(dataloader, model)


    all_acc, all_conf = [], []
    degree_bin_trips = get_trips_by_cat(data)
    num_trips_in_bin = [len(x) for x in degree_bin_trips.values()]

    for deg, deg_trips in degree_bin_trips.items():
        model_eval = EvaluationBySample(deg_trips, data, True, bs=1024, device=args.device)

        probs, hits10 = model_eval.evaluate(model)
        probs, hits10 = np.array(probs), np.array(hits10)

        acc = np.mean(hits10)
        conf = np.mean(probs)
        all_acc.append(acc) ; all_conf.append(conf)

        print(f"{deg} Bin ==>  Acc: {acc}   |   Conf: {conf}")
    
    total_ece = 0
    for i in range(len(num_trips_in_bin)):
        diff = np.abs(all_acc[i] - all_conf[i])
        bin_weight = num_trips_in_bin[i] / sum(num_trips_in_bin)
        total_ece += bin_weight * diff

    print(f"Total ECE: {total_ece}")
    print(f"Low Degree ECE: {np.abs(all_acc[1] - all_conf[1])}")
    print("\n\n")


        
def main():
    data = getattr(kgpy.datasets, args.dataset.upper())(inverse=True)

    model_perf_by_degree(data)
    acc_conf_by_degree(data)
    degree_stats(data)


if __name__ == "__main__":
    main()
