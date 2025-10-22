#!/usr/bin/env python3
"""
UMAP + AgglomerativeClustering (distance_threshold sweep) optimized for AMI.

Features:
- Loads embeddings and image_paths from .npz
- Extracts ground-truth labels from filename token before .jpg (last numeric token)
- Reduces embeddings with UMAP
- Samples a subset and computes hierarchical linkage distances to choose meaningful tau range
- Sweeps taus, computes AMI, saves CSV and per-tau .npz files
- Optionally: only run sweep on the sample (fast), then run best tau on full dataset

Usage example:
python3 umap_ac_ami_search.py \
  --embedding_file ../no_fish_embeddings/resnet_embeddings_train_no_fish.npz \
  --output_dir ../umap_ac_clusters/ \
  --n_components 50 \
  --sample_size 2000 \
  --n_steps 20 \
  --evaluate_on_sample_only
"""
import argparse
import os
import time
import re
import csv
from pathlib import Path

import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
import umap
from scipy.cluster import hierarchy
   from collections import defaultdict


def extract_labels_from_paths(image_paths):
    """
    Extract numeric label from filename: assumes last numeric token before .jpg is the label.
    Examples:
      - img_123_5.jpg -> 5
      - some/path/abc-3.jpg -> 3
    """
    labels = []
    for p in image_paths:
        fname = os.path.basename(str(p))
        # find last group of digits before file extension
        m = re.findall(r'(\d+)(?=(?:\.[^.]+$))|(\d+)(?=[^0-9]*$)', fname)
        # above pattern returns tuples; simpler fallback:
        if not m:
            # fallback: find all digits in name and take last
            digs = re.findall(r'(\d+)', fname)
            if not digs:
                raise ValueError(f"Couldn't extract numeric label from filename: {fname}")
            label = int(digs[-1])
        else:
            # m may contain tuples, flatten to get the last matched group
            flat = [g for tup in m for g in tup if g]
            if not flat:
                digs = re.findall(r'(\d+)', fname)
                label = int(digs[-1])
            else:
                label = int(flat[-1])
        labels.append(label)
    return np.array(labels, dtype=int)


def compute_linkage_distances(sample_emb, method='ward'):
    """
    Compute hierarchical linkage on the sample and return merge distances array.
    This returns the third column of the linkage matrix (the distances).
    """
    # hierarchical linkage expects 2D array of samples x features
    Z = hierarchy.linkage(sample_emb, method=method)
    # column 2 is distance for each merge
    distances = Z[:, 2]
    return distances


def save_npz_for_tau(out_dir, tau, reduced_embeddings, image_paths, cluster_labels):
    fname = os.path.join(out_dir, f"clusters_tau{tau:.6f}.npz")
    np.savez_compressed(fname,
                        reduced_embeddings=reduced_embeddings,
                        image_paths=image_paths,
                        cluster_labels=cluster_labels,
                        tau=tau)
    return fname


def main():
    parser = argparse.ArgumentParser(description="UMAP + AgglomerativeClustering AMI sweep using distance_threshold.")
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to .npz file (embeddings, image_paths)')
    parser.add_argument('--output_dir', type=str, default='ac_results', help='Directory to store results')
    parser.add_argument('--n_components', type=int, default=50, help='UMAP output dimensionality')
    parser.add_argument('--n_neighbors', type=int, default=30, help='UMAP n_neighbors')
    parser.add_argument('--min_dist', type=float, default=0.0, help='UMAP min_dist')
    parser.add_argument('--sample_size', type=int, default=2000,
                        help='Number of samples used to estimate linkage distances (for tau range). '
                             'Increase for more accurate range, decrease for speed.')
    parser.add_argument('--tau_min_percentile', type=float, default=5.0,
                        help='Lower percentile (on sample linkage distances) to use as tau_min')
    parser.add_argument('--tau_max_percentile', type=float, default=95.0,
                        help='Upper percentile (on sample linkage distances) to use as tau_max')
    parser.add_argument('--n_steps', type=int, default=20, help='Number of tau values to test between min and max')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for UMAP and sampling')
    parser.add_argument('--evaluate_on_sample_only', action='store_true',
                        help='If set: evaluate tau sweep only on the sample (fast). '
                             'You can then run best tau on full dataset separately.')
    parser.add_argument('--full_run_after_best', action='store_true',
                        help='If set and evaluate_on_sample_only was used, run best tau on full dataset and save final .npz')
    parser.add_argument('--linkage', type=str, default='ward', choices=['ward', 'complete', 'average', 'single'],
                        help='Linkage criterion for Agglomerative clustering and linkage computation.')
    parser.add_argument('--umap_verbose', action='store_true', help='Show UMAP progress')
    args = parser.parse_args()

    # ---- prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'ami_scores.csv')

    # ---- load
    data = np.load(args.embedding_file, allow_pickle=True)
    if 'embeddings' not in data or 'image_paths' not in data:
        raise ValueError("Input .npz must contain 'embeddings' and 'image_paths' arrays.")
    embeddings = data['embeddings']
    image_paths = data['image_paths']

    n_total = embeddings.shape[0]
    print(f"Loaded {n_total} embeddings, dim={embeddings.shape[1]}")

    # ---- extract labels
    labels_true = extract_labels_from_paths(image_paths)
    print(f"Extracted {len(np.unique(labels_true))} unique ground-truth labels (range {labels_true.min()}..{labels_true.max()})")

    # ---- UMAP reduction
    print(f"Running UMAP: n_components={args.n_components}, n_neighbors={args.n_neighbors}")
    reducer = umap.UMAP(n_neighbors=args.n_neighbors,
                        n_components=args.n_components,
                        min_dist=args.min_dist,
                        metric='euclidean',
                        random_state=args.random_state,
                        verbose=args.umap_verbose)
    t0 = time.time()
    reduced = reducer.fit_transform(embeddings)
    t_umap = time.time() - t0
    print(f"UMAP done in {t_umap:.1f}s. Reduced shape: {reduced.shape}")

    # ---- sample for linkage distances


def stratified_sample(labels, sample_size, random_state=42):
    rng = np.random.RandomState(random_state)
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)

    # target per class
    per_class = sample_size // n_classes

    indices = []
    for cls in unique:
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) <= per_class:
            chosen = cls_idx  # take all if fewer
        else:
            chosen = rng.choice(cls_idx, per_class, replace=False)
        indices.extend(chosen)

    return np.array(indices, dtype=int)

    print(f"Computing hierarchical linkage on sample of size {sample_size} (method={args.linkage}) to estimate meaningful tau range.")
    t0 = time.time()
    # compute linkage distances on the sample
    distances = compute_linkage_distances(sample_emb, method=args.linkage)
    t_link = time.time() - t0
    print(f"Linkage on sample took {t_link:.1f}s. Number of merges: {len(distances)}")

    # use percentiles of distances to select meaningful tau range
    tau_min = float(np.percentile(distances, args.tau_min_percentile))
    tau_max = float(np.percentile(distances, args.tau_max_percentile))
    if tau_min == tau_max:
        # fallback: expand a bit around median
        med = float(np.median(distances))
        tau_min, tau_max = med * 0.8, med * 1.2
        print("Warning: tau_min equals tau_max from percentiles; using expanded median range.")

    print(f"Using tau range from sample distances: [{tau_min:.6f}, {tau_max:.6f}] (percentiles {args.tau_min_percentile}..{args.tau_max_percentile})")

    tau_values = np.linspace(tau_min, tau_max, args.n_steps)

    # ---- CSV header
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['tau', 'n_clusters', 'AMI', 'time_seconds', 'on_sample_only'])

    best = {'ami': -1.0, 'tau': None, 'n_clusters': None, 'time': None, 'file': None}

    # helper to run clustering and score
    def eval_and_save(tau, emb, image_paths_local, labels_local, out_prefix, on_sample_only=False):
        t0 = time.time()
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=tau, linkage=args.linkage)
        pred = model.fit_predict(emb)
        t_elapsed = time.time() - t0
        n_clusters = len(set(pred))
        ami = adjusted_mutual_info_score(labels_local, pred)
        out_file = save_npz_for_tau(out_prefix, tau, emb, image_paths_local, pred)
        return ami, n_clusters, t_elapsed, out_file, pred

    # ---- evaluate taus
    for i, tau in enumerate(tau_values):
        print(f"[{i+1}/{len(tau_values)}] Testing tau={tau:.6f} ...")
        if args.evaluate_on_sample_only:
            # evaluate on sample only (fast)
            emb_to_use = sample_emb
            img_paths_local = image_paths[sample_idx]
            labels_local = labels_true[sample_idx]
            on_sample_only = True
        else:
            emb_to_use = reduced
            img_paths_local = image_paths
            labels_local = labels_true
            on_sample_only = False

        try:
            ami, n_clusters, elapsed, saved_file, pred = eval_and_save(tau, emb_to_use, img_paths_local, labels_local, args.output_dir, on_sample_only=on_sample_only)
        except MemoryError as me:
            print(f"MemoryError at tau={tau}: {me}. Skipping this tau.")
            ami, n_clusters, elapsed, saved_file = -1.0, None, None, None
        except Exception as e:
            print(f"Error at tau={tau}: {e}. Skipping this tau.")
            ami, n_clusters, elapsed, saved_file = -1.0, None, None, None

        # append CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([tau, n_clusters, ami, elapsed, args.evaluate_on_sample_only])

        print(f" -> clusters={n_clusters}, AMI={ami:.4f}, time={elapsed:.1f}s, saved={saved_file}")

        if ami > best['ami']:
            best.update({'ami': ami, 'tau': tau, 'n_clusters': n_clusters, 'time': elapsed, 'file': saved_file})

    print("Sweep finished.")
    if best['tau'] is not None:
        print(f"Best tau: {best['tau']:.6f} with AMI={best['ami']:.4f} ({best['n_clusters']} clusters), file: {best['file']}")

    # ---- Optionally run best tau on full dataset if sampling-only was used
    if args.evaluate_on_sample_only and args.full_run_after_best and best['tau'] is not None:
        print(f"Running best tau={best['tau']:.6f} on FULL reduced dataset ({n_total} samples). This may take a long time...")
        try:
            ami_full, n_clusters_full, elapsed_full, saved_full, pred_full = eval_and_save(best['tau'], reduced, image_paths, labels_true, args.output_dir, on_sample_only=False)
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([best['tau'], n_clusters_full, ami_full, elapsed_full, False])
            print(f"Full run done: AMI={ami_full:.4f}, clusters={n_clusters_full}, time={elapsed_full:.1f}s, saved={saved_full}")
        except Exception as e:
            print(f"Failed to run best tau on full data: {e}")

    print(f"All results (CSV + per-tau .npz) saved to {args.output_dir}")


if __name__ == '__main__':
    main()
