import os
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from sklearn.metrics import roc_curve, auc
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import pickle
import argparse
import json
import math
from numpy.lib.stride_tricks import sliding_window_view

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

# Local imports
from src.utils.plotting import plot_similarity_distributions, plot_images
from src.data.data_new import CDPImageDataset
from src.utils.default_datasets import image_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_to_square_array(lst):
    """
    Reshape a flat list of length n*n into an n×n NumPy array.

    Parameters
    ----------
    lst : list or 1D array
        Input sequence of length n*n.

    Returns
    -------
    arr : ndarray of shape (n, n)
        Reshaped square array.
    """
    length = len(lst)
    n = int(math.isqrt(length))  # integer square root
    if n * n != length:
        raise ValueError(f"Length {length} is not a perfect square.")
    return np.array(lst).reshape(n, n)

def combined_block_scores(S, n, modifier=None):
    """
    Average every contiguous n×n patch in S (no padding).
    Each output entry corresponds to a (128·n)×(128·n) region.

    Parameters
    ----------
    S : array_like of shape (M, M)
        Square matrix of per-128×128 scores.
    n : int
        Window size in blocks.

    Returns
    -------
    A : ndarray of shape (M-n+1, M-n+1)
        A[i, j] is the mean of S[i:i+n, j:j+n].
        Use A.ravel().tolist() if you need a flat list.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square 2D array.")
    M = S.shape[0]
    if not (1 <= n <= M):
        raise ValueError(f"n must be in [1, {M}].")

    windows = sliding_window_view(S, (n, n))        # (M-n+1, M-n+1, n, n)
    if modifier is not None:
        if modifier == "mean":
            #print("Using mean modifier")
            return windows.mean(axis=(-1, -2))
        elif modifier == "max":
            #print("Using max modifier")
            return windows.max(axis=(-1, -2))
        elif modifier == "min":
            #print("Using min modifier")
            return windows.min(axis=(-1, -2))
        elif modifier == "median":
            #print("Using median modifier")
            return np.median(windows, axis=(-1, -2))
        elif modifier == "50percentlowest_average":
            # In each window select 50% lowest scores 
            # and return their average
            # Step 1: flatten the last two dimensions
            flattened_windows = windows.reshape(windows.shape[:-2] + (-1,))
            # Step 2: sort the flattened windows
            sorted_windows = np.sort(flattened_windows, axis=-1)
            if n % 2 == 0:
                half_n = n*n // 2
            else:
                half_n = (n*n // 2) + 1
            # Step 3: take the first half_n elements
            lowest_half = sorted_windows[..., :half_n]
            # Step 4: calculate the mean
            return lowest_half.mean(axis=-1)
            
    return windows.mean(axis=(-1, -2))              # no padding used


def get_default_metrics():
    """Default metrics for image similarity evaluation."""
    return {
        "MSE": lambda x, y: np.mean((x - y) ** 2),
        "SSIM": lambda x, y: -1 * ssim(x, y, data_range=1.0),
        # "SSIM_Roman": lambda x, y: -1 * SSIM(reduction="none", sigma=2.5, data_range=(0, 1))(
        #     torch.tensor(x).unsqueeze(0).unsqueeze(0), 
        #     torch.tensor(y).unsqueeze(0).unsqueeze(0)
        # ).item(),
        "PCC": lambda x, y: -1 * np.corrcoef(x.flatten(), y.flatten())[0, 1]
    }

def save_results_as_text(results, output_path, reference_dataset, mode, test_uid_range=None, dedicated_class=None):
    """
    Save evaluation results to a formatted text file.
    
    Args:
        results (dict): Results dictionary containing performance metrics.
        output_path (str): Full path where the text file should be saved.
        reference_dataset (str): Name of the reference dataset used.
        mode (str): Comparison mode used for evaluation.
        test_uid_range (list, optional): UID range if filtering was applied.
        dedicated_class (str, optional): Dedicated class information from metadata.
    """
    with open(output_path, 'w') as f:
        f.write(f"Reference Authentication Performance using {reference_dataset} (Mode: {mode})\n")
        if test_uid_range:
            f.write(f"UID Range: {test_uid_range}\n")
        if dedicated_class:
            f.write(f"Dedicated Class: {dedicated_class}\n")
        f.write("="*80 + "\n\n")
        
        # Write summary by dataset pair
        f.write("Performance by Dataset Pair:\n")
        f.write("-"*40 + "\n")
        for pair, metrics_results in results["results"].items():
            f.write(f"{pair}:\n")
            for metric_name, metric_data in metrics_results.items():
                f.write(f"  {metric_name}:\n")
                f.write(f"    AUC = {metric_data['auc']:.4f}\n")
                f.write(f"    Mean Ref vs Orig = {metric_data['mean_ref_vs_orig']:.4f} ± {metric_data['std_ref_vs_orig']:.4f}\n")
                f.write(f"    Mean Ref vs Fake = {metric_data['mean_ref_vs_fake']:.4f} ± {metric_data['std_ref_vs_fake']:.4f}\n")
            f.write("\n")
        
        # Write summary by metric
        f.write("Performance by Metric:\n")
        f.write("-"*40 + "\n")
        
        # Get all available metrics from the first result pair
        if results["results"]:
            first_pair = next(iter(results["results"].values()))
            available_metrics = first_pair.keys()
        else:
            available_metrics = []
            
        for metric_name in available_metrics:
            f.write(f"{metric_name}:\n")
            
            # Collect AUC scores
            auc_scores = [pair_results[metric_name]["auc"] 
                         for pair_results in results["results"].values() 
                         if metric_name in pair_results]
            
            # Collect mean scores for reference vs original and reference vs fake
            mean_ref_orig_scores = [pair_results[metric_name]["mean_ref_vs_orig"] 
                                   for pair_results in results["results"].values() 
                                   if metric_name in pair_results]
            mean_ref_fake_scores = [pair_results[metric_name]["mean_ref_vs_fake"] 
                                   for pair_results in results["results"].values() 
                                   if metric_name in pair_results]
            
            # Write AUC statistics
            f.write(f"  AUC Statistics:\n")
            f.write(f"    Mean: {np.nanmean(auc_scores):.4f}\n")
            f.write(f"    Min:  {np.nanmin(auc_scores):.4f}\n")
            f.write(f"    Max:  {np.nanmax(auc_scores):.4f}\n")
            f.write(f"    Std:  {np.nanstd(auc_scores):.4f}\n")
            
            # Write metric value statistics
            f.write(f"  Metric Value Statistics:\n")
            f.write(f"    Mean Ref vs Orig: {np.nanmean(mean_ref_orig_scores):.4f} ± {np.nanstd(mean_ref_orig_scores):.4f}\n")
            f.write(f"    Mean Ref vs Fake: {np.nanmean(mean_ref_fake_scores):.4f} ± {np.nanstd(mean_ref_fake_scores):.4f}\n")
            f.write("\n")

def get_score(reference, probe, metric_fn, mode="average_blocks", ensemble_mode=None):
    """
    Calculate the similarity score between reference and probe images using the specified metric function.
    
    Args:
        reference (CDPImage) or np.array(CDPImage) for ensembling:  Reference image object.
        probe (CDPImage): Probe image object.
        metric_fn (callable): Metric function to compute similarity.
        
    Returns:
        list[float]: One or more similarity scores depending on the mode.
    """
    
    if ensemble_mode is None:
        if mode == "average_blocks" or mode == "blocks_separately" or mode.split("-")[0]=="blocks_combine":
            # For block-based comparison, average the similarity scores across all blocks
            ref_blocks = reference.get_all_blocks()
            probe_blocks = probe.get_all_blocks()
            
            if len(ref_blocks) != len(probe_blocks):
                raise ValueError(f"Reference {len(ref_blocks)} and probe {len(probe_blocks)} images have different number of blocks")

            scores = [metric_fn(ref_block, probe_block) for ref_block, probe_block in zip(ref_blocks, probe_blocks)]
            if mode == "average_blocks":
                return [np.mean(scores)]
            elif mode.split("-")[0]=="blocks_combine":
                if len(mode.split("-"))==2:
                    return combined_block_scores(list_to_square_array(scores), n=int(mode.split("-")[1])).ravel().tolist()
                else:
                    return combined_block_scores(list_to_square_array(scores), n=int(mode.split("-")[1]), modifier=mode.split("-")[2]).ravel().tolist()
            else:
                return scores
        if mode == "full_images":
            # For full image comparison, use the metric directly on the images
            ref_image = reference.get_image()
            probe_image = probe.get_image()
            return metric_fn(ref_image, probe_image)
    
    elif ensemble_mode == "average_reference":
        if mode == "average_blocks" or mode == "blocks_separately" or mode.split("-")[0]=="blocks_combine":
            # For block-based ensemble, average the similarity scores across all references
            ensemble_ref_blocks = [one_ref.get_all_blocks() for one_ref in reference]
            ref_blocks = np.mean(ensemble_ref_blocks, axis=0)
            probe_blocks = probe.get_all_blocks()
            if len(ref_blocks) != len(probe_blocks):
                raise ValueError("Reference and probe images have different number of blocks")
            scores = [metric_fn(ref_block, probe_block) for ref_block, probe_block in zip(ref_blocks, probe_blocks)]
            if mode == "average_blocks":
                return [np.mean(scores)]
            elif mode.split("-")[0]=="blocks_combine":
                if len(mode.split("-"))==2:
                    return combined_block_scores(list_to_square_array(scores), n=int(mode.split("-")[1])).ravel().tolist()
                else:
                    return combined_block_scores(list_to_square_array(scores), n=int(mode.split("-")[1]), modifier=mode.split("-")[2]).ravel().tolist()
            else:
                return scores
        if mode == "full_images":
            # For full image ensemble, average the reference images and compare to probe
            ref_images = [one_ref.get_image() for one_ref in reference]
            avg_ref_image = np.mean(ref_images, axis=0)
            probe_image = probe.get_image()
            return metric_fn(avg_ref_image, probe_image)

    elif ensemble_mode == "per_pixel_closest_reference":
        if mode == "average_blocks" or mode == "blocks_separately" or mode.split("-")[0]=="blocks_combine":
            # For block-based ensemble, average the similarity scores across all references
            ensemble_ref_blocks = np.array([one_ref.get_all_blocks() for one_ref in reference])
            probe_blocks = probe.get_all_blocks()
            ref_blocks = find_closest_blocks_per_pixel(ensemble_ref_blocks, probe_blocks)
            if len(ref_blocks) != len(probe_blocks):
                raise ValueError("Reference and probe images have different number of blocks")
            scores = [metric_fn(ref_block, probe_block) for ref_block, probe_block in zip(ref_blocks, probe_blocks)]
            if mode == "average_blocks":
                return [np.mean(scores)]
            elif mode.split("-")[0]=="blocks_combine":
                if len(mode.split("-"))==2:
                    return combined_block_scores(list_to_square_array(scores), n=int(mode.split("-")[1])).ravel().tolist()
                else:
                    return combined_block_scores(list_to_square_array(scores), n=int(mode.split("-")[1]), modifier=mode.split("-")[2]).ravel().tolist()
            else:
                return scores
        if mode == "full_images":
            # For full image comparison, use the metric directly on the images
            ensemble_ref_image = np.array([one_ref.get_image() for one_ref in reference])
            probe_image = probe.get_image()
            ref_image = find_closest_blocks_per_pixel([ensemble_ref_image], [probe_image])[0]
            score = metric_fn(ref_image, probe_image)
            return [score]
    
    else:
        raise ValueError(f"Unknown ensemble mode: {ensemble_mode}")
            
            
            

def find_closest_blocks_per_pixel(ensemble_ref_blocks, probe_blocks):
    """
    Optimized version using vectorized numpy operations.
    For each pixel in each probe block, find the closest pixel value from the ensemble of reference blocks at the same position.
    
    Args:
        ensemble_ref_blocks: Array of shape (n_references, n_blocks, height, width)
        probe_blocks: List of arrays, each of shape (height, width)
    
    Returns:
        List of optimal reference blocks, each of shape (height, width)
    """
    optima_ref_block_list = []
    
    for i_block in range(len(probe_blocks)):
        probe_block = probe_blocks[i_block]
        reference_blocks = ensemble_ref_blocks[:, i_block]  # Shape: (n_references, height, width)
        
        # Calculate distances for all pixels at once
        # probe_block: (height, width) -> (1, height, width)
        # reference_blocks: (n_references, height, width)
        # Broadcasting: (n_references, height, width) - (1, height, width) -> (n_references, height, width)
        probe_expanded = probe_block[np.newaxis, :, :]  # Shape: (1, height, width)
        distances = np.abs(reference_blocks - probe_expanded)  # Shape: (n_references, height, width)
        
        # Find closest reference for each pixel position
        closest_indices = np.argmin(distances, axis=0)  # Shape: (height, width)
        
        # Use advanced indexing to select the optimal pixels efficiently
        height, width = probe_block.shape
        h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        optimal_ref_block = reference_blocks[closest_indices, h_indices, w_indices]
        optima_ref_block_list.append(optimal_ref_block)
    
    return optima_ref_block_list

def evaluate_image_authentication_roc(reference_dataset="tem", 
                                      original_dataset="o76iP12", fake_dataset="f76iP12", 
                                      metrics=None, 
                                      dataset_base_path="data/wifs2024dataset/wifs2024dataset",
                                      reference_base_path=None, 
                                      plot_output_dir="plots/test/roc", 
                                      shot_probe=None,
                                      shot_reference=None,
                                      mode="average_blocks", reference_data_structure=None, test_uid_range=None, skip_plots=False,
                                      n_comparison_rows=5,
                                      ensemble_mode=None, 
                                      image_side=684,
                                      block_side=128,
                                      block_stride=64):
    """
    Evaluate ROC curves for distinguishing between fake and original images using template as reference.
    
    Args:
        reference_dataset (str): Dataset name for template images.
        original_dataset (str): Dataset name for original capture images.
        fake_dataset (str): Dataset name for fake capture images.
        metrics (dict, optional): Dictionary of metric functions to use. If None, defaults to MSE, SSIM, and PCC.
        dataset_base_path (str): Base path to the dataset (for original and fake datasets).
        reference_base_path (str, optional): Base path for the reference dataset. If None, uses dataset_base_path.
        plot_output_dir (str): Directory to save ROC plots.
        shot (int or None, optional): Specific shot number to use. If None, all common shots will be used.
        mode (str): Comparison mode. Options:
            - "full": Compare full images (default)
            - "average_blocks": Load all blocks, compare corresponding blocks, and average the results
        reference_data_structure (str, optional): Structure type of the reference dataset (e.g., 'template', 'default', 'generated').
        test_uid_range (list, optional): A list of [min_uid, max_uid] to filter UIDs by their numeric value.
        skip_plots (bool): If True, skip generating plots and only produce data output files.
        n_comparison_rows (int): Number of rows to include in reference vs original/fake comparison plots (default: 5).
        
    Returns:
        dict: Dictionary containing ROC data (fpr, tpr, auc) for each metric.
    """
    # Create CDPImageDataset for all three datasets
    
    if isinstance(reference_dataset, list) and isinstance(reference_data_structure, str):
        reference_data_structure = [reference_data_structure] * len(reference_dataset)
    
    block_settings = {
            'block_h': block_side, 'block_w': block_side,
            'stride_h': block_stride, 'stride_w': block_stride}
    
    dataset = CDPImageDataset.from_image_datasets(
        dataset_names=reference_dataset+[original_dataset, fake_dataset],
        dataset_base_path=dataset_base_path,
        reference_base_path=reference_base_path,
        structures={refd : refstr for refd, refstr in zip(reference_dataset, reference_data_structure)},
        block_settings=block_settings
    )
    #dataset.print_info()
    #print()
    
    if shot_probe is None and shot_reference is None:
        # Filter to common UIDs and shots
        dataset.filter_common_uids_shots()
        single_shot = False
    else:
        if shot_reference is None:
            shot_reference = shot_probe
        # Filter to common UIDs and specified shots
        dataset.filter_shots(2, include_dataset_names=reference_dataset)
        #dataset.print_all_available_shots()
        dataset.filter_shots(1, include_dataset_names=[original_dataset, fake_dataset])
        dataset.filter_common_uids()
        single_shot = True
    
    #dataset.print_info()
    print(test_uid_range)
    # Filter UIDs by range if specified
    if test_uid_range is not None:
        dataset.filter_uids_by_range(test_uid_range)
    
    #dataset.print_info()
    
    # Get UIDs after filtering (all datasets should have the same UIDs after filter_common_uids_shots)
    common_uids = set(dataset.get_uids(original_dataset))
    
    if not common_uids:
        logger.error(f"No common UIDs found across {reference_dataset}, {original_dataset}, {fake_dataset}")
        raise ValueError("No common UIDs found after filtering. Check your dataset and filters.")
    
    logger.info(f"Found {len(common_uids)} common UIDs across all datasets")
    
    # Define metrics if not provided
    if metrics is None:
        metrics = get_default_metrics()
    
    # Store similarity scores for each metric
    sim_scores = {metric_name: {"original": [], "fake": []} for metric_name in metrics.keys()}
    
    # Store sample images for plotting (limit to specified number of samples)
    sample_images = {
        "reference": [],
        "original": [],
        "fake": [],
        "uids": [],
        "shots": [],
        "metrics": [],
        "per_block_MSE_original": []
    }
    max_samples_for_plot = n_comparison_rows
    
    # Track statistics
    processed_pairs = 0
    
    # Process each UID and its shots
    for uid in sorted(common_uids):
        # Get available shots for this UID (shots already filtered at dataset level)
        available_shots = dataset.get_shots(original_dataset, uid)
        
        if not single_shot:
            # Process each shot for this UID
            for shot_str in sorted(available_shots):
                metric_values = {}
                
                # Get reference, original, and fake images for this UID-shot pair
                if ensemble_mode is not None:
                    reference = [dataset.get_CDPimage(ref_name, uid) for ref_name in reference_dataset]
                else:
                    reference = dataset.get_CDPimage(reference_dataset[0], uid)
                orig_image = dataset.get_CDPimage(original_dataset, uid, shot_str)
                fake_image = dataset.get_CDPimage(fake_dataset, uid, shot_str)
                
                for metric_name, metric_fn in metrics.items():
                    # Compute similarity for original
                    orig_score = get_score(reference, orig_image, metric_fn, mode=mode, ensemble_mode=ensemble_mode)
                    # Compute similarity for fake
                    fake_score = get_score(reference, fake_image, metric_fn, mode=mode, ensemble_mode=ensemble_mode)

                    sim_scores[metric_name]["original"].extend(orig_score)
                    sim_scores[metric_name]["fake"].extend(fake_score)

                    # Store metric values for plotting (use mean for average_blocks/full, all for blocks_separately)
                    metric_values[metric_name + "_orig"] = np.mean(orig_score)
                    metric_values[metric_name + "_fake"] = np.mean(fake_score)

                # Collect sample images for plotting (first few samples only)
                if len(sample_images["reference"]) < max_samples_for_plot:
                    if ensemble_mode is not None:
                        sample_images["reference"].append(reference[0].get_all_blocks()[0].copy())
                    else:
                        sample_images["reference"].append(reference.get_all_blocks()[0].copy())
                    sample_images["original"].append(orig_image.get_all_blocks()[0].copy())
                    sample_images["fake"].append(fake_image.get_all_blocks()[0].copy())
                    sample_images["uids"].append(uid)
                    sample_images["shots"].append(shot_str)
                    sample_images["metrics"].append(metric_values.copy())
            processed_pairs += 1
        else:
            # Process only the first shot for each UID
            metric_values = {}    
            # Get reference, original, and fake images for this UID-shot pair
            if ensemble_mode is not None:
                reference = [dataset.get_CDPimage(ref_name, uid) for ref_name in reference_dataset]
            else:
                reference = dataset.get_CDPimage(reference_dataset[0], uid)
            orig_image = dataset.get_CDPimage(original_dataset, uid)
            fake_image = dataset.get_CDPimage(fake_dataset, uid)
            
            for metric_name, metric_fn in metrics.items():
                # Compute similarity for original
                orig_score = get_score(reference, orig_image, metric_fn, mode=mode, ensemble_mode=ensemble_mode)
                # Compute similarity for fake
                fake_score = get_score(reference, fake_image, metric_fn, mode=mode, ensemble_mode=ensemble_mode)

                sim_scores[metric_name]["original"].extend(orig_score)
                sim_scores[metric_name]["fake"].extend(fake_score)

                # Store metric values for plotting (use mean for average_blocks/full, all for blocks_separately)
                metric_values[metric_name + "_orig"] = np.mean(orig_score)
                metric_values[metric_name + "_fake"] = np.mean(fake_score)

            # Collect sample images for plotting (first few samples only)
            if len(sample_images["reference"]) < max_samples_for_plot:
                if ensemble_mode is not None:
                    sample_images["reference"].append(reference[0].get_all_blocks()[0].copy())
                else:
                    sample_images["reference"].append(reference.get_all_blocks()[0].copy())
                sample_images["original"].append(orig_image.get_all_blocks()[0].copy())
                sample_images["fake"].append(fake_image.get_all_blocks()[0].copy())
                sample_images["uids"].append(uid)
                sample_images["shots"].append("?")
                sample_images["metrics"].append(metric_values.copy())
            processed_pairs += 1

    
    logger.info(f"Processed {processed_pairs} UID-shot pairs successfully in {mode} mode")
    print(f"Processed {processed_pairs} UID-shot pairs successfully in {mode} mode")
    
    # If no scores were calculated, exit
    if not any(scores["original"] for scores in sim_scores.values()):
        logger.error("No valid scores were calculated. Cannot generate ROC curves.")
        return {}
    
    # Compute ROC curves, AUC, and metric statistics for each metric
    results = {}
    
    for metric_name, scores in sim_scores.items():
        # Prepare data for ROC curve
        y_true = [0] * len(scores["original"]) + [1] * len(scores["fake"])  # 0 for original, 1 for fake
        y_scores = scores["original"] + scores["fake"]  # Higher score should indicate fake
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        results[metric_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc,
            "mean_ref_vs_orig": np.mean(scores["original"]),
            "std_ref_vs_orig": np.std(scores["original"]),
            "mean_ref_vs_fake": np.mean(scores["fake"]),
            "std_ref_vs_fake": np.std(scores["fake"])
        }
    
    # Generate all plots at the end if not skipped
    if not skip_plots:
        # Create directory for saving plots
        os.makedirs(plot_output_dir, exist_ok=True)
        
        # Plot the distributions of similarity scores
        plot_similarity_distributions(sim_scores, plot_output_dir)
        
        # Plot individual ROC curves for each metric
        for metric_name, metric_results in results.items():
            plt.figure(figsize=(8, 8))  # Square figure
            plt.plot(metric_results["fpr"], metric_results["tpr"], lw=2, label=f'{metric_name} (AUC = {metric_results["auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])  # Match xlim for square aspect
            plt.gca().set_aspect('equal', adjustable='box')  # Force square aspect ratio
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve: {metric_name} for Image Authentication')
            plt.legend(loc="lower right")
            plt.grid(True)
            
            # Save the plot
            output_path = os.path.join(plot_output_dir, f"roc_{metric_name}.png")
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()
            
            logger.info(f"ROC curve for {metric_name} saved to {output_path} (AUC: {metric_results['auc']:.3f})")
        
        # Plot all ROC curves together
        plt.figure(figsize=(10, 10))  # Square figure
        for metric_name, metric_results in results.items():
            plt.plot(metric_results["fpr"], metric_results["tpr"], lw=2, label=f'{metric_name} (AUC = {metric_results["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])  # Match xlim for square aspect
        plt.gca().set_aspect('equal', adjustable='box')  # Force square aspect ratio
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Image Authentication')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save the combined plot
        combined_output_path = os.path.join(plot_output_dir, "roc_all_metrics.png")
        plt.savefig(combined_output_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        logger.info(f"Combined ROC curves saved to {combined_output_path}")
        
        # Generate comparison plots if we have sample images
        if sample_images["reference"]:
            # Plot reference vs original comparison
            plot_reference_vs_original_comparison(
                reference_images=sample_images["reference"],
                original_images=sample_images["original"],
                uids=sample_images["uids"],
                shots=sample_images["shots"],
                metrics_data=sample_images["metrics"],
                output_dir=plot_output_dir,
                n_rows=min(n_comparison_rows, len(sample_images["reference"])),
                reference_name="Reference",
                original_name="Original",
                per_block_mse_data=sample_images.get("per_block_MSE_original", None)
            )
    else:
        logger.info("Plotting skipped, only data files will be generated")
    
    return results

def evaluate_reference_performance(reference_dataset="tem", 
                                  metrics=None, 
                                  dataset_base_path="data/wifs2024dataset/wifs2024dataset",
                                  reference_base_path=None,
                                  output_dir=None, 
                                  plot_output_dir=None,
                                  output_name=None,
                                  shot_reference=None,
                                  shot_probe=None,
                                  original_datasets="o55Epson",
                                  mode="average_blocks",
                                  reference_data_structure=None,
                                  test_uid_range=None,
                                  skip_plots=False,
                                  n_comparison_rows=5,
                                  ensemble_mode=None,
                                  image_side=684,
                                  block_side=128,
                                  block_stride=64):
    """
    Evaluate AUC scores for using a reference dataset to authenticate all original-fake dataset pairs.
    
    Args:
        reference_dataset (str): Dataset name for reference images.
        metrics (dict, optional): Dictionary of metric functions to use. If None, defaults to MSE, SSIM, and PCC.
        dataset_base_path (str): Base path to the dataset (for original and fake datasets).
        reference_base_path (str, optional): Base path for the reference dataset. If None, uses dataset_base_path.
        output_dir (str): Directory to save results.
        output_name (str, optional): Custom name for output files. If None, uses default naming.
        shot (int or None, optional): Specific shot number to use. If None, all common shots will be used.
        original_datasets (str or list): Dataset name(s) for original images. Fake datasets will be derived 
                                       by replacing the first letter 'o' with 'f'.
        mode (str): Comparison mode. Options:
            - "full": Compare full images
            - "average_blocks": Load all blocks, compare corresponding blocks, and average the results
        reference_data_structure (str, optional): Structure type of the reference dataset (e.g., 'template', 'default', 'generated').
        test_uid_range (list, optional): A list of [min_uid, max_uid] to filter UIDs by their numeric value.
        skip_plots (bool): If True, skip generating plots and only produce data output files.
        n_comparison_rows (int): Number of rows to include in reference vs original/fake comparison plots (default: 5).
        
    Returns:
        dict: Dictionary containing AUC scores for each original-fake pair and metric.
    """
    # Convert original_datasets to a list if it's a string
    if isinstance(original_datasets, str):
        original_datasets = [original_datasets]
    
    # Create a dictionary mapping original datasets to fake datasets
    original_to_fake_pairs = {}
    for original in original_datasets:
        if original.startswith('o'):
            fake = 'f' + original[1:]
            original_to_fake_pairs[original] = fake
        else:
            logger.warning(f"Original dataset '{original}' doesn't start with 'o', skipping")
    
    if not original_to_fake_pairs:
        raise ValueError("No valid original-fake pairs could be created. Original datasets should start with 'o'.")
    
    logger.info(f"Created original-fake pairs: {original_to_fake_pairs}")

    # Initialize results dictionary
    results = {"results": {}, "metadata": {}}
    
    # Log UID range if specified
    if test_uid_range is not None:
        min_uid, max_uid = test_uid_range
        logger.info(f"Evaluating with UID range: [{min_uid}, {max_uid}]")
    
    if output_dir is None:
        output_dir = f"{mode}"
        if test_uid_range is not None:
            output_dir = f"{output_dir}_uid_{min_uid}_{max_uid}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics if not provided (same as in evaluate_image_authentication_roc)
    if metrics is None:
        metrics = get_default_metrics()
    
    # Evaluate each original-fake pair
    for original_dataset, fake_dataset in original_to_fake_pairs.items():
        logger.info(f"Evaluating reference authentication for pair: {original_dataset} - {fake_dataset} (mode: {mode})")
        
        # Use existing function to evaluate this specific pair
        roc_results = evaluate_image_authentication_roc(
            reference_dataset=reference_dataset,
            original_dataset=original_dataset,
            fake_dataset=fake_dataset,
            metrics=metrics,
            dataset_base_path=dataset_base_path,
            reference_base_path=reference_base_path,
            plot_output_dir=f"{plot_output_dir}/{original_dataset}_vs_{fake_dataset}",
            shot_reference=shot_reference,
            shot_probe=shot_probe,
            mode=mode,
            reference_data_structure=reference_data_structure,
            test_uid_range=test_uid_range,
            skip_plots=skip_plots,
            n_comparison_rows=n_comparison_rows,
            ensemble_mode=ensemble_mode,
            image_side=image_side,
            block_side=block_side,
            block_stride=block_stride
        )
        
        # Extract all metric data for each metric
        if roc_results:
            results["results"][f"{original_dataset}_vs_{fake_dataset}"] = {
                metric_name: {
                    "auc": data["auc"],
                    "mean_ref_vs_orig": data["mean_ref_vs_orig"],
                    "std_ref_vs_orig": data["std_ref_vs_orig"],
                    "mean_ref_vs_fake": data["mean_ref_vs_fake"],
                    "std_ref_vs_fake": data["std_ref_vs_fake"]
                } for metric_name, data in roc_results.items()
            }
    
    
    main_reference_dataset = reference_dataset[0] if isinstance(reference_dataset, list) else reference_dataset
    if len(reference_dataset) == 1:
        refernce_dataset_naming = main_reference_dataset
    else:
        refernce_dataset_naming = main_reference_dataset + f"_ensemble{len(reference_dataset)}_{ensemble_mode}" 
    
    # Read dedicated class from metadata.json
    metadata_path = Path(reference_base_path or dataset_base_path) / main_reference_dataset / "metadata.json"
    dedicated_class = None
    if metadata_path.exists():
        with open(metadata_path, "r") as metadata_file:
            metadata_from_json = json.load(metadata_file)
            dedicated_class = metadata_from_json.get("dedicated_class", None)
            logger.info(f"Read dedicated class from metadata of the first reference provided: {dedicated_class}")
        results["metadata"]=metadata_from_json  # Save the dedicated class and other metadata
    else:
        logger.warning(f"metadata.json not found at {metadata_path}")

    # Add metadata
    results["metadata"]["reference_dataset"] = refernce_dataset_naming
    # Extract the last folder name if reference_dataset is a path
    if "/" in refernce_dataset_naming:
        results["metadata"]["reference_print_label"] = Path(refernce_dataset_naming).name
    else:
        results["metadata"]["reference_print_label"] = refernce_dataset_naming

    # Determine output file prefix
    if output_name:
        file_prefix = output_name
    else:
        file_prefix = f"reference_performance_results_{mode}"
    
    # Save results as pickle
    pickle_path = os.path.join(output_dir, f"{file_prefix}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Results saved as pickle to {pickle_path}")
    
    # Save results as text
    txt_path = os.path.join(output_dir, f"{file_prefix}.txt")
    save_results_as_text(results, txt_path, main_reference_dataset, mode, test_uid_range, dedicated_class)
    logger.info(f"Results saved as text to {txt_path}")
    
    return results

def plot_reference_vs_original_comparison(reference_images, original_images, uids, shots, 
                                        metrics_data, output_dir, n_rows=5, 
                                        reference_name="Reference", original_name="Original",
                                        per_block_mse_data=None):
    """
    Plot comparison between reference and original images with metrics.
    
    Args:
        reference_images (list): List of reference images (numpy arrays).
        original_images (list): List of original images (numpy arrays).
        uids (list): List of UIDs corresponding to each image pair.
        shots (list): List of shots corresponding to each image pair.
        metrics_data (list): List of dictionaries containing metric values for each pair.
        output_dir (str): Directory to save the plot.
        n_rows (int): Number of rows to plot (default: 5).
        reference_name (str): Name to display for reference images.
        original_name (str): Name to display for original images.
        per_block_mse_data (list, optional): List of per-block MSE arrays to display as 9x9 heatmaps.
    """
    if not reference_images or not original_images:
        logger.warning("No images provided for plotting")
        return
    
    # Limit the number of samples to plot
    num_samples = min(n_rows, len(reference_images))
    if num_samples == 0:
        logger.warning("No samples available for plotting")
        return
    
    # Define columns: Reference, Original, Difference, Per-Block MSE, Metrics
    columns = [reference_name, original_name, "Difference", "Per-Block MSE", "Metrics"]
    
    fig, axes = plt.subplots(num_samples, len(columns), figsize=(4 * len(columns), 4 * num_samples))
    
    # Ensure axes is 2D even for single row
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        ref_img = reference_images[i]
        orig_img = original_images[i]
        uid = uids[i] if i < len(uids) else "N/A"
        shot = shots[i] if i < len(shots) else "N/A"
        metrics = metrics_data[i] if i < len(metrics_data) else {}
        
        # Ensure images are 2D (remove channel dimension if present)
        if ref_img.ndim == 3 and ref_img.shape[2] == 1:
            ref_img = ref_img[:, :, 0]
        elif ref_img.ndim == 3 and ref_img.shape[0] == 1:
            ref_img = ref_img[0, :, :]
            
        if orig_img.ndim == 3 and orig_img.shape[2] == 1:
            orig_img = orig_img[:, :, 0]
        elif orig_img.ndim == 3 and orig_img.shape[0] == 1:
            orig_img = orig_img[0, :, :]
        
        # Plot reference image
        axes[i, 0].imshow(ref_img, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"{reference_name}\nUID: {uid}, Shot: {shot}")
        axes[i, 0].axis("off")
        
        # Plot original image
        axes[i, 1].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"{original_name}")
        axes[i, 1].axis("off")
        
        # Plot difference
        diff = orig_img - ref_img
        mse_diff = np.mean((orig_img - ref_img) ** 2)
        axes[i, 2].imshow(diff, cmap="bwr", vmin=-1, vmax=1)
        axes[i, 2].set_title(f"Difference ({original_name} - {reference_name})\nMSE: {mse_diff:.4f}")
        axes[i, 2].axis("off")
        
        # Plot per-block MSE as 9x9 heatmap
        if per_block_mse_data and i < len(per_block_mse_data) and per_block_mse_data[i] is not None:
            per_block_mse = per_block_mse_data[i]
            if len(per_block_mse) == 81:  # 9x9 = 81 blocks
                mse_grid = np.array(per_block_mse).reshape(9, 9)
                im = axes[i, 3].imshow(mse_grid, cmap="viridis", interpolation='nearest')
                axes[i, 3].set_title(f"Per-Block MSE\nMin: {np.min(mse_grid):.4f}, Max: {np.max(mse_grid):.4f}")
                # Add colorbar
                plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
            else:
                axes[i, 3].text(0.5, 0.5, f"Invalid MSE data\n({len(per_block_mse)} values)", 
                               transform=axes[i, 3].transAxes, ha='center', va='center')
                axes[i, 3].set_title("Per-Block MSE (Error)")
        else:
            axes[i, 3].text(0.5, 0.5, "No per-block\nMSE data", 
                           transform=axes[i, 3].transAxes, ha='center', va='center')
            axes[i, 3].set_title("Per-Block MSE")
        axes[i, 3].axis("off")
        
        # Plot metrics text
        axes[i, 4].axis("off")
        metric_text = "Metrics:\n"
        if metrics:
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    metric_text += f"{metric_name}: {metric_value:.4f}\n"
                else:
                    metric_text += f"{metric_name}: {metric_value}\n"
        else:
            metric_text += "No metrics available"
        
        axes[i, 4].text(0.1, 0.5, metric_text, transform=axes[i, 4].transAxes, 
                       fontsize=12, verticalalignment='center', fontfamily='monospace')
        axes[i, 4].set_title("Similarity Metrics\naverage over full image")
    
    plt.suptitle(f"{reference_name} vs {original_name} Comparison", fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{reference_name.lower()}_vs_{original_name.lower()}_comparison.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    logger.info(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    # Set up argument Parser
    parser = argparse.ArgumentParser(description="Evaluate reference performance for image authentication")
    parser.add_argument("--reference", nargs='+', type=str, required=True, help="Reference dataset name")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--plot_output_dir", type=str, required=False, help="Directory to save plots")
    parser.add_argument("--output_name", type=str, default=None, help="Custom name for output files")
    parser.add_argument("--shot_probe", type=str, default=None, help="Specific shot number to use in probes")
    parser.add_argument("--shot_reference", type=str, default=None, help="Specific shot number to use for reference")
    parser.add_argument("--test_uid_range", type=str, help="UID range in format 'min,max', e.g., '193,288'")
    parser.add_argument("--originals", nargs='+', required=True, help="List of original dataset names")
    parser.add_argument("--mode", type=str, default="average_blocks", help="Comparison mode")
    parser.add_argument("--reference_path", type=str, default=None, 
                        help="Base path for reference dataset (default: same as dataset_base_path)")
    parser.add_argument("--dataset_path", type=str, 
                        default="data/wifs2024dataset/wifs2024dataset", 
                        help="Base path for dataset")
    parser.add_argument("--reference_data_structure", type=str, required=True, 
                        help="Structure type of the reference dataset (e.g., 'template', 'default', 'generated').")
    parser.add_argument("--no_plots", action="store_false", 
                        help="Skip generating plots and only produce data output files")
    parser.add_argument("--ensemble_mode", type=str, default=None, 
                        help="If several references are provided deside how to ensemble them. possible values: 'average_reference', 'average_score', 'max_score', 'min_score', 'per_pixel_closest_reference'. Default is None, which means no ensemble.")
    parser.add_argument("--block_side", type=int, default=128, 
                        help="Size of the blocks to use for comparison (default: 128)")
    parser.add_argument("--block_stride", type=int, default=64, 
                        help="Stride for the blocks (default: 64)")
    parser.add_argument("--image_side", type=int, default=684, 
                        help="Size of the images to use for comparison (default: 684)")
    
    args = parser.parse_args()

    if len(args.reference) > 1 and args.ensemble_mode is None:
        parser.error("If multiple references are provided, --ensemble_mode must be specified.")
    if len(args.reference) == 1 and args.ensemble_mode is not None:
        parser.error("If only one reference is provided, --ensemble_mode should not be specified.")

    if args.plot_output_dir is None:
        if len(args.reference) == 1:
            if Path(args.reference[0]).is_dir():
                args.plot_output_dir = args.reference + "/plots"
        else:
            args.plot_output_dir = f"{args.output_dir}/plots"
    
    # Parse test_uid_range
    uid_range = None
    if args.test_uid_range:
        try:
            min_uid, max_uid = map(int, args.test_uid_range.split(','))
            uid_range = [min_uid, max_uid]
        except ValueError:
            print(f"Error: Invalid test_uid_range format. Use 'min,max', e.g., '193,288'")
            exit(1)
    
    # Parse shot to integer if it's a numeric value
    shot_probe = args.shot_probe
    if shot_probe and isinstance(shot_probe, str):
        shot_probe = int(shot_probe)#
        
    shot_reference = args.shot_reference
    if args.shot_reference:
        shot_reference = int(args.shot_reference)
    
    # Run evaluation
    evaluate_reference_performance(
        reference_dataset=args.reference,
        output_dir=args.output_dir,
        plot_output_dir=args.plot_output_dir,
        output_name=args.output_name,
        shot_probe=shot_probe,
        shot_reference=shot_reference,
        original_datasets=args.originals,
        mode=args.mode,
        reference_base_path=args.reference_path,
        dataset_base_path=args.dataset_path,
        reference_data_structure=args.reference_data_structure,
        test_uid_range=uid_range,
        skip_plots=args.no_plots,
        ensemble_mode=args.ensemble_mode,
        block_side=args.block_side,
        block_stride=args.block_stride,
        image_side=args.image_side
    )