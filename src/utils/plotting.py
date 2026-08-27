import os
import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logger
logger = logging.getLogger(__name__)

def plot_similarity_distributions(sim_scores, output_dir):
    """
    Plot histograms of similarity scores for fake and original images.
    
    Args:
        sim_scores (dict): Dictionary containing similarity scores for each metric.
        output_dir (str): Directory to save the plot files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_name, scores in sim_scores.items():
        plt.figure(figsize=(10, 6))
        
        # Determine global min and max for consistent binning
        all_scores = scores["original"] + scores["fake"]
        min_score = min(all_scores)
        max_score = max(all_scores)
        bins = np.linspace(min_score, max_score, 31)  # 30 bins across the range
        
        # Plot histograms with shared bins
        plt.hist(scores["original"], bins=bins, alpha=0.7, label=f"Original (n={len(scores['original'])})", color="green")
        plt.hist(scores["fake"], bins=bins, alpha=0.7, label=f"Fake (n={len(scores['fake'])})", color="red")
        
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {metric_name} Similarity Scores")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        output_path = os.path.join(output_dir, f"dist_{metric_name}.png")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        logger.info(f"Distribution of {metric_name} similarity scores saved to {output_path}")
    
    # Create a combined plot with all metrics side by side
    n_metrics = len(sim_scores)
    if n_metrics > 0:
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), sharey=True)
        
        # Handle case of only one metric
        if n_metrics == 1:
            axes = [axes]
        
        for i, (metric_name, scores) in enumerate(sim_scores.items()):
            # Determine global min and max for consistent binning
            all_scores = scores["original"] + scores["fake"]
            min_score = min(all_scores)
            max_score = max(all_scores)
            bins = np.linspace(min_score, max_score, 31)  # 30 bins across the range
            
            # Plot histograms with shared bins
            axes[i].hist(scores["original"], bins=bins, alpha=0.7, label=f"Original (n={len(scores['original'])})", color="green")
            axes[i].hist(scores["fake"], bins=bins, alpha=0.7, label=f"Fake (n={len(scores['fake'])})", color="red")
            
            axes[i].set_xlabel("Similarity Score")
            if i == 0:
                axes[i].set_ylabel("Frequency")
            axes[i].set_title(f"{metric_name}")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        combined_output_path = os.path.join(output_dir, "dist_all_metrics.png")
        plt.savefig(combined_output_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        logger.info(f"Combined distribution plot saved to {combined_output_path}")
        
        
def plot_images(images, labels, output_filename, output_dir=None, figsize=None, show_dimensions=False):
    """
    Plot multiple images side by side and save to a file.
    
    Args:
        images (list): List of images to plot.
        labels (list): List of labels for each image.
        output_filename (str): Filename for the saved plot.
        output_dir (str, optional): Directory to save the plot. If None, saves in current directory.
        figsize (tuple, optional): Figure size (width, height) in inches.
        show_dimensions (bool, optional): If True, display image dimensions on each plot.
    """
    n_images = len(images)
    if n_images != len(labels):
        raise ValueError("Number of images must match number of labels")
    
    # Determine figure size if not specified
    if figsize is None:
        # Default width per image * number of images, height fixed
        figsize = (6 * n_images, 6)
    
    # Create figure and axes
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    # Handle case of only one image
    if n_images == 1:
        axes = [axes]
    
    # Plot each image
    for i, (img, label) in enumerate(zip(images, labels)):
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)  # Set fixed color scale [0,1]
        axes[i].set_title(label)
        axes[i].axis('off')  # Turn off axis numbers
        
        # Add dimension text if requested
        if show_dimensions:
            height, width = img.shape[:2]
            dim_text = f"{width}×{height} px"
            # Position at the top of the image with a slight offset
            text = axes[i].text(
                width * 0.05,  # 5% from the left
                height * 0.05,  # 5% from the top
                dim_text,
                color='white',
                fontsize=10,
                horizontalalignment='left',
                verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.7, pad=2)
            )
    
    plt.tight_layout()
    
    # Create output path
    output_path = output_filename
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
    
    # Save figure with tight bbox and high dpi
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Images saved as {output_path}")