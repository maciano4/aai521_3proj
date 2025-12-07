"""
Evaluation script for trained flood detection models.
Evaluates models on test set and generates comprehensive reports.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent))

import config
from dataset import create_dataloaders
from models import create_model
from metrics import SegmentationMetrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate flood detection models')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, required=True,
                        choices=['unet++', 'deeplabv3+', 'segformer'],
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help=f'Batch size (default: {config.BATCH_SIZE})')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save prediction masks')
    
    return parser.parse_args()


def evaluate_model(model, test_loader, device, save_predictions=False, output_dir=None):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use
        save_predictions: Whether to save prediction masks
        output_dir: Directory to save predictions
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes=config.NUM_CLASSES)
    
    all_predictions = []
    all_targets = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            metrics.update(predictions, targets)
            
            # Store for visualization
            if save_predictions:
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
    
    # Compute final metrics
    results = {
        'iou': metrics.compute_iou(),
        'dice': metrics.compute_dice(),
        'precision_recall_f1': metrics.compute_precision_recall_f1(),
        'pixel_accuracy': metrics.compute_pixel_accuracy(),
        'confusion_matrix': metrics.confusion_matrix.tolist()
    }
    
    # Save predictions if requested
    if save_predictions and output_dir:
        pred_dir = Path(output_dir) / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        np.save(pred_dir / 'predictions.npy', all_predictions)
        np.save(pred_dir / 'targets.npy', all_targets)
        print(f"Predictions saved to: {pred_dir}")
    
    return results, metrics


def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")


def plot_metrics_comparison(results, class_names, output_path):
    """Plot per-class metrics comparison."""
    iou_per_class = results['iou']['per_class']
    dice_per_class = results['dice']['per_class']
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width/2, iou_per_class, width, label='IoU', alpha=0.8)
    rects2 = ax.bar(x + width/2, dice_per_class, width, label='Dice', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics comparison saved to: {output_path}")


def create_evaluation_report(results, output_path):
    """Create text evaluation report."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FLOOD DETECTION MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean IoU:           {results['iou']['mean']:.4f}\n")
        f.write(f"Mean Dice:          {results['dice']['mean']:.4f}\n")
        f.write(f"Pixel Accuracy:     {results['pixel_accuracy']:.4f}\n")
        f.write(f"Macro Precision:    {results['precision_recall_f1']['precision']:.4f}\n")
        f.write(f"Macro Recall:       {results['precision_recall_f1']['recall']:.4f}\n")
        f.write(f"Macro F1:           {results['precision_recall_f1']['f1']:.4f}\n")
        f.write("\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<30} {'IoU':>10} {'Dice':>10}\n")
        f.write("-"*80 + "\n")
        
        for i, class_name in enumerate(config.CLASS_NAMES):
            iou = results['iou']['per_class'][i]
            dice = results['dice']['per_class'][i]
            f.write(f"{class_name:<30} {iou:>10.4f} {dice:>10.4f}\n")
        
        f.write("\n")
    
    print(f"Evaluation report saved to: {output_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}\n")
    
    # Create test dataloader
    print("Creating test dataloader...")
    _, _, test_loader = create_dataloaders(
        train_dir=config.PROCESSED_TRAIN_DIR,
        val_dir=config.PROCESSED_VAL_DIR,
        test_dir=config.PROCESSED_TEST_DIR,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS
    )
    print(f"Test batches: {len(test_loader)}\n")
    
    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(
        args.model,
        num_classes=config.NUM_CLASSES,
        in_channels=6,
        **config.MODEL_CONFIGS.get(args.model, {})
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint metrics: {checkpoint.get('metrics', {})}\n")
    
    # Evaluate
    results, metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=args.device,
        save_predictions=args.save_predictions,
        output_dir=output_dir
    )
    
    # Save results as JSON
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Print formatted metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80 + "\n")
    print(format_metrics(metrics, config.CLASS_NAMES))
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Confusion matrix
    cm = np.array(results['confusion_matrix'])
    plot_confusion_matrix(cm, config.CLASS_NAMES, output_dir / 'confusion_matrix.png')
    
    # Metrics comparison
    plot_metrics_comparison(results, config.CLASS_NAMES, output_dir / 'metrics_comparison.png')
    
    # Text report
    create_evaluation_report(results, output_dir / 'evaluation_report.txt')
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80 + "\n")


# ============================================================================
# ADDITIONAL EVALUATION UTILITIES
# ============================================================================

def run_inference(model, dataloader, device, store_images=True):
    """
    Run inference on entire dataset and return predictions (memory-efficient).
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to use
        store_images: Whether to store images (set False for large datasets to save memory)
        
    Returns:
        Tuple of (images, predictions, targets) as numpy arrays
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_images = [] if store_images else None
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Store results as numpy arrays immediately to save memory
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
            # Only store first 100 images to save memory
            if store_images and len(all_images) < 100:
                all_images.append(images.cpu().numpy())
            
            # Clear CUDA cache periodically to prevent OOM
            if device.type == 'cuda' and (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
    
    # Concatenate all batches as numpy arrays
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    if store_images and all_images:
        all_images = np.concatenate(all_images, axis=0)
    else:
        # Create a small dummy array for compatibility
        all_images = np.zeros((min(100, len(all_predictions)), 6, 256, 256), dtype=np.float32)
    
    return all_images, all_predictions, all_targets


def compute_sample_iou(predictions, targets):
    """
    Compute IoU for each sample individually.
    
    Args:
        predictions: Predicted labels tensor [N, H, W]
        targets: Ground truth labels tensor [N, H, W]
        
    Returns:
        Array of IoU scores per sample
    """
    ious = []
    
    for pred, target in zip(predictions, targets):
        # Compute IoU for this sample
        intersection = (pred == target).sum().item()
        union = pred.numel()
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)
    
    return np.array(ious)


def analyze_error_distribution(predictions, targets, class_names):
    """
    Analyze error distribution across classes.
    
    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        class_names: List of class names
        
    Returns:
        Dictionary with error statistics
    """
    import torch
    
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.numpy()
    if torch.is_tensor(targets):
        targets = targets.numpy()
    
    preds_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    error_stats = {
        'class_distribution': {},
        'misclassifications': {}
    }
    
    # Class distribution
    for i, class_name in enumerate(class_names):
        true_count = (targets_flat == i).sum()
        pred_count = (preds_flat == i).sum()
        total = len(targets_flat)
        
        error_stats['class_distribution'][class_name] = {
            'ground_truth_count': int(true_count),
            'predicted_count': int(pred_count),
            'ground_truth_pct': float(100 * true_count / total),
            'predicted_pct': float(100 * pred_count / total)
        }
    
    # Misclassifications
    for true_class in range(len(class_names)):
        mask = targets_flat == true_class
        if mask.sum() > 0:
            misclassified = preds_flat[mask]
            unique, counts = np.unique(misclassified, return_counts=True)
            
            misclass_dict = {}
            for pred_class, count in zip(unique, counts):
                if pred_class != true_class:
                    pct = 100 * count / mask.sum()
                    if pct > 1.0:  # Only significant misclassifications
                        misclass_dict[class_names[int(pred_class)]] = {
                            'count': int(count),
                            'percentage': float(pct)
                        }
            
            if misclass_dict:
                error_stats['misclassifications'][class_names[true_class]] = misclass_dict
    
    return error_stats


def save_evaluation_results(
    model_name: str,
    checkpoint_path: Path,
    test_metrics: Dict,
    metrics_df: pd.DataFrame,
    sample_ious: np.ndarray,
    output_dir: Path,
    timestamp: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Save evaluation results to JSON and CSV files.
    
    Args:
        model_name: Name of the model
        checkpoint_path: Path to model checkpoint
        test_metrics: Dictionary of test metrics
        metrics_df: DataFrame with per-class metrics
        sample_ious: Array of per-sample IoU scores
        output_dir: Directory to save results
        timestamp: Optional timestamp string
        
    Returns:
        Tuple of (json_file_path, csv_file_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare results dictionary
    results = {
        'model': model_name,
        'timestamp': timestamp,
        'checkpoint': str(checkpoint_path),
        'test_set_size': len(sample_ious),
        'overall_metrics': {
            'mean_iou': float(test_metrics['mean_iou']),
            'mean_dice': float(test_metrics['mean_dice']),
            'mean_f1': float(test_metrics['mean_f1']),
            'accuracy': float(test_metrics['accuracy'])
        },
        'per_class_metrics': metrics_df.to_dict('records'),
        'sample_ious': {
            'mean': float(sample_ious.mean()),
            'std': float(sample_ious.std()),
            'min': float(sample_ious.min()),
            'max': float(sample_ious.max()),
            'median': float(np.median(sample_ious))
        }
    }
    
    # Save to JSON
    results_file = output_dir / f'{model_name}_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metrics DataFrame as CSV
    csv_file = output_dir / f'{model_name}_metrics_{timestamp}.csv'
    metrics_df.to_csv(csv_file, index=False)
    
    return results_file, csv_file


def export_predictions(
    model_name: str,
    predictions: np.ndarray,
    test_metrics: Dict,
    class_names: List[str],
    output_dir: Path,
    timestamp: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Export predictions and metadata for submission.
    
    Args:
        model_name: Name of the model
        predictions: Prediction array
        test_metrics: Dictionary of test metrics
        class_names: List of class names
        output_dir: Directory to save predictions
        timestamp: Optional timestamp string
        
    Returns:
        Tuple of (predictions_file, metadata_file)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save predictions
    if torch.is_tensor(predictions):
        predictions = predictions.numpy()
    
    predictions_file = output_dir / f'{model_name}_predictions.npy'
    np.save(predictions_file, predictions)
    
    # Save metadata
    metadata = {
        'model': model_name,
        'num_samples': len(predictions),
        'num_classes': len(class_names),
        'class_names': class_names,
        'mean_iou': float(test_metrics['mean_iou']),
        'timestamp': timestamp
    }
    
    metadata_file = output_dir / f'{model_name}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return predictions_file, metadata_file


def print_evaluation_summary(
    model_name: str,
    model_params: int,
    test_size: int,
    test_metrics: Dict,
    sample_ious: np.ndarray,
    class_names: List[str]
):
    """
    Print a comprehensive evaluation summary.
    
    Args:
        model_name: Name of the model
        model_params: Number of model parameters
        test_size: Size of test set
        test_metrics: Dictionary of test metrics
        sample_ious: Array of per-sample IoU scores
        class_names: List of class names
    """
    print("="*80)
    print("FLOOD DETECTION - EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nModel: {model_name}")
    print(f"Parameters: {model_params:,}")
    print(f"Test Samples: {test_size}")
    
    print("\nOverall Performance:")
    print(f"  Mean IoU:  {test_metrics['mean_iou']:.4f}")
    print(f"  Mean Dice: {test_metrics['mean_dice']:.4f}")
    print(f"  Mean F1:   {test_metrics['mean_f1']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    
    print("\nPer-Class IoU:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:20s}: {test_metrics['iou_per_class'][i]:.4f}")
    
    print("\nSample Statistics:")
    print(f"  Mean IoU:   {sample_ious.mean():.4f}")
    print(f"  Median IoU: {np.median(sample_ious):.4f}")
    print(f"  Std Dev:    {sample_ious.std():.4f}")
    
    best_class_idx = np.argmax(test_metrics['iou_per_class'])
    worst_class_idx = np.argmin(test_metrics['iou_per_class'])
    
    print("\nKey Findings:")
    print(f"  Best class: {class_names[best_class_idx]} (IoU: {test_metrics['iou_per_class'][best_class_idx]:.4f})")
    print(f"  Worst class: {class_names[worst_class_idx]} (IoU: {test_metrics['iou_per_class'][worst_class_idx]:.4f})")
    
    print("="*80)


if __name__ == '__main__':
    main()
