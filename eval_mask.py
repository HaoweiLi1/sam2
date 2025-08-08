# import numpy as np
# import cv2
# from scipy.ndimage import distance_transform_edt
# from skimage.segmentation import find_boundaries
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm

# from scipy.optimize import linear_sum_assignment

# def calculate_optimal_iou(gt_mask, pred_mask):
#     """Calculate IoU with optimal label matching using Hungarian algorithm."""
#     # Get unique labels (excluding background)
#     gt_labels = np.unique(gt_mask)
#     gt_labels = gt_labels[gt_labels != 0]
    
#     pred_labels = np.unique(pred_mask)
#     pred_labels = pred_labels[pred_labels != 0]
    
#     # If either mask has no segments, return 0
#     if len(gt_labels) == 0 or len(pred_labels) == 0:
#         return 0
    
#     # Create cost matrix (negative IoU for each pair of segments)
#     cost_matrix = np.zeros((len(gt_labels), len(pred_labels)))
    
#     for i, gt_label in enumerate(gt_labels):
#         gt_segment = gt_mask == gt_label
        
#         for j, pred_label in enumerate(pred_labels):
#             pred_segment = pred_mask == pred_label
            
#             # Calculate IoU
#             intersection = np.logical_and(gt_segment, pred_segment).sum()
#             union = np.logical_or(gt_segment, pred_segment).sum()
            
#             iou = intersection / union if union > 0 else 0
#             # Negative because Hungarian algorithm minimizes cost
#             cost_matrix[i, j] = -iou
    
#     # Find optimal assignment
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
#     # Calculate mean IoU with optimal assignment
#     total_iou = -cost_matrix[row_ind, col_ind].sum()
#     mean_iou = total_iou / len(gt_labels)
    
#     return mean_iou

# def normalize_masks(gt_mask, pred_mask, visualization=False):
#     """
#     Normalize masks by relabeling them to have consistent IDs.
#     Returns masks where corresponding segments have the same IDs.
#     """
#     # Get unique labels (excluding background)
#     gt_labels = np.unique(gt_mask)
#     if 0 in gt_labels:
#         gt_labels = gt_labels[gt_labels != 0]
    
#     pred_labels = np.unique(pred_mask)
#     if 0 in pred_labels:
#         pred_labels = pred_labels[pred_labels != 0]
    
#     # Create normalized masks
#     gt_norm = np.zeros_like(gt_mask)
#     pred_norm = np.zeros_like(pred_mask)
    
#     # Create IoU matrix
#     iou_matrix = np.zeros((len(gt_labels), len(pred_labels)))
    
#     for i, gt_label in enumerate(gt_labels):
#         gt_segment = gt_mask == gt_label
#         gt_norm[gt_segment] = i + 1  # New label starting from 1
        
#         for j, pred_label in enumerate(pred_labels):
#             pred_segment = pred_mask == pred_label
            
#             # Calculate IoU
#             intersection = np.logical_and(gt_segment, pred_segment).sum()
#             union = np.logical_or(gt_segment, pred_segment).sum()
            
#             iou_matrix[i, j] = intersection / union if union > 0 else 0
    
#     # Find optimal assignment
#     row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
#     # Create a mapping from original pred labels to matched gt labels
#     label_mapping = {}
#     for i, j in zip(row_ind, col_ind):
#         pred_label = pred_labels[j]
#         new_label = i + 1  # To match gt_norm
#         label_mapping[pred_label] = new_label
    
#     # Apply mapping to prediction mask
#     for pred_label, new_label in label_mapping.items():
#         pred_segment = pred_mask == pred_label
#         pred_norm[pred_segment] = new_label
    
#     # Visualization for debugging
#     if visualization:
#         import matplotlib.pyplot as plt
        
#         fig, axes = plt.subplots(2, 2, figsize=(10, 10))
#         axes[0, 0].imshow(gt_mask)
#         axes[0, 0].set_title('Original GT')
#         axes[0, 1].imshow(pred_mask)
#         axes[0, 1].set_title('Original Pred')
#         axes[1, 0].imshow(gt_norm)
#         axes[1, 0].set_title('Normalized GT')
#         axes[1, 1].imshow(pred_norm)
#         axes[1, 1].set_title('Normalized Pred')
#         plt.tight_layout()
#         plt.show()
    
#     return gt_norm, pred_norm

# def load_segmentation_mask(path):
#     """Load a segmentation mask and convert it to a label map."""
#     mask = cv2.imread(path)
#     if mask is None:
#         raise ValueError(f"Could not load mask from {path}")
    
#     # Convert RGB to a unique integer label per color
#     # This assumes different colors represent different segments
#     flattened = mask.reshape(-1, 3)
#     unique_colors = np.unique(flattened, axis=0)
    
#     # Create label map (each unique color gets a unique label)
#     label_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
#     for i, color in enumerate(unique_colors):
#         # Find all pixels with this color
#         mask_r = mask[:, :, 0] == color[0]
#         mask_g = mask[:, :, 1] == color[1]
#         mask_b = mask[:, :, 2] == color[2]
#         color_mask = mask_r & mask_g & mask_b
#         label_map[color_mask] = i + 1  # +1 to reserve 0 for background
    
#     return label_map

# def calculate_iou(gt_mask, pred_mask):
#     """Calculate mean IoU across all segments."""
#     # Get unique labels from both masks (excluding 0 if it's background)
#     gt_labels = np.unique(gt_mask)
#     gt_labels = gt_labels[gt_labels != 0]
    
#     pred_labels = np.unique(pred_mask)
#     pred_labels = pred_labels[pred_labels != 0]
    
#     # Calculate IoU for each segment in ground truth
#     ious = []
#     for gt_label in gt_labels:
#         gt_segment = gt_mask == gt_label
#         best_iou = 0
        
#         # Find the best matching segment in prediction
#         for pred_label in pred_labels:
#             pred_segment = pred_mask == pred_label
            
#             # Calculate intersection and union
#             intersection = np.logical_and(gt_segment, pred_segment).sum()
#             union = np.logical_or(gt_segment, pred_segment).sum()
            
#             # Calculate IoU
#             if union > 0:
#                 iou = intersection / union
#                 best_iou = max(best_iou, iou)
        
#         ious.append(best_iou)
    
#     # Return mean IoU
#     return np.mean(ious) if ious else 0

# def calculate_pixel_accuracy(gt_mask, pred_mask):
#     """Calculate pixel-wise accuracy."""
#     # Create a map where each unique gt label is mapped to its best matching pred label
#     gt_labels = np.unique(gt_mask)
#     pred_labels = np.unique(pred_mask)
    
#     # Skip background (0) if present
#     if 0 in gt_labels:
#         gt_labels = gt_labels[gt_labels != 0]
#     if 0 in pred_labels:
#         pred_labels = pred_labels[pred_labels != 0]
    
#     # For each gt label, find the pred label that has the most overlap
#     label_mapping = {}
#     for gt_label in gt_labels:
#         gt_segment = (gt_mask == gt_label)
#         best_overlap = 0
#         best_pred_label = 0
        
#         for pred_label in pred_labels:
#             pred_segment = (pred_mask == pred_label)
#             overlap = np.logical_and(gt_segment, pred_segment).sum()
            
#             if overlap > best_overlap:
#                 best_overlap = overlap
#                 best_pred_label = pred_label
        
#         label_mapping[gt_label] = best_pred_label
    
#     # Create a remapped prediction mask
#     remapped_pred = np.zeros_like(pred_mask)
#     for gt_label, pred_label in label_mapping.items():
#         mask = (pred_mask == pred_label)
#         remapped_pred[mask] = gt_label
    
#     # Calculate pixel accuracy
#     correct = (gt_mask == remapped_pred).sum()
#     total = gt_mask.size
    
#     return correct / total

# def calculate_boundary_f1(gt_mask, pred_mask, tolerance=2):
#     """Calculate boundary F1 score with tolerance."""
#     # Find boundaries
#     gt_boundary = find_boundaries(gt_mask, mode='thick')
#     pred_boundary = find_boundaries(pred_mask, mode='thick')
    
#     # Calculate distance transforms
#     gt_distance = distance_transform_edt(~gt_boundary)
#     pred_distance = distance_transform_edt(~pred_boundary)
    
#     # Calculate precision and recall
#     precision = (pred_boundary & (gt_distance <= tolerance)).sum() / pred_boundary.sum() if pred_boundary.sum() > 0 else 0
#     recall = (gt_boundary & (pred_distance <= tolerance)).sum() / gt_boundary.sum() if gt_boundary.sum() > 0 else 0
    
#     # Calculate F1 score
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
#     return f1, precision, recall

# def calculate_segmentation_covering(gt_mask, pred_mask):
#     """Calculate segmentation covering metric."""
#     gt_labels = np.unique(gt_mask)
#     if 0 in gt_labels:  # Skip background
#         gt_labels = gt_labels[gt_labels != 0]
    
#     total_pixels = np.sum(gt_mask > 0)
#     covering = 0
    
#     for gt_label in gt_labels:
#         gt_segment = (gt_mask == gt_label)
#         segment_size = np.sum(gt_segment)
        
#         # Find best matching segment in prediction
#         best_iou = 0
#         pred_labels = np.unique(pred_mask)
#         if 0 in pred_labels:  # Skip background
#             pred_labels = pred_labels[pred_labels != 0]
        
#         for pred_label in pred_labels:
#             pred_segment = (pred_mask == pred_label)
            
#             # Calculate IoU
#             intersection = np.logical_and(gt_segment, pred_segment).sum()
#             union = np.logical_or(gt_segment, pred_segment).sum()
            
#             if union > 0:
#                 iou = intersection / union
#                 best_iou = max(best_iou, iou)
        
#         # Weight by segment size
#         covering += segment_size * best_iou
    
#     # Normalize by total pixels
#     return covering / total_pixels if total_pixels > 0 else 0

# def evaluate_segmentation(gt_path, pred_path, visualize=False):
#     """Evaluate segmentation by calculating multiple metrics."""
#     # Load masks
#     gt_mask = load_segmentation_mask(gt_path)
#     pred_mask = load_segmentation_mask(pred_path)
    
#     # Normalize masks to have consistent labeling
#     gt_norm, pred_norm = normalize_masks(gt_mask, pred_mask, visualization=visualize)
    
#     # Calculate metrics with normalized masks
#     iou = calculate_optimal_iou(gt_norm, pred_norm)
#     accuracy = calculate_pixel_accuracy(gt_norm, pred_norm)
#     f1, precision, recall = calculate_boundary_f1(gt_norm, pred_norm)
#     covering = calculate_segmentation_covering(gt_norm, pred_norm)
    
#     # Visualize if needed
#     if visualize:
#         fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
#         # Display ground truth
#         axs[0].imshow(gt_mask, cmap='nipy_spectral')
#         axs[0].set_title('Ground Truth')
#         axs[0].axis('off')
        
#         # Display prediction
#         axs[1].imshow(pred_mask, cmap='nipy_spectral')
#         axs[1].set_title('Prediction')
#         axs[1].axis('off')
        
#         # Display boundaries
#         gt_boundary = find_boundaries(gt_mask, mode='thick')
#         pred_boundary = find_boundaries(pred_mask, mode='thick')
        
#         boundary_img = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
#         boundary_img[gt_boundary, 0] = 255  # Red for ground truth
#         boundary_img[pred_boundary, 1] = 255  # Green for prediction
#         boundary_img[gt_boundary & pred_boundary, 2] = 255  # Blue for overlap
        
#         axs[2].imshow(boundary_img)
#         axs[2].set_title('Boundaries (Red=GT, Green=Pred, Blue=Overlap)')
#         axs[2].axis('off')
        
#         plt.tight_layout()
#         plt.show()
    
#     metrics = {
#         'IoU': iou,
#         'Pixel Accuracy': accuracy,
#         'Boundary F1': f1,
#         'Boundary Precision': precision,
#         'Boundary Recall': recall,
#         'Segmentation Covering': covering
#     }
    
#     return metrics

# def evaluate_directory(gt_dir, pred_dir, output_file=None):
#     """Evaluate all segmentation masks in a directory."""
#     # Get list of files
#     gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg'))])
    
#     results = []
    
#     for gt_file in tqdm(gt_files):
#         # Find corresponding prediction file
#         pred_file = gt_file  # Assuming same filename
        
#         gt_path = os.path.join(gt_dir, gt_file)
#         pred_path = os.path.join(pred_dir, pred_file)
        
#         # Skip if prediction doesn't exist
#         if not os.path.exists(pred_path):
#             print(f"Warning: No prediction found for {gt_file}")
#             continue
        
#         # Evaluate
#         metrics = evaluate_segmentation(gt_path, pred_path)
#         metrics['filename'] = gt_file
#         results.append(metrics)
    
#     # Calculate overall metrics
#     overall = {key: np.mean([r[key] for r in results]) for key in results[0] if key != 'filename'}
    
#     # Print overall results
#     print("\nOverall metrics:")
#     for key, value in overall.items():
#         print(f"{key}: {value:.4f}")
    
#     # Save to file if specified
#     if output_file:
#         import pandas as pd
#         df = pd.DataFrame(results)
#         df.to_csv(output_file, index=False)
#         print(f"Results saved to {output_file}")
    
#     return overall, results

# # Example usage
# if __name__ == "__main__":
#     # For single image evaluation
#     gt_path = "/media/hdd2/users/haowei/Dataset/Replica/office0/instance_colors/instance_color000000.png"
#     pred_path = "/media/hdd2/users/haowei/Dataset/sam2/office0/mask/frame000000.png"
    
#     metrics = evaluate_segmentation(gt_path, pred_path, visualize=True)
#     print("Segmentation metrics:")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}")
    
#     # For directory evaluation
#     # gt_dir = "path/to/gt_masks"
#     # pred_dir = "path/to/sam2_masks"
#     # evaluate_directory(gt_dir, pred_dir, "evaluation_results.csv")

import pickle
import pandas as pd

# Load the pickle file
with open('object_color_mapping.pkl', 'rb') as f:
    data = pickle.load(f)

# If data is a dictionary
if isinstance(data, dict):
    # Convert to DataFrame
    df = pd.DataFrame(list(data.items()), columns=['Key', 'Value'])
    
    # For complex values like lists or nested dictionaries
    # Convert them to string for CSV compatibility
    if df['Value'].apply(lambda x: isinstance(x, (list, dict))).any():
        df['Value'] = df['Value'].apply(str)
    
    # Save to CSV
    df.to_csv('object_color_mapping.csv', index=False)
    print("Converted to CSV successfully!")

# If data is already a DataFrame
elif isinstance(data, pd.DataFrame):
    data.to_csv('object_color_mapping.csv', index=False)
    print("Converted to CSV successfully!")

else:
    print("Data format not suitable for direct CSV conversion")

