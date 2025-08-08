# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2 # Using OpenCV for image loading/saving
import time
import glob
from tqdm import tqdm
import random # For assigning random colors
import pickle # For saving/loading object tracking data
import argparse

# --- Add SAM2 repository to path (Optional: Adjust if needed) ---
# Assumes script is run from a location where 'sam2' package is importable
# Or uncomment and set the path:
# SAM2_REPO_PATH = '/path/to/your/sam2/repo'
# sys.path.append(SAM2_REPO_PATH)
# -----------------------------------

# Check if sam2 modules are importable
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    print(f"Error importing SAM2 modules: {e}")
    print("Please ensure the 'sam2' repository directory is in your Python path or install it (`pip install -e.` inside the repo).")
    sys.exit(1)

print("SAM2 libraries imported successfully.")

# --- Configuration ---
# <<< --- YOU MUST EDIT THESE PATHS --- >>>
INPUT_IMAGE_DIR = "/media/hdd2/users/haowei/Dataset/Replica/office3/frames" # Directory containing frameXXXXXX.jpg files
OUTPUT_MASKS_DIR = "/media/hdd2/users/haowei/Dataset/sam2/office3/mask" # Directory where colored masks will be saved
SAM2_CHECKPOINT_PATH = "checkpoints/sam2.1_hiera_large.pt" # Path to the downloaded checkpoint
MODEL_CONFIG_FILE = "configs/sam2.1/sam2.1_hiera_l.yaml" # Path to the corresponding config file (relative to sam2 repo structure)
# <<< --- END OF PATHS TO EDIT --- >>>

# --- Add consistency parameters ---
# Whether to use consistent colors across frames
USE_CONSISTENT_COLORS = True
# IoU threshold for considering two masks (from different frames) to be the same object
MASK_IOU_THRESHOLD = 0.5
# File to store the object tracking data (color assignments)
COLOR_TRACKING_FILE = "object_color_mapping.pkl"

# --- Other Settings ---
IMAGE_FORMAT = "jpg" # Format of the input frames
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Auto-select device

# --- Automatic Mask Generator Parameters (Based on notebook example [2]) ---
# These parameters aim for potentially better results, especially on smaller objects,
# compared to the absolute defaults.
GENERATOR_PARAMS = {
    "points_per_side": 64,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.7,
    "stability_score_thresh": 0.92,
    "stability_score_offset": 0.7, # Note: Effect might depend on model version
    "box_nms_thresh": 0.7,
    "crop_n_layers": 1, # Enable 1 layer of cropping for potentially better small object detection
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 25, # Minimum area in pixels for a mask region to be kept (float in notebook, using int here)
    "use_m2m": True, # Specific technique mentioned in notebook
}
# --- End Configuration ---


# --- Helper Functions for Visualization and Object Tracking ---
def compute_iou(mask1, mask2):
    """
    Compute Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1 (np.ndarray): First binary mask
        mask2 (np.ndarray): Second binary mask
        
    Returns:
        float: IoU score between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def match_objects_across_frames(prev_masks, curr_masks):
    """
    Match mask objects between previous and current frame based on IoU.
    
    Args:
        prev_masks (list): List of mask dictionaries from previous frame
        curr_masks (list): List of mask dictionaries from current frame
        
    Returns:
        dict: Mapping from current mask index to previous mask index
    """
    matches = {}
    
    # If either list is empty, no matches possible
    if not prev_masks or not curr_masks:
        return matches
    
    # Create IoU matrix (rows=curr_masks, cols=prev_masks)
    iou_matrix = np.zeros((len(curr_masks), len(prev_masks)))
    
    # Compute IoU for each pair
    for i, curr_mask in enumerate(curr_masks):
        for j, prev_mask in enumerate(prev_masks):
            iou_matrix[i, j] = compute_iou(curr_mask['segmentation'], prev_mask['segmentation'])
    
    # Match each current mask to the previous mask with highest IoU above threshold
    for i in range(len(curr_masks)):
        best_match = np.argmax(iou_matrix[i])
        if iou_matrix[i, best_match] > MASK_IOU_THRESHOLD:
            matches[i] = best_match
    
    return matches

def create_colored_instance_map(masks_data, input_shape, color_map=None, prev_masks=None):
    """
    Creates an image where each detected instance mask is filled with a unique color.
    If color_map is provided, uses consistent colors across frames.

    Args:
        masks_data (list): The list of dictionaries output by SAM2AutomaticMaskGenerator.
        input_shape (tuple): The shape (height, width) of the original image.
        color_map (dict, optional): Map from object IDs to BGR colors for consistency.
        prev_masks (list, optional): List of masks from previous frame for object tracking.

    Returns:
        np.ndarray: An image (H, W, 3) with unique colors per instance mask (BGR format).
        dict: Updated color_map with any new objects assigned colors.
    """
    if not masks_data:
        return np.zeros((input_shape[0], input_shape[1], 3), dtype=np.uint8), color_map or {}

    # Sort masks by area (draw smaller masks last/on top)
    sorted_masks = sorted(masks_data, key=(lambda x: x['area']), reverse=False)

    # Create output image (BGR format for cv2.imwrite)
    output_image = np.zeros((input_shape[0], input_shape[1], 3), dtype=np.uint8)
    
    # Initialize color map if not provided
    if color_map is None:
        color_map = {}
    
    # If using consistent colors and we have previous masks, match objects
    object_matches = {}
    if USE_CONSISTENT_COLORS and prev_masks is not None:
        object_matches = match_objects_across_frames(prev_masks, sorted_masks)

    # Process each mask
    for idx, ann in enumerate(sorted_masks):
        m = ann['segmentation']  # This is a boolean mask (H, W)
        
        # Determine the color for this mask
        if USE_CONSISTENT_COLORS:
            # If this mask matches a previous one, use its color
            if idx in object_matches:
                prev_idx = object_matches[idx]
                object_id = f"obj_{prev_idx}"
                if object_id not in color_map:
                    # Should not happen, but just in case
                    color = [random.randint(30, 255) for _ in range(3)]
                    color_map[object_id] = color
            else:
                # This is a new object, assign a new color
                object_id = f"obj_{len(color_map)}"
                color = [random.randint(30, 255) for _ in range(3)]
                color_map[object_id] = color
            
            color = color_map[object_id]
        else:
            # Original behavior: generate random color
            color = [random.randint(30, 255) for _ in range(3)]
        
        # Apply color where the mask is True
        output_image[m] = color

    return output_image, color_map

def get_processed_frames(output_dir):
    """Returns a set of frame IDs that have already been processed"""
    if not os.path.exists(output_dir):
        return set()
    
    # Get list of completed mask files
    existing_files = glob.glob(os.path.join(output_dir, "*.png"))
    # Extract frame IDs from filenames
    processed_frames = set()
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        frame_id = os.path.splitext(filename)[0]  # Remove extension
        processed_frames.add(frame_id)
    
    return processed_frames

# Add this function to parse command line arguments (right before the main() function)
def parse_args():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="SAM2 Automatic Mask Generation for Image Sequences")
    
    # Add arguments for paths that can be specified via command line
    parser.add_argument("--input-dir", type=str, default=INPUT_IMAGE_DIR,
                        help=f"Directory containing frame images (default: {INPUT_IMAGE_DIR})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_MASKS_DIR,
                        help=f"Directory where colored masks will be saved (default: {OUTPUT_MASKS_DIR})")
    parser.add_argument("--checkpoint", type=str, default=SAM2_CHECKPOINT_PATH,
                        help=f"Path to the SAM2 checkpoint (default: {SAM2_CHECKPOINT_PATH})")
    parser.add_argument("--config", type=str, default=MODEL_CONFIG_FILE,
                        help=f"Path to the model config file (default: {MODEL_CONFIG_FILE})")
    
    return parser.parse_args()

# --- Main Script Logic ---
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Override the global variables with command line arguments
    global INPUT_IMAGE_DIR, OUTPUT_MASKS_DIR, SAM2_CHECKPOINT_PATH, MODEL_CONFIG_FILE, DEVICE
    INPUT_IMAGE_DIR = args.input_dir
    OUTPUT_MASKS_DIR = args.output_dir
    SAM2_CHECKPOINT_PATH = args.checkpoint
    MODEL_CONFIG_FILE = args.config
    
    # global DEVICE  # Add this line to access the global DEVICE variable
    print(f"--- SAM2 Automatic Mask Generation for Image Sequence ---")
    print(f"Input directory: {INPUT_IMAGE_DIR}")
    print(f"Output directory: {OUTPUT_MASKS_DIR}")
    print(f"Checkpoint: {SAM2_CHECKPOINT_PATH}")
    print(f"Model Config: {MODEL_CONFIG_FILE}")
    print(f"Device: {DEVICE}")
    print(f"Generator Params: {GENERATOR_PARAMS}")
    print("-" * 60)

    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA specified but not available. Falling back to CPU.")
        DEVICE = "cpu" # Update device if CUDA not found

    # --- Verify paths ---
    if not os.path.isdir(INPUT_IMAGE_DIR):
        print(f"Error: Input directory not found: {INPUT_IMAGE_DIR}")
        sys.exit(1)
    if not os.path.exists(SAM2_CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found: {SAM2_CHECKPOINT_PATH}")
        sys.exit(1)
    if not os.path.exists(MODEL_CONFIG_FILE):
         # Try to construct path relative to potential repo structure if it's not absolute
         script_dir = os.path.dirname(os.path.abspath(__file__))
         potential_cfg_path = os.path.join(os.path.dirname(script_dir), MODEL_CONFIG_FILE) # Assumes script is one level down from repo root
         if os.path.exists(potential_cfg_path):
              print(f"Found config relative to script parent: {potential_cfg_path}")
              cfg_path_to_use = potential_cfg_path
         else:
              potential_cfg_path = os.path.join(script_dir, MODEL_CONFIG_FILE) # Assumes script is in repo root
              if os.path.exists(potential_cfg_path):
                   print(f"Found config relative to script: {potential_cfg_path}")
                   cfg_path_to_use = potential_cfg_path
              else:
                   print(f"Error: Model config file not found: {MODEL_CONFIG_FILE}")
                   print("Please provide a valid absolute path or ensure it's relative to the sam2 repo structure.")
                   sys.exit(1)
    else:
         cfg_path_to_use = MODEL_CONFIG_FILE # Use the provided path if it exists

    # --- 1. Initialize Automatic Mask Generator ---
    print("Initializing SAM2 Automatic Mask Generator...")
    try:
        # Build the SAM model first
        sam_model = build_sam2(cfg_path_to_use, SAM2_CHECKPOINT_PATH, device=DEVICE)

        # Initialize the generator with the model and specified parameters
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam_model,
            **GENERATOR_PARAMS # Unpack the dictionary of parameters
        )
    except Exception as e:
        print(f"Error initializing model or generator: {e}")
        sys.exit(1)
    print("Generator initialized.")

    # --- 2. Find and Sort Input Frames ---
    image_pattern = os.path.join(INPUT_IMAGE_DIR, f"*.{IMAGE_FORMAT}")
    frame_files = sorted(glob.glob(image_pattern))

    if not frame_files:
        print(f"Error: No images found matching pattern '{image_pattern}' in directory '{INPUT_IMAGE_DIR}'")
        sys.exit(1)

    processed_frames = get_processed_frames(OUTPUT_MASKS_DIR)
    print(f"Found {len(frame_files)} frames to process.")

    # --- 3. Process Each Frame ---
    print(f"Processing frames and saving instance maps to: {OUTPUT_MASKS_DIR}...")
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

    total_start_time = time.time()
    skipped_frames = 0
    
    # Initialize object tracking variables
    color_map = {}
    prev_masks = None

    for frame_path in tqdm(frame_files, desc="Processing Frames"):
        frame_basename = os.path.basename(frame_path)
        output_filename = os.path.splitext(frame_basename)[0] + ".png"  # Get base name without extension, then add png
        frame_id = os.path.splitext(frame_basename)[0]
        output_path = os.path.join(OUTPUT_MASKS_DIR, output_filename)

        if frame_id in processed_frames:
            skipped_frames += 1
            continue
    
        try:
            # Load image using OpenCV (keeps BGR order)
            image_bgr = cv2.imread(frame_path)
            if image_bgr is None:
                print(f"Warning: Skipping invalid image file: {frame_path}")
                continue

            # Convert BGR to RGB for SAM2 model
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Generate masks for the current frame
            # Note: SAM2 expects HWC, uint8, RGB image
            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                masks_data = mask_generator.generate(image_rgb)

            # Create the colored instance map
            if masks_data:
                # Pass shape without channel: (H, W)
                instance_map_bgr, color_map = create_colored_instance_map(
                    masks_data, 
                    image_rgb.shape[:2],
                    color_map=color_map,
                    prev_masks=prev_masks
                )
                # Save the colored instance map (using BGR)
                cv2.imwrite(output_path, instance_map_bgr)
                
                # Store the current masks for the next frame
                prev_masks = masks_data
            else:
                # Save a black image if no masks were found
                black_image = np.zeros_like(image_bgr)
                cv2.imwrite(output_path, black_image)
                prev_masks = None

        except Exception as e:
            print(f"\nError processing frame {frame_path}: {e}")
            # Continue to the next frame instead of exiting
            continue
            # sys.exit(1) # Exit on first error (commented out)
    
    # Optionally save the color mapping for future use
    if USE_CONSISTENT_COLORS and color_map:
        with open(COLOR_TRACKING_FILE, 'wb') as f:
            pickle.dump(color_map, f)
        print(f"Saved object color mapping to {COLOR_TRACKING_FILE}")

    total_time = time.time() - total_start_time
    avg_time = total_time / len(frame_files) if frame_files else 0
    print("\nProcessing complete.")
    print(f"Processed {len(frame_files)} frames in {total_time:.2f} seconds ({avg_time:.3f} seconds/frame).")
    print(f"Colored instance maps saved in {OUTPUT_MASKS_DIR}")

if __name__ == "__main__":
    main()