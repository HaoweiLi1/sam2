import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def blend_images(rgb_img, mask_img, alpha):
    """
    Blend RGB image with a mask using specified alpha value.
    
    Args:
        rgb_img: Original RGB image
        mask_img: Mask image (SAM2)
        alpha: Blending factor (0.0 to 1.0)
        
    Returns:
        Blended image
    """
    # Ensure mask is same size as rgb image
    if rgb_img.shape[:2] != mask_img.shape[:2]:
        mask_img = cv2.resize(mask_img, (rgb_img.shape[1], rgb_img.shape[0]))
    
    # Convert grayscale mask to RGB if needed
    if len(mask_img.shape) == 2 or mask_img.shape[2] == 1:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    
    # Blend images
    blended = cv2.addWeighted(rgb_img, 1.0 - alpha, mask_img, alpha, 0)
    return blended

def process_scene(scene_name, rgb_base_path, mask_base_path, output_base_path, alpha):
    """
    Process a single scene with the specified alpha value.
    
    Args:
        scene_name: Name of the scene (e.g., 'office0')
        rgb_base_path: Base path for RGB images
        mask_base_path: Base path for SAM2 masks
        output_base_path: Base path for output blended images
        alpha: Blending factor
    """
    # Construct paths for this scene
    rgb_path = os.path.join(rgb_base_path, scene_name, "frames")
    mask_path = os.path.join(mask_base_path, scene_name, "mask_sam2")
    output_path = os.path.join(output_base_path, scene_name, f"blender_sam2_{alpha:.1f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get list of files in RGB directory
    rgb_files = [f for f in os.listdir(rgb_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing scene {scene_name} with alpha {alpha:.1f} - {len(rgb_files)} images")
    
    # Process each file
    for rgb_file in tqdm(rgb_files):
        # Construct file paths
        rgb_file_path = os.path.join(rgb_path, rgb_file)
        
        # Determine mask file name (might have different extension)
        mask_file_base = os.path.splitext(rgb_file)[0]
        potential_mask_files = [
            f"{mask_file_base}.png",
            f"{mask_file_base}.jpg",
            f"{mask_file_base}.jpeg"
        ]
        
        # Find matching mask file
        mask_file_path = None
        for potential_file in potential_mask_files:
            full_path = os.path.join(mask_path, potential_file)
            if os.path.exists(full_path):
                mask_file_path = full_path
                break
        
        # Skip if mask doesn't exist
        if mask_file_path is None:
            print(f"Warning: No matching mask found for {rgb_file}")
            continue
        
        # Output file path
        output_file_path = os.path.join(output_path, rgb_file)
        
        # Read images
        rgb_img = cv2.imread(rgb_file_path)
        mask_img = cv2.imread(mask_file_path)
        
        # Skip if either image couldn't be read
        if rgb_img is None or mask_img is None:
            print(f"Warning: Could not read {rgb_file} or its mask")
            continue
            
        # Blend images
        blended_img = blend_images(rgb_img, mask_img, alpha)
        
        # Save blended image
        cv2.imwrite(output_file_path, blended_img)

def main():
    parser = argparse.ArgumentParser(description="Blend RGB images with SAM2 masks")
    parser.add_argument("--rgb_base", type=str, default="/media/hdd2/users/haowei/Replica",
                        help="Base path for RGB images")
    parser.add_argument("--mask_base", type=str, default="/media/hdd2/users/haowei/Replica",
                        help="Base path for SAM2 masks")
    parser.add_argument("--output_base", type=str, default="/media/hdd2/users/haowei/Replica",
                        help="Base path for output blended images")
    parser.add_argument("--alpha_start", type=float, default=0.1,
                        help="Starting alpha value")
    parser.add_argument("--alpha_end", type=float, default=0.9,
                        help="Ending alpha value")
    parser.add_argument("--alpha_step", type=float, default=0.1,
                        help="Alpha step size")
    parser.add_argument("--scenes", type=str, default="office0,office1,office2,office3,office4,room0,room1,room2",
                        help="Comma-separated list of scenes to process")
    
    args = parser.parse_args()
    
    # Generate alpha values
    alpha_values = np.arange(args.alpha_start, args.alpha_end + 0.001, args.alpha_step)
    
    # Get list of scenes
    scenes = args.scenes.split(",")
    
    # Process each scene with each alpha value
    for scene in scenes:
        for alpha in alpha_values:
            process_scene(scene, args.rgb_base, args.mask_base, args.output_base, alpha)
    
    print("Blending complete!")

if __name__ == "__main__":
    main()