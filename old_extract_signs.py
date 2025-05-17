import cv2
import numpy as np
import os
import kagglehub
import glob

def extract_sign_with_bounding_box(image_path, output_dir="extracted_signs"):
    """
    Extract signs from an image using bounding box detection and save as JPG
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save extracted signs
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return 0
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area to find potential signs
    min_area = 500  # Minimum area threshold
    sign_count = 0
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > min_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract sign using the bounding box
            sign = image[y:y+h, x:x+w]
            
            # Save the extracted sign as JPG
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"sign_{base_filename}_{i}.jpg")
            cv2.imwrite(output_path, sign)
            sign_count += 1
            
            # Draw bounding box on the original image (for visualization)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Save the annotated image as JPG regardless of input format
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    annotated_path = os.path.join(output_dir, f"annotated_{base_filename}.jpg")
    cv2.imwrite(annotated_path, image)
    
    print(f"Extracted {sign_count} potential signs from {image_path}")
    print(f"Saved annotated image to {annotated_path}")
    
    return sign_count

def main():
    # Download the German Traffic Sign Detection Benchmark dataset
    dataset_path = kagglehub.dataset_download("safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb")
    print(f"Dataset downloaded to {dataset_path}")
    
    # Specifically target the TrainIJCNN2013 folder containing PPM files
    target_folder = os.path.join(dataset_path, "TrainIJCNN2013", "TrainIJCNN2013")
    
    # Check if the target folder exists
    if not os.path.exists(target_folder):
        print(f"ERROR: Target folder not found: {target_folder}")
        print("Exploring dataset directory structure to locate the PPM files:")
        for root, dirs, files in os.walk(dataset_path):
            level = root.replace(dataset_path, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            if level < 3:  # Limit depth for readability
                for file in files[:5]:  # Show at most 5 files per directory
                    print(f"{indent}    {file}")
                if len(files) > 5:
                    print(f"{indent}    ... ({len(files) - 5} more files)")
        
        # Try to find any folders with 'train' in the name (case insensitive)
        train_folders = []
        for root, dirs, files in os.walk(dataset_path):
            for dir_name in dirs:
                if 'train' in dir_name.lower():
                    train_folders.append(os.path.join(root, dir_name))
        
        if train_folders:
            print("\nPotential training folders found:")
            for folder in train_folders:
                print(f"  - {folder}")
            
            # Try the first matching folder
            target_folder = train_folders[0]
            print(f"\nAttempting to use: {target_folder}")
    
    # Find all PPM files in the target directory
    ppm_files = glob.glob(os.path.join(target_folder, "*.ppm"))
    
    if not ppm_files:
        print(f"No PPM files found in {target_folder}")
        print("Trying to find PPM files anywhere in the dataset...")
        ppm_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith('.ppm'):
                    ppm_files.append(os.path.join(root, file))
    
    # Sort the PPM files to ensure consistent selection
    ppm_files.sort()
    
    print(f"Found {len(ppm_files)} PPM files.")
    
    if len(ppm_files) == 0:
        print("ERROR: No PPM files found in the dataset.")
        return
    
    # Select the first 10 images (or all if less than 10)
    selected_images = ppm_files[:10]
    print(f"Selected {len(selected_images)} images for processing:")
    for img in selected_images:
        print(f"  - {img}")
    
    # Set output directory
    output_dir = "extracted_signs"
    
    # Process each of the selected images
    total_signs = 0
    for image_path in selected_images:
        print(f"Processing: {image_path}")
        signs = extract_sign_with_bounding_box(image_path, output_dir)
        if signs is not None:
            total_signs += signs
    
    print(f"Total signs extracted: {total_signs}")

# Add this to call main() when the script is run
if __name__ == "__main__":
    main()