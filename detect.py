import time
import os
import pandas as pd
from deepface import DeepFace
import cv2
import numpy as np
import sys
# Function to find matches and log results
def find_and_log(img_dir, db_path, model_name, distance_metric, threshold, output_dir):
    all_results = pd.DataFrame()  # Reset results DataFrame for each trial

    # Get a list of all image files in the source directory
    img_files = [file for file in os.listdir(img_dir) if file.endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)

        # Perform face recognition with current parameters
        results = DeepFace.find(
            img_path=img_path, 
            db_path=db_path, 
            enforce_detection=False, 
            model_name=model_name, 
            distance_metric=distance_metric, 
            threshold=threshold)

        if not results[0].empty:
            # Add necessary information to results DataFrame
            results[0]['source_image_name'] = img_file
            results[0]['model_name'] = model_name
            results[0]['distance_metric'] = distance_metric
            results[0]['threshold'] = threshold
            all_results = pd.concat([all_results, results[0]], ignore_index=True)

    # Function to extract digits from strings
    def extract_digits(s):
        return ''.join(filter(str.isdigit, s))

    # Compare the digits in 'identity' and 'source_image_name', assign 1 if they match, else 0
    all_results['match'] = all_results.apply(lambda row: 1 if extract_digits(row['identity']) == extract_digits(row['source_image_name']) else 0, axis=1)

    # Minimum distance and groupby to select the best match for each source image
    idx = all_results.groupby('source_image_name')['distance'].idxmin()
    best_matches = all_results.loc[idx]

    # Save best matches to CSV
    best_matches.to_csv('matches_results.csv', index=False)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Resize dimensions
    desired_width = 300
    desired_height = 300

    # Function to resize images while maintaining aspect ratio
    def resize_image(img, width, height):
        aspect_ratio = img.shape[1] / img.shape[0]
        if img.shape[1] > img.shape[0]:
            new_width = width
            new_height = int(width / aspect_ratio)
        else:
            new_height = height
            new_width = int(height * aspect_ratio)
        return cv2.resize(img, (new_width, new_height))

    # Concatenate matching images and save them to the labeled_images folder
    for _, row in best_matches.iterrows():
        source_img_path = os.path.join(img_dir, row['source_image_name'])
        match_img_path = row['identity']

        # Load images using OpenCV
        source_img = cv2.imread(source_img_path)
        match_img = cv2.imread(match_img_path)

        if source_img is None or match_img is None:
            print(f"Warning: Could not read images for {row['source_image_name']}. Skipping.")
            continue

        # Resize images
        source_img_resized = resize_image(source_img, desired_width, desired_height)
        match_img_resized = resize_image(match_img, desired_width, desired_height)

        # Create a new blank image with the desired dimensions
        width = source_img_resized.shape[1] + match_img_resized.shape[1]
        height = max(source_img_resized.shape[0], match_img_resized.shape[0])
        combined_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Place the source and matching images horizontally next to each other
        combined_img[0:source_img_resized.shape[0], 0:source_img_resized.shape[1]] = source_img_resized
        combined_img[0:match_img_resized.shape[0], source_img_resized.shape[1]:] = match_img_resized

        # Save the concatenated image to the labeled_images folder
        combined_filename = os.path.join(output_dir, f"{os.path.splitext(row['source_image_name'])[0]}_match.jpg")
        cv2.imwrite(combined_filename, combined_img)

# Setup parameters
models = ["Facenet512"]
metrics = ["euclidean_l2"]
thresholds = [1.3]
db_path = "comp_label"
img_dir = "comp_img"
output_dir = "labeled_images"

# Iterate over each model, metric, and threshold combination
start = time.time()
for model in models:
    for metric in metrics:
        find_and_log(img_dir, db_path, model_name=model, distance_metric=metric, threshold=1.3, output_dir=output_dir)

end = time.time()
print(f"Executed whole script in {end - start} seconds.")

