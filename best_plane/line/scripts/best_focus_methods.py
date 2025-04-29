import os

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

from image_enhance.wide.src import filtering_utils as filters


def glcm_features(image, distances=[1, 5, 10, 15], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    contrasts = []
    correlations = []
    energies = []
    homogeneities = []

    for distance in distances:
        for angle in angles:
            glcm = graycomatrix(image, distances=[distance], angles=[angle], symmetric=True, normed=True)
            contrasts.append(graycoprops(glcm, 'contrast')[0, 0])
            correlations.append(graycoprops(glcm, 'correlation')[0, 0])
            energies.append(graycoprops(glcm, 'energy')[0, 0])
            homogeneities.append(graycoprops(glcm, 'homogeneity')[0, 0])

    # Compute statistical measures (mean, std) for each property
    contrast_mean = np.mean(contrasts)
    contrast_std = np.std(contrasts)
    correlation_mean = np.mean(correlations)
    correlation_std = np.std(correlations)
    energy_mean = np.mean(energies)
    energy_std = np.std(energies)
    homogeneity_mean = np.mean(homogeneities)
    homogeneity_std = np.std(homogeneities)

    return [contrast_mean, contrast_std, correlation_mean, correlation_std, energy_mean, energy_std, homogeneity_mean,
            homogeneity_std]


def edge_density(image):
    image = (image / filters.gauss_blur(image.astype('float'), 101, 51))
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = filters.apply_clahe(image)
    image = cv2.medianBlur(image, 5)
    sobel_x1 = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y1 = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)
    abs_sobel_x1 = cv2.convertScaleAbs(sobel_x1)
    abs_sobel_y1 = cv2.convertScaleAbs(sobel_y1)
    sobel_im1 = cv2.addWeighted(abs_sobel_x1, 0.5, abs_sobel_y1, 0.5, 25)
    _, threshold_im1 = cv2.threshold(sobel_im1, 35, 255, cv2.THRESH_BINARY)

    # edges = cv2.Canny(image, 50, 150)
    return np.mean(threshold_im1)


def preprocess_image(image):
    # Adaptive histogram equalization
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    image_equalized = cv2.equalizeHist(image_blurred)
    return image_equalized


def normalized_gray_level_variance(image):
    # Preprocess the image
    image = preprocess_image(image)

    mean_intensity = np.mean(image)
    variance = np.var(image)

    # Add a small constant to the denominator to avoid division by zero
    return variance / (mean_intensity + 1e-8) if mean_intensity != 0 else 0


def main():
    # Load the CSV file with bounding boxes and class information
    df = pd.read_csv('/home/dorozco/best_depth/datasets/office_3patients_old_wv/500_lines_500_ws/merged.csv')

    # Initialize a list to store results
    results = []

    # Process each frame that has detections
    for frame_name in df['frame_name'].unique():
        # Read the image
        frame_path = os.path.join(
            '/home/dorozco/best_depth/datasets/office_3patients_old_wv/500_lines_500_ws/test/images/', frame_name)
        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        # Check if image is loaded correctly
        if image is None:
            print(f"Failed to load {frame_path}")
            continue

        # Get bounding boxes for this frame
        frame_bboxes = df[df['frame_name'] == frame_name]

        # Process each bounding box
        for bbox_id, bbox in frame_bboxes.iterrows():
            left = int(bbox['left'])
            top = int(bbox['top'])
            width = int(bbox['width'])
            height = int(bbox['height'])

            # Extract the bounding box region
            bbox_region = image[top:top + height, left:left + width]

            # Calculate focus quality measures
            glcm_vals = glcm_features(bbox_region)
            nglv_val = normalized_gray_level_variance(bbox_region)
            edge_density_val = edge_density(bbox_region)

            # Append the result
            results.append({
                'frame_name': frame_name,
                'class': bbox['class'],
                'bbox_id': bbox_id,
                'glcm_correlation': glcm_vals[1],
                'nglv': nglv_val,
                'edge_density': edge_density_val
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    (results_df.to_csv
     ('/home/dorozco/best_depth/datasets/office_3patients_old_wv/500_lines_500_ws/best_focus_quality_results.csv',
      index=False))


if __name__ == "__main__":
    main()
