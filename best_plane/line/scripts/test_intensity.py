import os

import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def mean_intensity(image):
    return np.mean(image)


def median_intensity(image):
    return np.median(image)


def intensity_variance(image):
    return np.var(image)


def intensity_range(image):
    return np.max(image) - np.min(image)


def histogram_skewness(image):
    return skew(image.flatten())


def histogram_kurtosis(image):
    return kurtosis(image.flatten())


def main():
    # Load the CSV file with bounding boxes and class information
    df = pd.read_csv(
        '/home/dorozco/best_depth/datasets/office_3patients_old_wv/500_lines_500_ws/organized/intensity/use_4_intensity_png.csv')

    # Initialize a list to store results
    results = []

    # Process each frame that has detections
    for frame_name in df['frame_name'].unique():
        # Read the image
        frame_path = (
            os.path.join('/home/dorozco/best_depth/datasets/office_3patients_old_wv/500_lines_500_ws/test/images/',
                         frame_name))
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

            # Calculate intensity measures
            mean_int = mean_intensity(bbox_region)
            median_int = median_intensity(bbox_region)
            variance_int = intensity_variance(bbox_region)
            range_int = intensity_range(bbox_region)
            skewness_int = histogram_skewness(bbox_region)
            kurtosis_int = histogram_kurtosis(bbox_region)

            # Append the result
            results.append({
                'frame_name': frame_name,
                'class': bbox['class'],
                'bbox_id': bbox_id,
                'mean_intensity': mean_int,
                'median_intensity': median_int,
                'intensity_variance': variance_int,
                'intensity_range': range_int,
                'histogram_skewness': skewness_int,
                'histogram_kurtosis': kurtosis_int
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(
        '/home/dorozco/best_depth/datasets/office_3patients_old_wv/500_lines_500_ws/organized/intensity/results/use_4_intensity.csv',
        index=False)


if __name__ == "__main__":
    main()
