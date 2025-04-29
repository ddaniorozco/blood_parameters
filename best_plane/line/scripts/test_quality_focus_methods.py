import cv2
import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import os


def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def tenengrad(image):
    Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    return np.sqrt(Gx ** 2 + Gy ** 2).mean()


def brenner_gradient(image):
    shifted = np.roll(image, 1, axis=1)
    return np.sum((image - shifted) ** 2)


def glcm_features(image):
    glcm = graycomatrix(image, distances=[5], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0] # este
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, correlation, energy, homogeneity]


def lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    return hist.tolist()


def edge_density(image):
    edges = cv2.Canny(image, 50, 150)
    return np.mean(edges)


def hog_features(image):
    image_resized = cv2.resize(image, (64, 128))
    hog_desc = cv2.HOGDescriptor()
    h = hog_desc.compute(image_resized)
    return h.flatten().tolist()


def fourier_features(image):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    return magnitude_spectrum.flatten().mean()


def sum_modified_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.sum(np.abs(laplacian - image))


def normalized_gray_level_variance(image): # este
    mean_intensity = np.mean(image)
    variance = np.var(image)
    return variance / mean_intensity if mean_intensity != 0 else 0


def main():
    # Load the CSV file with bounding boxes and class information
    df = pd.read_csv('/home/dorozco/best_depth/datasets/office_3patients_old_wv/500_lines_500_ws/merged.csv')

    # Initialize a list to store results
    results = []

    # Process each frame that has detections
    for frame_name in df['frame_name'].unique():
        # Read the image
        frame_path = os.path.join('/home/dorozco/best_depth/datasets/office_3patients_old_wv/500_lines_500_ws/test/images/', frame_name)
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
            laplacian_var = laplacian_variance(bbox_region)
            tenengrad_val = tenengrad(bbox_region)
            brenner_val = brenner_gradient(bbox_region)
            glcm_vals = glcm_features(bbox_region)
            lbp_vals = lbp_features(bbox_region)
            edge_density_val = edge_density(bbox_region)
            hog_vals = hog_features(bbox_region)
            fourier_val = fourier_features(bbox_region)
            sml_val = sum_modified_laplacian(bbox_region)
            nglv_val = normalized_gray_level_variance(bbox_region)

            # Append the result
            results.append({
                'frame_name': frame_name,
                'class ': bbox['class'],
                'bbox_id': bbox_id,
                'laplacian_variance': laplacian_var,
                'tenengrad': tenengrad_val,
                'brenner_gradient': brenner_val,
                'glcm_contrast': glcm_vals[0],
                'glcm_correlation': glcm_vals[1],
                'glcm_energy': glcm_vals[2],
                'glcm_homogeneity': glcm_vals[3],
                'lbp': lbp_vals,
                'edge_density': edge_density_val,
                'hog': hog_vals,
                'fourier': fourier_val,
                'sml': sml_val,
                'nglv': nglv_val
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    (results_df.to_csv
     ('/home/dorozco/best_depth/datasets/office_3patients_old_wv/500_lines_500_ws/focus_quality_results.csv',
      index=False))


if __name__ == "__main__":
    main()
