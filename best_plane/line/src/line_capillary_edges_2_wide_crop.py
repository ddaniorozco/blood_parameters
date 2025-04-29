import pandas as pd
import numpy as np

# Read the CSV files
line_csv = pd.read_csv('/home/dorozco/best_depth/detections/wide_and_line_frames/wide_line_z_scan_2024_07_07/capillaries_extrapolation/best_line_wide_crop_frames.csv')
wide_csv = pd.read_csv('/home/dorozco/best_depth/datasets/wide_and_line_frames/wide_line_z_scan_2024_07_07/line_position.csv')

# Image sizes
line_image_size = (500, 1280)
wide_image_size = (1216, 1368)
wide_crop_image_size = (360, 360)



def map_line_to_wide(line_start_line, line_width_line, line_image_size, line_begin_x_wide, line_end_x_wide, line_begin_y_wide, line_end_y_wide):
    line_end_line = line_start_line + line_width_line

    # Calculate the relative positions in line_image
    rel_line_start = line_start_line / line_image_size[1]
    rel_line_end = line_end_line / line_image_size[1]
    inverted_rel_line_start = 1 - rel_line_end
    inverted_rel_line_end = 1 - rel_line_start

    # Map the relative positions to wide_image using provided coordinates
    wide_line_start_x = line_begin_x_wide + inverted_rel_line_start * (line_end_x_wide - line_begin_x_wide)
    wide_line_end_x = line_begin_x_wide + inverted_rel_line_end * (line_end_x_wide - line_begin_x_wide)
    wide_line_start_y = line_begin_y_wide + inverted_rel_line_start * (line_end_y_wide - line_begin_y_wide)
    wide_line_end_y = line_begin_y_wide + inverted_rel_line_end * (line_end_y_wide - line_begin_y_wide)

    return (wide_line_start_x, wide_line_start_y), (wide_line_end_x, wide_line_end_y)


# Function to normalize and map coordinates from wide_image to wide_crop_image
def map_wide_to_crop(coords, crop_boundaries, wide_image_size, wide_crop_image_size):
    left, top, right, bottom = crop_boundaries
    crop_width = right - left
    crop_height = bottom - top

    mapped_coords = []
    for coord in coords:
        wide_x, wide_y = coord

        # Normalize the coordinates relative to the wide_image_size
        norm_x = wide_x / wide_image_size[1]
        norm_y = wide_y / wide_image_size[0]

        # Map to the crop boundaries
        crop_x = (norm_x * wide_image_size[1] - left) / crop_width * wide_crop_image_size[1]
        crop_y = (norm_y * wide_image_size[0] - top) / crop_height * wide_crop_image_size[0]
        mapped_coords.append((crop_x, crop_y))

    return mapped_coords

results = []

# Iterate through the line_csv and match with wide_csv
for _, line_row in line_csv.iterrows():
    recording_id = line_row['recording_id']
    wide_image_name = line_row['wide_image_name']

    # Extract crop boundaries from the wide_image_name
    parts = wide_image_name.split('_')
    left, top, right, bottom = int(parts[-4]), int(parts[-3]), int(parts[-2]), int(parts[-1])

    # Filter the wide_csv for the current recording_id
    wide_rows = wide_csv[wide_csv['recording_id'] == recording_id]

    for _, wide_row in wide_rows.iterrows():
        # Perform the mapping from line_image to wide_image
        wide_coords = map_line_to_wide(
            line_row['left'], line_row['width'], line_image_size, wide_image_size,
            wide_row['line_begin_x'], wide_row['line_end_x'],
            wide_row['line_begin_y'], wide_row['line_end_y']
        )

        # Perform the mapping from wide_image to wide_crop_image for the line coordinates
        crop_coords = map_wide_to_crop(wide_coords, (left, top, right, bottom), wide_image_size, wide_crop_image_size)

        # Map the additional coordinates from wide_csv to wide_crop_image
        additional_coords = [
            (wide_row['line_center_x'], wide_row['line_center_y']),
            (wide_row['line_begin_x'], wide_row['line_begin_y']),
            (wide_row['line_end_x'], wide_row['line_end_y'])
        ]
        mapped_additional_coords = map_wide_to_crop(additional_coords, (left, top, right, bottom), wide_image_size, wide_crop_image_size)

        result = line_row.to_dict()
        result.update({
            'start_coord': crop_coords[0],
            'end_coord': crop_coords[1],
            'line_center_coord': mapped_additional_coords[0],
            'line_begin_coord': mapped_additional_coords[1],
            'line_end_coord': mapped_additional_coords[2]
        })
        results.append(result)

results_df = pd.DataFrame(results)
results_df.to_csv('/home/dorozco/best_depth/detections/wide_and_line_frames/wide_line_z_scan_2024_07_07/capillaries_extrapolation/final/wide_crop_capillary_coords.csv', index=False)

print("Mapping completed and saved")
