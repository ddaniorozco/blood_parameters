import pandas as pd
import re

# Function to extract the recording ID from the frame name
def extract_recording_id(frame_name):
    match = re.match(r'line_\w+_(\d+)_\w+_\w+_\w+_\d', frame_name)
    return match.group(1) if match else None

# Function to check if two bounding boxes are close enough to be considered the same
def are_bboxes_similar(bbox1, bbox2, tolerance=20):
    return abs(bbox1[3] - bbox2[3]) <= tolerance and abs(bbox1[5] - bbox2[5]) <= tolerance

# Load the CSV file
df = pd.read_csv('/home/dorozco/best_depth/datasets/line_sliced_frames/yolo_detections/sorted_dets_yolov_640_baseline_yolov8s_do0_line_sliced_frames_test.csv')

# Extract recording ID and frame number
df['recording_id'] = df['frame_name'].apply(extract_recording_id)
df['frame_number'] = df['frame_name'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))

# Sort by recording_id and frame_number
df.sort_values(by=['recording_id', 'frame_number'], inplace=True)

# Group by recording_id
grouped = df.groupby('recording_id')

filtered_rows = []

# Process each recording separately
for recording_id, group in grouped:
    group = group.reset_index(drop=True)
    bbox_counts = {}

    # Iterate over frames in the group
    for i in range(len(group) - 2):
        frame1 = group.iloc[[i]]
        frame2 = group.iloc[[i + 1]]
        frame3 = group.iloc[[i + 2]]

        # Check for bbox continuity across three consecutive frames
        for bbox1 in frame1.itertuples(index=False, name=None):
            for bbox2 in frame2.itertuples(index=False, name=None):
                for bbox3 in frame3.itertuples(index=False, name=None):
                    if are_bboxes_similar(bbox1, bbox2) and are_bboxes_similar(bbox2, bbox3):
                        bbox_key = (bbox1[3], bbox1[5])  # (left, width)
                        if bbox_key in bbox_counts:
                            bbox_counts[bbox_key] += 1
                        else:
                            bbox_counts[bbox_key] = 1

    if recording_id == '900001670':
        pass
    # Find the bbox that appears the most in consecutive frames
    if bbox_counts:
        most_frequent_bbox = max(bbox_counts, key=bbox_counts.get)
        for i in range(len(group)):
            frame = group.iloc[[i]]
            for bbox in frame.itertuples(index=False, name=None):
                if abs(bbox[3] - most_frequent_bbox[0]) <= 20 and abs(bbox[5] - most_frequent_bbox[1]) <= 20:
                    filtered_rows.append(frame.iloc[0].to_dict())

# Convert filtered rows to DataFrame and drop duplicates
filtered_df = pd.DataFrame(filtered_rows).drop_duplicates()
# Save the filtered rows to a new CSV file
filtered_output_path = '/home/dorozco/best_depth/datasets/line_sliced_frames/yolo_detections/results_longest_bbox/results.csv'
filtered_df.to_csv(filtered_output_path, index=False)

print(f"Filtered CSV saved to {filtered_output_path}")
