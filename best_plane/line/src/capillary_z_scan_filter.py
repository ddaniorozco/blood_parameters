import pandas as pd
import re
from collections import defaultdict
import argparse
import os



def extract_recording_id(frame_name):
    match = re.match(r'line_\w+_(\d+)_\w+_\w+_\w+_\d', frame_name)
    return match.group(1) if match else None


def are_bboxes_similar(bbox1, bbox2, tolerance=20):
    return abs(bbox1[3] - bbox2[3]) <= tolerance and abs(bbox1[5] - bbox2[5]) <= tolerance


def main(df_path, filtered_df_path):

    df = pd.read_csv(df_path)

    # Extract recording ID and frame number
    df['recording_id'] = df['frame_name'].apply(extract_recording_id)
    df['frame_number'] = df['frame_name'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))

    # Sort by recording_id and frame_number
    df.sort_values(by=['recording_id', 'frame_number'], inplace=True)

    # Group by recording_id
    grouped = df.groupby('recording_id')
    filtered_df = pd.concat([group for name, group in grouped if len(group) > 2])
    grouped = filtered_df.groupby('recording_id')

    # group_names = [group_name for group_name, _ in grouped]
    # num_groups = len(group_names)

    result_rows = []

    # Process each recording separately
    for recording_id, group in grouped:
        group = group.reset_index(drop=True)
        bbox_consecutive_counts = defaultdict(lambda: {'current_streak': 0, 'max_streak': 0, 'frames': []})

        # Iterate over frames in the group
        for i in range(len(group) - 2):
            current_frame = group.iloc[[i]]
            next_frame = group.iloc[[i + 1]]
            after_next_frame = group.iloc[[i + 2]]

            # Check bboxes across three consecutive frames
            for bbox1 in current_frame.itertuples(index=False, name=None):
                for bbox2 in next_frame.itertuples(index=False, name=None):
                    for bbox3 in after_next_frame.itertuples(index=False, name=None):
                        if are_bboxes_similar(bbox1, bbox2) and are_bboxes_similar(bbox2, bbox3):
                            bbox_key = (bbox1[3], bbox1[5])
                            bbox_consecutive_counts[bbox_key]['current_streak'] += 3
                            bbox_consecutive_counts[bbox_key]['frames'].extend([current_frame.iloc[0].to_dict(),
                                                                                next_frame.iloc[0].to_dict(),
                                                                                after_next_frame.iloc[0].to_dict()])
                            if (bbox_consecutive_counts[bbox_key]['current_streak'] >
                                    bbox_consecutive_counts[bbox_key]['max_streak']):
                                bbox_consecutive_counts[bbox_key]['max_streak'] = (
                                    bbox_consecutive_counts)[bbox_key]['current_streak']
                        else:
                            bbox_key = (bbox1[3], bbox1[5])
                            bbox_consecutive_counts[bbox_key]['current_streak'] = 0



        # Find the maximum streak for each bbox
        max_streaks = []
        for bbox, data in bbox_consecutive_counts.items():
            if data['max_streak'] > 0:
                max_streaks.append(data['max_streak'])

        if max_streaks:
            max_streak = max(max_streaks)
            for bbox, data in bbox_consecutive_counts.items():
                if data['max_streak'] == max_streak:
                    result_rows.extend(data['frames'])

    result_df = pd.DataFrame(result_rows).drop_duplicates()
    df_name = os.path.basename(df_path).replace('.csv', '')
    result_df.to_csv(filtered_df_path + f'filtered_{df_name}.csv', index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Longest capillary in each recording")
    parser.add_argument('--df_path', type=str, required=True, help='Path to csv with detections of capillaries')
    parser.add_argument('--filtered_df_path', type=str, required=True, help='Path to save the filtered csv')

    args = parser.parse_args()

    main(df_path=args.df_path, filtered_df_path=args.filtered_df_path)


if __name__ == '__main__':
    parse_args()
