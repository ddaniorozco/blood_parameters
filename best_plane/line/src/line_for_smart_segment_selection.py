#%%
import os
import re

import pandas as pd
import torch
from loguru import logger
from skimage.color import gray2rgb
from skimage.io import imread

from capillary_classifier.line.src.inference import \
    LineCapillaryClassifierModel
from capillary_detector.line.yolo.src.inference import \
    YoloLineCapillaryDetectorModel
from dataset_utils.src.line_dataset_loader import LineDatasetLoader
from interfaces.inference.image_inference_model import ImageInferenceModel


class LineDataSSS:

    def __init__(self, line_capillary_detector: ImageInferenceModel = None,
                 line_capillary_classifier: ImageInferenceModel = None,
                 default_device: torch.device = None, half_precision=False, to_device_on_init: bool = False,
                 to_cpu_after_infer: bool = False, verbosity: int = 0):

        self.line_capillary_classifier = line_capillary_classifier
        self.line_capillary_detector = line_capillary_detector
        self.default_device = default_device
        self.half_precision = half_precision
        self.to_device_on_init = to_device_on_init
        self.to_cpu_after_infer = to_cpu_after_infer
        self.verbosity = verbosity
        self._models_init()

    def _models_init(self):
        """
        Initialize the frame classifier and capillary detector.
        """
        models = {'line_capillary_detector': (self.line_capillary_detector, YoloLineCapillaryDetectorModel),
                  'line_capillary_classifier': (self.line_capillary_classifier, LineCapillaryClassifierModel)}

        for attr, (model, model_class) in models.items():
            if model is not None:
                model.default_device = self.default_device
                model.half_precision = self.half_precision
                model.to_device_on_init = self.to_device_on_init
                model.to_cpu_after_infer = self.to_cpu_after_infer
                model.verbosity = self.verbosity
                setattr(self, attr, model)
            else:
                setattr(self, attr, model_class(
                    default_device=self.default_device,
                    half_precision=self.half_precision,
                    to_device_on_init=self.to_device_on_init,
                    to_cpu_after_infer=self.to_cpu_after_infer,
                    verbosity=self.verbosity
                ))

    @staticmethod
    def extract_recording_id(frame_name):
        match = re.match(r'line_\w+_(\d+)_\w+_\w+_\w+_\d', frame_name)
        return match.group(1) if match else None

    @staticmethod
    def are_bboxes_similar(bbox1, bbox2, tolerance=50):
        return abs(bbox1[4] - bbox2[4]) <= tolerance and abs(bbox1[6] - bbox2[6]) <= tolerance

    def _filter_consecutive_detections(self, df, min_consecutive_capillaries=3):

        # Step 1: Extract recording ID and frame number
        df['recording_id'] = df['line_image_name'].apply(self.extract_recording_id)
        df['frame_number'] = df['line_image_name'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))

        # Step 2: Sort dataframe by recording_id and frame_number
        df.sort_values(by=['recording_id', 'frame_number'], inplace=True)

        # Step 3: Group by recording_id
        grouped = df.groupby('recording_id')
        grouped_list = [group for name, group in grouped if len(group) > 0]
        if len(grouped_list) == 0:
            raise ValueError('No capillaries were detected in the line images.')
        elif len(grouped_list) == 1:
            filtered_df = grouped_list[0]
        else:
            filtered_df = pd.concat(grouped_list)
        # filtered_df = grouped_list[0] if len(grouped_list) == 1 else pd.concat(grouped_list)
        # filtered_df = pd.concat([group for name, group in grouped if len(group) > 2])
        grouped = filtered_df.groupby('recording_id')

        # Step 4: Filter consecutive bounding boxes
        result_rows = []

        for recording_id, group in grouped:
            group = group.reset_index(drop=True)
            temp_consecutive_detections = []

            for i in range(len(group) - 1):
                current_detection = group.iloc[i]
                next_detection = group.iloc[i + 1]

                if self.are_bboxes_similar(current_detection, next_detection):
                    temp_consecutive_detections.append(current_detection.to_dict())
                else:
                    if len(temp_consecutive_detections) >= min_consecutive_capillaries - 1:
                        temp_consecutive_detections.append(current_detection.to_dict())
                        result_rows.extend(temp_consecutive_detections)
                    temp_consecutive_detections = []

            # Check for the last detection in the group
            if len(temp_consecutive_detections) >= min_consecutive_capillaries - 1:
                temp_consecutive_detections.append(group.iloc[-1].to_dict())
                result_rows.extend(temp_consecutive_detections)

        # Step 5: Create DataFrame from the filtered results
        result_df = pd.DataFrame(result_rows).drop_duplicates()

        return result_df

      

    @staticmethod
    def _crop_arrays(df, detected_arrays):
        """
               Crops arrays based on detection data and bounding boxes.

               Args:
                   df (pd.DataFrame): Dataframe containing the detection data.

               Returns:
                   list: List of tuples containing frame index, sequence number, and cropped image arrays.
               """

        cropped_arrays = []
        for array, row in zip(detected_arrays, df.iterrows()):
            center_x = int((row[1][4] + row[1][6]) // 2)

            left = int(center_x - 128)
            right = int(center_x + 128)
            top = int(row[1][5])
            bottom = int(row[1][7])

            left = max(0, left)
            # _, (_, _, np_array) = array
            right = min(array.shape[1], right)
            top = max(0, top)
            bottom = min(array.shape[0], bottom)

            cropped_array = array[top:bottom, left:right]

            cropped_arrays.append(cropped_array)

        return cropped_arrays

    @staticmethod
    def save_capillary_detections_2_df(results, line_images_paths):

        df = pd.DataFrame([
            {
                'frame': results[0][i],
                'line_image_name': os.path.splitext(os.path.basename(line_images_paths[i]))[0],
                'class': int(detection[5]),
                'confidence': detection[4],
                'left': detection[0],
                'top': detection[1],
                'width': detection[2],
                'height': detection[3],
            }  # 'class': int(detection[5]),

            for i in range(len(results[0]))
            for detection in results[1][i]
        ])

        return df

    @staticmethod
    def merge_dfs(df_best_frames, frame_sync_df):

        merged_df = pd.merge(df_best_frames,
                             frame_sync_df[['line_image_name', 'wide_image_name', 'z_position', 'glass_position']],
                             how='left',
                             on='line_image_name')
        return merged_df

    def pipeline(self, dataset_path: str, device=None, min_consecutive_capillaries=0, return_all_detections=True, chunk_size=100):

        images_path = os.path.join(dataset_path, 'test', 'images')
        frame_sync_df = pd.read_csv(os.path.join(dataset_path, 'frame_sync.csv'))
        line_position_df = pd.read_csv(os.path.join(dataset_path, 'line_position.csv'))

        line_images_paths = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.startswith("line")])
        ds_line = LineDatasetLoader(dataset_path, is_slice=True, imgs_suffix='png')
        wide_images_paths = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.startswith("wide")])
        first_wide_image_rgb = gray2rgb(imread(wide_images_paths[0], as_gray=True))
        wide_crop_image_size = first_wide_image_rgb.shape

        all_filtered_dfs = []
        line_video_ids = ds_line.get_video_ids()

        for line_vid in line_video_ids:
            line_images_rgb = ds_line.read_video(line_vid)
            chunk_paths = ds_line.get_video_imgs_names(line_vid)
            line_images_rgb = [gray2rgb(line_image) for line_image in line_images_rgb]
            
            line_image_size = line_images_rgb[0].shape

            results_capillary_detection = self.line_capillary_detector.batch_infer(images_source=line_images_rgb,
                                                                                   device=device)
            df_capillary_coordinates = self.save_capillary_detections_2_df(results_capillary_detection, chunk_paths)
            if len(df_capillary_coordinates) == 0:
                logger.warning(f'No capillaries were detected in the line images of video {line_vid}.')
                continue

            filtered_df = self._filter_consecutive_detections(df_capillary_coordinates, min_consecutive_capillaries)
            indices_to_keep = filtered_df['frame'].tolist()
            line_images_rgb_filtered = [line_images_rgb[j] for j in indices_to_keep]
            line_crops_filtered = self._crop_arrays(filtered_df, line_images_rgb_filtered)

            results = self.line_capillary_classifier.batch_infer(images_source=line_crops_filtered, device=device)
            filtered_df['classification'], filtered_df['classification_probability'] = results[1]
            all_filtered_dfs.append(filtered_df)

        # Concatenate all filtered dataframes
        final_filtered_df = pd.concat(all_filtered_dfs, ignore_index=True)
        df_best_frames = self._choose_frame(final_filtered_df) if not return_all_detections else final_filtered_df
        merged_df = self.merge_dfs(df_best_frames, frame_sync_df)
        df_final = self.line_edges_2_wide_crop(line_position_df, merged_df, line_image_size, wide_crop_image_size)
        output_dir = os.path.join(dataset_path, 'df_line_wide_extrapolated')

        try:
            os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

            # Save df_final to CSV in the new directory
            df_final_path = os.path.join(output_dir, 'df_final.csv')
            df_final.to_csv(df_final_path, index=False)
        except Exception as e:
            dataset_name = os.path.basename(dataset_path)
            output_dir = f"/home/omri/Datasets/{dataset_name}/df_line_wide_extrapolated"
            os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

            # Save df_final to CSV in the new directory
            df_final_path = os.path.join(output_dir, 'df_final.csv')
            df_final.to_csv(df_final_path, index=False)

    @staticmethod
    def _choose_frame(df):
        """
        Chooses the best frame for each recording_id based on classification results and updates the DataFrame.

        Args:
            df (pd.DataFrame): Dataframe containing the classification results.
        """

        if not df.empty:
            # df['best'] = ''

            grouped = df.groupby('recording_id')

            best_rows = []

            for recording_id, group_df in grouped:
                good_rows = group_df[group_df['classification'] == 1]

                if len(good_rows) > 0:
                    if len(good_rows) == 1:
                        chosen_row = good_rows.iloc[0]
                        logger.info(f'One good frame was detected for recording {recording_id}.')
                    else:  # Take the middle one
                        mid = len(good_rows) // 2
                        chosen_row = good_rows.iloc[mid]
                        logger.info(
                            f'More than 1 good frame was detected for recording {recording_id} - taking the middle one.')
                else:
                    bad_rows = group_df[group_df['class'] == 0]
                    mid = len(bad_rows) // 2
                    chosen_row = bad_rows.iloc[mid]
                    logger.warning(
                        f'No good frames were detected for recording {recording_id} - taking the middle of the bad frames.')

                best_rows.append(chosen_row)

            best_df = pd.DataFrame(best_rows)

            return best_df
        else:
            logger.warning(f'There were 0 detections in the entire image stack')
            return df

    @staticmethod
    def line_edges_2_wide_crop(wide_csv, line_csv, line_image_size, wide_crop_image_size):

        # line_image_size = (500, 1280)
        # wide_crop_image_size = (360, 360)
        wide_image_size = (1216, 1368)


        def map_line_to_wide(line_start_line, line_end_line, line_image_size, line_begin_x_wide, line_end_x_wide,
                             line_begin_y_wide, line_end_y_wide):

            # Calculate the relative positions in line_image
            rel_line_start = line_start_line / line_image_size[1]
            rel_line_end = line_end_line / line_image_size[1]
            # inverted_rel_line_start = 1 - rel_line_end
            # inverted_rel_line_end = 1 - rel_line_start
            inverted_rel_line_start = rel_line_start
            inverted_rel_line_end = rel_line_end

            # Map the relative positions to wide_image using provided coordinates
            wide_line_start_x = line_begin_x_wide + inverted_rel_line_start * (line_end_x_wide - line_begin_x_wide)
            wide_line_end_x = line_begin_x_wide + inverted_rel_line_end * (line_end_x_wide - line_begin_x_wide)
            wide_line_start_y = line_begin_y_wide + inverted_rel_line_start * (line_end_y_wide - line_begin_y_wide)
            wide_line_end_y = line_begin_y_wide + inverted_rel_line_end * (line_end_y_wide - line_begin_y_wide)

            return (wide_line_start_x, wide_line_start_y), (wide_line_end_x, wide_line_end_y)

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
        line_csv['recording_id'] = line_csv['recording_id'].astype(int)

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
                exposure_time = wide_row['exposure']
                # Perform the mapping from line_image to wide_image
                wide_coords = map_line_to_wide(
                    line_row['left'], line_row['width'], line_image_size,
                    wide_row['line_begin_x'], wide_row['line_end_x'],
                    wide_row['line_begin_y'], wide_row['line_end_y']
                )

                # Perform the mapping from wide_image to wide_crop_image for the line coordinates
                crop_coords = map_wide_to_crop(wide_coords, (left, top, right, bottom), wide_image_size,
                                               wide_crop_image_size)

                # Map the additional coordinates from wide_csv to wide_crop_image
                additional_coords = [
                    (wide_row['line_center_x'], wide_row['line_center_y']),
                    (wide_row['line_begin_x'], wide_row['line_begin_y']),
                    (wide_row['line_end_x'], wide_row['line_end_y'])
                ]
                mapped_additional_coords = map_wide_to_crop(additional_coords, (left, top, right, bottom),
                                                            wide_image_size, wide_crop_image_size)

                result = line_row.to_dict()
                result.update({
                    'start_coord': crop_coords[0],
                    'end_coord': crop_coords[1],
                    'line_center_coord': mapped_additional_coords[0],
                    'line_begin_coord': mapped_additional_coords[1],
                    'line_end_coord': mapped_additional_coords[2],
                    'exposure_time': exposure_time
                })
                results.append(result)
        results = pd.DataFrame(results)

        return results

#%% Testing the LineDataSSS class
lds = LineDataSSS()
lds.pipeline(dataset_path="/home/datasets/examples/wide_and_line_frames/segment1_2024_09_17")