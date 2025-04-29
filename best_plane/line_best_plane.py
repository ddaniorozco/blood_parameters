from typing import Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from skimage.color import gray2rgb

from capillary_classifier.line.src.inference import LineCapillaryClassifierModel
from capillary_detector.line.yolo.src.inference import YoloLineCapillaryDetectorModel
from ext.util.design_patterns import Singleton
from ext.util.performance import timer
from image_enhance.shared.src.basic_utils import scale_image
from interfaces.inference.image_inference_model import ImageInferenceModel


class LineBestPlaneConfig(Singleton):
    MAX_DETECTION_TOL = 75
    MIN_CONSECUTIVE_SEQUENCE = 3
    DETECTION_IOU = 1e-10
    DETECTION_CONF = 0.5
    CAPILLARY_CROP_WIDTH = 256


# TODO - Line capillary classifier should not return (0,6) shaped numpy arrays
# TODO - erase rgb convertion one it is assign in the detection model
class LineBestPlaneModel:

    def __init__(self, line_capillary_detector: YoloLineCapillaryDetectorModel = None,
                 line_capillary_classifier: LineCapillaryClassifierModel = None,
                 default_device: torch.device = None, half_precision=False, to_device_on_init: bool = False,
                 to_cpu_after_infer: bool = False, verbosity: int = 0,
                 min_consecutive_capillaries: int = LineBestPlaneConfig.MIN_CONSECUTIVE_SEQUENCE,
                 max_tolerance: int = LineBestPlaneConfig.MAX_DETECTION_TOL):

        """
             Initializes the LineBestPlaneModel with given parameters.

             Args:
                 line_capillary_detector (ImageInferenceModel): Model for detecting capillaries in images.
                 line_capillary_classifier (ImageInferenceModel): Model for classifying frames.
                 default_device (torch.device): Default device for model inference.
                 half_precision (bool): Flag for using half precision during inference.
                 to_device_on_init (bool): Flag to move models to device during initialization.
                 to_cpu_after_infer (bool): Flag to move models back to CPU after inference.
                 verbosity (int): Verbose logging level.
                 min_consecutive_capillaries (int): Minimum number of consecutive capillary detections,
                  to use for estimation when there are no good classifications
                 max_tolerance (int): Maximum tolerance for bounding box similarity,
                  in respect to the center and width of the detections
             """

        self.max_tolerance = max_tolerance
        self.min_consecutive_sequence = min_consecutive_capillaries
        self.global_index = 0
        self.classify_called = False
        self.detections_df = pd.DataFrame()
        self.loc_estimation = {'classification_success': False, 'image_id': -1, 'segment_id': -1, 'z': -1}
        self.output_cols = ['image_id', 'segment_id', 'sequence_id', 'sequence_len', 'is_good', 'bbox']

        # Models initialization
        self.line_capillary_classifier = line_capillary_classifier
        self.line_capillary_detector = line_capillary_detector
        self.default_device = default_device
        self.half_precision = half_precision
        self.to_device_on_init = to_device_on_init
        self.to_cpu_after_infer = to_cpu_after_infer
        self.verbosity = verbosity
        self._models_init()
        self.line_capillary_detector.set_iou_threshold(LineBestPlaneConfig.DETECTION_IOU)
        self.line_capillary_detector.set_confidence_threshold(LineBestPlaneConfig.DETECTION_CONF)

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

    def reset(self):
        logger.info('Resetting Model')
        self.classify_called = False
        self.global_index = 0
        self.detections_df = pd.DataFrame()
        self.loc_estimation = {'classification_success': False, 'image_id': -1, 'segment_id': -1, 'z': -1}

    def detect(self, images: Union[list, np.ndarray], device: torch.device = None):
        """
        This method gets a line frame or a batch of line frames.
        The method detects capillaries.
        The frames are split into half as part of the preprocessing,
        and the detections are in respect to each split

        Args:
            images (list np.ndarray | np.ndarray): Line frame or a batch of line frames
            device (torch.device): Device for performing inference.

        Returns:
            A dictionary of the detections

        """
        if isinstance(images, np.ndarray):
            images = [images]
        if self.classify_called:
            self.reset()

        logger.info(f'Splitting and detecting capillaries in {len(images)} images')

        batch_detections = []
        for frame in images:  # TODO: dont need loop, it should work for batches
            frames_splits = self._split_frames_by_height(frame)
            rgb_frames_split = [gray2rgb(split[2]) for split in frames_splits]
            detections = self.line_capillary_detector.batch_infer(images_source=rgb_frames_split, device=device)[1]

            for dets, (frame_id, split_id, frame_split) in zip(detections, frames_splits):
                dets = [det.tolist() for det in dets] if len(dets) > 0 else [[]]
                for det_id, det in enumerate(dets):
                    batch_detections.append(self._detection_dict(frame_id, split_id, det_id, frame_split, det))

        batch_detections_df = pd.DataFrame(batch_detections)
        self.detections_df = pd.concat([self.detections_df, batch_detections_df])
        self.detections_df.reset_index(inplace=True, drop=True)

        n_capillaries = len(batch_detections_df.dropna())
        if n_capillaries > 0:
            logger.info(f'Found {n_capillaries} capillaries in line image batch')
        else:
            logger.warning('No capillaries found in line image batch')
        return {'found_capillaries_in_batch': n_capillaries > 0}

    def classify(self, segment_centers: np.array, device: torch.device = None):
        """
        This method should be called after the detect method ran on the entire z-scan.
        It uses the detections from the detect method (self.detections_df)
        This methods get the center location of each frame split in mm
        It classifies each detection as 'good' or 'bad'
        It finds which detections are consecutive along several frame splits
        Using these details it estimates the best line z plane

        Args:
            segment_centers (np.ndarray): These are the locations [mm] of each frame split.
                                          Each location is the motor z location while capturing the line frame.
                                          The location is the location of the middle row of the line frame split
            device (torch.device): Device for performing inference.

        Returns:
            A dictionary of the with detail of the best plane
            {'classification_success' , 'image_id', 'segment_id', 'z'}
        """

        # Validate inputs
        self._validate_detect_input(segment_centers)

        # no detections
        if len(self.detections_df.dropna()) == 0:
            self.loc_estimation['classification_success'] = False
            logger.warning('Did not detect capillaries -> bad z-scan')

        else:
            logger.info(f"Classifying line capillary detections")
            self._classify_detections(device)

            logger.info(f"Estimating best line z plane")
            self._estimate_best_frame()

        self.classify_called = True

        self.loc_estimation['detections'] = self.detections_df.dropna()[self.output_cols].to_dict(orient='records')

        return self.loc_estimation

    def _validate_detect_input(self, segment_centers):
        """
        Validation of the frame split z location (segment_centers),
        If the validation succeed the data merged into the detection dataframe (self.detections_df)
        """

        frame_splits_cols = ['image_id', 'segment_id']
        frame_splits_df = self.detections_df[frame_splits_cols].drop_duplicates().sort_values(frame_splits_cols)

        assert len(segment_centers) == len(frame_splits_df), \
            f"Segment center array must hold {len(frame_splits_df)} values - twice the number of input frames"

        frame_splits_df['z'] = segment_centers
        self.detections_df = pd.merge(self.detections_df, frame_splits_df, on=frame_splits_cols)

    def _split_frames_by_height(self, image_16bit: np.ndarray):
        """# TODO: refactor all method
        Split a frame height by half
        Returns: frane

        """
        image_8bit = scale_image(image_16bit).astype('uint8')  # TODO: is it correct
        height = image_8bit.shape[0]

        # Split the image into top and bottom halves
        top_half, bottom_half = image_8bit[:height // 2, :], image_8bit[height // 2:, :]

        # Create tuples for each half
        arrays_split = [
            (self.global_index, 0, top_half),
            (self.global_index, 1, bottom_half)
        ]
        self.global_index += 1
        return arrays_split

    @staticmethod
    def _detection_dict(frame_id, split_id, det_id, frame_split, det):
        """
        Constructs a dictionary representing detection details for a specific frame split.

        Args:
            frame_id (int): Identifier for the frame.
            split_id (int): Identifier for the frame split.
            det_id (int): Identifier for the detection.
            frame_split (np.ndarray): frame split.
            det (list): Detection data containing coordinates, confidence, and class.

        Returns:
            dict: A dictionary containing the detection details. If `det` is empty, values for detection-related keys
            are set to None or default.

        """
        if len(det) > 0:
            x1, y1, x2, y2, conf, cls = det
            return {'image_id': frame_id,
                    'segment_id': split_id,
                    'det_id': det_id,
                    'sequence_id': -1,
                    'sequence_len': -1,
                    'is_good': -1,
                    'frame_split': frame_split,
                    'detection_class': int(cls),
                    'confidence': conf,
                    'left': int(x1),
                    'top': int(y1),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1),
                    'bbox': det
                    }
        else:
            return {'image_id': frame_id,
                    'segment_id': split_id,
                    'det_id': det_id,
                    'sequence_id': -1,
                    'sequence_len': -1,
                    'is_good': -1,
                    'frame_split': None,
                    'detection_class': None,
                    'confidence': None,
                    'left': None,
                    'top': None,
                    'width': None,
                    'height': None,
                    'bbox': [],
                    }

    def _are_bboxes_similar(self, bbox1: pd.Series, bbox2: pd.Series):
        """
        Check if two bounding boxes are similar based on their 'middle' and 'width' data.
        If any bbox is empty the method returns false

        :param bbox1: The first bounding box (dictionary).
        :param bbox2: The second bounding box (dictionary).
        :return: True if the bounding boxes are similar, False otherwise.
        """
        if bbox1.isna().any() or bbox2.isna().any():
            return False
        else:
            bbox1_middle = bbox1['left'] + bbox1['width'] / 2
            bbox2_middle = bbox2['left'] + bbox2['width'] / 2
            return (abs(bbox1_middle - bbox2_middle) <= self.max_tolerance and
                    abs(bbox1['width'] - bbox2['width']) <= self.max_tolerance)

    def _crop_capillaries_from_frames(self):
        """
        Crops frames based on detection data and bounding boxes.
        The cropped frames saved in self.detections_df
        """

        self.detections_df['cropped_frame'] = None

        for index, row in self.detections_df.iterrows():
            if row['detection_class'] is None or pd.isna(row['detection_class']):
                continue

            center_x = int(row['left'] + (row['width'] // 2))

            # Define the crop box
            left = int(center_x - LineBestPlaneConfig.CAPILLARY_CROP_WIDTH / 2)
            right = int(center_x + LineBestPlaneConfig.CAPILLARY_CROP_WIDTH / 2)
            top = int(row['top'])
            bottom = int(row['top']) + int(row['height'])

            # Ensure the crop box is within array bounds
            left = max(0, left)
            right = min(row['frame_split'].shape[1], right)
            top = max(0, top)
            bottom = min(row['frame_split'].shape[0], bottom)

            # Crop the array
            cropped_frame = row['frame_split'][top:bottom, left:right]

            self.detections_df.at[index, 'cropped_frame'] = cropped_frame

    def _classify_detections(self, device):
        """
        The method crops the line frames using the capillary detections bboxes
        Then it classifies each crop as 'good' or 'bad'
        More over it indicates if the z-scan had at least one good classification ('classification_success')

        Args:
            device: (torch.device): Default device for model inference.
        """

        self._crop_capillaries_from_frames()

        valid_entries = self.detections_df.dropna(subset=['cropped_frame'])
        rgb_cropped_frame = [gray2rgb(frame) for frame in valid_entries['cropped_frame']]

        # Perform inference
        with timer('Line Capillary Classifier Inference'):
            # I'm assuming the inference function returns a tuple where results are the second item
            results = self.line_capillary_classifier.batch_infer(images_source=rgb_cropped_frame, device=device)[1]

        # Assign results back to the DataFrame, aligning with the original rows
        # This assumes `results` is a list or array of results corresponding one-to-one with `rgb_cropped_frame`
        self.detections_df.loc[valid_entries.index, 'is_good'] = results
        self.detections_df.loc[
            list(set(self.detections_df.index) - set(valid_entries.index)), 'is_good'] = 0

        if np.sum(self.detections_df['is_good']) > 0:
            self.loc_estimation['classification_success'] = True
        else:
            self.loc_estimation['classification_success'] = False

    def _find_consecutive_detections_multi(self):
        """
        Updates the DataFrame by identifying and labeling consecutive detections based on their similarity.
        The method counts consecutive rows where detections are similar (as defined by `_are_bboxes_similar` method),
        and updates a 'capillary_length' column with the count of consecutive similar rows.

        This function assumes the DataFrame is pre-sorted and that each detection has an associated 'detection_class'
        field.

        Modifies:
            self.capillary_detection_df (pd.DataFrame): This DataFrame will have an additional 'sequence_len' column
            updated with the count of consecutive detections for each row.

        Returns:
            None: The function modifies the DataFrame in-place and does not return any value.
        """
        self.detections_df = self.detections_df.sort_values(['image_id', 'segment_id', 'det_id'])

        self.detections_df['sequence_len'] = 0
        self.detections_df['sequence_id'] = -1
        self.detections_df.loc[self.detections_df['detection_class'].notna(), 'sequence_len'] = 1
        self.detections_df.loc[self.detections_df['detection_class'].notna(), 'sequence_id'] = self.detections_df[
            self.detections_df['detection_class'].notna()].index

        detections_df = self.detections_df.copy()

        # Track the start index and count consecutive rows
        i_start = 0
        index_list = [i_start]
        sequence_list = []
        search_ended = False

        while len(detections_df) > 1:
            if not search_ended:  # should continue searching
                i_start = index_list[-1]
            elif search_ended:  # search ended
                if len(index_list) > 1:
                    sequence_list.append(index_list)
                detections_df = detections_df.drop(index=index_list)
                if len(detections_df) <= 1:
                    break
                i_start = detections_df.index[0]
                index_list = [i_start]
                if np.any(self.detections_df.loc[i_start].isna()):
                    continue

            image_id = detections_df.loc[i_start, 'image_id']
            segment_id = detections_df.loc[i_start, 'segment_id']

            optional_df = detections_df[((detections_df['image_id'] == image_id) &
                                         (detections_df['segment_id'] == segment_id + 1)) |
                                        ((detections_df['image_id'] == image_id + 1) &
                                         (detections_df['segment_id'] == segment_id - 1))]

            search_ended = True
            for optional_id, optional_det in optional_df.iterrows():
                if self._are_bboxes_similar(self.detections_df.loc[i_start], optional_det):
                    search_ended = False
                    index_list.append(optional_id)
                    break

        for sequence_id, sequence in enumerate(sequence_list):
            self.detections_df.loc[sequence, 'sequence_id'] = sequence_id + np.max(
                self.detections_df.index.tolist())
            self.detections_df.loc[sequence, 'sequence_len'] = len(sequence)

    def _find_consecutive_detections(self):
        """
        Updates the DataFrame by identifying and labeling consecutive detections based on their similarity.
        The method counts consecutive rows where detections are similar (as defined by `_are_bboxes_similar` method),
        and updates a 'capillary_length' column with the count of consecutive similar rows.

        This function assumes the DataFrame is pre-sorted and that each detection has an associated 'detection_class'
        field.

        Modifies:
            self.capillary_detection_df (pd.DataFrame): This DataFrame will have an additional 'sequence_len' column
            updated with the count of consecutive detections for each row.

        Returns:
            None: The function modifies the DataFrame in-place and does not return any value.
        """
        self.detections_df = self.detections_df.sort_values(['image_id', 'segment_id', 'det_id'])

        self.detections_df['sequence_len'] = 0
        self.detections_df['sequence_id'] = -1
        self.detections_df.loc[self.detections_df['detection_class'].notna(), 'sequence_len'] = 1
        self.detections_df.loc[self.detections_df['detection_class'].notna(), 'sequence_id'] = self.detections_df[
            self.detections_df['detection_class'].notna()].index

        # Track the start index and count consecutive rows
        start_index = 0  # Start index of consecutive segment
        consec_count = 1  # Start the consecutive count at 1 since we compare to previous

        for i in range(1, len(self.detections_df)):

            if self._are_bboxes_similar(self.detections_df.iloc[i], self.detections_df.iloc[i - 1]):
                consec_count += 1
            else:
                if consec_count > 1:
                    # Assign the consecutive count to the correct range (excluding the current row)
                    self.detections_df.loc[start_index:i - 1, 'sequence_len'] = consec_count
                    self.detections_df.loc[start_index:i - 1, 'sequence_id'] = start_index
                # Reset for the next group
                start_index = i
                consec_count = 1 if np.all(self.detections_df.iloc[i] is not None) else 0

        # After loop, handle the last sequence if it was consecutive
        if consec_count > 1:
            self.detections_df.loc[start_index:, 'sequence_len'] = consec_count
            self.detections_df.loc[start_index:, 'sequence_id'] = start_index

    @staticmethod
    def _find_middle_elements(lst: list):
        """
        Given a list of indices the method finds the middle index for odd len list and the middle indices for even len

        Args:
            lst: list of indices

        Returns: middle index/indices

        """
        n = len(lst)
        if n % 2 == 1:
            # List has odd length, return the middle element
            return int(lst[n // 2])
        else:
            # List has even length, return the two middle elements
            return [lst[n // 2 - 1], lst[n // 2]]

    def _select_the_best_index(self, df):
        """
        Given a sequence of detections, this method finds the best index.
            1. If it contains good detections, finds the most middle and shallow good detection
            1. If it does not contain good detections, finds the middle and shallow index
        Args:
            df: a sequence of detection

        Returns:
            best index
        """

        # Step 1: Calculate the actual middle index of the DataFrame
        middle_indices = self._find_middle_elements(df.index)
        if isinstance(middle_indices, int):
            middle_index = middle_indices
        elif df.loc[middle_indices, 'is_good'].sum() == 1:
            return df.loc[df.index.isin(middle_indices) & df['is_good'] == 1, 'is_good'].index[0]
        elif df.loc[middle_indices[0], 'z'] == df['z'].min():
            middle_index = middle_indices[0]
        else:
            middle_index = middle_indices[1]

        # Step 2: Filter the DataFrame for rows where `is_good` is 1
        good_df = df[df['is_good'] == 1].copy()

        if len(good_df) == 0:
            return middle_index
        elif len(good_df) == 1:
            return good_df.index[0]

        # Step 3: Calculate the absolute difference from the actual middle index
        good_df['distance_to_middle'] = np.abs(np.array(good_df.index - middle_index))

        # Step 4: Find the minimum distance
        min_distance = good_df['distance_to_middle'].min()

        # Step 5: Filter rows with the minimum distance and find the one with the lowest 'z' value
        closest_row = good_df[good_df['distance_to_middle'] == min_distance].nsmallest(1, 'z')

        # Output the index of the row
        closest_index = closest_row.index[0]

        return closest_index

    def _select_longest_shallowest(self, df):
        """
        Given sequences of detections, this method finds the best index.
            1. Finds the longest sequences
            2. If they do not contain good detections and they shorter than min_consecutive_sequence, return None
            3. Finds the shallowest sequence
            4. If it contains good detections, finds the most middle and shallow good detection
            5. If it does not contain good detections, finds the middle and shallow index

        Args:
            df: dataframe with sequences to be tested

        Returns:
            best index
        """

        # Take the longest sequences
        best_sequences = df[df['sequence_len'] == df['sequence_len'].max()]

        # If it does not contain good detection and is smaller than min_consecutive_sequence, return None
        if (np.sum(best_sequences['is_good']) == 0 and
                best_sequences['sequence_len'].max() < self.min_consecutive_sequence):
            return None

        # Take the shallowest sequence
        z_mean_per_sequence = best_sequences.groupby('sequence_id')['z'].transform('min')
        best_sequence = best_sequences[z_mean_per_sequence == z_mean_per_sequence.min()]

        # find the best index
        best_index = self._select_the_best_index(best_sequence)

        return best_index

    def _estimate_best_frame(self):
        """
        Chooses the best frame based on the capillary detections and their classification as 'good' or bad.
        The method first selects the best consecutive sequence, and then select the best frame split within it.

        The importance of the selection of the best consecutive sequence:
            1. Number of good detections the consecutive sequence.
            2. Length of the consecutive sequence.
            3. How shallow the consecutive sequence.

        Once we have the best consecutive sequence:
            1. If we have good detections in that sequence,
               we select the good detection that is closet to the middle of the sequence and is the shallowest
            2. If we don't have good detections in that sequence,
                we select the middle of the sequence and is the shallowest
        """
        best_index = None

        logger.info(f"Searching consecutive sequences of detections")
        self._find_consecutive_detections_multi()

        # no good detections find the longest and shallowest sequence
        if np.sum(self.detections_df['is_good']) == 0:
            best_index = self._select_longest_shallowest(self.detections_df)
            if best_index is None:
                logger.warning('No good detections and no minimal length sequence -> bad z-scan')

        # one good detection
        elif np.sum(self.detections_df['is_good']) == 1:
            best_index = self.detections_df[self.detections_df['is_good'] == 1].index[0]

        # more than one good detection
        else:
            good_per_sequence = self.detections_df.groupby('sequence_id')['is_good'].transform('sum')
            self.detections_df['good_per_sequence'] = good_per_sequence
            best_sequences = self.detections_df[good_per_sequence == good_per_sequence.max()]
            best_sequences_indices = best_sequences['sequence_id']
            num_best_sequences = best_sequences_indices.nunique()

            # all the good detections from the same sequence
            if num_best_sequences == 1:
                best_index = self._select_the_best_index(best_sequences)

            # the good detections from more than one sequence, and one sequence has the most good detections
            elif best_sequences.loc[good_per_sequence == good_per_sequence.max(), 'sequence_id'].nunique() == 1:
                best_index = self._select_the_best_index(best_sequences)

            # the good detections from more than one sequence, and they have same amount of good detections
            elif best_sequences.loc[good_per_sequence == good_per_sequence.max(), 'sequence_id'].nunique() > 1:
                best_index = self._select_longest_shallowest(best_sequences)
                if best_index is None:
                    logger.warning('Found no minimal length sequence -> bad z-scan')

            else:
                assert best_index is not None, "Should not get here"

        if best_index is not None:
            for key in self.loc_estimation.keys():
                if key in self.detections_df.columns:
                    self.loc_estimation[key] = float(self.detections_df.at[best_index, key])

            logger.info('Successfully estimated the best line z plane')
        else:
            logger.warning('Failed to estimate the best line z plane')
