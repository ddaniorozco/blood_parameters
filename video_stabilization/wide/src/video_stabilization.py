"""
VideoStabilization Class

The `VideoStabilization` class provides methods to stabilize video frames by calculating shifts in consecutive frames
with the first one as ground truth, and applying these shifts to minimize the movement between consecutive frames.
The class includes various methods for pre-processing, stabilizing frames, calculating stability metrics,
and saving the results.

Methods:
- __init__: Initializes the VideoStabilization class with specified parameters.
- setup_directories: Creates necessary directories for saving output files.
- calculate_average_image: Calculates and saves the average image from the original and stabilized frames.
- stabilize_frames: Applies the computed shift to stabilize the frame.
- compute_stability_metrics: Computes and saves, into a csv, stability metrics from the shifts between frames.
- log_scale_diff: Computes the logarithmic scale difference between two values.
- make_crop: Crops the frame based on the crop size parameter.
- cross_correlation_using_sobel: Applies Sobel filters and computes the shift between frames using phase correlation.
- pre_process: Pre-processes the current and first frames using filters and normalization.
- max_distance_consecutive_frames_between_points: Computes the distance between two shifts.
- start_csv: Initializes a CSV file for writing shift data.
- save_shifts_to_csv: Saves the corrected shifts to a CSV file.
- process_image: Processes an image frame to calculate the shift for stabilization.
- stabilize_recording_frames:  Main method to start the stabilization process.
Stabilizes the frames in a recording and saves the results.
- apply_filter_to_shifts: Applies filtering logic to the shifts and saves the corrected shifts.
"""

import argparse
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from image_enhance.wide.src import filtering_utils as filters


class VideoStabilization:
    def __init__(self, images_path: str = None, output_path: str = None, crop_size: float = 1,
                 save_all_output: bool = False,
                 save_stab_images: bool = False, distance_threshold: float = 4, max_workers: int = 8,
                 confidence: int = 20):

        """
              Initializes the VideoStabilization class with the specified parameters.

              Parameters:
              - images_path (str): Path to the directory containing input image frames.
              - output_path (str): Path to save the output images and CSV files.
              - crop_size (float): Fraction of the original frame size to use for stabilization.
              - save_all_output (bool): Boolean flag to save all output files (intermediate steps).
              - save_stab_images (bool): Boolean flag to save only the stabilized images.
              - distance_threshold (float): Maximum allowed distance between consecutive frame shifts.
              - max_workers (int): Maximum number of parallel workers for processing.
              - confidence (int): Minimum confidence value for considering a shift valid.
              """

        self.images_path = images_path
        self.output_path = output_path
        self.crop_size = crop_size
        self.save_all_output = save_all_output
        self.save_stab_images = save_stab_images
        self.distance_threshold = distance_threshold
        self.max_workers = max_workers
        self.confidence = confidence
        self.corrected_all_shifts = {}
        self.distances = []
        self.lock = threading.Lock()
        self.total_frames = 0
        self.sum_original = 0
        self.sum_stabilized = 0
        self.distance_consecutive_frames = 0
        self.good_shift = (0, 0)
        self.cropped_current_frame = None
        self.average_images = None
        self.csv_files = None
        self.stabilized_images = None
        self.current_recording = None
        self.writer_all = None
        self.reference_frame = None
        self.writer_metrics = None
        self.caller = None
        self.prev_im_filter = None  # intermediate result for reference frame (for runtime optimization)

    def setup_directories(self) -> tuple | str:

        """
       Creates necessary directories for saving output files.

       Returns:
       - tuple or str: Paths to the created directories.
       """

        if self.save_all_output:
            subdirectories = ['stabilized_images', 'average_images', 'csv_files']
            directories = [os.path.join(self.output_path, subdir) for subdir in subdirectories]
            for dir_path in directories:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    logger.info(f"Created subdirectory: {dir_path}")
            return tuple(directories)
        else:
            subdirectory = 'stabilized_images'
            directory = os.path.join(self.output_path, subdirectory)
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created subdirectory: {directory}")
            return directory

    def calculate_average_image(self) -> np.ndarray | None:

        """
        Calculates and saves the average image from the original and stabilized frames.

        Returns:
        - np.ndarray: The average stabilized image.
        """

        if self.total_frames > 0:
            # Calculate the average of original and stabilized frames
            avg_original = self.sum_original / self.total_frames
            avg_stabilized = self.sum_stabilized / self.total_frames

            # Normalize to ensure the values are in the range [0, 255]
            avg_original_normalized = np.clip(avg_original, 0, 255).astype(np.uint8)
            avg_stabilized_normalized = np.clip(avg_stabilized, 0, 255).astype(np.uint8)

            cv2.imwrite(os.path.join(self.average_images, 'average_original.png'), avg_original_normalized)
            cv2.imwrite(os.path.join(self.average_images, 'average_stabilized.png'), avg_stabilized_normalized)

            # Load the average stabilized image and return it
            average_stabilized_path = os.path.join(self.average_images, 'average_stabilized.png')
            average_stabilized_image = cv2.imread(average_stabilized_path, cv2.IMREAD_GRAYSCALE)
            return average_stabilized_image
        else:
            logger.info("No frames to process.")
            return None

    @staticmethod
    def stabilize_frames(shift: tuple, frame: np.ndarray):
        """
         Applies the computed shift to stabilize the frame.

         Parameters:
         - shift (tuple): Tuple containing the x and y shifts.
         - frame (np.ndarray): The image frame to be stabilized.

         Returns:
         - np.ndarray: The stabilized frame.
         """

        # Create the transformation matrix for the given shift
        m = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])

        # Apply the transformation to stabilize the frame
        stabilized_frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))
        return stabilized_frame

    def compute_stability_metrics(self, shifts_dict: dict, crop_average_stabilized_frame: np.ndarray,
                                  cropped_first_image: np.ndarray):
        """
        Computes and saves stability metrics from the shifts between frames.

        Parameters:
        - shifts_dict (dict): Dictionary containing frame shifts.
        - crop_average_stabilized_frame (np.ndarray): Cropped average stabilized frame.
        - cropped_first_image (np.ndarray): Cropped first image frame.
        """

        # Define the path to save the CSV
        csv_path = os.path.join(self.csv_files, 'metrics_shifts.csv')

        shift_values = np.array(list(shifts_dict.values()))

        # Calculate shifts between consecutive frames
        consecutive_shifts = np.diff(shift_values, axis=0)

        # Calculate magnitudes of consecutive shifts
        consecutive_shift_magnitudes = np.linalg.norm(consecutive_shifts, axis=1)

        # Function to calculate metrics
        def calculate_metrics(shifts):
            return {
                'average': np.mean(shifts),
                'std_dev': np.std(shifts),
                'max': np.max(shifts),
                'min': np.min(shifts),
                'median': np.median(shifts),
                'cumulative': np.sum(shifts)
            }

        # Calculate general metrics
        general_metrics = calculate_metrics(consecutive_shift_magnitudes)

        # Calculate metrics for x and y components
        components = ['x', 'y']
        component_metrics = {}
        for i, component in enumerate(components):
            component_metrics[component] = calculate_metrics(consecutive_shifts[:, i])

        avg_all = crop_average_stabilized_frame / cropped_first_image
        additional_mean = avg_all.mean()
        additional_std = avg_all.std()

        # Prepare data for DataFrame
        data = {
            'Metric': list(general_metrics.keys()),
            'General': list(general_metrics.values())
        }
        for component in components:
            data[component.upper()] = list(component_metrics[component].values())

        # Convert to DataFrame
        df = pd.DataFrame(data).set_index('Metric').transpose()

        # Add the additional row with specified values
        additional_row = pd.Series(
            {'average': additional_mean, 'std_dev': additional_std},
            name='avg_stab_frame / first_frame'
        )

        # Append the additional row to the DataFrame
        df = pd.concat([df, additional_row.to_frame().T])

        # Save to CSV
        df.to_csv(csv_path)

    @staticmethod
    def log_scale_diff(value1: float, value2: float):
        """
               Computes the logarithmic scale difference between two values.

               Parameters:
               - value1 (float): First value.
               - value2 (float): Second value.

               Returns:
               - float: Logarithmic scale difference.
               """

        # Add a small epsilon to avoid log of zero
        epsilon = 0.01
        adjusted_value1 = np.abs(value1) + epsilon
        adjusted_value2 = np.abs(value2) + epsilon

        # Compute the logarithmic values
        log_value1 = np.log(adjusted_value1)
        log_value2 = np.log(adjusted_value2)

        # Calculate the difference in logarithmic scale
        log_diff = np.abs(log_value1 - log_value2)

        # If the signs are different, add log(2) to the difference
        if np.sign(value1) != np.sign(value2):
            log_diff += np.log(2)

        return log_diff

    def make_crop(self, frame: np.ndarray):
        """
            Crops the frame based on the crop size parameter.

            Parameters:
            - frame (np.ndarray): The image frame to be cropped.

            Returns:
            - np.ndarray: The cropped frame.
            """

        # Calculate the crop dimensions
        crop_height = int(frame.shape[0] * self.crop_size)
        crop_width = int(frame.shape[1] * self.crop_size)
        crop_y = (frame.shape[0] - crop_height) // 2
        crop_x = (frame.shape[1] - crop_width) // 2

        # Crop the frame and return
        cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        return cropped_frame

    @staticmethod
    def cross_correlation_using_sobel(cur_im_filter: np.ndarray, prev_im_filter: np.ndarray):
        """
            Applies Sobel filters to detect edges and computes the shift between frames using phase correlation.

            Parameters:
            - cur_im_filter (np.ndarray): Filtered current image frame.
            - prev_im_filter (np.ndarray): Filtered previous image frame.

            Returns:
            - tuple: Computed shift and confidence.
        """

        # Apply Sobel filters to detect edges
        sobel_x1 = cv2.Sobel(cur_im_filter, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y1 = cv2.Sobel(cur_im_filter, cv2.CV_16S, 0, 1, ksize=3)
        sobel_x2 = cv2.Sobel(prev_im_filter, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y2 = cv2.Sobel(prev_im_filter, cv2.CV_16S, 0, 1, ksize=3)
        abs_sobel_x1 = cv2.convertScaleAbs(sobel_x1)
        abs_sobel_y1 = cv2.convertScaleAbs(sobel_y1)
        abs_sobel_x2 = cv2.convertScaleAbs(sobel_x2)
        abs_sobel_y2 = cv2.convertScaleAbs(sobel_y2)
        sobel_im1 = cv2.addWeighted(abs_sobel_x1, 0.5, abs_sobel_y1, 0.5, 25)
        sobel_im2 = cv2.addWeighted(abs_sobel_x2, 0.5, abs_sobel_y2, 0.5, 25)
        _, threshold_im1 = cv2.threshold(sobel_im1, 55, 255, cv2.THRESH_BINARY)
        _, threshold_im2 = cv2.threshold(sobel_im2, 55, 255, cv2.THRESH_BINARY)

        # Use phase correlation to compute the shift
        shift, confidence = cv2.phaseCorrelate(np.float32(threshold_im2), np.float32(threshold_im1))

        return shift, confidence

    def pre_process(self, cropped_current_frame: np.ndarray):
        """
           Pre-processes the current and previous frames using filters and normalization.

           Parameters:
           - cropped_current_frame (np.ndarray): Cropped current image frame.

           Returns:
           - tuple: Computed shift and confidence.
           """
        current_frame = cropped_current_frame / filters.gauss_blur(cropped_current_frame.astype('float'),
                                                                   201, 51)

        # Normalize, CLAHE, blur the reference frame:
        if self.prev_im_filter is None:
            # do it only once for the reference frame
            frame_x_prev = cv2.normalize(self.reference_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame_x_prev = filters.apply_clahe(frame_x_prev)
            prev_im_filter = cv2.medianBlur(frame_x_prev, 11)
            self.prev_im_filter = prev_im_filter

        # Normalize, CLAHE, blur the current frame:
        frame_x = cv2.normalize(current_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame_x = filters.apply_clahe(frame_x)
        cur_im_filter = cv2.medianBlur(frame_x, 11)

        cur_shift, confidence = self.cross_correlation_using_sobel(cur_im_filter=cur_im_filter.copy(),
                                                                   prev_im_filter=self.prev_im_filter.copy())
        return {'shift': cur_shift, 'confidence': confidence, 'distance': np.linalg.norm(cur_shift)}

    @staticmethod
    def shift_distance(shift_1: tuple, shift_2: tuple):
        """
        Computes the distance between two shifts.

        Parameters:
        - shift_1 (tuple): First shift.
        - shift_2 (tuple): Second shift.

        Returns:
        - float: Distance.
        """

        if shift_1 and shift_2 is not None:
            shift_1 = np.array(shift_1)
            shift_2 = np.array(shift_2)

            # Calculate the Euclidean distance between the shifts
            distance = np.linalg.norm(shift_2 - shift_1)
            return distance

    @staticmethod
    def start_csv(file: object):
        """
        Initializes a CSV file for writing shift data.

        Parameters:
        - file (file object): The CSV file to write to.

        Returns:
        - csv.writer: CSV writer object.
        """
        writer_all = csv.writer(file)
        writer_all.writerow(
            ['frame_number', 'x_axis_shift', 'y_axis_shift', 'shift', 'confidence', 'distance_consecutive_frames'])
        return writer_all

    def save_shifts_to_csv(self):
        """
        Saves the corrected shifts to a CSV file.
        """
        csv_path = os.path.join(self.csv_files, 'corrected_shifts.csv')
        with open(csv_path, 'w', newline='') as file:
            writer = self.start_csv(file)
            for idx, shift_data in self.corrected_all_shifts.items():
                x_axis_shift, y_axis_shift = shift_data['shift']
                shift_magnitude = np.sqrt(x_axis_shift ** 2 + y_axis_shift ** 2)
                confidence = shift_data['confidence']
                distance_consecutive_frames = shift_data['distance_consecutive_frames']
                writer.writerow(
                    [idx, x_axis_shift, y_axis_shift, shift_magnitude, confidence, distance_consecutive_frames])

    def set_reference_image(self, reference_image: np.ndarray):
        logger.info('Set stabilization reference frame')
        self.reference_frame = self.make_crop(reference_image)
        self.prev_im_filter = None

    def compute_shift(self, target_image: np.ndarray):
        assert self.reference_frame is not None, "Reference frame was never set - call set_reference_frame first"
        logger.info('Computing stabilization shift')
        cropped_frame = self.make_crop(target_image)
        return self.pre_process(cropped_frame)

    def process_image(self, idx: int, image: np.ndarray):
        """
         Processes an image frame to calculate the shift for stabilization.

         Parameters:
         - idx (int): Index of the frame.
         - image (np.ndarray): Image frame.

         Returns:
         - tuple: Index and shift data.
         """

        cropped_current_frame = self.make_crop(image)

        if idx == 0:
            return idx, {'shift': self.good_shift, 'confidence': 1}

        cur_shift, confidence = self.pre_process(cropped_current_frame)
        return idx, {'shift': cur_shift, 'confidence': confidence}

    # def stabilize_recording_frames(self, images: list, file_paths: list):
    def stabilize_recording_frames(self):

        """
        Stabilizes the frames in a recording and saves the results.

        Parameters:
        - images (list): List of image frames.
        - file_paths (list): List of file paths for the image frames.

        Returns:
        - dict: Dictionary of shifts for each frame.

        """
        file_paths = sorted(glob(os.path.join(self.images_path, '*.png')))
        images = [cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) for file_path in tqdm(file_paths, desc="Loading images")]

        # Setup directories and stabilize frames based on save options
        if self.save_all_output:
            self.stabilized_images, self.average_images, self.csv_files = self.setup_directories()

        elif self.save_stab_images:
            self.stabilized_images = self.setup_directories()

        # Process the first image
        first_image = images[0]
        cropped_first_image = self.make_crop(first_image)
        self.reference_frame = cropped_first_image / filters.gauss_blur(cropped_first_image.astype('float'), 201, 51)
        all_shifts = {}
        if images:
            self.sum_original = np.zeros_like(images[0], dtype=np.float64)
            self.sum_stabilized = np.zeros_like(images[0], dtype=np.float64)
        else:
            self.sum_original = None
            self.sum_stabilized = None
        lock = threading.Lock()

        self.total_frames = len(images)

        # Process each frame to compute the shifts using parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # logger.info(executor._max_workers)
            futures = {executor.submit(self.process_image, idx, image): idx for idx, image in enumerate(images)}
            for future in tqdm(as_completed(futures), total=len(images), desc="Calculating shifts"):
                idx, result = future.result()
                all_shifts[idx] = result

        # Apply filtering to the shifts
        all_shifts = dict(sorted(all_shifts.items()))
        self.apply_filter_to_shifts(all_shifts)
        shifts_dict = {index: value['shift'] for index, value in self.corrected_all_shifts.items()}

        # Apply the stabilization and save frames if required
        if self.save_stab_images or self.save_all_output:
            def stabilize_and_save(idx, image, file_path):
                shift = self.corrected_all_shifts[idx]['shift']
                stabilized_frame = self.stabilize_frames(shift, image)

                # Update sum of original and stabilized frames
                with lock:
                    self.sum_original += image.astype(np.float64)
                    self.sum_stabilized += stabilized_frame.astype(np.float64)

                # Save stabilized frames with original names prefixed by "stabilized_"
                directory, original_file_name = os.path.split(file_path)
                stabilized_file_name = f"stabilized_{original_file_name}"
                stabilized_file_path = os.path.join(self.stabilized_images, stabilized_file_name)
                cv2.imwrite(stabilized_file_path, stabilized_frame)

            # Process and save each stabilized frame using parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(stabilize_and_save, idx, image, file_path): idx for idx, (image, file_path)
                           in enumerate(zip(images, file_paths))}
                for _ in tqdm(as_completed(futures), total=len(images),
                              desc="Stabilizing and saving frames and calculating metrics recording"):
                    pass

        # Calculate average images and metrics if required
        if self.save_all_output:
            average_stabilized_frame = self.calculate_average_image()
            crop_average_stabilized_frame = self.make_crop(average_stabilized_frame)
            self.compute_stability_metrics(shifts_dict, crop_average_stabilized_frame, cropped_first_image)
            self.save_shifts_to_csv()

        return shifts_dict

    def apply_filter_to_shifts(self, all_shifts: dict):
        """
        Applies the filtering logic to the self.all_shifts dictionary and saves the corrected shifts in
        self.corrected_all_shifts.

        Parameters:
        - all_shifts (dict): Dictionary containing all shifts.
        """
        confidence_threshold = (self.confidence / 100)
        log_diff_threshold = 0.1
        last_good_shift = None

        first_key = sorted(all_shifts.keys())[0]

        for frame_idx in tqdm(sorted(all_shifts.keys()), desc="Filtering shifts"):
            shift_data = all_shifts[frame_idx]
            cur_shift = shift_data['shift']

            confidence = shift_data['confidence']

            if frame_idx == first_key:
                last_good_shift = cur_shift
                self.corrected_all_shifts[frame_idx] = {
                    'shift': (0, 0),
                    'distance_consecutive_frames': 0,
                    'confidence': 1
                }
                continue

            if last_good_shift is not None:
                distance_consecutive_frames = self.shift_distance(last_good_shift,
                                                                  cur_shift)
            else:
                distance_consecutive_frames = 0

            if confidence > confidence_threshold and distance_consecutive_frames < self.distance_threshold:
                last_good_shift = cur_shift
                self.corrected_all_shifts[frame_idx] = {
                    'shift': cur_shift,
                    'distance_consecutive_frames': distance_consecutive_frames,
                    'confidence': confidence
                }
            else:
                is_good_shift = False
                if last_good_shift is not None:
                    shift_diff_x = self.log_scale_diff(last_good_shift[0], cur_shift[0])
                    shift_diff_y = self.log_scale_diff(last_good_shift[1], cur_shift[1])

                    if ((shift_diff_x < log_diff_threshold or shift_diff_y < log_diff_threshold) and confidence > 0.10
                            and distance_consecutive_frames < self.distance_threshold):
                        last_good_shift = cur_shift
                        is_good_shift = True

                if is_good_shift:
                    self.corrected_all_shifts[frame_idx] = {
                        'shift': cur_shift,
                        'distance_consecutive_frames': distance_consecutive_frames,
                        'confidence': confidence
                    }
                else:
                    self.corrected_all_shifts[frame_idx] = {
                        'shift': last_good_shift if last_good_shift is not None else cur_shift,
                        'distance_consecutive_frames': 0,
                        'confidence': confidence
                    }

        # self.smooth_shifts()

    # def smooth_shifts(self):
    #     shifts_x = []
    #     shifts_y = []
    #
    #     frame_indices = sorted(self.corrected_all_shifts.keys())
    #
    #     # Skip the first frame
    #     for frame_idx in frame_indices[1:]:
    #         shifts_x.append(self.corrected_all_shifts[frame_idx]['shift'][0])
    #         shifts_y.append(self.corrected_all_shifts[frame_idx]['shift'][1])
    #
    #     # Define a low-pass Butterworth filter
    #     b, a = butter(3, 0.1)
    #
    #     # Apply filtfilt to smooth the data
    #     shifts_x_smooth = filtfilt(b, a, shifts_x)
    #     shifts_y_smooth = filtfilt(b, a, shifts_y)
    #
    #     # Re-assign the smoothed values back to the dictionary, skipping the first frame
    #     for i, frame_idx in enumerate(frame_indices[1:]):
    #         self.corrected_all_shifts[frame_idx]['shift'] = (shifts_x_smooth[i], shifts_y_smooth[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Stabilization")
    parser.add_argument('--images_path', type=str, help='Path to original images directory')  # required=True
    parser.add_argument('--output_path', type=str, help='Path to stabilized data')
    parser.add_argument('--crop_size', type=float, default=0.7,
                        help='Crop size as a fraction of original size (default: 1)')
    parser.add_argument('--save_all_output', action='store_true',
                        help='Set this flag for saving csv with shifts, norm, confidence')
    parser.add_argument('--save_stab_images', action='store_true',
                        help='Set this flag for saving the stabilized images')
    parser.add_argument('--distance_threshold', type=float, default=4, help='Distance threshold (default: 4)')
    parser.add_argument('--max_workers', type=int, default=8, help='Workers for parallelization (default: 8)')
    parser.add_argument('--confidence', type=int, default=20, help='Confidence for filtering shifts (default: 20)')

    args = parser.parse_args()

    if args.output_path and not (args.save_all_output or args.save_stab_images):
        parser.error("--output_path can only be used if --save_all_output or --save_stab_images are also provided")

    # if not args.output_path and args.save_all_output or args.save_stab_images:
    #     parser.error("--output_path is needed if --save_all_output or --save_stab_images are provided")

    VideoStabilization(images_path=args.images_path, output_path=args.output_path, crop_size=args.crop_size,
                       save_all_output=args.save_all_output, save_stab_images=args.save_stab_images,
                       distance_threshold=args.distance_threshold, max_workers=args.max_workers,
                       confidence=args.confidence).stabilize_recording_frames()
