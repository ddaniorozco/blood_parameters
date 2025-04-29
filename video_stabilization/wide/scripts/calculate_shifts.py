import argparse

import cv2
import numpy as np

from gen_utils.src.logger import init_loggers
from image_enhance.wide.src import filtering_utils as filters


class VideoStabilization:
    def __init__(self, distance_threshold=4):

        # self.crop_size = crop_size
        self.distance_consecutive_frames = 0
        self.good_shift = (0, 0)
        self.distances = []
        self.cropped_current_frame = None
        self.distance_threshold = distance_threshold

    @staticmethod
    def stabilize_frames(shift, frame):
        m = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
        stabilized_frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))
        return stabilized_frame

    @staticmethod
    def stabilize_using_sobel(cur_im_filter, prev_im_filter):

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

        shift, response = cv2.phaseCorrelate(np.float32(threshold_im2), np.float32(threshold_im1))

        return shift, response

    @staticmethod
    def log_scale_diff(value1, value2):
        epsilon = 0.01
        adjusted_value1 = np.abs(value1) + epsilon
        adjusted_value2 = np.abs(value2) + epsilon

        log_value1 = np.log(adjusted_value1)
        log_value2 = np.log(adjusted_value2)

        log_diff = np.abs(log_value1 - log_value2)

        if np.sign(value1) != np.sign(value2):
            log_diff += np.log(2)

        return log_diff

    @staticmethod
    def make_crop(frame, crop_size):
        crop_height = int(frame.shape[0] * crop_size)
        crop_width = int(frame.shape[1] * crop_size)
        crop_y = (frame.shape[0] - crop_height) // 2
        crop_x = (frame.shape[1] - crop_width) // 2
        cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        return cropped_frame

    def pre_process(self, cropped_first_frame):
        first_frame = (cropped_first_frame / filters.gauss_blur(cropped_first_frame.astype('float'), 201, 51))
        cur_frame = self.cropped_current_frame / filters.gauss_blur(self.cropped_current_frame.astype('float'), 201, 51)
        frame_x = cv2.normalize(cur_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame_x_prev = cv2.normalize(first_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        frame_x = filters.apply_clahe(frame_x)
        frame_x_prev = filters.apply_clahe(frame_x_prev)

        cur_im_filter = cv2.medianBlur(frame_x, 11)
        prev_im_filter = cv2.medianBlur(frame_x_prev, 11)
        cur_shift, response = self.stabilize_using_sobel(cur_im_filter=cur_im_filter.copy(),
                                                         prev_im_filter=prev_im_filter.copy())

        return cur_shift, response

    @staticmethod
    def max_distance_consecutive_frames_between_points(shift_1, shift_2):
        if shift_1 and shift_2 is not None:
            shift_1 = np.array(shift_1)
            shift_2 = np.array(shift_2)
            distance = np.linalg.norm(shift_2 - shift_1)
            return distance

    def stabilization_just_shifts(self, first_frame: np.ndarray, current_frame: np.ndarray, crop_size: float):

        # log_diff_threshold = 0.1

        cropped_first_frame = self.make_crop(first_frame, crop_size)
        self.cropped_current_frame = self.make_crop(current_frame, crop_size)

        cur_shift, response = self.pre_process(cropped_first_frame)
        self.distance_consecutive_frames = self.max_distance_consecutive_frames_between_points(self.good_shift,
                                                                                               cur_shift)
        if response > 0.20 and self.distance_consecutive_frames < self.distance_threshold:
            self.good_shift = cur_shift

        else:
            self.good_shift = cur_shift

        return self.good_shift


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Stabilization")
    parser.add_argument('--first_frame', type=np.ndarray, required=True, help='np.ndarray from the 1st frame')
    parser.add_argument('--current_frame', type=np.ndarray, help='np.ndarray from each consecutive frame')
    parser.add_argument('--crop_size', type=float, default=1,
                        help='Crop size as a fraction of original size (default: 1)')
    parser.add_argument('--distance_threshold', type=float, default=4, help='Distance threshold (default: 4)')

    args = parser.parse_args()

    stabilizer = VideoStabilization(first_frame=args.first_frame, current_frame=args.current_frame,
                                    crop_size=args.crop_size, distance_threshold=args.distance_threshold)
    stabilizer.stabilization_just_shifts()
    init_loggers()
