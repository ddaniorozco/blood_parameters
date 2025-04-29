#%% imports
from typing import Sequence

import numpy as np
from loguru import logger
from skimage.color import gray2rgb

from interfaces.inference.image_inference_model import ImageInferenceModel
from primitives.src.bounding_box import BoundingBox2D, BoundingBoxCollection

#%% Wide field capillary detection utility functions

def detect_capillaries(wide_capillary_detector: ImageInferenceModel, image: np.ndarray):
    """
    Detect capillaries in an image using a wide capillary detector.

    Parameters:
    -----------
    wide_capillary_detector : ImageInferenceModel
        The wide capillary detector model.
    image : np.ndarray
        The image in which to detect capillaries.
    """
    detections = wide_capillary_detector.infer(gray2rgb(image))[0]  # TODO - to remove once inference is set
    filtered_detections = filter_nested_bboxes(detections)
    return filtered_detections

def filter_nested_bboxes(detections: list[list[int]]) -> list[list[int]]:
    """
    Filter nested bounding boxes from a list of bounding boxes.

    Parameters:
    -----------
    detections : list[list[int]]
        A list of bounding boxes where each bounding box is represented by a list of integers.

    Returns:
    --------
    list[list[int]]
        A list of bounding boxes with nested bounding boxes removed.
    """
    bboxes = [BoundingBox2D(bbox[:4]) for bbox in detections]
    filtered_bboxes = BoundingBoxCollection.filter_nested_bboxes(bboxes)
    filtered_detections = np.array([detections[i] for i, bbox in enumerate(bboxes) if bbox in filtered_bboxes])
    return filtered_detections

def crop_bounding_boxes_from_image(bboxes, image):
    """
    Crop bounding boxes from an image.
    """
    if isinstance(bboxes, np.ndarray) and len(bboxes.shape) == 1:
        bboxes = [bboxes]
        bboxes = [tuple(int(coordinate) for coordinate in bbox[:4]) for bbox in bboxes]
    else:
        bboxes = [tuple(int(coordinate) for coordinate in bbox[:4]) for bbox in bboxes]

    crops = []

    for bbox in bboxes:
        crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        crops.append(crop)

    return crops, bboxes

def sort_detections_by_distance(query_point: np.ndarray | Sequence, detections: np.ndarray):
        """
        Sorts the given detections by their distance to a query point.

        Parameters:
        query_point (np.ndarray | Sequence): The reference point to which distances are calculated. 
                                             If a sequence is provided, it is converted to a numpy array.
                                             If a numpy array is provided, the center of the image is used.
        detections (np.ndarray): An array of detections where each detection is represented by a bounding box 
                                 and additional information. The bounding box is expected to be in the first 
                                 four columns of each detection.

        Returns:
        np.ndarray: The detections sorted by their distance to the query point, from closest to farthest.
        """
        if isinstance(query_point, Sequence):
            image_center = np.array(query_point)
        else:
            image_center = np.array([query_point.shape[1] / 2, query_point.shape[0] / 2])

        bounding_boxes = detections[:, :4]
        bboxes_centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in bounding_boxes])
        distances = np.linalg.norm(bboxes_centers - image_center, axis=1)

        closest_indices = np.argsort(distances)
        closest_bboxes_to_center = detections[closest_indices]
        logger.info(f'Returning the closest capillary to location = {image_center.tolist()}')
        return closest_bboxes_to_center
