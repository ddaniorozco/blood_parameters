import functools
import itertools
import multiprocessing as mp
import pathlib
import time
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from skimage.color import gray2rgb

import best_segment.wide.utils.cross_section_utils as csu
import best_segment.wide.utils.skeleton_utils as su
from best_segment.wide.utils.detect_crop_utils import WideCapillaryDetector
from best_segment.wide.utils.masking_utils import Masker
from best_segment.wide.utils.sharpness_utils import SharpnessEstimator
from capillary_detector.wide.src.inference import YoloWideCapillaryDetectorModel
from capillary_segmentation.wide.src.inference import YoloWideCapillarySegmentorModel
from iae_config import ModelPaths
from interfaces.inference.image_inference_model import ImageInferenceModel


# ----------------------------------------------------------------------------------------------------------------------
#                                                
# ----------------------------------------------------------------------------------------------------------------------


def calculate_sharpness(crops_masks_bboxes_chunk, max_capillaries):
    sharpness_th = 23
    capillaries_found = 0
    selected_items = []
    sharpness_estimator = SharpnessEstimator()

    if max_capillaries is None:
        max_capillaries = len(crops_masks_bboxes_chunk)

    for crop, mask, bbox in crops_masks_bboxes_chunk:
        if mask is None:
            continue
        sharpness_2append = sharpness_estimator.estimate_sharpness(crop, mask)
        if sharpness_2append >= sharpness_th:
            # logger.info(f'Producing Skeletons')
            # capillaries_found acts as the idx_crop
            selected_items.append((capillaries_found, crop, bbox, mask, sharpness_2append))
            capillaries_found += 1

        if max_capillaries and capillaries_found >= max_capillaries:
            break

    # If no capillaries were found, you might want to log or handle it
    if capillaries_found == 0:
        logger.info('No capillaries found that meet the sharpness threshold.')

    return selected_items


def compute_features(crops_masks_bboxes_chunk, depth, exposure_time, wide_capillary_segmentor, max_capillaries,
                     max_segments_per_capillary):
    segment_size = 20  # TODO - move these magic numbers to some config
    features_2append = []
    additional_output = {'masks': [], 'skeletons': [], 'detections': []}
    masker = Masker(wide_capillary_segmentor=wide_capillary_segmentor)
    cross_sectioner = csu.CrossSection()

    crops_masks_bboxes_chunk = calculate_sharpness(crops_masks_bboxes_chunk, max_capillaries)

    logger.info('Starting Features Calculations')
    for idx_crop, crop, bbox, mask, sharpness in crops_masks_bboxes_chunk:
        # logger.info(f'Crop: {idx_crop + 1}/{len(crops_masks_bboxes_chunk)}')
        # logger.info(f'Producing Skeletons')
        skeletons = masker.produce_skeletons(crop, mask)

        additional_output['detections'].append(bbox)
        additional_output['masks'].append(mask)
        additional_output['skeletons'].append(skeletons)
        # logger.info(f'{len(skeletons)} Skeletons were detected')

        segments_2append = []
        normals_crop = []

        for idx, sk in enumerate(skeletons):
            # logger.info(f'Skeleton: {idx + 1}/{len(skeletons)}')

            features_sk = []
            normals_sk = []
            # logger.info(f'Sampling Skeleton')

            spline_x, spline_y, ts, _ = su.sample_skeleton(sk, segment_size)
            sampled_skeleton = np.array([spline_y(ts), spline_x(ts)]).T
            segments_2append.append(sampled_skeleton)
            # logger.info(f'{len(sampled_skeleton)} Segments sampled')
            if max_segments_per_capillary is None:
                num_segments = len(ts)
            else:
                num_segments = min(len(ts), max_segments_per_capillary)

            for i, ts0 in enumerate(ts[:num_segments]):
                # logger.info(f'Analyzing Segment: {i + 1}/{len(ts)}')

                t = np.linspace(ts0 - segment_size / 2, ts0 + segment_size / 2, 10)
                normals_calc = su.calculate_skeleton_normals((spline_x, spline_y, t))
                # logger.info(f'Calculating Curvature')

                curvatures = su.calculate_skeleton_curvature((spline_x, spline_y, t))

                # logger.info(f'Calculating Fit Parameters')

                cross_sectioner.set_normal_cross_section_points(np.array([spline_y(t), spline_x(t)]).T,
                                                                normals_calc)

                _, fit_params = cross_sectioner.fit(crop, return_dict=True, average=True)
                features_sk.append({"depth": depth,
                                    "exposure_time": exposure_time,
                                    "sharpness": sharpness,
                                    "curvature": np.mean(curvatures),
                                    "segment_point": np.array(
                                        [spline_y(ts0) + bbox[1], spline_x(ts0) + bbox[0]]).astype(int),
                                    "idx_crop": int(idx_crop)})

                features_sk[-1].update(fit_params)
                normals_mean = np.mean(normals_calc, axis=0)
                normals_sk.append(normals_mean / np.linalg.norm(normals_mean))
                features_sk[-1].update({"angle": np.arctan2(normals_mean[0], normals_mean[1]) * 180 / np.pi})

            normals_crop.append(normals_sk)
            features_2append.append(features_sk)

    logger.info(f'Finished Features Calculations for {len(additional_output["detections"])} crops')
    return features_2append, additional_output


def best_segment(rfc, features):
    logger.info('Starting analysis of Best Segments')

    ignored_columns = ['exposure_time', 'segment_point', 'angle', 'idx_crop']

    features = list(itertools.chain(*features))
    segment_points = [f.pop('segment_point').tolist() for f in features]
    idx_crops = [f.pop('idx_crop') for f in features]
    features_df = pd.DataFrame(features)
    features_df['segment_point'] = segment_points
    features_df['idx_crop'] = idx_crops
    columns_to_standardize = features_df.columns.difference(ignored_columns)
    features_df = features_df.dropna()
    assert not features_df.empty, "Segment Features Data Frame is EMPTY"
    logger.info(f'Classifying {len(features_df)} Segments with RandomForest')
    probabilities = rfc.predict_proba(features_df[columns_to_standardize])
    scores = probabilities[:, 1]
    sorted_indices = np.argsort(scores)[::-1]
    sorted_segments = features_df.iloc[sorted_indices]
    sorted_scores = scores[sorted_indices]

    x_y_theta_score = []
    for i, item in enumerate(zip(sorted_segments["segment_point"], sorted_segments["angle"], sorted_scores,
                                 sorted_segments["idx_crop"])):
        y, x = item[0]
        y = float(y)
        x = float(x)
        theta = float(item[1])
        score = float(item[2])

        res = [x, y, theta, score, item[3]]  # x, y, theta, score, bbox_idx

        # corresponding_bbox = bboxes[res[-1]]
        # res.extend(corresponding_bbox)

        x_y_theta_score.append(res)

    return x_y_theta_score


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class SegmentDetectionModel:

    def __init__(self,
                 wide_capillary_detector: ImageInferenceModel = None,
                 wide_capillary_segmentor: ImageInferenceModel = None,
                 feature_classifier_weights_path: str | pathlib.Path = ModelPaths.SEGMENT_FEATURE_CLASSIFIER,
                 default_device: torch.device = None,
                 half_precision: bool = False,
                 to_device_on_init: bool = False,
                 to_cpu_after_infer: bool = False,
                 verbosity: int = 0):

        # Default values
        self.multiprocessing = False
        self.max_capillaries = None
        self.max_segments_per_capillary = None

        # Models initialization
        self.wide_capillary_detector = wide_capillary_detector
        self.wide_capillary_segmentor = wide_capillary_segmentor
        self.default_device = default_device
        self.half_precision = half_precision
        self.to_device_on_init = to_device_on_init
        self.to_cpu_after_infer = to_cpu_after_infer
        self.verbosity = verbosity
        self._models_init()
        logger.info(f'Successfully initialized inference models')

        # Tree Classifier Load:
        self.rfc = joblib.load(feature_classifier_weights_path)
        logger.info(f'Successfully loaded TreeClassifier weights from {feature_classifier_weights_path}')

    def _models_init(self):
        """
        Initialize the frame classifier and capillary detector.
        """
        models = {'wide_capillary_detector': (self.wide_capillary_detector, YoloWideCapillaryDetectorModel),
                  'wide_capillary_segmentor': (self.wide_capillary_segmentor, YoloWideCapillarySegmentorModel)}

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

    def _iterate_crops_masks_bboxes(self, crops, masks, bboxes, depth, exposure_time):

        logger.info(f'Iterating each crop')

        # non_picklable_attrs = check_picklability(self)
        # if non_picklable_attrs:
        #     raise ValueError(f'Non-picklable attributes: {non_picklable_attrs}')

        num_processes = 4
        crops_masks_bboxes = list(zip(crops, masks, bboxes))
        chunk_size = len(crops) // num_processes
        chunks = [crops_masks_bboxes[i:i + chunk_size] for i in range(0, len(crops), chunk_size)]

        calculate_features_partial = functools.partial(
            compute_features,
            depth=depth,
            exposure_time=exposure_time,
            wide_capillary_segmentor=self.wide_capillary_segmentor,
            max_capillaries=self.max_capillaries,
            max_segments_per_capillary=self.max_segments_per_capillary
        )

        logger.info('Calculating Features')
        start_time = time.time()

        with mp.Pool(processes=num_processes) as pool:
            features = pool.map(calculate_features_partial, chunks)

        # with mp.Pool(processes=num_processes) as pool:
        #     features = pool.map(calculate_features, chunks)

        elapsed_time = time.time() - start_time
        logger.info(f'Time taken for calculating features: {elapsed_time:.2f} seconds for {len(crops)} '
                    f'crops with num_processes = {num_processes}')

        # flattened_results = [item for sublist in results for item in sublist]

        return features

    def _sort_detections_by_distance(self, query_point: np.ndarray | Sequence, detections: np.ndarray):

        if isinstance(query_point, Sequence):
            image_center = np.array(query_point)
        else:
            image_center = np.array([query_point.shape[1] / 2, query_point.shape[0] / 2])

        bounding_boxes = detections[:, :4]
        bboxes_centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in bounding_boxes])
        distances = np.linalg.norm(bboxes_centers - image_center, axis=1)

        if self.max_capillaries == 1:
            closest_index = np.argsort(distances)[:10]
            closest_bbox_to_center = detections[closest_index]
            logger.info(f'Returning the closest capillary to location = {image_center.tolist()}')
            return closest_bbox_to_center
        else:
            if self.max_capillaries is not None:
                closest_indices = np.argsort(distances)[:(int(self.max_capillaries * 1.5))]
                logger.info(f'Returning the closest {self.max_capillaries} capillaries to location '
                            f'({query_point[0]},{query_point[1]})')
                closest_indices = closest_indices[:self.max_capillaries]
            else:
                closest_indices = np.argsort(distances)[:self.max_capillaries]
            closest_bboxes_to_center = detections[closest_indices]
            return closest_bboxes_to_center

    @staticmethod
    def _convert_to_float(lst):
        return [float(x) for x in lst]

    def compute_features(self, image: np.ndarray,
                         depth: float,
                         exposure_time: int,
                         multiprocessing: bool = False,
                         query_point: Optional[tuple[float, float]] = None,
                         max_capillaries: Optional[int] = None,
                         max_segments_per_capillary: Optional[int] = None) -> dict:

        self.multiprocessing = multiprocessing
        self.max_capillaries = max_capillaries
        self.max_segments_per_capillary = max_segments_per_capillary

        logger.info('Starting Smart Segment Selection')

        detections = self.wide_capillary_detector.infer(gray2rgb(image))[0]  # TODO - to remove once inference is set
        # TODO: filter nested bboxes
        if len(detections) == 0:
            logger.warning('Found no capillaries in wide image')
            return {}
        logger.info('Finished detecting capillaries')

        if query_point is None:
            logger.info('Query point not provided - using center of the image')
            detections = self._sort_detections_by_distance(image, detections)
        else:
            logger.info(f'Query point provided - ({query_point[0]},{query_point[1]})')
            detections = self._sort_detections_by_distance(query_point, detections)

        crops, bboxes = WideCapillaryDetector.crop_bounding_boxes_from_image(detections, image)
        bboxes = [tuple(self._convert_to_float(bbox)) for bbox in bboxes]
        logger.info(f'{len(bboxes)} Capillaries were detected')

        logger.info(f'Producing Masks of {len(crops)} crops')
        stack_crops = [gray2rgb(crop) for crop in crops]  # TODO - to remove once inference is set
        masks = self.wide_capillary_segmentor.batch_infer(stack_crops)[1]
        if len(masks) == 0:
            logger.warning('No masks generated')
            return {}
        logger.info(f'Finished Masks of {len(masks)} crops')

        if self.multiprocessing:
            raise NotImplementedError('Multiprocessing needs testing and refactoring')

        else:
            crops_masks_bboxes = list(zip(crops, masks, bboxes))
            features, additional_output = compute_features(crops_masks_bboxes, depth, exposure_time,
                                                           self.wide_capillary_segmentor, self.max_capillaries,
                                                           self.max_segments_per_capillary)
            # TODO - return bbox score and conf along side the segment proposals (bbox of length 6 instead 4)
            if not features:
                return {}

            segment_proposals = best_segment(self.rfc, features)

            logger.info('Smart Segment Computation Completed')

            outputs = {'segment_proposals': segment_proposals}
            outputs.update(additional_output)
            return outputs
