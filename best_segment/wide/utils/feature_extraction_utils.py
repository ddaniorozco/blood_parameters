# %% Importing libraries
import numpy as np

import best_segment.wide.utils.cross_section_utils as csu
import best_segment.wide.utils.skeleton_utils as su
from best_segment.wide.utils.sharpness_utils import SharpnessEstimator


# %% Feature extraction utility functions

def update_idx_crop(data, idx=0):
        """
        Recursively updates the 'idx_crop' key in a nested dictionary or list structure with a serial index.

        Args:
            data (dict or list): The nested dictionary or list structure to update.
            idx (int, optional): The starting index for the serial update. Defaults to 0.

        Returns:
            int: The next index value after the update.

        Example:
            data = {
                'a': {'idx_crop': 0, 'b': {'idx_crop': 0}},
                'c': [{'idx_crop': 0}, {'idx_crop': 0}]
            }
            update_idx_crop(data)
            # data will be updated to:
            # {
            #     'a': {'idx_crop': 0, 'b': {'idx_crop': 1}},
            #     'c': [{'idx_crop': 2}, {'idx_crop': 3}]
            # }
        """
        if isinstance(data, dict):
            for k, v in data.items():
                if k == 'idx_crop':
                    data[k] = idx
                    idx += 1
                else:
                    idx = update_idx_crop(v, idx)
        elif isinstance(data, list):
            for item in data:
                idx = update_idx_crop(item, idx)
        return idx

def extract_capillary_segment_features(skeleton_splines, t0, crop, mask, cross_sectioner, features_list, segment_size):
    """
    Extracts features from a capillary segment based on the provided skeleton splines and other parameters.

    Parameters:
    skeleton_splines (tuple): A tuple containing the x and y coordinates of the skeleton splines.
    t0 (float): The central point of the segment to be analyzed.
    crop (ndarray): The cropped image data.
    mask (ndarray): The mask to be applied to the image data.
    cross_sectioner (object): An object that handles cross-section operations.
    features_list (list): A list of features to be extracted.
    segment_size (float): The size of the segment to be analyzed.

    Returns:
    dict: A dictionary containing the extracted features, including curvature, angle, fit parameters, and capillary features.
    """
    spline_x, spline_y = skeleton_splines
    t = np.linspace(t0-segment_size/2, t0+segment_size/2, 10)

    normals = su.calculate_skeleton_normals((spline_x, spline_y, t)) 
    normals_mean = np.mean(normals, axis=0)
    angle = np.arctan2(normals_mean[0], normals_mean[1]) * 180 / np.pi          
    curvatures = su.calculate_skeleton_curvature((spline_x, spline_y, t)) if "curvature" in features_list else np.nan
    curvature = np.mean(curvatures)
    curv_angle_dict = {"curvature": curvature, "angle": angle}
    cross_sectioner.set_normal_cross_section_points(np.array([spline_y(t), spline_x(t)]).T, normals)
    cross_sections = cross_sectioner(crop, cross_sectioner.points, False)
    _, fit_params = csu.super_gaussian_fit(cross_sectioner.raw_normal_axis, cross_sections, return_dict=True, average=True, features_list=features_list)
    capi_features = cross_sectioner.get_capillary_features(crop, mask, css=cross_sections, return_dict=True, average=True, features_list=features_list)
    combined_features = {**curv_angle_dict, **fit_params, **capi_features}
    return combined_features

def iterate_over_crops(crops: np.ndarray | list, masks: np.ndarray | list, bboxes: np.ndarray | list, 
                           sharpness_estimator: SharpnessEstimator = None, skeletonizer: su.Skeletonizer = None, 
                           cross_sectioner: csu.CrossSection = None, features_list: list = None, segment_size: int = 20, 
                           sharpness_th: int = 23, max_segments_per_capillary: int = 4, depth: float = np.nan,
                           exposure_time: int = np.nan, intersection_angle: float = np.nan, multiprocessing=False) -> dict:
        """
        Iterates over the crops and masks to extract features.
        
        Args:
            crops (ndarray): The crops.
            masks (ndarray): The masks.
            bboxes (ndarray): The bounding boxes.
        """
        valid_crops = []
        valid_bboxes = []
        valid_sharpness = []
        valid_masks = []
        valid_skeletons = []
        valid_segments = []
        valid_features = []
        idx_crop = -1
        if multiprocessing:
            crops = [crops]
            masks = [masks]
            bboxes = [bboxes]
        for crop, mask, bbox in zip(crops, masks, bboxes):
            if "sharpness" in features_list:
                sharpness = sharpness_estimator.estimate_sharpness(crop, mask)  # TODO: Check for correlation between sharpness and score. If there is a strong one, consider using it as a filter for the crops.
            else:
                sharpness = np.nan
            if sharpness < sharpness_th:
                continue
            skeletons = skeletonizer.produce_skeletons(image=crop.astype(float), mask=mask.astype(float))
            if len(skeletons) == 0:
                continue
            idx_crop += 1
            valid_crops.append(crop)
            valid_bboxes.append(bbox)
            valid_sharpness.append(sharpness)
            valid_masks.append(mask)
            valid_skeletons.append(skeletons)
            segments = []
            features = []
            segments_per_skeleton_segment = su.distribute_segments_on_skeleton(skeletons, max_segments_per_capillary)
            for sk, segs_per_sk in zip(skeletons, segments_per_skeleton_segment):
                features_sk = []
                spline_x, spline_y, ts, segment_size = su.sample_skeleton(sk, segment_size, use_two_splines=True, max_segments_per_capillary=segs_per_sk)
                sampled_skeleton = np.array([spline_y(ts), spline_x(ts)]).T
                segments.append(sampled_skeleton)
                for ts0 in ts:
                    features_sk.append({"depth": depth,
                                        "exposure_time": exposure_time,
                                        "intersection_angle": intersection_angle,
                                        "sharpness": sharpness,
                                        "segment_point": np.array([spline_y(ts0) + bbox[1], spline_x(ts0) + bbox[0]]).astype(int), 
                                        "idx_crop": int(idx_crop)})
                    
                    capillary_segment_features = extract_capillary_segment_features((spline_x, spline_y), ts0, crop, mask, cross_sectioner, features_list, segment_size)
                    features_sk[-1].update(capillary_segment_features)
                features.append(features_sk)
            valid_features.append(features)
            valid_segments.append(segments)
        outputs = {'crops': valid_crops, 'bboxes': valid_bboxes, 'sharpness': valid_sharpness, 'masks': valid_masks, 'skeletons': valid_skeletons, 'segments': valid_segments, 'features': valid_features}
        return outputs

    
#%% Old code
# class SegmentDetectionModel:
#     """
#     SegmentDetectionModel is a class designed for detecting and extracting features from capillary segments in images.
#     Attributes:
#         features_list (list): List of features to extract.
#         multiprocessing (bool): Flag for multiprocessing.
#         max_capillaries (int): Maximum number of capillaries to detect.
#         max_segments_per_capillary (int): Maximum number of segments per capillary.
#         segment_size (int): Size of each segment.
#         sharpness_th (int): Threshold for sharpness.
#         wide_capillary_detector (ImageInferenceModel): Model for detecting wide capillaries.
#         wide_capillary_segmentor (ImageInferenceModel): Model for segmenting wide capillaries.
#         default_device (torch.device): Default device for computation.
#         half_precision (bool): Flag for half precision.
#         to_device_on_init (bool): Flag to move models to device on initialization.
#         to_cpu_after_infer (bool): Flag to move models to CPU after inference.
#         verbosity (int): Verbosity level.
#         feature_classifier (object): Classifier for features.
#         sharpness_estimator (SharpnessEstimator): Estimator for sharpness.
#         skeletonizer (Skeletonizer): Skeletonizer for segments.
#         cross_sectioner (CrossSection): Cross sectioner for segments.
#         crops (list): List of crops.
#         bboxes (list): List of bounding boxes.
#         detections (list): List of detections.
#         sharpness (list): List of sharpness values.
#         masks (list): List of masks.
#         skeletons (list): List of skeletons.
#         segments (list): List of segments.
#         features (list): List of features.
#         normals (list): List of normals.
#         best_segment (tuple): Best segment point and angle.
#         ignored_columns (list): List of ignored columns.
#         cropping_time (float): Time taken for cropping.
#         masking_time (float): Time taken for masking.
#         skeleton_time (list): List of times taken for skeleton processing.
#         feature_time (list): List of times taken for feature extraction.
#         crops_times (list): List of times taken for cropping.
#         skeletons_times (list): List of times taken for skeleton processing.
#         sharpness_times (list): List of times taken for sharpness estimation.
#         skeletoning_times (list): List of times taken for skeletoning.
#         sampling_times (list): List of times taken for sampling.
#         cross_sectioning_times (list): List of times taken for cross-sectioning.
#         fitting_times (list): List of times taken for fitting.
#         capi_feat_times (list): List of times taken for capillary feature extraction.
#         init_time (float): Initialization time.
#     Methods:
#         __init__: Initializes the SegmentDetectionModel with given parameters.
#         _models_init: Initializes the frame classifier and capillary detector.
#         detect_crops: Detects the crop and bounding box for the given image.
#         _convert_to_float: Converts a list of values to float.
#         mask_crops: Masks the given crop.
#         get_features: Extracts features from the given image.
#         compute_features: Computes features and returns sorted segments.
#         get_sorted_segments: Sorts segments by the probability of being a capillary.
#         get_best_segment: Gets the best segment from the given image.
#         _sort_detections_by_distance: Sorts detections by distance from a query point.
#         plot_ordered_segments: Plots the ordered segments on the given image.
#         get_timings: Returns the timings for the different steps of the feature extraction.
#         plot_timings: Plots the timings for the different steps of the feature extraction.
# """
#     def __init__(self,
#                  wide_capillary_detector: ImageInferenceModel = None,
#                  wide_capillary_segmentor: ImageInferenceModel = None,
#                  feature_classifier_weights_path: str | pathlib.Path = ModelPaths.SEGMENT_FEATURE_CLASSIFIER,
#                  default_device: torch.device = None,
#                  half_precision: bool = False,
#                  to_device_on_init: bool = False,
#                  to_cpu_after_infer: bool = False,
#                  verbosity: int = 0):
#         """ Possible features to extract:
#         self.features_list = ["depth", "exposure_time", "sharpness", "intersection_angle", 
#                               "curvature", "fit_center", "fit_amp", "fit_width", "fit_fwhm", 
#                               "fit_exponent", "fit_offset", "mask_width", "capillary_background_ratio", 
#                               "capillary_background_min_max", "capillary_std", "capillary_sharpness"]
#             Features without fitting:
#         self.features_list = ["depth", "exposure_time", "sharpness", "intersection_angle", 
#                               "curvature", "mask_width", "capillary_background_ratio", 
#                               "capillary_background_min_max", "capillary_std", "capillary_sharpness"]

#         Features without capillary features:
#         self.features_list = ["depth", "exposure_time", "sharpness", "intersection_angle", 
#                               "curvature", "fit_center", "fit_amp", "fit_width", "fit_fwhm", 
#                               "fit_exponent", "fit_offset"]
#         """
#         self.features_list = ["depth", "exposure_time", "sharpness", "intersection_angle", 
#                               "curvature", "fit_center", "fit_amp", "fit_width", "fit_fwhm", 
#                               "fit_exponent", "fit_offset"]
        
#         # Begin initialization timing
#         t0 = time.time()
        
#         # Default values
#         self.multiprocessing = False
#         self.max_capillaries = None
#         self.max_segments_per_capillary = None
#         self.segment_size = 20
#         self.sharpness_th = 23

#         # Models initialization
#         self.wide_capillary_detector = wide_capillary_detector
#         self.wide_capillary_segmentor = wide_capillary_segmentor
#         self.default_device = default_device
#         self.half_precision = half_precision
#         self.to_device_on_init = to_device_on_init
#         self.to_cpu_after_infer = to_cpu_after_infer
#         self.verbosity = verbosity
#         self._models_init()
#         logger.info(f'Successfully initialized inference models')

#         # Feature Classifier Load:
#         self.feature_classifier = joblib.load(feature_classifier_weights_path)
#         logger.info(f'Successfully loaded FeatureClassifier weights from {feature_classifier_weights_path}')

#         # Set feature extraction modules
#         self.sharpness_estimator = SharpnessEstimator()
#         self.skeletonizer = su.Skeletonizer(segment_size=self.segment_size)
#         self.cross_sectioner = csu.CrossSection()
#         logger.info(f'Successfully initialized feature extraction modules')
#         # Set detections and features lists
#         self.crops = []
#         self.bboxes = []
#         self.detections = []
#         self.sharpness = []
        
#         self.masks = []
#         self.skeletons = []
#         self.segments = []
        
#         self.features = []
#         self.normals = []
#         self.best_segment = None
#         self.ignored_columns = ['name', 'velocity', 'image_root', 'data_file', 'index_in_file', 'segment_point', 'idx_crop', 'angle', 'exposure_time', 'intersection_angle',
#                                 "mask_width", "capillary_background_ratio", 
#                                 "capillary_background_min_max", "capillary_std", 
#                                 "capillary_sharpness"]

#         # Set timings
#         self.cropping_time = None
#         self.masking_time = None
#         self.skeleton_time = []
#         self.feature_time = []
#         self.crops_times = []
#         self.skeletons_times = []
#         self.sharpness_times = []
#         self.skeletoning_times = []
#         self.sampling_times = []
#         self.cross_sectioning_times = []
#         self.fitting_times = []
#         self.capi_feat_times = []
#         self.init_time = time.time() - t0
#         logger.info(f'Successfully initialized SegmentDetectionModel')
    
#     def _models_init(self):
#         """
#         Initialize the frame classifier and capillary detector.
#         """
#         models = {'wide_capillary_detector': (self.wide_capillary_detector, YoloWideCapillaryDetectorModel),
#                   'wide_capillary_segmentor': (self.wide_capillary_segmentor, YoloWideCapillarySegmentorModel)}

#         for attr, (model, model_class) in models.items():
#             if model is not None:
#                 model.default_device = self.default_device
#                 model.half_precision = self.half_precision
#                 model.to_device_on_init = self.to_device_on_init
#                 model.to_cpu_after_infer = self.to_cpu_after_infer
#                 model.verbosity = self.verbosity
#                 setattr(self, attr, model)
#             else:
#                 setattr(self, attr, model_class(
#                     default_device=self.default_device,
#                     half_precision=self.half_precision,
#                     to_device_on_init=self.to_device_on_init,
#                     to_cpu_after_infer=self.to_cpu_after_infer,
#                     verbosity=self.verbosity
#                 ))


#     def detect_crops(self, image: np.ndarray, query_point: np.ndarray | Sequence | None=None):
#         """
#         image (np.ndarray): The image in which to detect capillaries.
#         query_point (np.ndarray | Sequence | None, optional): The point to use as a reference for sorting detections. 
#             If None, the center of the image is used. Defaults to None.
#         tuple: A tuple containing the crops, bounding boxes, and detections.
#         Detects the crop and bounding box for the given image.
        
#         Args:
#             image_path (str): The path to the image.
            
#         Returns:
#             tuple: A tuple containing the crops and bounding boxes.
#         """
#         detections = self.wide_capillary_detector.infer(gray2rgb(image))[0]  # TODO - to remove once inference is set
#         bboxes = [BoundingBox2D(bbox[:4]) for bbox in detections]
#         filtered_bboxes = BoundingBoxCollection.filter_nested_bboxes(bboxes)
#         detections = np.array([detections[i] for i, bbox in enumerate(bboxes) if bbox in filtered_bboxes])
#         num_of_capillaries = len(detections)
#         if num_of_capillaries == 0:
#             logger.info('No Capillaries were detected')
#             return [], [], []
#         if self.max_capillaries <= 0:
#             logger.info('max_capillaries set to 0 - skipping detection')
#             return [], [], []
        
#         num_of_capillaries = min(num_of_capillaries, self.max_capillaries) if self.max_capillaries is not None else num_of_capillaries
        
        
#         logger.info(f'{len(detections)} Capillaries were detected, from which {num_of_capillaries} will be considered')
#         if query_point is None:
#             logger.info('Query point not provided - using center of the image')
#             detections = self._sort_detections_by_distance(image, detections)
#         else:
#             logger.info(f'Query point provided - ({query_point[0]},{query_point[1]})')
#             detections = self._sort_detections_by_distance(query_point, detections)
        
#         detections = detections[:num_of_capillaries]
#         logger.info(f'Finished sorting detections - returning {len(detections)} detections')
        
       
#         crops, bboxes = WideCapillaryDetector.crop_bounding_boxes_from_image(detections, image)
#         logger.info(f'Finished detecting capillaries - returning {len(crops)} crops')
#         bboxes = [tuple(self._convert_to_float(bbox)) for bbox in bboxes]
#         return crops, bboxes, detections

#     @staticmethod
#     def _convert_to_float(lst):
#         return [float(x) for x in lst]
    

#     def mask_crops(self, crops: np.ndarray | list) -> np.ndarray | list:
#         """
#         Masks the given crop.
        
#         Args:
#             crop (ndarray): The cropped image.
            
#         Returns:
#             ndarray: The masked image.
#         """
#         crops = [gray2rgb(crop) for crop in crops]
#         masks = self.wide_capillary_segmentor.batch_infer(crops)[1] # TODO - to remove once inference is set
#         logger.info(f'Finished producing {len(masks)} masks for {len(crops)} crops')
#         return masks


#     @staticmethod
#     def iterate_over_crops(crops: np.ndarray | list, masks: np.ndarray | list, bboxes: np.ndarray | list, 
#                            sharpness_estimator: SharpnessEstimator = None, skeletonizer: su.Skeletonizer = None, 
#                            cross_sectioner: csu.CrossSection = None, features_list: list = None, segment_size: int = 20, 
#                            sharpness_th: int = 23, max_segments_per_capillary: int = 4, depth: float = np.nan,
#                            exposure_time: int = np.nan, intersection_angle: float = np.nan, multiprocessing=False) -> dict:
#         """
#         Iterates over the crops and masks to extract features.
        
#         Args:
#             crops (ndarray): The crops.
#             masks (ndarray): The masks.
#             bboxes (ndarray): The bounding boxes.
#         """
#         valid_crops = []
#         valid_bboxes = []
#         valid_sharpness = []
#         valid_masks = []
#         valid_skeletons = []
#         valid_segments = []
#         valid_features = []
#         valid_normals = []
#         idx_crop = -1
#         if multiprocessing:
#             crops = [crops]
#             masks = [masks]
#             bboxes = [bboxes]
#         for crop, mask, bbox in zip(crops, masks, bboxes):
#             # t0_crop = time.time()
#             if "sharpness" in features_list:
#                 sharpness = sharpness_estimator.estimate_sharpness(crop, mask)  # TODO: Check for correlation between sharpness and score. If there is a strong one, consider using it as a filter for the crops.
#             else:
#                 sharpness = np.nan
#             # self.sharpness_times.append(time.time() - t0_crop)
#             if np.isnan(sharpness) or sharpness >= sharpness_th:
#                 skeletons = skeletonizer.produce_skeletons(image=crop.astype(float), mask=mask.astype(float))
#                 if len(skeletons) > 0:
#                     idx_crop += 1
#             #         self.skeletoning_times.append(time.time() - t0_crop - self.sharpness_times[-1])
#                     skeleton_segments_lengths = [su.skeleton_length(sk) for sk in skeletons]
#                     total_length = sum(skeleton_segments_lengths)
#                     segments_per_skeleton_segment = [int(np.floor(length / total_length * max_segments_per_capillary)) for length in skeleton_segments_lengths]
#                     total_segments = np.sum(segments_per_skeleton_segment)
#                     segment_diff = max_segments_per_capillary - total_segments
#                     segments_per_skeleton_segment_modulo = [(length * max_segments_per_capillary) % total_length for length in skeleton_segments_lengths]
#                     while segment_diff > 0:
#                         max_idx = np.argmax(segments_per_skeleton_segment_modulo)
#                         segments_per_skeleton_segment[max_idx] += 1
#                         segment_diff -= 1
#                         segments_per_skeleton_segment_modulo[max_idx] = 0

#                     valid_crops.append(crop)
#                     valid_bboxes.append(bbox)
#                     valid_sharpness.append(sharpness)
#                     valid_masks.append(mask)
#                     valid_skeletons.append(skeletons)
#                     segments = []
#                     features = []
#                     normals_crop = []
#                     for sk, segs_per_sk in zip(skeletons, segments_per_skeleton_segment):
#             #             t0_sk = time.time()
#                         features_sk = []
#                         normals_sk = []
#                         spline_x, spline_y, ts, segment_size = su.sample_skeleton(sk, segment_size, use_two_splines=True, max_segments_per_capillary=segs_per_sk)
#                         sampled_skeleton = np.array([spline_y(ts), spline_x(ts)]).T
#             #             self.sampling_times.append(time.time() - t0_sk)
#                         segments.append(sampled_skeleton)
#                         for ts0 in ts:
#                             features_sk.append({})
#             #                 t0_feature = time.time()
#                             t = np.linspace(ts0-segment_size/2, ts0+segment_size/2, 10)
#                             normals = su.calculate_skeleton_normals((spline_x, spline_y, t))
#                             curvatures = su.calculate_skeleton_curvature((spline_x, spline_y, t)) if "curvature" in features_list else np.nan
#                             cross_sectioner.set_normal_cross_section_points(
#                                 np.array([spline_y(t), spline_x(t)]).T, normals)
#                             cross_sections = cross_sectioner(crop, cross_sectioner.points, False)
#             #                 self.cross_sectioning_times.append(time.time() - t0_feature)
#                             _, fit_params = csu.super_gaussian_fit(cross_sectioner.raw_normal_axis, cross_sections,
#                                                                    return_dict=True, average=True, features_list=features_list)
#             #                 self.fitting_times.append(time.time() - t0_feature - self.cross_sectioning_times[-1])
#                             capi_features = cross_sectioner.get_capillary_features(crop, mask, css=cross_sections, return_dict=True, average=True, features_list=features_list)
#             #                 self.capi_feat_times.append(time.time() - t0_feature - self.cross_sectioning_times[-1] - self.fitting_times[-1])
#                             features_sk[-1].update({"depth": depth,
#                                                 "exposure_time": exposure_time,
#                                                 "intersection_angle": intersection_angle,
#                                                 "sharpness": sharpness,
#                                                 "curvature": np.mean(curvatures),
#                                                 "segment_point": np.array(
#                                                     [spline_y(ts0) + bbox[1], spline_x(ts0) + bbox[0]]).astype(int), 
#                                                 "idx_crop": int(idx_crop)})
#                             features_sk[-1].update(fit_params)
#                             features_sk[-1].update(capi_features)
#                             normals_mean = np.mean(normals, axis=0)
#                             normals_sk.append(normals_mean / np.linalg.norm(normals_mean))
#                             features_sk[-1].update({"angle": np.arctan2(normals_mean[0], normals_mean[1]) * 180 / np.pi})
#             #                 self.feature_time.append(time.time() - t0_feature)
#                         normals_crop.append(normals_sk)
#                         features.append(features_sk)
#             #             self.skeletons_times.append(time.time() - t0_sk)
#                     valid_normals.append(normals_crop)
#                     valid_features.append(features)
#                     valid_segments.append(segments)
#             #         self.crops_times.append(time.time() - t0_crop)
#         outputs = {'crops': valid_crops, 'bboxes': valid_bboxes, 'sharpness': valid_sharpness, 'masks': valid_masks, 'skeletons': valid_skeletons, 'segments': valid_segments, 'features': valid_features, 'normals': valid_normals}
#         return outputs
    
#     def crop_iterators(self, crops: np.ndarray | list, masks: np.ndarray | list, bboxes: np.ndarray | list):
#         if not self.multiprocessing:
#             logger.info('Starting sequential feature extraction')
#             return self.iterate_over_crops(crops, masks, bboxes, 
#                                            sharpness_estimator=self.sharpness_estimator, 
#                                            skeletonizer=self.skeletonizer, cross_sectioner=self.cross_sectioner, 
#                                            features_list=self.features_list, segment_size=self.segment_size, 
#                                            sharpness_th=self.sharpness_th, max_segments_per_capillary=self.max_segments_per_capillary,
#                                            depth=self.depth, exposure_time=self.exposure_time, intersection_angle=self.intersection_angle)
#         else:
#             logger.info('Starting parallel feature extraction')
#             num_processes = 4
#             chunks = list(zip(crops, masks, bboxes))
#             iterate_over_crops_partial = functools.partial(self.iterate_over_crops, sharpness_estimator=self.sharpness_estimator, 
#                                                            skeletonizer=self.skeletonizer, cross_sectioner=self.cross_sectioner, 
#                                                            features_list=self.features_list, segment_size=self.segment_size, 
#                                                            sharpness_th=self.sharpness_th, max_segments_per_capillary=self.max_segments_per_capillary,
#                                                            depth=self.depth, exposure_time=self.exposure_time, 
#                                                            intersection_angle=self.intersection_angle, multiprocessing=True)
#             with mp.Pool(processes=num_processes) as pool:
#                 crops_results_list = pool.starmap(iterate_over_crops_partial, chunks)

#             crops_results_dict = {key: [d[key][0] for d in crops_results_list if len(d['crops']) > 0] for key in crops_results_list[0].keys()}
#             update_idx_crop(crops_results_dict)
#             # Remove any empty lists from crops_results_dict
            
#             # for idx_crop in range(len(crops_results_dict['features'])):
#             #     crops_results_dict['features']['idx_crop'][idx_crop] = idx_crop
#             # crops_results_dict['features']['idx_crop'] = np.arange(len(crops_results_dict['features']['idx_crop']))
#             return crops_results_dict        

    
#     def get_features(self, image: np.ndarray,
#                          depth: float,
#                          exposure_time: int,
#                          multiprocessing: bool = False,
#                          query_point: Optional[tuple[float, float]] = None,
#                          max_capillaries: Optional[int] = None,
#                          max_segments_per_capillary: Optional[int] = None) -> dict:
#         """
#         Extracts features from the given image.
#         Parameters:
#         -----------
#         image : np.ndarray
#             The input image from which features are to be extracted.
#         depth : float
#             The depth value to be used in feature extraction.
#         exposure_time : int
#             The exposure time value to be used in feature extraction.
#         multiprocessing : bool, optional
#             Flag to indicate whether to use multiprocessing for feature extraction. Default is False.
#         query_point : Optional[tuple[float, float]], optional
#             A tuple representing the query point coordinates. Default is None.
#         max_capillaries : Optional[int], optional
#             The maximum number of capillaries to consider. Default is None.
#         max_segments_per_capillary : Optional[int], optional
#             The maximum number of segments per capillary to consider. Default is None.
#         Returns:
#         --------
#         dict
#             A dictionary containing the extracted features.
#         """
#         self.multiprocessing = multiprocessing
#         self.max_capillaries = max_capillaries
#         self.max_segments_per_capillary = max_segments_per_capillary
#         self.depth = depth if "depth" in self.features_list else np.nan
#         self.exposure_time = exposure_time if "exposure_time" in self.features_list else np.nan
#         self.intersection_angle = 90 if "intersection_angle" in self.features_list else np.nan

#         logger.info('Starting Smart Segment Selection')
#         t0 = time.time()
#         crops, bboxes, _ = self.detect_crops(image=image, query_point=query_point)
#         self.cropping_time = time.time() - t0
#         masks = self.mask_crops(crops)
#         self.masking_time = time.time() - t0 - self.cropping_time
#         logger.info('Starting Features Extraction')
#         crops_results_dict = self.crop_iterators(crops, masks, bboxes)
        
#         self.crops = crops_results_dict['crops']
#         self.bboxes = crops_results_dict['bboxes']
#         self.sharpness = crops_results_dict['sharpness']
#         self.masks = crops_results_dict['masks']
#         self.skeletons = crops_results_dict['skeletons']
#         self.segments = crops_results_dict['segments']
#         self.features = crops_results_dict['features']
#         self.normals = crops_results_dict['normals']
#         logger.info('Finished extracting features')
#         return self.features
        

#     def compute_features(self, image: np.ndarray,
#                          depth: float,
#                          exposure_time: int,
#                          multiprocessing: bool = False,
#                          query_point: Optional[tuple[float, float]] = None,
#                          max_capillaries: Optional[int] = None,
#                          max_segments_per_capillary: Optional[int] = None) -> dict:
#         """
#         Compute features from the given image and additional parameters.

#         Parameters:
#         -----------
#         image : np.ndarray
#             The input image from which features are to be extracted.
#         depth : float
#             The depth information of the image.
#         exposure_time : int
#             The exposure time of the image capture.
#         multiprocessing : bool, optional
#             Flag to indicate if multiprocessing should be used (default is False).
#         query_point : Optional[tuple[float, float]], optional
#             A specific point in the image to query (default is None).
#         max_capillaries : Optional[int], optional
#             Maximum number of capillaries to consider (default is None).
#         max_segments_per_capillary : Optional[int], optional
#             Maximum number of segments per capillary to consider (default is None).

#         Returns:
#         --------
#         dict
#             A dictionary containing the computed features, including segment proposals,
#             detections, masks, and skeletons.
#         """
#         features = self.get_features(image, depth, exposure_time, multiprocessing, query_point, max_capillaries, max_segments_per_capillary)
#         x_y_theta_score = []
#         if len(features) > 0:
#             logger.info('Sorting segments based on probability of being a good capillary')
#             features, scores = self.get_sorted_segments(features)
#             for segment_point, angle, score, idx_crop in zip(features["segment_point"], features["angle"], scores, features["idx_crop"]):
#                 y, x = segment_point[0], segment_point[1]
#                 y = float(y)
#                 x = float(x)
#                 theta = float(angle)
#                 score = float(score)
#                 res = [x, y, theta, score, idx_crop]
#                 x_y_theta_score.append(res)
#         else:
#             logger.info('No segments were found')
#         outputs = {'segment_proposals': x_y_theta_score}
#         additional_output = {"detections": self.bboxes, "masks": self.masks, "skeletons": self.skeletons}
#         outputs.update(additional_output)
#         logger.info('Smart Segment Computation Completed')
#         return outputs

    
#     def get_sorted_segments(self, features: pd.DataFrame | list=None):
#         """
#         Sorts the segments based on the probability of being a capillary.

#         Parameters:
#         -----------
#         features : pd.DataFrame or list, optional
#             The features to be sorted. If None, the instance's features attribute is used.
#             If a DataFrame is provided, it is converted to a list of DataFrames.

#         Returns:
#         --------
#         sorted_segments : pd.DataFrame
#             The segments sorted by the probability of being a capillary.
#         sorted_scores : np.ndarray
#             The scores corresponding to the sorted segments.
#         """
#         if features is None:
#             features = self.features
#         if len(features) == 0:
#             logger.info('No segments were found')
#             return pd.DataFrame(), np.array([])
#         else:
#             if isinstance(features, pd.DataFrame):
#                 features = [features]
#             features = list(itertools.chain(*features))
#             features = pd.concat([pd.DataFrame(f) for f in features], ignore_index=True)
#             columns_to_standardize = features.columns.difference(self.ignored_columns)

#             # Calculate the scores
#             probabilities = self.feature_classifier.predict_proba(features[columns_to_standardize])
#             scores = probabilities[:, 1]
#             # Sort the segments by the probability of being a capillary
#             sorted_indices = np.argsort(scores)[::-1]
#             sorted_segments = features.iloc[sorted_indices]
#             sorted_scores = scores[sorted_indices]
#             return sorted_segments, sorted_scores
    
#     def get_best_segment(self, features: pd.DataFrame | list=None):
#         """
#         Get the best segment from the given image.

#         Args:
#             image (str or ndarray): An image or a path to the image.

#         Returns:
#             tuple: A tuple containing the best segment point and angle.
#         """
#         sorted_segments, sorted_scores = self.get_sorted_segments(features)
#         if len(sorted_segments) == 0:
#             logger.info('No segments were found')
#             return None, None
#         best_segment = sorted_segments.iloc[0]
#         self.best_segment = best_segment["segment_point"], best_segment["angle"]
#         return self.best_segment
    
#     def _sort_detections_by_distance(self, query_point: np.ndarray | Sequence, detections: np.ndarray):
#         """
#         Sorts detections by their distance to a given query point.

#         Parameters:
#         -----------
#         query_point : np.ndarray or Sequence
#             The reference point to which distances are calculated. If a Sequence is provided, it is assumed to be the 
#             coordinates of the point. If an np.ndarray is provided, it is assumed to be an image from which the center 
#             point is calculated.
#         detections : np.ndarray
#             An array of detections where each detection is represented by a bounding box (x1, y1, x2, y2) and possibly 
#             other attributes.

#         Returns:
#         --------
#         np.ndarray
#             An array of detections sorted by their distance to the query point. The number of returned detections depends 
#             on the value of `self.max_capillaries`. If `self.max_capillaries` is 1, the closest 10 detections are returned. 
#             Otherwise, the closest `self.max_capillaries` detections are returned. If `self.max_capillaries` is None, all 
#             detections are returned sorted by distance.

#         Notes:
#         ------
#         - The method logs information about the number of capillaries being returned and their proximity to the query point.
#         - The `self.max_capillaries` attribute determines the number of closest detections to return.
#         """

#         if isinstance(query_point, Sequence):
#             image_center = np.array(query_point)
#         else:
#             image_center = np.array([query_point.shape[1] / 2, query_point.shape[0] / 2])

#         bounding_boxes = detections[:, :4]
#         bboxes_centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in bounding_boxes])
#         distances = np.linalg.norm(bboxes_centers - image_center, axis=1)

#         if self.max_capillaries == 1:
#             closest_index = np.argsort(distances)[:10]
#             closest_bbox_to_center = detections[closest_index]
#             logger.info(f'Returning the closest capillary to location = {image_center.tolist()}')
#             return closest_bbox_to_center
#         else:
#             if self.max_capillaries is not None:
#                 closest_indices = np.argsort(distances)[:(int(self.max_capillaries * 1.5))]
#                 logger.info(f'Returning the closest {self.max_capillaries} capillaries to location '
#                             f'({query_point[0]},{query_point[1]})')
#                 closest_indices = closest_indices[:self.max_capillaries]
#             else:
#                 closest_indices = np.argsort(distances)[:self.max_capillaries]
#             closest_bboxes_to_center = detections[closest_indices]
#             return closest_bboxes_to_center

#     def plot_ordered_segments(self, image: np.ndarray ,features: pd.DataFrame | list=None):
#         """
#         Plots the ordered segments on the given image.
        
#         Args:
#             image (ndarray): The image.
#         """
#         if features is None:
#             features = self.features
#         if isinstance(features, pd.DataFrame):
#             features = [features]
#         sorted_segments, scores = self.get_sorted_segments(features)
#         plt.imshow(image, cmap='gray')
#         plt.axis('off')
#         # plot each segment point in different colors
#         cmp = plt.get_cmap('viridis')
#         norm = plt.Normalize(vmin=0, vmax=len(sorted_segments))
#         for idx in range(len(sorted_segments)):
#             segment = sorted_segments.iloc[idx]
#             plt.scatter(segment["segment_point"][1], segment["segment_point"][0], c=idx, s=3, cmap=cmp, norm=norm, edgecolor='none')
#         plt.show()
#         return sorted_segments

#     def get_timings(self):
#         """
#         Returns the timings for the different steps of the feature extraction.
        
#         Returns:
#             tuple: A tuple containing the timings.
#         """
#         return self.init_time, self.cropping_time, self.masking_time, self.crops_times, self.skeletons_times, self.feature_time

#     def plot_timings(self):
#         """
#         Plots the timings for the different steps of the feature extraction.
#         """
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(14, 14))
#         plt.subplot(3, 3, 1)
#         # plot a histogram of the cropping times
#         plt.hist(self.crops_times, bins=20)
#         plt.title("Time per crop")

#         plt.subplot(3, 3, 2)
#         # plot a histogram of the skeleton times
#         plt.hist(self.skeletons_times, bins=20)
#         plt.title("Time per skeleton segment")
#         plt.subplot(3, 3, 3)
#         # plot a histogram of the feature times
#         plt.hist(self.feature_time, bins=20)
#         plt.title("Time per segment point")

#         plt.subplot(3, 3, 4)
#         # plot a histogram of the feature times
#         plt.hist(self.sharpness_times, bins=20)
#         plt.title("Time per sharpness estimation")

#         plt.subplot(3, 3, 5)
#         # plot a histogram of the skeletoning times
#         plt.hist(self.skeletoning_times, bins=20)
#         plt.title("Time per skeletoning")

#         plt.subplot(3, 3, 6)
#         # plot a histogram of the sampling times
#         plt.hist(self.sampling_times, bins=20)
#         plt.title("Time per sampling")

#         plt.subplot(3, 3, 7)
#         # plot a histogram of the cross sectioning times
#         plt.hist(self.cross_sectioning_times, bins=20)
#         plt.title("Time per cross sectioning")

#         plt.subplot(3, 3, 8)
#         # plot a histogram of the fitting times
#         plt.hist(self.fitting_times, bins=20)
#         plt.title("Time per fitting")

#         plt.subplot(3, 3, 9)
#         # plot a histogram of the capillary feature extraction times
#         plt.hist(self.capi_feat_times, bins=20)
#         plt.title("Time per feature extraction")

#         plt.show()


# if __name__ == "__main__":
    
#     image_path = FS.GCP_DATASETS/"wide_full_frames"/"capillary_mask_2024_02_21"/"test"/"images"/"wide_0004205_000076419_0000.png"
#     # image_path = "/home/datasets/wide_full_frames/capillary_mask_2024_02_21/test/images/wide_0004259_000077642_0000.png"
#     sharpness_th = 23
#     segment_size = 20
#     time0 = time.time()
#     image = Image.open(image_path)
#     image_np = np.array(image.convert('L'))

#     t_load_image = time.time() - time0

#     segment_analyzer = SegmentDetectionModel(feature_classifier_weights_path=ModelPaths.SEGMENT_FEATURE_CLASSIFIER)

#     #%%
#     # Load the image
#     image = Image.open(image_path)

#     # # Prepare to draw on the image
#     draw = ImageDraw.Draw(image)

#     features = segment_analyzer.get_features(image_np, multiprocessing=True, depth=0.5, exposure_time=1000, query_point=None, max_capillaries=10, max_segments_per_capillary=4)

#     best_segment, angle = segment_analyzer.get_best_segment(features)
#     if best_segment is not None and angle is not None:
#         angle = angle * np.pi / 180

#         # # Assuming bboxes is a list of tuples (x1, y1, x2, y2)
#         for bbox in segment_analyzer.bboxes:
#             draw.rectangle(bbox[:4], outline="green")

#         # draw a cross on the best segment

#         draw.line([(int(best_segment[1] - 10 * np.cos(angle)), int(best_segment[0] - 10 * np.sin(angle))),
#                 (int(best_segment[1] + 10 * np.cos(angle)), int(best_segment[0] + 10 * np.sin(angle)))], fill="red",
#                 width=4)
#     # draw.line([(best_segment[1], best_segment[0]-10), (best_segment[1], best_segment[0]+10)], fill="red", width=2)

#     # Display the image
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')  # Hide axis
#     plt.show()
#     #%%
#     # segment_analyzer.get_features(image_np)
    
#     plt.figure(figsize=(14, 6))
#     for i, (crop, mask, skeletons, segments, sharpness, features, normals) in enumerate(
#             zip(segment_analyzer.crops, segment_analyzer.masks, segment_analyzer.skeletons, segment_analyzer.segments,
#                 segment_analyzer.sharpness, segment_analyzer.features, segment_analyzer.normals)):
#         plt.subplot(1, len(segment_analyzer.crops), i + 1)
#         plt.imshow(crop, cmap='gray')
#         plt.axis('off')
#         plt.title(f"{sharpness:.0f}")

#         # plot the contour of the mask
#         plt.contour(mask, levels=[0.1], colors='red')

#         for sk, segs, features_sk, normals_sk in zip(skeletons, segments, features, normals):
#             spline_x, spline_y, ts, segment_size = su.sample_skeleton(sk, segment_analyzer.segment_size*10, use_two_splines=True)
#             sk_len = su.skeleton_length(sk)
#             t = np.linspace(0, sk_len, 100)
#             # plt.plot(spline_x(t), spline_y(t), 'b')
#             plt.plot(sk[:, 1], sk[:, 0], 'b')
#             for seg, feature, norm in zip(segs, features_sk, normals_sk):
#                 # cntr = feature["fit_center"]
#                 fwhm = feature["mask_width"]# - np.abs(cntr)
#                 fwhm_fit = feature["fit_fwhm"]
#                 # plt.arrow(seg[1] + cntr * norm[1], seg[0] + cntr * norm[0], norm[1] * fwhm / 2, norm[0] * fwhm / 2, color='g', width=1, head_width=0.2, head_length=0.2, alpha=1)
#                 # plt.arrow(seg[1] + cntr * norm[1], seg[0] + cntr * norm[0], -norm[1] * fwhm / 2, -norm[0] * fwhm / 2, color='g', width=1, head_width=0.2, head_length=0.2, alpha=1)
#                 plt.arrow(seg[1], seg[0], norm[1] * fwhm_fit / 2, norm[0] * fwhm_fit / 2, color='y', width=1, head_width=0.2, head_length=0.2, alpha=1)
#                 plt.arrow(seg[1], seg[0], -norm[1] * fwhm_fit / 2, -norm[0] * fwhm_fit / 2, color='y', width=1, head_width=0.2, head_length=0.2, alpha=1)
#                 plt.arrow(seg[1], seg[0], norm[1] * fwhm / 2, norm[0] * fwhm / 2, color='g', linestyle='dashed', width=1, head_width=0.2, head_length=0.2, alpha=1)
#                 plt.arrow(seg[1], seg[0], -norm[1] * fwhm / 2, -norm[0] * fwhm / 2, color='g', linestyle='dashed', width=1, head_width=0.2, head_length=0.2, alpha=1)

#     plt.show()
#     #%% Running time Analysis
#     segment_analyzer.plot_timings()
#     #%% Running time Analysis
#     segment_analyzer.plot_ordered_segments(image_np);