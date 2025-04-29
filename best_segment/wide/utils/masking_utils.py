import best_segment.wide.utils.skeleton_utils as su
from best_segment.wide.utils.cross_section_utils import CrossSection
import torch
from torch import nn
import numpy as np
from capillary_segmentation.wide.src.inference import YoloWideCapillarySegmentorModel


class Masker(nn.Module):
    def __init__(self, scale_factor=1, is_center_skeleton=False, segment_point=None, segment_size=20,
                 wide_capillary_segmentor: YoloWideCapillarySegmentorModel = None):
        super(Masker, self).__init__()
        if wide_capillary_segmentor is None:
            wide_capillary_segmentor = YoloWideCapillarySegmentorModel()  # TODO - default arguments?
        self.model = wide_capillary_segmentor
        self.segment_point = segment_point
        self.segment_size = segment_size
        self.scale_factor = scale_factor
        self.is_center_skeleton = is_center_skeleton
        self.skeletonizer = su.Skeletonizer(segment_size=self.segment_size)
        self.image = None
        self.mask = None
        self.skeleton = None
        self.skeleton_params = None
        self.skelespline = None
        self.curvature = None
        self.tangents = None
        self.normals = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def clear_data(self):
        self.image = None
        self.mask = None
        self.skeleton = None
        self.skeleton_params = None
        self.skelespline = None
        self.curvature = None
        self.tangents = None
        self.normals = None

    def set_segment_point(self, segment_point):
        self.segment_point = segment_point

    def load_image(self, image=None):
        if image is not None:
            self.clear_data()
            self.image = image
        elif self.image is None:
            raise ValueError("No image loaded")

    def mask_image(self):
        if isinstance(self.image, np.ndarray):
            if self.image.ndim == 2:
                image = np.stack([self.image, self.image, self.image], axis=-1)
                mask = self.model.infer(image)
            else:
                image = [np.stack([img, img, img], axis=-1) for img in self.image]
                mask = self.model.batch_infer(image)[1]
        else:
            image = [np.stack([img, img, img], axis=-1) for img in self.image]
            mask = self.model.batch_infer(image)[1]
        return mask

    @staticmethod
    def mask_image_hemoscope(crops):
        image = [np.stack([img, img, img], axis=-1) for img in crops]
        return image

    def produce_mask(self, image=None, line_points=None):
        self.load_image(image)
        mask = self.mask_image()
        mask = np.expand_dims(mask, axis=0) if mask.ndim == 2 else mask
        if line_points is not None:
            selected_masks = []
            for m in mask:
                if any(m[int(point[0]), int(point[1])] for point in line_points):
                    selected_masks.append(m)
            if len(selected_masks) == 0:
                return None
            else:
                self.mask = np.logical_or.reduce(selected_masks, axis=0)
        else:
            self.mask = np.logical_or.reduce(mask, axis=0)
        return self.mask

    def produce_skeleton(self, image=None, mask=None, segment_points=None):
        self.produce_skeletons(image=image, mask=mask)
        if segment_points is None:
            segment_points = self.segment_point

        self.skeleton = self.skeletonizer.produce_skeleton(segment_points=segment_points)
        return self.skeleton

    def produce_skeletons(self, image=None, mask=None):
        self.load_image(image)
        if mask is None:
            if self.mask is None:
                mask = self.produce_mask()
            else:
                mask = self.mask

        # Ensure mask is a numpy array
        if not isinstance(mask, np.ndarray):
            raise ValueError("Mask must be a numpy array")

        # print(f"Mask type: {type(mask)}")  # Debugging statement
        # print(f"Mask shape: {mask.shape}")  # Debugging statement

        skeletons = self.skeletonizer.produce_skeletons(image=self.image, mask=mask.astype(float))
        return skeletons

    @property
    def skeletons(self):
        return self.skeletonizer.skeletons

    def produce_skeleton_params(self, image=None):
        self.load_image(image)
        if self.skeleton is None:
            self.produce_skeleton()
        (spline_y, spline_x), t = su.calculate_skeleton_parametrization(self.skeleton, True, False)
        self.skeleton_params = spline_x, spline_y, t
        return self.skeleton_params

    def produce_skelespline(self, image=None):
        self.load_image(image)
        if self.skeleton_params is None:
            self.produce_skeleton_params()
        spline_x, spline_y, t = self.skeleton_params
        tu = np.arange(int(t[0]), int(t[-1]), self.scale_factor)
        self.skelespline = np.array([spline_y(tu), spline_x(tu)]).T
        return self.skelespline

    def produce_curvature(self, image=None, scale_sample=False):
        self.load_image(image)
        if self.skeleton_params is None:
            self.produce_skeleton_params()
        scale_factor = self.scale_factor if scale_sample else None

        self.curvature = su.calculate_skeleton_curvature(self.skeleton_params, scale_factor=scale_factor)
        return self.curvature

    def produce_tangents(self, image=None, scale_sample=True):
        self.load_image(image)
        if self.skeleton_params is None:
            self.produce_skeleton_params()
        scale_factor = self.scale_factor if scale_sample else None

        self.tangents = su.calculate_skeleton_tangents(self.skeleton_params, scale_factor=scale_factor)

        return self.tangents

    def produce_normals(self, image=None):
        self.load_image(image)
        if self.skeleton_params is None:
            self.produce_skeleton_params()

        self.normals = su.calculate_skeleton_normals(self.skeleton_params)

        return self.normals

    def produce_all(self, image=None, features=[]):
        self.load_image(image)
        output_dict = {}
        if "mask" in features or len(features) == 0:
            output_dict["mask"] = self.produce_mask()
        if "skeleton" in features or len(features) == 0:
            output_dict["skeleton"] = self.produce_skeleton()
        if "skeleton_params" in features or len(features) == 0:
            output_dict["skeleton_params"] = self.produce_skeleton_params()
        if "skelespline" in features or len(features) == 0:
            output_dict["skelespline"] = self.produce_skelespline()
        if "curvature" in features or len(features) == 0:
            output_dict["curvature"] = self.produce_curvature()
        if "tangents" in features or len(features) == 0:
            output_dict["tangents"] = self.produce_tangents()
        if "normals" in features or len(features) == 0:
            output_dict["normals"] = self.produce_normals()

        return output_dict

    def center_skeleton(self):
        cross_sectioner = CrossSection(type="curve")
        self.produce_tangents(scale_sample=False)
        self.produce_normals()
        cross_sectioner.set_normal_cross_section_points(self.skeleton, self.normals)
        _, params = cross_sectioner.fit(self.image, return_dict=True)
        self.skeleton = self.skeleton.copy() + params['fit_center'][:, np.newaxis] * self.normals
        self.skeleton = su.bezier_interpolation_with_perpendiculars(self.skeleton, 2)
        # self.skeleton = self.skeleton[:, 0:2]
        self.produce_skeleton_params()
        self.produce_skelespline()
        self.produce_curvature()
        self.produce_tangents()
        self.produce_normals()

    def return_all(self, features=[]):
        output_dict = {}
        if "mask" in features or len(features) == 0:
            output_dict["mask"] = self.mask
        if "skeleton" in features or len(features) == 0:
            output_dict["skeleton"] = self.skeleton
        if "skeleton_params" in features or len(features) == 0:
            output_dict["skeleton_params"] = self.skeleton_params
        if "skelespline" in features or len(features) == 0:
            output_dict["skelespline"] = self.skelespline
        if "curvature" in features or len(features) == 0:
            output_dict["curvature"] = self.curvature
        if "tangents" in features or len(features) == 0:
            output_dict["tangents"] = self.tangents
        if "normals" in features or len(features) == 0:
            output_dict["normals"] = self.normals

        return output_dict

    def forward(self, image=None, features=[], is_center_skeleton=None):
        self.load_image(image)
        if is_center_skeleton is None:
            is_center_skeleton = self.is_center_skeleton

        if is_center_skeleton:
            output_dict = self.produce_all(image=image, features=["mask", "skeleton"])
            self.center_skeleton()
            output_dict = self.return_all(features=features)
        else:
            output_dict = self.produce_all(image=image, features=features)

        return output_dict
