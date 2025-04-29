#%% Imports
import numpy as np
from scipy.interpolate import CubicSpline

from skeleton.src.skeleton2d import Skeleton2D


# %% Omri's skeleton utility functions
def calculate_skeleton_parametrization(skeleton, return_t=False, return_vals=True):
    """
    Calculate the parametrization of a skeleton curve.
    
    Args:
        skeleton (ndarray): The skeleton curve represented as an array of points.
        return_t (bool, optional): Whether to return the parameter values. 
            Defaults to False.
    
    Returns:
        tuple: A tuple containing the interpolated x and y coordinates of the skeleton curve.
        If `return_t` is True, the tuple also includes the parameter values.
    """
    t = np.zeros(skeleton.shape[0])
    t[1:] = np.sqrt(np.sum(np.diff(skeleton, axis=0)**2, axis=1))
    t = np.cumsum(t)
    x = skeleton[:, 1]
    y = skeleton[:, 0]

    # Interpolate the x and y coordinates using a cubic spline
    spline_x = CubicSpline(t, x)
    spline_y = CubicSpline(t, y)

    if return_vals:
        spline_x = spline_x(t)
        spline_y = spline_y(t)
        spline = np.array([spline_y, spline_x]).T
    else:
        spline = spline_y, spline_x

    if return_t:
        return spline, t
    else:
        return spline
    

def calculate_skeleton_tangents(skeleton_params, scale_factor=None):
    """
    Calculate the tangent vectors of a skeleton curve.
    
    Parameters:
    - skeleton_params: A cubic spline of a skeleton (spline_x, spline_y, t) where t is a parametrization variable.
    
    Returns:
    - tangent: A numpy array representing the tangent vectors of the skeleton curve.
    """
    spline_x, spline_y, t = skeleton_params
    if scale_factor is not None:
        t = np.arange(int(t[0]), int(t[-1]), scale_factor)
    # Calculate the derivative at each point
    dx_dt = spline_x.derivative()(t)
    dy_dt = spline_y.derivative()(t)

    # The tangent vector at each point is given by the derivative
    tangent = np.array([dy_dt, dx_dt]).T



    # Normalize the normal vectors
    tangent /= np.linalg.norm(tangent, axis=1)[:, np.newaxis]
    
    return tangent


def calculate_skeleton_normals(skeleton_params):
    """
    Calculate the normal vectors of a skeleton.

    Parameters:
    - skeleton_params: A numpy array representing the skeleton parameters.

    Returns:
    - normals: A numpy array representing the normal vectors of the skeleton.
    """
    tangents = calculate_skeleton_tangents(skeleton_params)
    # Calculate the normal vector by rotating the tangent vector by 90 degrees
    normals = np.array([-tangents[:, 1], tangents[:, 0]]).T

    # Normalize the normal vectors
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    
    return normals

def calculate_skeleton_curvature(skeleton_params, scale_factor=None):
    """
    Calculate the curvature of a skeleton curve.
    
    Parameters:
    - skeleton_params: A cubic spline of a skeleton (spline_x, spline_y, t) where t is a parametrization variable.
    
    Returns:
    - curvature: A numpy array representing the curvature of the skeleton curve.
    """
    spline_x, spline_y, t = skeleton_params
    if scale_factor is not None:
        t = np.arange(int(t[0]), int(t[-1]), scale_factor)
    # Calculate the first and second derivatives of the spline
    dx = spline_x(t, 1) # First derivative of x
    ddx = spline_x(t, 2) # Second derivative of x
    dy = spline_y(t, 1) # First derivative of y
    ddy = spline_y(t, 2) # Second derivative of y

    # Calculate the curvature using the formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
    
    return curvature

def filter_skeletons_by_position(skeletons, segment_points):
    """
    Filter skeletons by their proximity to a set of segment points.

    Parameters:
    - skeletons: A list of skeletons, where each skeleton is an ndarray of shape (m, 2).
    - segment_points: An ndarray of shape (n, 2) containing multiple segment points.

    Returns:
    - best_skeleton: The skeleton that is closest to any of the segment points.
    - min_skel_ind: The index of the best skeleton.
    - closest_points: The closest point on the best skeleton to any of the segment points.
    - min_inds: The index of the closest point on the best skeleton.
    - min_distance: The minimum distance between the best skeleton and the segment points.
    """
    min_distances = np.full(len(skeletons), np.inf)  # Initialize with infinity
    min_inds = np.full(len(skeletons), -1)  # Initialize with -1
    closest_points = []

    for ind_skel, skeleton in enumerate(skeletons):
        distances = np.linalg.norm(skeleton[:, np.newaxis, :] - segment_points[np.newaxis, :, :], axis=2)
        min_dist, min_ind = np.min(distances), np.unravel_index(np.argmin(distances), distances.shape)
        min_distances[ind_skel] = min_dist
        min_inds[ind_skel] = min_ind[0]
        closest_points.append(skeleton[min_ind[0]])

    min_distance, min_skel_ind = np.min(min_distances), np.argmin(min_distances)
    best_skeleton = skeletons[min_skel_ind]
    closest_point = closest_points[min_skel_ind]

    return best_skeleton, min_skel_ind, closest_point, min_inds[min_skel_ind], min_distance

def filter_skeletons_by_length(skeletons):
    skel_lengths = np.array([np.linalg.norm(np.diff(skeleton, axis=0), axis=1).sum() for skeleton in skeletons])
    max_length, max_skel_ind = np.max(skel_lengths), np.argmax(skel_lengths)
    best_skeleton = skeletons[max_skel_ind]
    return best_skeleton, max_skel_ind, max_length

def skeleton_length(skeleton):
    if len(skeleton) == 0:
        return 0
    else:
        return np.linalg.norm(np.diff(skeleton, axis=0), axis=1).sum()

def sample_skeleton(skeleton, segment_length=20, use_two_splines=True, max_segments_per_capillary=None):
    """
    Samples a skeleton by interpolating points at a fixed distance along the curve.

    Parameters:
    - skeleton: An ndarray of shape (m, 2) where each row is an [x, y] point.
    - N: The distance between consecutive points on the sampled skeleton.

    Returns:
    - A list of ndarrays, each representing a segment of the skeleton.
    """
    # Calculate the cumulative distance along the points as the parameter t
    t = np.zeros(skeleton.shape[0])
    t[1:] = np.sqrt(np.sum(np.diff(skeleton, axis=0) ** 2, axis=1))
    t = np.cumsum(t)
    skeleton_length = t[-1]
    if skeleton_length < segment_length:
        segment_length = t[-1]
    
    number_of_segments = int(skeleton_length / segment_length)
    if max_segments_per_capillary is not None:
        number_of_segments = min(number_of_segments, max_segments_per_capillary)
    # Interpolate the x and y coordinates using a cubic spline
    spline_x = CubicSpline(t, skeleton[:, 1])
    spline_y = CubicSpline(t, skeleton[:, 0])

    if use_two_splines:
        ts = np.linspace(t[0], t[-1], number_of_segments + 2)
        sampled_skeleton = np.array([spline_y(ts), spline_x(ts)]).T
    else:
        ts = np.linspace(0, len(t)-1, number_of_segments + 2)
        sampled_skeleton = skeleton[ts.astype(int)]
    spline_x_new = CubicSpline(ts, sampled_skeleton[:, 1])
    spline_y_new = CubicSpline(ts, sampled_skeleton[:, 0])
    return spline_x_new, spline_y_new, ts[1:-1], segment_length

def distribute_segments_on_skeleton(skeleton, max_segments_per_capillary=None):
    """
    Distributes segments along a skeleton based on the length of each skeleton segment.
    Parameters:
    skeleton (list): A list of skeleton segments, where each segment is represented in a format that can be processed by the `skeleton_length` function.
    max_segments_per_capillary (int, optional): The maximum number of segments to distribute across the entire skeleton. If None, the function returns a list of None values.
    Returns:
    list: A list of integers representing the number of segments assigned to each skeleton segment. If `max_segments_per_capillary` is None, returns a list of None values.
    """
    skeleton_segments_lengths = [skeleton_length(sk) for sk in skeleton]
    total_length = sum(skeleton_segments_lengths)
    if max_segments_per_capillary is None:
        segments_per_skeleton_segment = [None for _ in skeleton_segments_lengths]
        return segments_per_skeleton_segment
    
    segments_per_skeleton_segment = [int(np.floor(length / total_length * max_segments_per_capillary)) for length in skeleton_segments_lengths]
    total_segments = np.sum(segments_per_skeleton_segment)
    segment_diff = max_segments_per_capillary - total_segments
    segments_per_skeleton_segment_modulo = [(length * max_segments_per_capillary) % total_length for length in skeleton_segments_lengths]
    while segment_diff > 0:
        max_idx = np.argmax(segments_per_skeleton_segment_modulo)
        segments_per_skeleton_segment[max_idx] += 1
        segment_diff -= 1
        segments_per_skeleton_segment_modulo[max_idx] = 0
    return segments_per_skeleton_segment

# %% BoneProcessor class
class Skeletonizer:
    """
    A class that processes bone data and produces a skeleton.

    Attributes:
        bp.filtered_skeleton_components: The filtered skeleton components.

    Methods:
        clear_data(): Clears the data in the BoneProcessor instance.
        produce_skeleton(image, mask): Produces a skeleton based on the given image and mask.

    """

    def __init__(self, segment_size=20):
        self.segment_size = segment_size
        self.skeletonizer = None
        self.skeletons = None
    
    def produce_skeletons(self, *args, **kwargs):
        self.skeletonizer = Skeleton2D(*args, **kwargs)
        skeletons = self.skeletonizer.segment_coordinates()
        self.skeletons = [skeleton for skeleton in skeletons if skeleton_length(skeleton) >= self.segment_size]
        return self.skeletons
    
    def produce_skeleton(self, segment_points=None, *args, **kwargs):
        skeletons = self.produce_skeletons(*args, **kwargs)
        if len(skeletons) == 0:
            return None
        if segment_points is None:
            best_skeleton, _, _ = filter_skeletons_by_length(skeletons)
        else:
            best_skeleton, _, _, _, min_distance = filter_skeletons_by_position(skeletons, segment_points)
        return best_skeleton
        