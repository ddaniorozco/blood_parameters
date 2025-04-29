import numpy as np
from skimage.measure import profile_line as sk_profile_line
from scipy.ndimage import map_coordinates as scp_map_coordinates, gaussian_filter1d
from scipy.optimize import curve_fit


# %% ~~~~~~~~~~~~~~~~~~~~ Getting cross-section profiles utility functions ~~~~~~~~~~~~~~~~~~~~
def profile_line(image, points, **kwargs):
    return sk_profile_line(image, points[0], points[-1], **kwargs)

def map_coordinates(image, points, **kwargs):
    return scp_map_coordinates(image, np.transpose(points), **kwargs)


# %% ~~~~~~~~~~~~~~~~~~~~ Profiler class ~~~~~~~~~~~~~~~~~~~~
class Profiler:
    """
    A class for profiling an image using different types of profiles.

    Args:
        points (list, optional): List of points to profile. Defaults to None.
        type (str, optional): Type of profile to use. Defaults to 'line'.
        mode (str, optional): Interpolation mode. Defaults to 'nearest'.
        **kwargs: Additional keyword arguments.

    Attributes:
        points (list): List of points to profile.
        type (str): Type of profile to use.
        mode (str): Interpolation mode.
        func (function): Profile function based on the type.

    Methods:
        set_points: Set the points to profile.
        set_type: Set the type of profile.
        set_mode: Set the interpolation mode.
        __call__: Perform the profiling on an image.

    """

    def __init__(self, points=None, type='curve', mode='nearest', **kwargs):
        self.points = points
        self.type = type
        self.mode = mode
        self.func = profile_line if type == 'line' else map_coordinates
    
    def set_points(self, points):
        """
        Set the points to profile.

        Args:
            points (list): List of points to profile.

        """
        self.points = points
    
    def set_type(self, type):
        """
        Set the type of profile.

        Args:
            type (str): Type of profile to use.

        """
        self.type = type
        self.func = profile_line if type == 'line' else map_coordinates
    
    def set_mode(self, mode):
        """
        Set the interpolation mode.

        Args:
            mode (str): Interpolation mode.

        """
        self.mode = mode
    
    def __call__(self, image, points=None):
        """
        Perform the profiling on an image.

        Args:
            image: The image to profile.
            points (list, optional): List of points to profile. Defaults to None.

        Returns:
            The profiled values.

        """
        if points is None:
            points = self.points
        return self.func(image, points, mode=self.mode)


# %% ~~~~~~~~~~~~~~~~~~~~ Cross-section utility functions ~~~~~~~~~~~~~~~~~~~~
class CrossSection:
    def __init__(self, normal_half_length=15, **kwargs):
        self.profiler = Profiler(**kwargs)
        self.normal_half_length = normal_half_length
    
    @property
    def raw_normal_axis(self):
        return np.arange(-self.normal_half_length, self.normal_half_length + 1)
    
    @property
    def points(self):
        return self.profiler.points
    
    @points.setter
    def points(self, points):
        self.profiler.set_points(points)

    def set_points(self, points):
        self.points = points
    
    def set_normal_cross_section_points(self, center, normal, normal_half_length=None):
        if normal_half_length is not None:
            self.normal_half_length = normal_half_length
        self.points = center + np.outer(self.raw_normal_axis, normal).reshape((self.raw_normal_axis.shape[0], normal.shape[0], normal.shape[1]))
        return self.points
    
    def get_axis(self, points=None, length_scale=True, is_centered=True):
        if points is None:
            points = self.points
        axs = []
        if points.ndim == 2:
            points = [points]
        for pnts in points:
            if length_scale:
                # Calculate the cumulative distance along the points as the parameter t
                t = np.zeros(pnts.shape[0])
                t[1:] = np.sqrt(np.sum(np.diff(pnts, axis=0)**2, axis=1))
                t = np.cumsum(t)
            else:
                t = np.arange(pnts.shape[0])
            if is_centered:
                t = t - t[-1] / 2
            axs.append(t)
        return np.squeeze(np.array(axs))
    
    def __call__(self, image, points=None, average=False):
        if points is None:
            points = self.points
        if image.ndim == 2:
            image = [image]
        if points.ndim == 2:
            points = [points]
        cross_sections = []
        for img in image:
            for pnts in points:
                cross_section = self.profiler(img, pnts)
                cross_sections.append(cross_section)
        cross_sections = np.squeeze(np.array(cross_sections).T)
        if average:
            cross_sections = np.mean(cross_sections, axis=0, keepdims=True)
        return cross_sections
    
    def fit(self, image, points=None, return_dict=False, average=False):
        if points is None:
            points = self.points
        return super_gaussian_fit(self.raw_normal_axis, self(image, points, False), return_dict=return_dict, average=average)
    
    def get_capillary_features(self, image, mask, css=None, points=None, return_dict=False, average=False, features_list: list=[]):
        if not set(["mask_width", "capillary_background_ratio", "capillary_background_min_max", "capillary_std", "capillary_sharpness"]) & set(features_list):
            return {"mask_width": np.nan,
                    "capillary_background_ratio": np.nan,
                    "capillary_background_min_max": np.nan,
                    "capillary_std": np.nan,
                    "capillary_sharpness": np.nan}
        if points is None:
            points = self.points
        x = self.raw_normal_axis
        css = self(image, points, False) if css is None else css
        # smooth the cross-sections keeping the same shape and avoid boundary effects
        css = gaussian_filter1d(css, 1, axis=1, mode='nearest')
        
        mask_css = self(mask, points, False)
        widths = []
        bg_ratios = []
        min_maxs = []
        cap_stds = []
        cap_sharps = []
        for cs, mask_cs in zip(css, mask_css):
            width = get_mask_width_around_zero(x, mask_cs) if "mask_width" in features_list else np.nan
            

            # if width <= 2 or np.isnan(width) or width >= np.max(x) - np.min(x) - 2:
            #     widths.append(np.nan)
            #     bg_ratios.append(np.nan)
            #     min_maxs.append(np.nan)
            #     cap_stds.append(np.nan)
            #     cap_sharps.append(np.nan)
            
            # else:
            widths.append(width)
            bg_ratio = capillary_background_ratio(cs, x, mask_cs) if "capillary_background_ratio" in features_list else np.nan
            bg_ratios.append(bg_ratio)

            min_max = capillary_background_min_max(cs, x, mask_cs) if "capillary_background_min_max" in features_list else np.nan
            min_maxs.append(min_max)

            cap_std = capillary_std(cs, x, mask_cs) if "capillary_std" in features_list else np.nan
            cap_stds.append(cap_std)

            cap_sharp = capillary_sharpness(cs, x, mask_cs, smooth_sigma=0) if "capillary_sharpness" in features_list else np.nan
            cap_sharps.append(cap_sharp)

        widths = np.array(widths)
        bg_ratios = np.array(bg_ratios)
        min_maxs = np.array(min_maxs)
        cap_stds = np.array(cap_stds)
        cap_sharps = np.array(cap_sharps)
        if average:
            widths = nanmean(widths)
            bg_ratios = nanmean(bg_ratios)
            min_maxs = nanmean(min_maxs)
            cap_stds = nanmean(cap_stds)
            cap_sharps = nanmean(cap_sharps)
        if return_dict:
            return {"mask_width": widths,
                    "capillary_background_ratio": bg_ratios,
                    "capillary_background_min_max": min_maxs,
                    "capillary_std": cap_stds,
                    "capillary_sharpness": cap_sharps}
        else:
            return widths, bg_ratios, min_maxs, cap_stds, cap_sharps
        
    
    

# %% ~~~~~~~~~~~~~~~~~~~~ Mask width utility functions ~~~~~~~~~~~~~~~~~~~~
def get_mask_edges_around_zero(x, mask_cs):
    # Ensure x and mask_cs are numpy arrays
    x = np.array(x)
    mask_cs = np.array(mask_cs)
    
    # Find indices where mask_cs is positive
    positive_indices = np.where(mask_cs > 0)[0]
    
    if len(positive_indices) == 0:
        return 0, 0  # No positive values in mask_cs
    
    # Find the index in x closest to zero
    zero_index = np.argmin(np.abs(x))
    
    # Find the range of positive indices around zero_index
    left_index = zero_index
    while left_index >= 0 and mask_cs[left_index] > 0:
        if left_index == 0:
            break
        else:
            left_index -= 1
    
    right_index = zero_index
    while right_index <= len(mask_cs) - 1 and mask_cs[right_index] > 0:
        if right_index == len(mask_cs) - 1:
            break
        else:
            right_index += 1
    
    return left_index, right_index

def get_mask_width_around_zero(x, mask_cs):
    # Get the indices of the mask edges around zero
    left_index, right_index = get_mask_edges_around_zero(x, mask_cs)
    # Calculate the width
    width = x[right_index] - x[left_index]
    
    return width

# %% ~~~~~~~~~~~~~~~~~~~~ Cross-section profile features utility functions ~~~~~~~~~~~~~~~~~~~~
def capillary_values(cs, x, mask_sc):
    left_index, right_index = get_mask_edges_around_zero(x, mask_sc)
    return cs[left_index:right_index]

def background_values(cs, x, mask_sc):
    left_index, right_index = get_mask_edges_around_zero(x, mask_sc)
    return np.concatenate((cs[:left_index], cs[right_index:]))

def capillary_background_values(cs, x, mask_sc):
    capillary = capillary_values(cs, x, mask_sc)
    background = background_values(cs, x, mask_sc)
    if np.mean(capillary) < np.mean(background):
        return capillary, background
    else:
        return np.nan, np.nan

def capillary_background_ratio(cs, x, mask_sc):
    capillary, background = capillary_background_values(cs, x, mask_sc)
    if np.isnan(capillary).any() or np.isnan(background).any():
        return np.nan
    return np.mean(capillary) / np.mean(background)

def capillary_background_min_max(cs, x, mask_sc):
    capillary, background = capillary_background_values(cs, x, mask_sc)
    if np.isnan(capillary).any() or np.isnan(background).any():
        return np.nan
    max_val = np.max(background)
    min_val = np.min(capillary)
    return np.subtract(max_val, min_val)

def capillary_std(cs, x, mask_sc):
    capillary, background = capillary_background_values(cs, x, mask_sc)
    if np.isnan(capillary).any() or np.isnan(background).any():
        return np.nan
    # bg_mean = np.mean(background)
    # capillary = bg_mean - capillary
    # return np.std(capillary) / np.mean(capillary)
    return np.std(capillary)

def capillary_sharpness(cs, x, mask_sc, smooth_sigma=0):
    if smooth_sigma > 0:
        cs = gaussian_filter1d(cs, smooth_sigma, axis=0, mode='nearest')
    cs_grad = np.gradient(cs, axis=0)[1:-1]
    sharpness = np.sqrt(np.mean(cs_grad**2))
    return sharpness
    

# %% ~~~~~~~~~~~~~~~~~~~~ Curve fitting utility functions ~~~~~~~~~~~~~~~~~~~~
def super_gaussian(x, a, b, c, d, e):
    """
    Calculate the value of a super Gaussian function at the specified x value.

    Args:
        x (float): The x value at which to calculate the function.
        a (float): The amplitude of the function.
        b (float): The center of the function.
        c (float): The width of the function.
        d (float): The exponent of the function.
        e (float): The offset of the function.

    Returns:
        float: The value of the super Gaussian function at the specified x value.

    Raises:
        None

    """
    if np.isnan(a) or np.isnan(b) or np.isnan(c) or np.isnan(d) or np.isnan(e):
        return np.full_like(x, np.nan, dtype=float)
    else:
        return -a * np.exp(-(np.abs(x - b) / c) ** d) + e


def fit_super_gaussian(x, y):
    """
    Fit a super Gaussian function to the specified data points.

    Args:
        x (np.array): The x values of the data points.
        y (np.array): The y values of the data points.

    Returns:
        tuple: A tuple containing the parameters of the super Gaussian function.

    Raises:
        None

    """
    init_a = np.max(y) - np.min(y)
    init_b = 0
    init_c = (np.max(x) - np.min(x)) / 2
    init_d = 2
    init_e = np.max(y)
    initial_guess = [init_a, init_b, init_c, init_d, init_e]

    # Set the bounds for the parameters of the super Gaussian function
    b_a = (0, np.max(y))
    b_b = (2*np.min(x), 2*np.max(x))
    b_c = (np.mean(np.diff(x)), np.max(x) - np.min(x))
    b_d = (0, 10)
    b_e = (0, np.max(y))
    bounds = ([b_a[0], b_b[0], b_c[0], b_d[0], b_e[0]], [b_a[1], b_b[1], b_c[1], b_d[1], b_e[1]])
    # Fit a super Gaussian function to the data points
    try:
        popt, _ = curve_fit(super_gaussian, x, y, p0=initial_guess, bounds=bounds)
    except RuntimeError:
        popt = np.full(5, np.nan, dtype=float)
    
    return popt
    

def super_gaussian_fit(x, y, return_dict=True, average=False, features_list:list=[]):
    """
    Fit a super Gaussian function to the specified data points and return the fitted function.

    Args:
        x (list): The x values of the data points.
        y (list): The y values of the data points.
        return_dict (bool): Whether to return the parameters of the fitted super Gaussian function in a dictionary.

    Returns:
        list: An array containing the values of the fitted super Gaussian function.

    Raises:
        None

    """
    if not set(["fit_amp", "fit_center", "fit_width", "fit_exponent", "fit_offset", "fit_fwhm"]) & set(features_list):
            return {"fit_amp": np.nan,
                    "fit_center": np.nan,
                    "fit_width": np.nan,
                    "fit_exponent": np.nan,
                    "fit_offset": np.nan,
                    "fit_fwhm": np.nan}
    y_fits = []
    params = []
    if x.ndim == 1:
        x = np.repeat(x[np.newaxis, :], y.shape[0], axis=0)
    
    for xi, yi in zip(x, y):
        # Fit a super Gaussian function to the data points
        popt = fit_super_gaussian(xi, yi)

        # Calculate the values of the fitted super Gaussian function
        y_fit = super_gaussian(xi, *popt)
        y_fits.append(y_fit)
        params.append(popt)

    if return_dict:
            params = params_to_dict(params.copy(), average=average)
    return np.array(y_fits), params


def params_to_dict(params, average=False):
    """
    Reorder the parameters of the super Gaussian fitted cross-sections [500, 5] --> [5, 500] and return as a dictionary.

    Args:
        params (list): The parameters of the super Gaussian function [500, 5].

    Returns:
        dict: The reordered parameters of the super Gaussian function as a dictionary [5, 500].

    Raises:
        None

    """
    # Reorder the parameters of the super Gaussian function
    fit_amp = []
    fit_ceneter = []
    fit_width = []
    fit_exponent = []
    fit_offset = []
    for param in params:
        fit_amp.append(param[0])
        fit_ceneter.append(param[1])
        fit_width.append(param[2])
        fit_exponent.append(param[3])
        fit_offset.append(param[4])
    
    if average:
        fit_amp = nanmean(fit_amp)
        fit_ceneter = nanmean(fit_ceneter)
        fit_width = nanmean(fit_width)
        fit_exponent = nanmean(fit_exponent)
        fit_offset = nanmean(fit_offset)
        return {"fit_amp": fit_amp, 
                "fit_center": fit_ceneter, 
                "fit_width": fit_width, 
                "fit_exponent": fit_exponent, 
                "fit_offset": fit_offset, 
                "fit_fwhm": 2 * fit_width * (np.log(2) ** (1 / fit_exponent))}
    else:
        return {"fit_amp": np.array(fit_amp), 
                "fit_center": np.array(fit_ceneter), 
                "fit_width": np.array(fit_width), 
                "fit_exponent": np.array(fit_exponent), 
                "fit_offset": np.array(fit_offset), 
                "fit_fwhm": 2 * np.array(fit_width) * (np.log(2) ** (1 / np.array(fit_exponent)))}

#%% ~~~~~~~~~~~~~~~~~~~~ Utility functions ~~~~~~~~~~~~~~~~~~~~
def nanmean(x):
    return np.nanmean(x) if not np.isnan(x).all() else np.nan
