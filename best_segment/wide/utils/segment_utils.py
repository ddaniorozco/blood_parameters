# %% Imports
import numpy as np
import pandas as pd
from scipy.optimize import fsolve


# %% Line class
class Line:
    def __init__(self, filename, recording_id=None):
        self.filename = filename
        self.recording_id = recording_id
        if recording_id is not None and filename is not None:
            self.endpoints_data = pd.read_csv(filename)
        else:
            self.endpoints_data = None

    def load_file(self, filename):
        self.filename = filename
        self.endpoints_data = pd.read_csv(filename)

    @property
    def x0(self):
        try:
            return float(
                self.endpoints_data[(self.endpoints_data['recording_id'] == self.recording_id)]['start_coord'].values[
                    0][1:-1].split(",")[0])
        except:
            raise ValueError('No image path provided')

    @property
    def y0(self):
        try:
            return float(
                self.endpoints_data[(self.endpoints_data['recording_id'] == self.recording_id)]['start_coord'].values[
                    0][1:-1].split(",")[1])
        except:
            raise ValueError('No image path provided')

    @property
    def x1(self):
        try:
            return float(
                self.endpoints_data[(self.endpoints_data['recording_id'] == self.recording_id)]['end_coord'].values[0][
                1:-1].split(",")[0])
        except:
            raise ValueError('No image path provided')

    @property
    def y1(self):
        try:
            return float(
                self.endpoints_data[(self.endpoints_data['recording_id'] == self.recording_id)]['end_coord'].values[0][
                1:-1].split(",")[1])
        except:
            raise ValueError('No image path provided')

        return self.x0

    def get_line_ends(self):
        return np.array([[self.y0, self.x0], [self.y1, self.x1]])

    def get_line_center(self):
        try:
            cntr_strs = \
            self.endpoints_data[(self.endpoints_data['recording_id'] == self.recording_id)]['line_center_coord'].values[
                0][1:-1].split(",")
            return np.array([float(cntr_strs[1]), float(cntr_strs[0])])
        except:
            raise ValueError('No image path provided')

    def x(self, s):
        return self.x0 + s * (self.x1 - self.x0)

    def y(self, s):
        return self.y0 + s * (self.y1 - self.y0)


# %% Sample line class
class SampleLine:
    def __init__(self, sample):
        self.sample = sample

    @property
    def x0(self):
        try:
            return float(self.sample['start_coord'][1:-1].split(",")[0])
        except:
            raise ValueError('No image path provided')

    @property
    def y0(self):
        try:
            return float(self.sample['start_coord'][1:-1].split(",")[1])
        except:
            raise ValueError('No image path provided')

    @property
    def x1(self):
        try:
            return float(self.sample['end_coord'][1:-1].split(",")[0])
        except:
            raise ValueError('No image path provided')

    @property
    def y1(self):
        try:
            return float(self.sample['end_coord'][1:-1].split(",")[1])
        except:
            raise ValueError('No image path provided')

    @property
    def x_line_0(self):
        try:
            return float(self.sample['line_begin_coord'][1:-1].split(",")[0])
        except:
            raise ValueError('No image path provided')

    @property
    def y_line_0(self):
        try:
            return float(self.sample['line_begin_coord'][1:-1].split(",")[1])
        except:
            raise ValueError('No image path provided')

    @property
    def x_line_1(self):
        try:
            return float(self.sample['line_end_coord'][1:-1].split(",")[0])
        except:
            raise ValueError('No image path provided')

    @property
    def y_line_1(self):
        try:
            return float(self.sample['line_end_coord'][1:-1].split(",")[1])
        except:
            raise ValueError('No image path provided')

    def get_line_ends(self):
        return np.array([[self.y0, self.x0], [self.y1, self.x1]])

    def get_whole_line_ends(self):
        return np.array([[self.y_line_0, self.x_line_0], [self.y_line_1, self.x_line_1]])

    def get_line_center(self):
        line_ends = self.get_line_ends()
        return np.mean(line_ends, axis=0)

    def get_line_unit_vector(self):
        vector = np.array([self.y_line_1 - self.y_line_0, self.x_line_1 - self.x_line_0])
        vector_norm = np.linalg.norm(vector)
        return vector / vector_norm

    def get_whole_line_center(self):
        try:
            cntr_strs = self.sample['line_center_coord'].values[0][1:-1].split(",")
            return np.array([float(cntr_strs[1]), float(cntr_strs[0])])
        except:
            raise ValueError('No image path provided')

    def x(self, s):
        return self.x0 + s * (self.x1 - self.x0)

    def y(self, s):
        return self.y0 + s * (self.y1 - self.y0)

    def x_line(self, s):
        return self.x_line_0 + s * (self.x_line_1 - self.x_line_0)

    def y_line(self, s):
        return self.y_line_0 + s * (self.y_line_1 - self.y_line_0)
    
    def get_line(self, s=None, n=100):
        if s is None:
            s = np.linspace(0, 1, n)
        return np.array([self.y(s), self.x(s)]).T


# %% Solver class
class Intersection:
    def __init__(self, spline=None, line=None):
        if spline is not None:
            self.spline_x, self.spline_y, self.t = spline
        else:
            self.spline_x, self.spline_y, self.t = None, None, None
        self.line = line

    def equations(self, vars):
        try:
            t, s = vars
            return [self.spline_x(t) - self.line.x(s), self.spline_y(t) - self.line.y(s)]
        except:
            raise ValueError('No spline or line provided')

    def equations_whole(self, vars):
        try:
            t, s = vars
            return [self.spline_x(t) - self.line.x_line(s), self.spline_y(t) - self.line.y_line(s)]
        except:
            raise ValueError('No spline or line provided')

    @property
    def initial_guess(self):
        try:
            return [self.t[-1] / 2, 0.5]
        except:
            raise ValueError('No spline provided')

    def solve(self, whole=False):
        if whole:
            solution = fsolve(self.equations_whole, self.initial_guess)
            t_int, s_int = solution
            return s_int
        else:
            solution = fsolve(self.equations, self.initial_guess)
            t_int, s_int = solution
            return self.spline_x(t_int), self.spline_y(t_int), t_int

    def equations_center_y(self, vars):
        try:
            s = vars
            center = self.line.get_line_center()
            return self.line.y_line(s) - center[0]
        except:
            raise ValueError('No spline or line provided')

    def equations_center_x(self, vars):
        try:
            s = vars
            center = self.line.get_line_center()
            return self.line.x_line(s) - center[1]
        except:
            raise ValueError('No spline or line provided')

    @property
    def initial_guess_center(self):
        try:
            return 0.5
        except:
            raise ValueError('No spline provided')

    def solve_center(self):
        s_center_x = fsolve(self.equations_center_x, self.initial_guess_center)
        s_center_y = fsolve(self.equations_center_y, self.initial_guess_center)
        return (s_center_x + s_center_y) / 2


def intersect(line_0, line_1):
    min_distance = np.inf  # Initialize with infinity
    closest_points = (None, None)  # Initialize closest points
    closest_inds = (None, None)  # Initialize closest indices

    for ind_0, point_0 in enumerate(line_0):
        for ind_1, point_1 in enumerate(line_1):
            distance = np.linalg.norm(point_0 - point_1)  # Euclidean distance
            if distance < min_distance:
                min_distance = distance
                closest_points = (point_0, point_1)
                closest_inds = (ind_0, ind_1)

    return closest_points, closest_inds,  min_distance