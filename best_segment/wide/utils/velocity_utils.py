import numpy as np
from scipy.signal import correlate2d
from skimage.transform import radon

import video_utils as vu


class VelocityEstimator:
    def __init__(self, input_size=None, fps=166.64, pix_to_um=0.88, scale_factor=None, max_vel=7,
                 uniform_axis='velocity', **kwargs):
        self.fps = fps  # Hemoscope standard FPS
        self.pix_to_um = pix_to_um  # Standard Pixel Size in Wide Camera
        if scale_factor is None:
            self.scale_factor = int(1.39 / pix_to_um * 1450 / fps)  # 1.39 um - pixel size in paper ,
            # 1450 - FPS in paper
            # Scale factor to space points on skeleton and on angle to velocity conversion accordingly
        else:
            self.scale_factor = scale_factor
        self.input_size = input_size  # Image Shape
        self.max_vel = max_vel  # Max velocity we allow
        self.uniform_axis = uniform_axis  # Whether to uniformly sample along velocity axis or theta axis

    @property
    def n_proj(self):
        if self.input_size is None:
            return None
        else:
            return int(np.pi * max(self.input_size) / 2) # The recommend number of projects in a Radon transform

    @property
    def min_vel(self):
        if self.max_vel is None:
            return None
        else:
            return -self.max_vel

    @property
    def max_theta(self):
        if self.min_vel is None:
            return 180  # Max theta in Radon Transform
        else:
            return velocity_to_angle(self.min_vel, pix_to_um=self.pix_to_um, fps=self.fps,
                                     scale_factor=self.scale_factor)

    @property
    def min_theta(self):
        if self.max_vel is None:
            return 0
        else:
            return velocity_to_angle(self.max_vel, pix_to_um=self.pix_to_um, fps=self.fps,
                                     scale_factor=self.scale_factor)

    @property
    def velocities(self):
        if self.uniform_axis == 'velocity':
            return np.linspace(self.min_vel, self.max_vel, self.n_proj)
        elif self.uniform_axis == 'angle':
            return angle_to_velocity(self.thetas, pix_to_um=self.pix_to_um, fps=self.fps,
                                     scale_factor=self.scale_factor)
        else:
            raise ValueError('uniform_axis must be either "velocity" or "angle"')

    @property
    def thetas(self):
        if self.uniform_axis == 'velocity':
            return velocity_to_angle(self.velocities, pix_to_um=self.pix_to_um, fps=self.fps,
                                     scale_factor=self.scale_factor)
        elif self.uniform_axis == 'angle':
            return np.linspace(self.min_theta, self.max_theta, self.n_proj)
        else:
            raise ValueError('uniform_axis must be either "velocity" or "angle"')

    def preprocess(self, x, dt_ac=3, **kwargs):
        x_prep = preprocess_to_radon_transform(x, dt_ac)
        self.input_size = x_prep.shape
        return x_prep

    def radon_transform(self, x, **kwargs):
        # TODO - circle= True? False? N_projections?
        return radon(x, theta=self.thetas, circle=False)

    def cross_sec_radon(self, x, collapse=True, **kwargs):
        x_prep = self.preprocess(x, **kwargs)
        x_radon = self.radon_transform(x_prep)
        if collapse:
            x_radon = np.max(x_radon, axis=-2)
            x_radon = x_radon / x_radon.sum(axis=-1, keepdims=True)
        return x_radon

    def velocity(self, x, **kwargs):
        radon_1d = self.cross_sec_radon(x, **kwargs)
        max_idx = np.argmax(radon_1d, axis=-1)
        vel = self.velocities[max_idx]
        return vel


# %% Auto-correlation utility functions
def autocorr(x):
    return correlate2d(x, x, mode='full')


def autocorr_diff(x_ac, dt_ac=3, **kwargs):
    # dt_ac - in how many pixels to move the array. Omri said that they wanted to move the temporal axis, but emprically
    # the longest axis move was superior.
    if x_ac.shape[-1] > x_ac.shape[-2]:
        x_ac_shifted = np.roll(x_ac, -dt_ac, axis=-1)
    else:
        x_ac_shifted = np.roll(x_ac, -dt_ac, axis=-2)
    return x_ac - x_ac_shifted


def autocorr_crop_center(x_ac_2, **kwargs):
    u, v = x_ac_2.shape
    # Take center third 1/3 -> 2/3 of the image - to recieve ~ same skeleton segment size
    return x_ac_2[u // 3:2 * u // 3, v // 3:2 * v // 3]


# %% Radon transform utility functions
def preprocess_to_radon_transform(x, dt_ac=3, **kwargs):
    # x (n_spatial_samples, n_temporal_samples) <-> (n_points on skeleton segment, n_frames in window)
    x_ac = autocorr(x)
    x_ac_diff = autocorr_diff(x_ac, dt_ac)
    x_ac_2 = autocorr(x_ac_diff)
    x_prep = autocorr_crop_center(x_ac_2)
    return x_prep


def radon_transform(x, return_theta=False, vel_max=None, **kwargs):
    n_angles = int(np.pi * max(x.shape) / 2)
    theta_min, theta_max = theta_limits(vel_max, **kwargs)
    # n_angles = 1800
    # theta_min, theta_max = 0, 180
    endpoint = False if theta_max - theta_min == 180 else True
    theta = np.linspace(theta_min, theta_max, n_angles, endpoint=endpoint)
    x_radon = radon(x, theta=theta, circle=False)
    if return_theta:
        return x_radon, theta
    else:
        return x_radon


def cross_sec_radon(x, n_theta=1800, line_sum=True, **kwargs):
    x_prep = preprocess_to_radon_transform(x, **kwargs)
    x_radon = radon_transform(x_prep, n_theta)
    if line_sum:
        x_radon = np.max(x_radon, axis=-2)
        x_radon = x_radon / x_radon.sum(axis=-1, keepdims=True)
    return x_radon


def center_radon(x_radon, theta):
    # Not working
    theta_center = np.argmax(x_radon, axis=-1)
    theta_centered = (theta - 90 - theta[theta_center]) % 180
    theta_centered[theta_centered > 90 + theta[theta_center]] -= 180
    theta_roll = x_radon.shape[-1] // 2 - theta_center
    # theta_centered = np.roll(theta_centered, theta_roll)
    x_radon_centered = np.roll(x_radon, -theta_roll, axis=-1)
    return x_radon_centered, theta_centered


# %% Velocity utility functions
def angle_to_velocity(theta, pix_to_um=0.88, fps=166.64, scale_factor=1, **kwargs):
    """
    Calculates the velocity based on the angle of motion.

    Parameters:
    - theta (float): The angle of motion in degrees.
    - pix_to_um (float, optional): Conversion factor from pixels to micrometers. Default is 1.5.
    - fps (int, optional): The original frame rate of the video. Default is 166.64.

    Returns:
    - vel (float): The calculated velocity in mm/s.
    """
    vel = scale_factor * pix_to_um * fps * np.tan(np.deg2rad(90 - theta)) / 1e3
    return vel


def velocity_to_angle(vel, pix_to_um=0.88, fps=166.64, scale_factor=1, **kwargs):
    """
    Calculates the angle of motion based on the velocity.

    Parameters:
    - vel (float): The velocity in mm/s.
    - pix_to_um (float, optional): Conversion factor from pixels to micrometers. Default is 1.5.
    - fps (int, optional): The original frame rate of the video. Default is 166.64.

    Returns:
    - theta (float): The calculated angle of motion in degrees.
    """
    theta = 90 - np.rad2deg(np.arctan(1e3 * vel / (scale_factor * pix_to_um * fps)))
    return theta


def theta_limits(vel_max=None, **kwargs):
    if vel_max is None:
        return 0, 180
    else:
        theta_min = velocity_to_angle(vel_max, **kwargs)
        theta_max = velocity_to_angle(-vel_max, **kwargs)
        return theta_min, theta_max


# ----------------------------------------------------------------------------------------------------------------------
#                                                
# ----------------------------------------------------------------------------------------------------------------------

def get_temporal_velocity(line, vel_window_sec=70e-3, d_vel_window_sec=50e-3, heart_rate_window_sec=5,
                          original_frame_rate=166.64, t0=0, **kwargs):
    # Line - pixel indices, where to sample
    # vel_window_sec - temporal window
    print("hi")
    # time = np.arange(t0, heart_rate_window_sec, 1 / original_frame_rate)
    # Round to a numbe that will supply an integer number of frames
    vel_window_sec = int(vel_window_sec * original_frame_rate) / original_frame_rate
    # d_vel_window_sec = int(d_vel_window_sec * original_frame_rate) / original_frame_rate
    # heart_rate_window_sec - How big of a window do I compute heart rate on
    # t0 - initial time to start measuring
    time = np.arange(t0, heart_rate_window_sec, vel_window_sec / 4) # 1/4 of a window intersection between window to
    # window
    velocities = np.zeros_like(time)
    kwargs.pop('t0', None)
    kwargs.pop('t1', None)
    for idx, tt in enumerate(time):
        print('Processing frame {}/{}'.format(idx + 1, len(time)))
        cross_sections = get_cross_sections(line, t0=tt, t1=tt + vel_window_sec,
                                                 original_frame_rate=original_frame_rate, **kwargs)
        x_prep = preprocess_to_radon_transform(cross_sections, **kwargs)
        radon_transformed, theta = radon_transform(x_prep, return_theta=True, **kwargs)
        radon_collapsed_max = np.max(radon_transformed, axis=0)
        radon_collapsed = radon_collapsed_max / np.sum(radon_collapsed_max)
        max_index = np.argmax(radon_collapsed)
        best_angle = theta[max_index]
        best_velocity = angle_to_velocity(best_angle, original_frame_rate=original_frame_rate, **kwargs)
        # vel = file_to_velocity(line, t0=tt, t1=tt+vel_window_sec, original_frame_rate=original_frame_rate, **kwargs)
        vel = best_velocity
        velocities[idx] = vel

    return time, velocities


def plot_velocities(line, **kwargs):
    time, velocities = get_temporal_velocity(line, **kwargs)
    plt.plot(time, velocities)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/s)')
    plt.grid()
    plt.show()


def velocity_to_frequency(velocities, time, **kwargs):
    # Get heart rate via FFT from the velocity
    velocities = velocities - np.mean(velocities)  # Reduce DC component
    dt = np.mean(np.diff(time)) # Should be the same as time[1] - time[0]
    freqs = np.fft.rfftfreq(len(velocities), dt)
    fft = np.fft.rfft(velocities)
    return freqs, np.abs(fft)


def plot_frequency(velocities, time, **kwargs):
    freqs, fft = velocity_to_frequency(velocities, time)
    plt.plot(freqs, fft)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()


def get_heart_rate(velocities, time, **kwargs):
    freqs, fft = velocity_to_frequency(velocities, time)
    max_freq_idx = np.argmax(fft)
    heart_rate = freqs[max_freq_idx] * 60
    # *60 for transfer from beats per second -> Beats per minute range (good range e.g. : ~60-80)
    return heart_rate
