import cv2
import numpy as np
import os
import glob
from contextlib import contextmanager

#%% ~~~~~~~~~~~~~~ I/O functions ~~~~~~~~~~~~~~
# Load images as numpy arrays
def load_images(root, regex=None, n0=0, n1=None):
    # Get the list of image files
    if regex is None:
        regex = '*.png'
    files = sorted(glob.glob(os.path.join(root, regex)))[n0:]
    if n1 is not None:
        files = files[:n1]
    # Load each image file
    images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in files]
    return np.array(images)

def load_images_as_video(root, regex=None, fps=166.64, t0=0, t1=None):
    # Load the images
    n0 = int(t0 * fps)
    n1 = int(t1 * fps) if t1 is not None else None
    images = load_images(root, regex, n0=n0, n1=n1)
    # Create a time axis
    n1 = n0 + images.shape[0]
    time = np.arange(n0, n1) / fps
    return images, time

# %% ~~~~~~~~~~~~~~ Old video loading ~~~~~~~~~~~~~~
# Custom context manager for cv2.VideoCapture
@contextmanager
def VideoCapture(video):
    # Check if video is a string (file path) or a cv2.VideoCapture object
    if isinstance(video, str):
        cap = cv2.VideoCapture(video)
    elif isinstance(video, cv2.VideoCapture):
        cap = video
    else:
        raise TypeError('video must be a file path (str) or a cv2.VideoCapture object')
    try:
        yield cap
    finally:
        cap.release()

def get_fps(video):
    with VideoCapture(video) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def get_times(video, t0=0, t1=None, fps=None):
    # Get the frame rate
    if fps is None:
        fps = video.get(cv2.CAP_PROP_FPS)
    # Get the end time if not provided
    if t1 is None:
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        t1 = total_frames / fps
    # Return the time array
    return np.arange(t0, t1, 1 / fps)

def vid2array(video, t0=0, t1=None, fps=None):
    # Open the video file
    with VideoCapture(video) as cap:
        if fps is None:
            fps = video.get(cv2.CAP_PROP_FPS)
        # Set time scale
        time = get_times(cap, t0, t1, fps)
        
        # Set the start frame for cv2.VideoCapture
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(time[0] * fps))

        # Grab frames until the end time
        frames = []
        for t in time:
            # Read the next frame
            ret, frame = cap.read()
            
            # Check if the frame was read successfully
            if not ret:
                break
            
            # Blur the frame to remove noise and append to the list
            # frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frames.append(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), ksize=(9, 9), sigmaX=3, sigmaY=3))
        return np.array(frames)

def load_video(video, t0=0, t1=None, frame_rate=None):
    # Open the video file
    with VideoCapture(video) as cap:
        # Set time scale
        fps = cap.get(cv2.CAP_PROP_FPS) if frame_rate is None else frame_rate
        dt = 1 / fps
        time = get_times(cap, t0, t1, fps)
        frames = vid2array(cap, t0, t1, fps)
    return frames, time, fps, dt