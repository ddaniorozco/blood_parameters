#%% Imports
from sharpness_estimation.wide.src.laplacian_estimator import LaplacianSharpnessEstimator

# %% Simplified class
class SharpnessEstimator(LaplacianSharpnessEstimator):
    def __init__(self,
                 gaussian_kernel_size=(3, 3),
                 median_kernel_size=5,
                 laplacian_ksize=5,
                 outline_belt_size=9,
                 clipping_percentile=99.9):
        super().__init__(gaussian_kernel_size, median_kernel_size, laplacian_ksize, outline_belt_size, clipping_percentile)
    
    def estimate_sharpness(self, image, mask=None):
        self.process_image(image, mask)
        return self.get_sharpness_score()
        