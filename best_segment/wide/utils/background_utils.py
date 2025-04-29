import torch
from torch import nn
from torch.nn import functional as F


# %% Background subtractor class
class BackgroundSubtractor(nn.Module):
    """
    A class that performs background subtraction on input tensors.

    Args:
        window_size (int): The size of the sliding window used for convolution. Default is 5.
        std_factor (float): The factor multiplied by the standard deviation to control the amount of background subtraction. Default is 0.5.
        invert (bool): Whether to invert the background-subtracted tensor. Default is False.
        normalize (bool): Whether to normalize the background-subtracted tensor. Default is False.
    """

    def __init__(self, window_size=5, std_factor=0.5, invert=False, normalize=False, **kwargs):
        super(BackgroundSubtractor, self).__init__()
        # Ensure window_size is odd
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        self.std_factor = std_factor
        self.invert = invert
        self.normalize = normalize
        self.register_buffer("window", torch.ones((1, 1, window_size), dtype=torch.float32) / window_size)

    @property
    def device(self):
        return self.window.device

    @property
    def pad_size(self):
        return self.window_size // 2

    @staticmethod
    def t_size(tensor):
        tensor_t = tensor.transpose(0, 2).detach().clone()
        return tensor_t.size()

    def prepare_tensor(self, tensor):
        """
        Prepares the input tensor for background subtraction.

        Args:
            tensor (Tensor): The input tensor.

        Returns:
            Tensor: The prepared tensor.
            tuple: The size of the original tensor before padding, if `return_t_size` is True.
        """
        # Ensure tensor is float for accurate calculations
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        tensor_t = tensor.transpose(0, 2)  # New shape: (5, 5, 100) for example tensor
        padded_tensor = F.pad(tensor_t, (self.pad_size, self.pad_size), mode='reflect').reshape(-1, 1, tensor_t.shape[
            -1] + 2 * self.pad_size)
        return padded_tensor

    def get_signal_statistics(self, tensor):
        """
        Calculates the running mean and standard deviation of the input tensor.

        Args:
            tensor (Tensor): The input tensor.

        Returns:
            Tensor: The running mean of the input tensor.
            Tensor: The running standard deviation of the input tensor.
        """
        t_size = self.t_size(tensor)
        # Prepare tensor
        padded_tensor = self.prepare_tensor(tensor)
        # Perform 1D convolution
        convolved = F.conv1d(padded_tensor, self.window).reshape(t_size).transpose(0, 2)
        # Calculate squares for standard deviation
        convolved_squares = F.conv1d(padded_tensor ** 2, self.window).reshape(t_size).transpose(0, 2)
        # Calculate running mean and standard deviation
        running_mean = convolved
        running_std = torch.abs(convolved_squares - convolved ** 2).sqrt()
        return running_mean, running_std

    def get_background_signal(self, tensor):
        """
        Calculates the background signal by adding the running mean and a scaled running standard deviation.

        Args:
            tensor (Tensor): The input tensor.

        Returns:
            Tensor: The background signal.
        """
        running_mean, running_std = self.get_signal_statistics(tensor)
        return running_mean + self.std_factor * running_std

    def forward(self, tensor):
        """
        Performs background subtraction on the input tensor.

        Args:
            tensor (Tensor): The input tensor.

        Returns:
            numpy.ndarray: The background-subtracted tensor.
        """
        # Prepare tensor
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        # Get the background signal
        background_signal = self.get_background_signal(tensor)

        # Remove the background signal
        background_removed = tensor - background_signal

        if self.invert:
            background_removed = -background_removed

        # Keep only the negative values and take the absolute value
        background_removed = torch.where(background_removed < 0, background_removed.abs(),
                                         torch.zeros_like(background_removed))

        if self.normalize:
            background_removed = background_removed / background_removed.max() * 255

        return background_removed.squeeze().detach().numpy()
