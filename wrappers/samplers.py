import pytorch_lightning as pl
import torch
import warnings
from interpolators.pl_modules import TrilinearInterpolation


class TrilinearVolumeSampler(pl.LightningModule):
    def __init__(self, volume, volume_require_grad=True):
        """
        A trilinear sampler on volume(s)
        :param volume: volume(s) of shape (channel, x, y, z) or (batch, channel, x, y, z)
        :param volume_require_grad: whether volume require grad, default = True
        """
        super(TrilinearVolumeSampler, self).__init__()
        if volume.ndimension() == 5:
            batch_size, num_channels, num_d, height, width = volume.shape
            self.volume_batch_size = batch_size
            warnings.warn(f"The batch size of volume is fixed to {batch_size}, "
                          f"then your input must have the same batch size")
        elif volume.ndimension() == 4:
            num_channels, num_d, height, width = volume.shape
            self.volume_batch_size = 0
        else:
            raise Exception("Invalid dimension of volume")
        self.volume = torch.nn.Parameter(volume, requires_grad=volume_require_grad)
        scale_factor = torch.tensor([width - 1.0, height - 1.0, num_d - 1.0], dtype=torch.float)
        self.trilinear_interpolator = TrilinearInterpolation(_scale_factor=scale_factor)

    def forward(self, sample_indices):
        """
        Interpolate
        :param sample_indices: tensor of shape (batch, width, height, 3), range = [-1, 1]
        :return: interpolated tensor of shape (batch, volume_channel, width, height)
        """
        batch_size = sample_indices.shape[0]
        volume_not_batched = self.volume_batch_size == 0
        assert volume_not_batched or self.volume_batch_size == batch_size
        if volume_not_batched:
            volume = self.volume.repeat(batch_size, 1, 1, 1, 1)
        else:
            volume = self.volume
        return self.trilinear_interpolator(volume, sample_indices)
