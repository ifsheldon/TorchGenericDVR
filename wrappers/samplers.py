import pytorch_lightning as pl
import torch
import warnings
from interpolators.pl_modules import TrilinearInterpolation
import torch.nn.functional as F


class TransferFunctionModel1D(pl.LightningModule):
    def __init__(self, transfer_function_tensor, tf_require_grad=True, align_corners=True):
        """
        Init
        :param transfer_function_tensor: 2D tensor of shape (resolution, channel)
        :param tf_require_grad: whether TF requires gradients
        :param align_corners: If True, x=0 is mapped to tensor[0]; If false, x= 1/(2*resolution) is mapped to tensor[0]
        """
        super(TransferFunctionModel1D, self).__init__()
        shape = transfer_function_tensor.shape
        assert len(shape) == 2  # grid_num * channel_num
        self.tf = torch.nn.Parameter(transfer_function_tensor.reshape(1, shape[1], 1, shape[0]),
                                     requires_grad=tf_require_grad)  # (N=1, C, H=1, W)
        self.channel_num = shape[1]
        self.align_corners = align_corners

    def forward(self, scalars):
        """
        :param scalars: [0,1] values of shape (batch_num, sample_num)
        :return: tensor of shape(batch_num, channel, sample_num)
        """
        assert len(scalars.shape) == 2  # (N, num)
        batch, sample_num = scalars.shape
        x_indices = scalars * 2 - 1
        y_indices = -torch.ones_like(x_indices)
        xy_indices = torch.stack([x_indices, y_indices], dim=-1).unsqueeze(1)  # (N, 1, num, 2)
        sampled_values = F.grid_sample(self.tf.repeat(scalars.shape[0], 1, 1, 1), xy_indices,
                                       align_corners=self.align_corners)  # (N, C, 1, num)
        return sampled_values.view(batch, self.channel_num, sample_num)


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
        assert len(sample_indices.shape) == 4
        assert sample_indices.shape[-1] == 3
        batch_size = sample_indices.shape[0]
        volume_not_batched = self.volume_batch_size == 0
        assert volume_not_batched or self.volume_batch_size == batch_size
        if volume_not_batched:
            volume = self.volume.repeat(batch_size, 1, 1, 1, 1)
        else:
            volume = self.volume
        return self.trilinear_interpolator(volume, sample_indices)
