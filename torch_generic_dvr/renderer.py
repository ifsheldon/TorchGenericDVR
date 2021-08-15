import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from .interpolations import TrilinearInterpolation
from .utils import *


class DirectVolumeRendering(nn.Module):
    def __init__(self,
                 n_ray_samples,
                 feature_img_resolution,
                 depth_range,
                 fov,
                 feature_transfer_function,
                 alpha_transfer_function):
        """
        :param n_ray_samples: step num along a ray
        :param feature_img_resolution: resolution of rendered feature image
        :param depth_range: tuple (depth start, depth end)
        :param fov: (degree) field of view
        :param feature_transfer_function: transfer function for converting features,
         handling tensor of shape (batch, num of eval points, channel), output shape = (batch, new_channel, num of eval points)
        :param  alpha_transfer_function: transfer function for generating alpha values,
         handling tensor of shape (batch, num of eval points, channel), output shape = (batch, 1, num of eval points)
        """
        super(DirectVolumeRendering, self).__init__()
        self.n_ray_samples = n_ray_samples
        self.feature_img_resolution = feature_img_resolution
        self.depth_range = depth_range
        self.camera_matrix = nn.Parameter(self.get_camera_mat(fov))
        self.feature_tf = feature_transfer_function
        self.alpha_tf = alpha_transfer_function
        self.trilinear_interpolator = TrilinearInterpolation()

    def forward(self, volume, world_mat, add_noise, device):
        """
        forward pass
        :param volume: should be of shape [B,F,D,H,W], NOTICE!
        :param world_mat: matrices specifying camera's positions and poses
        :param add_noise: whether to add ray jitter AND scala noise
        :param device: which device tensors on
        :return: feature images rendered by DVR
        """
        assert volume.shape[0] == world_mat.shape[0]
        batch_size = world_mat.shape[0]
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        self.device = device
        feature_img = self.volume_render_image(volume, camera_mat, world_mat, batch_size, add_noise)
        return feature_img

    def image_points_to_world(self, image_points, camera_mat, world_mat, negative_depth=True):
        ''' Transforms points on image plane to world coordinates.

        In contrast to transform_to_world, no depth value is needed as points on
        the image plane have a fixed depth of 1.

        Args:
            image_points (tensor): image points tensor of size B x N x 2
            camera_mat (tensor): camera matrix
            world_mat (tensor): world matrix
        '''
        batch_size, n_pts, dim = image_points.shape
        assert (dim == 2)
        depth_image = torch.ones(batch_size, n_pts, 1).to(self.device)
        if negative_depth:
            depth_image *= -1.
        return self.transform_to_world(image_points, depth_image, camera_mat, world_mat)

    def cast_ray_and_get_eval_points(self, img_res, batch_size, depth_range, n_steps, camera_mat, world_mat,
                                     ray_jittering):
        # Arrange Pixels
        pixels = self.arrange_pixels((img_res, img_res), batch_size)[1].to(self.device)
        pixels[..., -1] *= -1.
        n_points = img_res * img_res
        # Project to 3D world
        pixel_pos_wc = self.image_points_to_world(pixels, camera_mat, world_mat)
        camera_pos_wc = self.origin_to_world(n_points, camera_mat, world_mat)
        # batch_size x n_points x n_steps
        step_depths = depth_range[0] + \
                      torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (
                              depth_range[1] - depth_range[0])
        step_depths = step_depths.to(self.device)
        step_depths = step_depths.repeat(batch_size, n_points, 1)
        if ray_jittering:
            step_depths = self.add_noise_to_interval(step_depths)
        point_pos_wc = self.get_evaluation_points(pixel_pos_wc, camera_pos_wc,
                                                  step_depths)  # shape (batch, num of eval points, 3)
        return point_pos_wc

    def volume_render_image(self, volume, camera_mat, world_mat, batch_size, add_noise):
        img_res = self.feature_img_resolution
        n_steps = self.n_ray_samples
        point_pos_wc = self.cast_ray_and_get_eval_points(img_res, batch_size, self.depth_range, n_steps,
                                                         camera_mat, world_mat,
                                                         add_noise)  # shape (batch, num of eval points, 3)
        density_scalars = self.trilinear_interpolator(volume,
                                                      point_pos_wc.unsqueeze(
                                                          1)).squeeze(2)  # shape (batch, channel=1, num of eval points)
        density_scalars = density_scalars.permute(0, 2, 1)  # shape (batch, num of eval points, channel=1,)
        # mask out out-of-bound positions
        padding = 0.01
        positions_valid = torch.all(point_pos_wc <= 1.0 + padding, dim=-1) & \
                          torch.all(point_pos_wc >= -1.0 - padding, dim=-1)
        positions_valid = positions_valid.unsqueeze(-1)\
            .repeat(1, 1, density_scalars.shape[2])  # shape (batch, num of eval points, channel=1,)
        density_scalars[~positions_valid] = 0.0
        if add_noise:
            # As done in NeRF, add noise during training
            density_scalars += torch.randn_like(density_scalars)

        features = self.feature_tf(density_scalars).permute(0, 2, 1)  # shape(batch, num of eval points, channel)
        alphas = self.alpha_tf(density_scalars).permute(0, 2, 1)  # shape(batch, num of eval points, 1)
        # Reshape
        n_points = img_res * img_res
        features = features.reshape(
            batch_size,
            n_points,
            n_steps,
            -1  # channels
        )
        alphas = alphas.reshape(
            batch_size,
            n_points,
            n_steps
        )
        # DVR composition
        weights = self.calc_volume_weights(alphas)
        feat_map = torch.sum(weights.unsqueeze(-1) * features, dim=-2)
        # Reformat output
        feat_map = feat_map.permute(0, 2, 1).reshape(batch_size, -1, img_res, img_res)  # B x feat x h x w
        feat_map = feat_map.permute(0, 1, 3, 2)  # new to flip x/y
        return feat_map

    @staticmethod
    def arrange_pixels(resolution, batch_size, image_range=(-1., 1.)):
        ''' Arranges pixels for given resolution in range image_range.

        The function returns the unscaled pixel locations as integers and the
        scaled float values.

        Args:
            resolution (tuple): image resolution
            batch_size (int): batch size
            image_range (tuple): range of output points (default [-1, 1])
        '''
        h, w = resolution
        # Arrange pixel location in scale resolution
        pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
        pixel_locations = torch.stack(
            [pixel_locations[0], pixel_locations[1]],
            dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
        pixel_scaled = pixel_locations.clone().float()

        # Shift and scale points to match image_range
        scale = (image_range[1] - image_range[0])
        loc = scale / 2
        pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
        pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

        return pixel_locations, pixel_scaled

    def origin_to_world(self, n_points, camera_mat, world_mat):
        """ Transforms origin (camera location) to world coordinates.

        Args:
            n_points (int): how often the transformed origin is repeated in the
                form (batch_size, n_points, 3)
            camera_mat (tensor): camera matrix
            world_mat (tensor): world matrix
        """
        batch_size = camera_mat.shape[0]
        # Create origin in homogen coordinates
        p = torch.zeros(batch_size, 4, n_points).to(self.device)
        p[:, -1] = 1.
        # Apply transformation
        p_world = world_mat @ camera_mat @ p
        # Transform points back to 3D coordinates
        p_world = p_world[:, :3].permute(0, 2, 1)
        return p_world

    @staticmethod
    def transform_to_world(pixels, depth, camera_mat, world_mat, use_absolute_depth=True):
        ''' Transforms pixel positions p with given depth value d to world coordinates.

        Args:
            pixels (tensor): pixel tensor of size B x N x 2
            depth (tensor): depth tensor of size B x N x 1
            camera_mat (tensor): camera matrix
            world_mat (tensor): world matrix
        '''
        assert (pixels.shape[-1] == 2)
        # Transform pixels to homogen coordinates
        pixels = pixels.permute(0, 2, 1)
        pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)
        # Project pixels into camera space
        if use_absolute_depth:
            pixels[:, :2] = pixels[:, :2] * depth.permute(0, 2, 1).abs()
            pixels[:, 2:3] = pixels[:, 2:3] * depth.permute(0, 2, 1)
        else:
            pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)
        # Transform pixels to world space
        p_world = world_mat @ camera_mat @ pixels
        # Transform p_world back to 3D coordinates
        p_world = p_world[:, :3].permute(0, 2, 1)
        return p_world

    @staticmethod
    def calc_volume_weights(alpha):
        # alpha: shape (batch, res*res, step_num)
        weights = alpha * \
                  torch.cumprod(torch.cat([
                      torch.ones_like(alpha[:, :, :1]),
                      (1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
        return weights

    @staticmethod
    def get_evaluation_points(pixels_world, camera_world, di):
        batch_size = pixels_world.shape[0]
        ray = pixels_world - camera_world
        p = camera_world.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * ray.unsqueeze(-2).contiguous()
        p = p.reshape(batch_size, -1, 3)
        return p

    @staticmethod
    def add_noise_to_interval(di):
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti

    @staticmethod
    def get_camera_mat(fov, invert=True):
        focal = 1. / np.tan(0.5 * np.radians(fov))
        focal = focal.astype(np.float32)
        mat = torch.tensor([
            [focal, 0., 0., 0.],
            [0., focal, 0., 0.],
            [0., 0., 1, 0.],
            [0., 0., 0., 1.]
        ]).reshape(1, 4, 4)
        if invert:
            mat = torch.inverse(mat)
        return mat


class DirectVolumeRenderer(pl.LightningModule):
    """ Direct Volume Renderer.

    Args:
        n_ray_samples (int): number of samples per ray
        depth_range (tuple): near and far depth plane
        feature_img_resolution (int): resolution of volume-rendered image
        neural_renderer (nn.Module): neural renderer
        fov (float): field of view
    """

    def __init__(self,
                 volume,
                 feature_transfer_function,
                 alpha_transfer_function,
                 n_ray_samples,
                 feature_img_resolution,
                 fov,
                 depth_range,
                 neural_renderer=None,
                 ):
        super().__init__()
        self.neural_renderer = None if neural_renderer is None else neural_renderer
        self.volume = nn.Parameter(volume.unsqueeze(0), requires_grad=False)
        self.dvr_op = DirectVolumeRendering(n_ray_samples, feature_img_resolution, depth_range, fov,
                                            feature_transfer_function, alpha_transfer_function)

    def forward(self, world_mat, mode="training"):
        batch_size = world_mat.shape[0]
        volume_batch = self.volume.repeat(batch_size, 1, 1, 1, 1)
        add_noise = mode == "training"
        feature_img = self.dvr_op(volume_batch, world_mat, add_noise, self.device)
        if self.neural_renderer is not None:
            rgb = self.neural_renderer(feature_img)
        else:
            rgb = feature_img
        return rgb

    def predict_step(self, batch, _batch_idx, _data_loader_idx=None):
        return self(batch, mode="predict")
