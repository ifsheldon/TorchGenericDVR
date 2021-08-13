import math
import torch
from torch import nn
import logging
import numpy as np
import pytorch_lightning as pl
from samplers import TrilinearVolumeSampler, TransferFunctionModel1D
from torchvision.transforms import ToPILImage
from utils import *
from torch.utils.data import Dataset


class RandomCameraPoses(Dataset):
    """ Random Camara Positions and Poses

        Args:
            range_u (tuple): rotation range (0 - 1)
            range_v (tuple): elevation range (0 - 1)
            range_radius(tuple): radius range
    """

    def __init__(self, length, range_u, range_v, range_radius):
        super(RandomCameraPoses, self).__init__()
        self.length = length
        self.range_u = range_u
        self.range_v = range_v
        self.range_radius = range_radius

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.get_random_pose().squeeze()

    def get_random_pose(self):
        batch_size = 1
        location = self.sample_on_sphere(self.range_u, self.range_v, batch_size)
        radius = self.range_radius[0] + \
                 torch.rand(batch_size) * (self.range_radius[1] - self.range_radius[0])
        location = location * radius.unsqueeze(-1)
        R = self.look_at(location.numpy(), np.array([0, 0, 0]), np.array([0, 0, 1]))
        RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
        RT[:, :3, :3] = R
        RT[:, :3, -1] = location
        return RT

    @staticmethod
    def sample_on_sphere(range_u, range_v, batch_size):
        u = torch.rand(batch_size) * (range_u[1] - range_u[0]) + range_u[0]
        v = torch.rand(batch_size) * (range_v[1] - range_v[0]) + range_v[0]
        pi = torch.tensor(math.pi)
        theta = 2 * pi * u
        phi = torch.arccos(1 - 2 * v)
        cx = torch.sin(phi) * torch.cos(theta)
        cy = torch.sin(phi) * torch.sin(theta)
        cz = torch.cos(phi)
        sample = torch.stack([cx, cy, cz], dim=-1).float()
        return sample

    @staticmethod
    def look_at(eye, at, up, eps=1e-5):
        at = at.astype(float).reshape(1, 3)
        up = up.astype(float).reshape(1, 3)
        eye = eye.reshape(-1, 3)
        up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
        eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

        z_axis = eye - at
        z_axis /= np.max(np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]))

        x_axis = np.cross(up, z_axis)
        x_axis /= np.max(np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]))

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.max(np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]))

        r_mat = np.concatenate(
            (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(
                -1, 3, 1)), axis=2)

        return torch.tensor(r_mat).float()


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
                 transfer_function_data,
                 n_ray_samples,
                 feature_img_resolution,
                 fov,
                 depth_range,
                 neural_renderer=None,
                 ):
        super().__init__()
        self.n_ray_samples = n_ray_samples
        self.feature_img_resolution = feature_img_resolution
        self.depth_range = depth_range
        self.camera_matrix = nn.Parameter(self.get_camera_mat(fov))
        self.volume_sampler = TrilinearVolumeSampler(volume)
        self.tf = TransferFunctionModel1D(transfer_function_data)
        self.neural_renderer = None if neural_renderer is None else neural_renderer

    def arrange_pixels(self, resolution, batch_size, image_range=(-1., 1.)):
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
            [pixel_locations[0].to(self.device), pixel_locations[1].to(self.device)],
            dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
        pixel_scaled = pixel_locations.clone().float()

        # Shift and scale points to match image_range
        scale = (image_range[1] - image_range[0])
        loc = scale / 2
        pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
        pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

        return pixel_locations, pixel_scaled

    def origin_to_world(self, n_points, camera_mat, world_mat):
        ''' Transforms origin (camera location) to world coordinates.

        Args:
            n_points (int): how often the transformed origin is repeated in the
                form (batch_size, n_points, 3)
            camera_mat (tensor): camera matrix
            world_mat (tensor): world matrix
            scale_mat (tensor): scale matrix
            invert (bool): whether to invert the matrices (default: False)
        '''
        batch_size = camera_mat.shape[0]
        # Create origin in homogen coordinates
        p = torch.zeros(batch_size, 4, n_points).to(self.device)
        p[:, -1] = 1.
        # Apply transformation
        p_world = world_mat @ camera_mat @ p
        # Transform points back to 3D coordinates
        p_world = p_world[:, :3].permute(0, 2, 1)
        return p_world

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

    def forward(self, world_mat, mode="training"):
        batch_size = world_mat.shape[0]
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        rgb_v = self.volume_render_image(camera_mat, world_mat, batch_size, mode)
        if self.neural_renderer is not None:
            rgb = self.neural_renderer(rgb_v)
        else:
            rgb = rgb_v
        return rgb

    def cast_ray_and_get_eval_points(self, img_res, batch_size, depth_range, n_steps, camera_mat, world_mat,
                                     ray_jittering):
        # Arrange Pixels
        pixels = self.arrange_pixels((img_res, img_res), batch_size)[1]
        pixels[..., -1] *= -1.
        n_points = img_res * img_res
        # Project to 3D world
        pixel_pos_wc = self.image_points_to_world(pixels, camera_mat, world_mat)
        camera_pos_wc = self.origin_to_world(n_points, camera_mat, world_mat)
        # batch_size x n_points x n_steps
        step_depths = depth_range[0] + \
                      torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (
                              depth_range[1] - depth_range[0])
        step_depths = step_depths.repeat(batch_size, n_points, 1).to(self.device)
        if ray_jittering:
            step_depths = self.add_noise_to_interval(step_depths)
        point_pos_wc = self.get_evaluation_points(pixel_pos_wc, camera_pos_wc,
                                                  step_depths)  # shape (batch, num of eval points, 3)
        return point_pos_wc

    def volume_render_image(self, camera_mat, world_mat, batch_size, mode):
        img_res = self.feature_img_resolution
        n_steps = self.n_ray_samples
        point_pos_wc = self.cast_ray_and_get_eval_points(img_res, batch_size, self.depth_range, n_steps,
                                                         camera_mat, world_mat, mode == "training")
        logging.debug(f"point pos wc shape = {point_pos_wc.shape}")
        density_scalars = self.volume_sampler(
            point_pos_wc.unsqueeze(1))  # shape (batch, channel=1, 1, num of eval points)
        density_scalars = density_scalars.view(batch_size, -1)
        if mode == 'training':
            # As done in NeRF, add noise during training
            density_scalars += torch.randn_like(density_scalars)

        feature_with_alpha = self.tf(density_scalars).permute(0, 2, 1)  # shape(batch, num of eval points, channel)
        # Reshape
        n_points = img_res * img_res
        feature_with_alpha = feature_with_alpha.reshape(batch_size,
                                                        n_points,
                                                        n_steps,
                                                        -1  # channels
                                                        )
        features = feature_with_alpha[:, :, :, :-1]
        alphas = feature_with_alpha[:, :, :, -1]
        # DVR composition
        weights = self.calc_volume_weights(alphas)
        feat_map = torch.sum(weights.unsqueeze(-1) * features, dim=-2)
        # Reformat output
        feat_map = feat_map.permute(0, 2, 1).reshape(batch_size, -1, img_res, img_res)  # B x feat x h x w
        feat_map = feat_map.permute(0, 1, 3, 2)  # new to flip x/y
        return feat_map

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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    # load volume data
    head_data = load_head_data().astype(np.float32).transpose([2, 1, 0])
    uint16_max = float(np.iinfo(np.uint16).max)
    normalized_head_data = head_data / uint16_max
    head_tensor = torch.from_numpy(normalized_head_data).unsqueeze(0)
    # load TF data
    tf = torch.from_numpy(load_transfer_function()).float()
    # setup random camera
    range_radius = (2.732, 2.732)
    range_u = (0., 0.5)
    range_v = (0., 0.5)
    random_camera_poses = RandomCameraPoses(100, range_u, range_v, range_radius)
    dvr = DirectVolumeRenderer(head_tensor, tf,
                               feature_img_resolution=256,
                               fov=49.13,
                               depth_range=[0.5, 6.],
                               n_ray_samples=600).eval().cuda()
    with torch.no_grad():
        img = dvr(random_camera_poses[0].unsqueeze(0).cuda(), mode="testing")
    to_pil_img = ToPILImage()
    for i in range(img.shape[0]):
        pil_img = to_pil_img(img[i])
        pil_img.save(f"test_{i}.png")
