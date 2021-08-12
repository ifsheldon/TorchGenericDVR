import torch
from torch import nn
import logging
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from samplers import TrilinearVolumeSampler, TransferFunctionModel1D
from torchvision.transforms import ToPILImage
from utils import *


def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return np.stack([cx, cy, cz], axis=-1)


def calc_volume_weights(alpha):
    # alpha: shape (batch, res*res, step_num)
    weights = alpha * \
              torch.cumprod(torch.cat([
                  torch.ones_like(alpha[:, :, :1]),
                  (1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
    return weights


def get_evaluation_points(pixels_world, camera_world, di):
    batch_size = pixels_world.shape[0]
    ray = pixels_world - camera_world
    p = camera_world.unsqueeze(-2).contiguous() + \
        di.unsqueeze(-1).contiguous() * ray.unsqueeze(-2).contiguous()
    logging.debug(f"p_i and ray_i shape = {p.shape}")
    p = p.reshape(batch_size, -1, 3)
    return p


def add_noise_to_interval(di):
    di_mid = .5 * (di[..., 1:] + di[..., :-1])
    di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
    di_low = torch.cat([di[..., :1], di_mid], dim=-1)
    noise = torch.rand_like(di_low)
    ti = di_low + (di_high - di_low) * noise
    return ti


class Generator(pl.LightningModule):
    ''' GIRAFFE Generator Class.

    Args:
        range_u (tuple): rotation range (0 - 1)
        range_v (tuple): elevation range (0 - 1)
        n_ray_samples (int): number of samples per ray
        range_radius(tuple): radius range
        depth_range (tuple): near and far depth plane
        feature_img_resolution (int): resolution of volume-rendered image
        neural_renderer (nn.Module): neural renderer
        fov (float): field of view
    '''

    def __init__(self,
                 volume,
                 transfer_function_data,
                 range_u=(0, 0), range_v=(0.25, 0.25), n_ray_samples=64,
                 range_radius=(2.732, 2.732), depth_range=[0.5, 6.],
                 feature_img_resolution=16,
                 neural_renderer=None,
                 fov=49.13,
                 ):
        super().__init__()
        self.n_ray_samples = n_ray_samples
        self.range_u = range_u
        self.range_v = range_v
        self.feature_img_resolution = feature_img_resolution
        self.range_radius = range_radius
        self.depth_range = depth_range
        self.camera_matrix = nn.Parameter(self.get_camera_mat(fov))
        self.volume_sampler = TrilinearVolumeSampler(volume)
        self.tf = TransferFunctionModel1D(transfer_function_data)
        self.neural_renderer = None if neural_renderer is None else neural_renderer

    def sample_on_sphere(self, range_u=(0, 1), range_v=(0, 1), batch_size=(1,)):
        u = np.random.uniform(*range_u, size=batch_size)
        v = np.random.uniform(*range_v, size=batch_size)

        sample = to_sphere(u, v)
        return torch.tensor(sample).float().to(self.device)

    def get_camera_mat(self, fov, invert=True):
        # fov = 2 * arctan( sensor / (2 * focal))
        # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))
        # in our case, sensor = 2 as pixels are in [-1, 1]
        focal = 1. / np.tan(0.5 * fov * np.pi / 180.)
        focal = focal.astype(np.float32)
        mat = torch.tensor([
            [focal, 0., 0., 0.],
            [0., focal, 0., 0.],
            [0., 0., 1, 0.],
            [0., 0., 0., 1.]
        ]).reshape(1, 4, 4)

        if invert:
            mat = torch.inverse(mat)
        return mat.to(self.device)

    def transform_to_world(self, pixels, depth, camera_mat, world_mat, scale_mat=None,
                           invert=True, use_absolute_depth=True):
        ''' Transforms pixel positions p with given depth value d to world coordinates.

        Args:
            pixels (tensor): pixel tensor of size B x N x 2
            depth (tensor): depth tensor of size B x N x 1
            camera_mat (tensor): camera matrix
            world_mat (tensor): world matrix
            scale_mat (tensor): scale matrix
            invert (bool): whether to invert matrices (default: true)
        '''
        assert (pixels.shape[-1] == 2)

        if scale_mat is None:
            scale_mat = torch.eye(4).unsqueeze(0).repeat(
                camera_mat.shape[0], 1, 1).to(self.device)

        # Invert camera matrices
        if invert:
            camera_mat = torch.inverse(camera_mat)
            world_mat = torch.inverse(world_mat)
            scale_mat = torch.inverse(scale_mat)

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
        p_world = scale_mat @ world_mat @ camera_mat @ pixels

        # Transform p_world back to 3D coordinates
        p_world = p_world[:, :3].permute(0, 2, 1)
        return p_world

    def arrange_pixels(self, resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                       subsample_to=None, invert_y_axis=False):
        ''' Arranges pixels for given resolution in range image_range.

        The function returns the unscaled pixel locations as integers and the
        scaled float values.

        Args:
            resolution (tuple): image resolution
            batch_size (int): batch size
            image_range (tuple): range of output points (default [-1, 1])
            subsample_to (int): if integer and > 0, the points are randomly
                subsampled to this value
        '''
        h, w = resolution
        n_points = resolution[0] * resolution[1]

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

        # Subsample points if subsample_to is not None and > 0
        if (subsample_to is not None and subsample_to > 0 and
                subsample_to < n_points):
            idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                                   replace=False)
            pixel_scaled = pixel_scaled[:, idx]
            pixel_locations = pixel_locations[:, idx]

        if invert_y_axis:
            assert (image_range == (-1, 1))
            pixel_scaled[..., -1] *= -1.
            pixel_locations[..., -1] = (h - 1) - pixel_locations[..., -1]

        return pixel_locations, pixel_scaled

    def origin_to_world(self, n_points, camera_mat, world_mat, scale_mat=None,
                        invert=False):
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
        device = camera_mat.device
        # Create origin in homogen coordinates
        p = torch.zeros(batch_size, 4, n_points).to(device)
        p[:, -1] = 1.

        if scale_mat is None:
            scale_mat = torch.eye(4).unsqueeze(
                0).repeat(batch_size, 1, 1).to(device)

        # Invert matrices
        if invert:
            camera_mat = torch.inverse(camera_mat)
            world_mat = torch.inverse(world_mat)
            scale_mat = torch.inverse(scale_mat)

        # Apply transformation
        p_world = scale_mat @ world_mat @ camera_mat @ p

        # Transform points back to 3D coordinates
        p_world = p_world[:, :3].permute(0, 2, 1)
        return p_world

    def look_at(self, eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
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

        return torch.tensor(r_mat).float().to(self.device)

    def get_random_pose(self, range_u, range_v, range_radius, batch_size=32,
                        invert=False):
        location = self.sample_on_sphere(range_u, range_v, batch_size=batch_size)
        radius = range_radius[0] + \
                 torch.rand(batch_size) * (range_radius[1] - range_radius[0])
        location = location * radius.unsqueeze(-1).to(self.device)
        R = self.look_at(location.cpu().numpy())
        RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
        RT[:, :3, :3] = R
        RT[:, :3, -1] = location

        if invert:
            RT = torch.inverse(RT)
        return RT.to(self.device)

    def image_points_to_world(self, image_points, camera_mat, world_mat, scale_mat=None,
                              invert=False, negative_depth=True):
        ''' Transforms points on image plane to world coordinates.

        In contrast to transform_to_world, no depth value is needed as points on
        the image plane have a fixed depth of 1.

        Args:
            image_points (tensor): image points tensor of size B x N x 2
            camera_mat (tensor): camera matrix
            world_mat (tensor): world matrix
            scale_mat (tensor): scale matrix
            invert (bool): whether to invert matrices (default: False)
        '''
        batch_size, n_pts, dim = image_points.shape
        assert (dim == 2)
        depth_image = torch.ones(batch_size, n_pts, 1).to(self.device)
        if negative_depth:
            depth_image *= -1.
        return self.transform_to_world(image_points, depth_image, camera_mat, world_mat,
                                       scale_mat, invert=invert)

    def forward(self, batch_size=32, mode="training"):
        camera_matrices = self.sample_random_camera(batch_size)
        logging.debug(f"\ncamera mat shape = {camera_matrices[0].shape}"
                      f"\nworld mat shape = {camera_matrices[1].shape}")
        rgb_v = self.volume_render_image(camera_matrices, batch_size, mode=mode)

        if self.neural_renderer is not None:
            rgb = self.neural_renderer(rgb_v)
        else:
            rgb = rgb_v
        return rgb

    def sample_random_camera(self, batch_size=32):
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat = self.get_random_pose(self.range_u, self.range_v, self.range_radius, batch_size)
        return camera_mat, world_mat

    def volume_render_image(self, camera_matrices, batch_size, mode='training'):
        img_res = self.feature_img_resolution
        n_steps = self.n_ray_samples
        depth_range = self.depth_range
        n_points = img_res * img_res
        camera_mat, world_mat = camera_matrices

        # Arrange Pixels
        pixels = self.arrange_pixels((img_res, img_res), batch_size, invert_y_axis=False)[1]
        pixels[..., -1] *= -1.
        # Project to 3D world
        pixel_pos_wc = self.image_points_to_world(
            pixels, camera_mat,
            world_mat)
        camera_pos_wc = self.origin_to_world(
            n_points, camera_mat,
            world_mat)

        logging.debug(f"pixel_pos_wc shape = {pixel_pos_wc.shape}")
        logging.debug(f"camera_pos_wc shape = {camera_pos_wc.shape}")
        logging.debug(f"camera_pos_wc = {camera_pos_wc}")

        # batch_size x n_points x n_steps
        step_depths = depth_range[0] + \
                      torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (
                              depth_range[1] - depth_range[0])
        step_depths = step_depths.repeat(batch_size, n_points, 1).to(self.device)
        logging.debug(f"di shape = {step_depths.shape}")
        if mode == 'training':
            step_depths = add_noise_to_interval(step_depths)

        point_pos_wc = get_evaluation_points(pixel_pos_wc, camera_pos_wc,
                                             step_depths)  # shape (batch, num of eval points, 3)
        logging.debug(f"point pos wc shape = {point_pos_wc.shape}")
        density_scalars = self.volume_sampler(
            point_pos_wc.unsqueeze(1))  # shape (batch, channel=1, 1, num of eval points)
        density_scalars = density_scalars.view(batch_size, -1)
        feature_with_alpha = self.tf(density_scalars).permute(0, 2, 1)  # shape(batch, num of eval points, channel)
        if mode == 'training':
            # As done in NeRF, add noise during training
            density_scalars += torch.randn_like(density_scalars)

        # Reshape
        feature_with_alpha = feature_with_alpha.reshape(batch_size,
                                                        n_points,
                                                        n_steps,
                                                        -1  # channels
                                                        )
        features = feature_with_alpha[:, :, :, :-1]
        alphas = feature_with_alpha[:, :, :, -1]
        # DVR composition
        weights = calc_volume_weights(alphas)
        feat_map = torch.sum(weights.unsqueeze(-1) * features, dim=-2)
        # Reformat output
        feat_map = feat_map.permute(0, 2, 1).reshape(batch_size, -1, img_res, img_res)  # B x feat x h x w
        feat_map = feat_map.permute(0, 1, 3, 2)  # new to flip x/y
        return feat_map


# FIXME: fix all .to() in this file and in pl_modules.py and fxxking mixing np and torch in this file
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    head_data = load_head_data().astype(np.float32).transpose([2, 1, 0])
    uint16_max = float(np.iinfo(np.uint16).max)
    normalized_head_data = head_data / uint16_max
    head_tensor = torch.from_numpy(normalized_head_data).unsqueeze(0)
    tf = torch.from_numpy(load_transfer_function()).float()
    gen = Generator(head_tensor, tf,
                    feature_img_resolution=256,
                    range_u=(0., 0.5),
                    range_v=(0., 0.5),
                    n_ray_samples=600).eval().cuda()
    with torch.no_grad():
        img = gen(batch_size=1, mode="testing")
    to_pil_img = ToPILImage()
    for i in range(img.shape[0]):
        pil_img = to_pil_img(img[i])
        pil_img.save(f"test_{i}.png")
