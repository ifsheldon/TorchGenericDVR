import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import math


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


def load_transfer_function():
    return np.array([0.0, 0.0, 0.0, 0.0,
                     0.0, 0.5, 0.5, 0.0,
                     0.0, 0.5, 0.5, 0.01,
                     0.0, 0.5, 0.5, 0.0,
                     0.5, 0.5, 0.0, 0.0,
                     0.5, 0.5, 0.0, 0.2,
                     0.5, 0.5, 0.0, 0.5,
                     0.5, 0.5, 0.0, 0.2,
                     0.5, 0.5, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0,
                     1.0, 0.0, 1.0, 0.0,
                     1.0, 0.0, 1.0, 0.8]).reshape(12, 4)


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        return data


def dump_data(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_head_data():
    return load_data("./skewed_head.pickle")
