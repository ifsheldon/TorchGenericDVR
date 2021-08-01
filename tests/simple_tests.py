import torch
from wrappers import samplers

POSITION_DIMS = 3


def test_shape_match():
    batch = 10
    feature_channel_num = 2
    volume_dims = (feature_channel_num, 3, 4, 5)
    data_points = torch.randn(*volume_dims)
    trilinear_sampler = samplers.TrilinearVolumeSampler(data_points, volume_require_grad=False)
    img_width = 6
    img_height = 7
    sample_point_pos = torch.rand(batch, img_width, img_height, POSITION_DIMS)
    interpolated_values = trilinear_sampler(sample_point_pos)
    assert interpolated_values.shape == (batch, feature_channel_num, img_width, img_height)


def test_vertex_values_match():
    batch = 1
    feature_channel_num = 1
    data_points = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.]).reshape(feature_channel_num, 2, 2, 2)
    img_width = 2
    img_height = 4
    sample_point_pos = torch.tensor([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [1., 1., 0.],
        [0., 0., 1.],
        [1., 0., 1.],
        [0., 1., 1.],
        [1., 1., 1.],
    ]).reshape(batch, img_width, img_height, POSITION_DIMS) * 2.0 - 1.0
    trilinear_sampler = samplers.TrilinearVolumeSampler(data_points, volume_require_grad=False)
    output = trilinear_sampler(sample_point_pos).reshape(-1)
    assert (output == data_points.reshape(-1)).all().item()


if __name__ == "__main__":
    test_shape_match()
    test_vertex_values_match()
