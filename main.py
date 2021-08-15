import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.transforms import ToPILImage
from torch_generic_dvr.renderer import DirectVolumeRenderer
from torch_generic_dvr.samplers import TransferFunctionModel1D
from torch_generic_dvr.utils import load_head_data, load_transfer_function, RandomCameraPoses

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    # load volume data
    head_data = load_head_data().astype(np.float32).transpose([2, 1, 0])
    uint16_max = float(np.iinfo(np.uint16).max)
    normalized_head_data = head_data / uint16_max
    head_tensor = torch.from_numpy(normalized_head_data).unsqueeze(0)
    # setup random camera
    dataset_size = 5
    range_radius = (3, 3)
    range_u = (0., 0.5)
    range_v = (0., 0.5)
    random_camera_poses = RandomCameraPoses(dataset_size, range_u, range_v, range_radius)
    # setup loader
    batch_size = 1
    data_loader = DataLoader(random_camera_poses, batch_size=batch_size)
    # create transfer functions
    tf = torch.from_numpy(load_transfer_function()).float()
    feature_tf = TransferFunctionModel1D(tf[:, :3])
    alpha_tf = TransferFunctionModel1D(tf[:, 3:])
    # setup DVR
    dvr = DirectVolumeRenderer(head_tensor,
                               feature_tf, alpha_tf,
                               feature_img_resolution=256,
                               fov=45.0,
                               depth_range=[0., 4.],
                               n_ray_samples=200)
    # setup trainer
    trainer = pl.Trainer(gpus=1, logger=False)
    img = trainer.predict(dvr, data_loader, return_predictions=True)
    img = torch.cat(img, dim=0)
    to_pil_img = ToPILImage()
    for i in range(img.shape[0]):
        pil_img = to_pil_img(img[i])
        pil_img.save(f"test_{i}.png")
