import numpy as np
import torch.nn as nn

def crop_images_and_voxels(voxels, images, patch_size):
    # voxels => (batch_size, n_channels, depth, height, width)
    # images => (batch_size, m_channels, image_size, image_size)
    batch_size = voxels.shape[0]
    voxel_dim = voxels.shape[2]  # voxel_size = 128
    image_dim = images.shape[2]  # image_size = 512
    image_voxel_factor = image_dim // voxel_dim

    if patch_size == voxel_dim:

        return nn.Identity()(voxels), nn.Identity()(images)

    voxel_start_point = np.random.uniform(low=0, high=voxel_dim - patch_size + 1, size=[2])
    image_start_point = voxel_start_point * image_voxel_factor

    voxel_patch = voxels[:, :, :,
                  voxel_start_point[0]:voxel_start_point[0] + patch_size, voxel_start_point[1]:voxel_start_point[1] + patch_size]
    # voxel_patch size => (batch_size, n_channels, depth, patch_size, patch_size)
    image_patch = images[:, :, image_start_point[0]:image_start_point[0] + patch_size * image_voxel_factor,
                  image_start_point[1]:image_start_point[1] + patch_size * image_voxel_factor]
    # image patch size => (batch_size, m_channels, patch_size * image_voxel_factor, patch_size * image_voxel_factor)

    return nn.Identity()(voxel_patch), nn.Identity()(image_patch)
