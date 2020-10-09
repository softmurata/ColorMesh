# create color mesh automatic generation model
"""
input: 2d single rgb image + camera pose (+ light position)
voxel transformation(TLNet(2d single rgb image))

3d voxel = generator(input)
# generator
convolution 3d + fc layer

3d voxel: (rgb value + occupancy(0 or 1)), shape=(4, voxel_size, voxel_size, voxel_size)

# projection unit
depth collapse: (3, depth, height, width) => (3 * depth, height, width)
transform_voxel_to_match_image() -> 2d image input (out_channels, height, width), camera pose

# upconvolution
2d image = upconvolution(2d image input)  shape=(3, height, width)
normal map + lighting

# loss function
color loss => (2d image, target 2d image)
shape loss => (3d voxel, target 3d voxel)


# dataset configuration
3d voxel
nview * (camera pose, 2d image)

"""

import argparse
import torch
from networks import Generator3D
from rotation_method import rotation_resampling, match_voxel_to_image
from preprocess import crop_images_and_voxels

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_number', type=int, default=0)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--real_voxel_size', type=int, default=64)
args = parser.parse_args()


device = 'cuda:{}'.format(args.gpu_number) if torch.cuda.is_available() else 'cpu'

# model_in(b, 1, 64, 64, 64), params_in(b, 3), real_image(b, 3, 512, 512)
# convert into camera coordinates  => rotation_resampling()(xyz => ijk) + match_voxel_to_image()(reshape method)
# pytorch meshgrid => default(indexing='ij')
new_res = 128  # new voxel size?

# create ml model and optimizer


for e in range(args.epoch):
    print('training start')
    if e < 5:
        batch_patch_size = new_res // 4
    else:
        batch_patch_size = new_res // 2

    for n, data in enumerate(train_dataloader):
        real_images, real_models, real_params = data
        # data shape
        # real images => (batch_size, 3, 512, 512)
        # real models => (batch_size, 1, 64, 64, 64)
        # real params => (batch_size, 3)
        rotated_models = rotation_resampling(voxel_array=real_models, view_params=real_params,
                                             size=args.real_voxel_size, new_size=new_res)
        rotated_models = match_voxel_to_image(rotated_models)

        # cropping images and voxels
        crop_voxels, crop_images = crop_images_and_voxels(voxels=rotated_models, images=real_images,
                                                          patch_size=batch_patch_size)
        # rendernet inference



