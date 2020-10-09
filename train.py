import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from voxel_generator import VoxelGenerator
from projection_unit import ProjectionUnit
from image_decoder import ImageDecoder
from dataset import VoxelDataset
from rotation_method import rotation_resampling, match_voxel_to_image

"""
# compositor
train.py
dataset.py
image_decoder.py
voxel_generator.py
projection_unit.py
rotation_method.py
layer_uti;s.py
"""

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_number', type=int, default=0)

parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--voxel_size', type=int, default=64)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--shape_loss_weight', type=float, default=1.0)
parser.add_argument('--color_loss_weight', type=float, default=1.0)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
args = parser.parse_args()


device = 'cuda:{}'.format(args.gpu_number) if torch.cuda.is_available() else 'cpu'

# build required module
voxel_gene = VoxelGenerator(in_channels=3, out_channels=4, image_size=args.image_size, voxel_size=args.voxel_size)
proj_uni = ProjectionUnit(in_channels=3, out_channels=32 * 16)
img_dec = ImageDecoder(in_channels=32 * 16, out_channels=3)


dataset = VoxelDataset(args)
dataset.create_dummy_data()  # for test
train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

params = list(voxel_gene.parameters()) + list(proj_uni.parameters()) + list(img_dec.parameters())
optimizer = Adam(params=params, lr=args.lr, betas=(args.beta1, args.beta2))

for e in range(args.epoch):
    losses = []
    for n, data in enumerate(train_dataloader):
        real_image, real_voxel_model, real_params = data

        # cropping data
        rotated_model = rotation_resampling(real_voxel_model, real_params, size=args.voxel_size, new_size=args.voxel_size)
        rotated_model = match_voxel_to_image(rotated_model)

        optimizer.zero_grad()
        # real_image = torch.randn(batch_size, 3, image_size, image_size)

        rgb_output, occupancy_output = voxel_gene(real_image)

        # projection unit
        # args => view_params(azimuth, elevation, scale)
        # transform(1, 64, 64, 64) => (32, 16, 16, 16)
        # depth collapse
        proj_output = proj_uni(rgb_output)

        # image decoder
        img_dec_output = img_dec(proj_output)

        # loss function
        # shape loss: bce_loss(occupancy_output, real_voxel_model)
        # color loss: mse_loss(image_decoder_output, real_input_image)
        shape_loss_criterion = nn.BCELoss()
        color_loss_criterion = nn.MSELoss()
        # print(occupancy_output.shape, img_dec_output.shape)
        shape_loss = shape_loss_criterion(occupancy_output, rotated_model)
        color_loss = color_loss_criterion(img_dec_output, real_image)

        loss = args.shape_loss_weight * shape_loss + args.color_loss_weight * color_loss

        loss.backward()

        optimizer.step()
        print('{} loss:{:.5f}'.format(n, loss.item()))
        losses.append(loss.item())

    mean_loss = np.mean(losses)

    print('epoch: {}  loss: {:.5f}'.format(e, mean_loss))








