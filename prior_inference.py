# inference
# input => multiview input images(3, n_view, height, width)
# output => voxel(1, voxel_size, voxel_size, voxel_size)
# output = decoder(image_extractor(input))

import argparse
import torch
import numpy as np
from prior_network import LNetwork
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--weight_path', type=str, default='./models/joint/checkpoint_00200.pth.tar')
parser.add_argument('--dataset_dir', type=str, defaultr='./datasets/')
parser.add_argument('--model_number', type=int, default=0)
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()


lnet = LNetwork(in_channels=3, hidden_channels=64, out_channels=1)

single_image_path = args.dataset_dir + args.model_number + '/' + '0.png'  # 1 of multiview images
single_image = Image.open(single_image_path).convert('RGB')
single_image = torch.from_numpy(np.asarray(single_image))  # (3, height, width) tensor
# ToDo: need reshape?(1, 3, height, width)

generated_voxel = lnet(single_image)  # (batch_size, voxel_size, voxel_size, voxel_size)


