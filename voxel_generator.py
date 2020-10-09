import torch.nn as nn
from layer_utils import ResBlock2d

class VoxelGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, image_size=512, voxel_size=64, input_tensor_size=4):
        super().__init__()

        # input image => (batch_size, 3, 512, 512)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2, padding=1)  # (batch_size, 32, 256, 256)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)  # (batch_size, 64, 128, 128)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # (batch_size, 128, 64, 64)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        self.resblock1 = ResBlock2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # (batch_size, 128, 64, 64, 64)
        self.resblock2 = ResBlock2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # (batch_size, 128, 64, 64, 64)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)  # (batch_size, 256, 32, 32)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)  # (batch_size, 512, 16, 16)
        self.bn5 = nn.BatchNorm2d(num_features=512)

        self.resblock3 = ResBlock2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # (batch_size, 512, 16, 16)
        self.resblock4 = ResBlock2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # (batch_size, 512, 16, 16)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)  # (batch_size, 1024, 16, 16)

        self.fc1 = nn.Linear(in_features=1024 * (image_size // 2 ** 6) ** 2, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)

        # tensor reshape
        # (batch_size, 64, 4, 4, 4)
        self.input_tensor_size = input_tensor_size

        self.conv3d_1 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)  # (batch_size, 32, 8, 8, 8)
        self.conv3d_2 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)  # (batch_size, 16, 16, 16, 16)
        self.conv3d_3 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)  # (batch_size, 8, 32, 32, 32)
        self.conv3d_4 = nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1)  # (batch_size, 4, 64, 64, 64)

        self.conv3d_rgb = nn.Conv3d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1)  # (batch_size, 3, 64, 64, 64)
        self.conv3d_occ = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)  # (batch_size, 1, 64, 64, 64)

    def forward(self, x):
        batch_size = x.shape[0]

        # convolution part1
        x = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))  # (batch_size, 32, 256, 256)
        x = nn.ReLU(inplace=True)(self.bn2(self.conv2(x)))  # (batch_size, 64, 128, 128)
        x = nn.ReLU(inplace=True)(self.bn3(self.conv3(x)))  # (batch_size, 128, 64, 64)

        conv_part1_output = x

        # resblock part1
        x = self.resblock1(x)  # (batch_size, 128, 64, 64)
        x = self.resblock2(x)  # (batch_size, 128, 64, 64)

        x = x + conv_part1_output  # skip connection

        # convolution part2
        x = nn.ReLU(inplace=True)(self.bn4(self.conv4(x)))  # (batch_size, 256, 32, 32)
        x = nn.ReLU(inplace=True)(self.bn5(self.conv5(x)))  # (batch_size, 512, 16, 16)

        conv_part2_output = x

        # resblock part2
        x = self.resblock3(x)  # (batch_size, 512, 16, 16)
        x = self.resblock4(x)  # (batch_size, 512, 16, 16)

        x = x + conv_part2_output  # skip connection

        x = self.conv6(x)  # (batch_size, 1024, 8, 8)
        x = x.reshape(batch_size, -1)  # (batch_size, 1024 * 8 * 8

        # fully connected layer
        x = nn.ReLU(inplace=True)(self.fc1(x))
        x = self.fc2(x)  # (batch_size, 4096)

        # reshape
        linear_channels = x.shape[-1]
        channels = int(linear_channels // (self.input_tensor_size) ** 3)
        x = x.reshape(batch_size, channels, self.input_tensor_size, self.input_tensor_size, self.input_tensor_size)

        # up convolution 3d part
        x = nn.PReLU()(self.conv3d_1(x))  # (batch_size, 32, 8, 8)
        x = nn.PReLU()(self.conv3d_2(x))  # (batch_size, 16, 16, 16)
        x = nn.PReLU()(self.conv3d_3(x))  # (batch_size, 8, 32, 32)
        x = nn.PReLU()(self.conv3d_4(x))  # (batch_size, 4, 64, 64)
        conv3d_output = x

        # create rgb output
        rgb_output = self.conv3d_rgb(conv3d_output)  # (batch_size, 3, 64, 64)
        # create occupancy output
        occupancy_output = nn.Sigmoid()(self.conv3d_occ(x))  # (batch_size, 1, 64, 64)

        return rgb_output, occupancy_output

"""
if __name__ == '__main__':
    import torch
    import numpy as np
    batch_size = 2
    image_size = 512
    n_channels = 3
    voxel_size = 64

    input_tensor = torch.from_numpy(np.random.randn(batch_size, n_channels, image_size, image_size)).type(torch.float32)

    voxel_generator = VoxelGenerator(in_channels=n_channels, out_channels=1, image_size=image_size, voxel_size=voxel_size)

    voxel_rgb_output, voxel_occupancy_output = voxel_generator(input_tensor)
    print(voxel_rgb_output.shape, voxel_occupancy_output.shape)
"""



