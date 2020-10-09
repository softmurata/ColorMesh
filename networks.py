import torch
import torch.nn as nn
from layer_utils import transform_3d_to_2d_tensor

class ResBlock3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # convolution 3d
        self.conv3d_1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3d_2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x_input = x
        x = nn.PReLU()(self.conv3d_1(x))
        x = self.conv3d_2(x)

        x = x + x_input  # skip connection

        return x


class ResBlock2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2d_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):

        x_input = x
        x = nn.PReLU()(self.conv2d_1(x))
        x = self.conv2d_2(x)

        # skip connection
        x = x + x_input

        return x


class Generator3D(nn.Module):

    def __init__(self, in_channels, out_channels, voxel_size):
        super().__init__()
        # out_channels => 3
        # (down 3d convolution + parametric relu) * 3
        # resblock3d * 10

        # down convolution part
        self.downconv3d_1 = nn.Conv3d(in_channels=in_channels, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.downconv3d_2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.downconv3d_3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # resblock part(3d)
        self.resblock3d_1 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.resblock3d_2 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.resblock3d_3 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.resblock3d_4 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.resblock3d_5 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.resblock3d_6 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.resblock3d_7 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.resblock3d_8 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.resblock3d_9 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.resblock3d_10 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        proj_size = voxel_size // 4  # 32
        # projection unit(reshape + depth collapsion learning)
        self.projection_unit = transform_3d_to_2d_tensor  # sampling (batch_size, channels * depth, height, width)
        self.depth_collapsion = nn.Conv2d(in_channels=32 * proj_size, out_channels=32 * proj_size, kernel_size=1,
                                          stride=1, padding=0)

        # resblock part 2d
        self.resblock2d_1 = ResBlock2d(in_channels=32 * proj_size, out_channels=32 * proj_size, kernel_size=3, stride=1,
                                       padding=1)
        self.resblock2d_2 = ResBlock2d(in_channels=32 * proj_size, out_channels=32 * proj_size, kernel_size=3, stride=1,
                                       padding=1)
        self.resblock2d_3 = ResBlock2d(in_channels=32 * proj_size, out_channels=32 * proj_size, kernel_size=3, stride=1,
                                       padding=1)
        self.resblock2d_4 = ResBlock2d(in_channels=32 * proj_size, out_channels=32 * proj_size, kernel_size=3, stride=1,
                                       padding=1)
        self.resblock2d_5 = ResBlock2d(in_channels=32 * proj_size, out_channels=32 * proj_size, kernel_size=3, stride=1,
                                       padding=1)
        self.resblock2d_6 = ResBlock2d(in_channels=32 * proj_size, out_channels=32 * proj_size, kernel_size=3, stride=1,
                                       padding=1)
        self.resblock2d_7 = ResBlock2d(in_channels=32 * proj_size, out_channels=32 * proj_size, kernel_size=3, stride=1,
                                       padding=1)
        self.resblock2d_8 = ResBlock2d(in_channels=32 * proj_size, out_channels=32 * proj_size, kernel_size=3, stride=1,
                                       padding=1)
        self.resblock2d_9 = ResBlock2d(in_channels=32 * proj_size, out_channels=32 * proj_size, kernel_size=3, stride=1,
                                       padding=1)
        self.resblock2d_10 = ResBlock2d(in_channels=32 * proj_size, out_channels=32 * proj_size, kernel_size=3, stride=1,
                                        padding=1)

        self.conv2d_1 = nn.Conv2d(in_channels=32 * proj_size, out_channels=16 * proj_size, kernel_size=3, stride=1, padding=1)

        # resblock part(5 resblock)
        self.resblock2d_11 = ResBlock2d(in_channels=16 * proj_size, out_channels=16 * proj_size, kernel_size=3, stride=1, padding=1)
        self.resblock2d_12 = ResBlock2d(in_channels=16 * proj_size, out_channels=16 * proj_size, kernel_size=3, stride=1, padding=1)
        self.resblock2d_13 = ResBlock2d(in_channels=16 * proj_size, out_channels=16 * proj_size, kernel_size=3, stride=1, padding=1)
        self.resblock2d_14 = ResBlock2d(in_channels=16 * proj_size, out_channels=16 * proj_size, kernel_size=3, stride=1, padding=1)
        self.resblock2d_15 = ResBlock2d(in_channels=16 * proj_size, out_channels=16 * proj_size, kernel_size=3, stride=1, padding=1)

        self.conv2d_2 = nn.Conv2d(in_channels=16 * proj_size, out_channels=8 * proj_size, kernel_size=3, stride=1, padding=1)

        # up convolution part
        self.upconv2d_1 = nn.ConvTranspose2d(in_channels=8 * proj_size, out_channels=4 * proj_size, kernel_size=4,
                                             stride=2, padding=1)
        self.upconv2d_2 = nn.ConvTranspose2d(in_channels=4 * proj_size, out_channels=2 * proj_size, kernel_size=4,
                                             stride=2, padding=1)
        self.upconv2d_3 = nn.ConvTranspose2d(in_channels=2 * proj_size, out_channels=proj_size, kernel_size=4,
                                             stride=2, padding=1)

        self.upconv2d_4 = nn.ConvTranspose2d(in_channels=proj_size, out_channels=proj_size // 2, kernel_size=4,
                                             stride=2, padding=1)
        self.upconv2d_5 = nn.ConvTranspose2d(in_channels=proj_size // 2, out_channels=out_channels, kernel_size=3,
                                             stride=1, padding=1)

    def forward(self, x):
        ##### 3d #####
        print('forward part in')
        # input shape => (batch_size, 1, 128, 128, 128)
        # down convolution 3d part
        x = nn.PReLU()(self.downconv3d_1(x))  # (batch_size, 8, 64, 64, 64)
        x = nn.PReLU()(self.downconv3d_2(x))  # (batch_size, 16, 32, 32, 32)
        x = nn.PReLU()(self.downconv3d_3(x))  # (batch_size, 32, 32, 32, 32)

        downconv3d_output = x

        # resblock 3d part
        x = self.resblock3d_1(x)  # (batch_size, 32, 32, 32, 32)
        x = self.resblock3d_2(x)  # (batch_size, 32, 32, 32, 32)
        x = self.resblock3d_3(x)  # (batch_size, 32, 32, 32, 32)
        x = self.resblock3d_4(x)  # (batch_size, 32, 32, 32, 32)
        x = self.resblock3d_5(x)  # (batch_size, 32, 32, 32, 32)
        x = self.resblock3d_6(x)  # (batch_size, 32, 32, 32, 32)
        x = self.resblock3d_7(x)  # (batch_size, 32, 32, 32, 32)
        x = self.resblock3d_8(x)  # (batch_size, 32, 32, 32, 32)
        x = self.resblock3d_9(x)  # (batch_size, 32, 32, 32, 32)
        x = self.resblock3d_10(x)  # (batch_size, 32, 32, 32, 32)

        x = x + downconv3d_output  # (batch_size, 32, 32, 32, 32)

        print('resblock 3d output shape:', x.shape)
        shape_output = x

        ####### 2d ########
        # projection unit
        proj_unit_output = self.projection_unit(x)
        proj_unit_output = self.depth_collapsion(proj_unit_output)  # (batch_size, 32 * 32, 32, 32)
        x = proj_unit_output

        # resblock2d part1
        x = self.resblock2d_1(x)  # (batch_size, 32 * 32, 32, 32)
        x = self.resblock2d_2(x)  # (batch_size, 32 * 32, 32, 32)
        x = self.resblock2d_3(x)  # (batch_size, 32 * 32, 32, 32)
        x = self.resblock2d_4(x)  # (batch_size, 32 * 32, 32, 32)
        x = self.resblock2d_5(x)  # (batch_size, 32 * 32, 32, 32)
        x = self.resblock2d_6(x)  # (batch_size, 32 * 32, 32, 32)
        x = self.resblock2d_7(x)  # (batch_size, 32 * 32, 32, 32)
        x = self.resblock2d_8(x)  # (batch_size, 32 * 32, 32, 32)
        x = self.resblock2d_9(x)  # (batch_size, 32 * 32, 32, 32)
        x = self.resblock2d_10(x)  # (batch_size, 32 * 32, 32, 32)

        x = x + proj_unit_output  # (batch_size, 32 * 32, 32, 32)

        x = nn.PReLU()(self.conv2d_1(x))  # (batch_size, 16 * 32, 32, 32)
        middle_conv2d_output = x

        # resblock2d part2
        x = self.resblock2d_11(x)  # (batch_size, 16 * 32, 32, 32)
        x = self.resblock2d_12(x)  # (batch_size, 16 * 32, 32, 32)
        x = self.resblock2d_13(x)  # (batch_size, 16 * 32, 32, 32)
        x = self.resblock2d_14(x)  # (batch_size, 16 * 32, 32, 32)
        x = self.resblock2d_15(x)  # (batch_size, 16 * 32, 32, 32)

        x = x + middle_conv2d_output  # (batch_size, 16 * 32, 32, 32)
        # adjust shape
        x = self.conv2d_2(x)  # (batch_size, 8 * 32, 32, 32)
        print('adjust reshape output shape:', x.shape)

        # upconvolution2d part
        x = nn.PReLU()(self.upconv2d_1(x))  # (batch_size, 4 * 32, 64, 64)
        x = nn.PReLU()(self.upconv2d_2(x))  # (batch_size, 2 * 32, 128, 128)
        x = nn.PReLU()(self.upconv2d_3(x))  # (batch_size, 32, 256, 256)
        x = nn.PReLU()(self.upconv2d_4(x))  # (batch_size, 16, 512, 512)
        x = nn.PReLU()(self.upconv2d_5(x))  # (batch_size, 3, 512, 512)

        x = nn.Sigmoid()(x)  # (batch_size, 3, 512, 512)
        print(x.shape)

        return x


if __name__ == '__main__':
    batch_size = 2
    voxel_size = 128
    tensor_empty = torch.empty((batch_size, 1, voxel_size, voxel_size, voxel_size))
    input_tensor = torch.nn.init.normal_(tensor_empty, 0, 1)
    print(input_tensor.shape)
    # create generator class
    generator = Generator3D(in_channels=1, out_channels=3, voxel_size=voxel_size)
    output = generator(input_tensor)

    print(output.shape)








