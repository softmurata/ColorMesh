import torch.nn as nn
from layer_utils import ResBlock3d, transform_3d_to_2d_tensor

class ProjectionUnit(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # input => (batch_size, 3, 64, 64, 64)
        # 3d extractor
        self.conv3d_1 = nn.Conv3d(in_channels=in_channels, out_channels=8, kernel_size=5, stride=2, padding=2)  # (batch_size, 8, 32, 32, 32)
        self.conv3d_2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)  # (batch_size, 16, 16, 16, 16)
        self.conv3d_3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # (batch_size, 32, 16, 16, 16)

        self.resblock3d_1 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.resblock3d_2 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.resblock3d_3 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.resblock3d_4 = ResBlock3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # reshape
        # depth collapsion
        self.depth_collapse = nn.Conv2d(in_channels=32 * 16, out_channels=out_channels, kernel_size=1, stride=1, padding=0)  # (batch_size, 32 * 16, 16, 16)

    def forward(self, x):
        # input => (batch_size, 3, 64, 64, 64)
        # conv 3d part
        x = nn.PReLU()(self.conv3d_1(x))  # (batch_size, 8, 32, 32, 32)
        x = nn.PReLU()(self.conv3d_2(x))  # (batch_size, 16, 16, 16, 16)
        x = nn.PReLU()(self.conv3d_3(x))  # (batch_size, 32, 16, 16, 1&)

        conv3d_part_output = x

        # resblock 3d part
        x = self.resblock3d_1(x)  # (batch_size, 32, 16, 16, 16)
        x = self.resblock3d_2(x)  # (batch_size, 32, 16, 16, 16)
        x = self.resblock3d_3(x)  # (batch_size, 32, 16, 16, 16)
        x = self.resblock3d_4(x)  # (batch_size, 32, 16, 16, 16)

        # reshape
        x = transform_3d_to_2d_tensor(x)  # (batch_size, 32 * 16, 16, 16)
        x = nn.ReLU(inplace=True)(self.depth_collapse(x))  # (batch_size, 32 * 16, 16, 16)

        return x
"""
if __name__ == '__main__':
    import torch
    import numpy as np
    batch_size = 2
    voxel_size = 64
    input_tensor = torch.from_numpy(np.random.randn(batch_size, 3, voxel_size, voxel_size, voxel_size)).type(torch.float32)

    projection_unit = ProjectionUnit(in_channels=3, out_channels=32 * 16)

    proj_output = projection_unit(input_tensor)

    print(proj_output.shape)
"""


