import torch.nn as nn
from layer_utils import ResBlock2d

class ImageDecoder(nn.Module):
    # reference: RenderNet 2d part
    def __init__(self, in_channels=32 * 16, out_channels=3):
        super().__init__()
        # input size => (batch_size, 32 * 16, 16, 16)

        # upconv2d part
        self.resblock2d_1 = ResBlock2d(in_channels=32 * 16, out_channels=32 * 16, kernel_size=3, stride=1,
                                       padding=1)  # (batch_size, 32 * 16, 16, 16)
        self.resblock2d_2 = ResBlock2d(in_channels=32 * 16, out_channels=32 * 16, kernel_size=3, stride=1,
                                       padding=1)  # (batch_size, 32 * 16, 16, 16)
        self.resblock2d_3 = ResBlock2d(in_channels=32 * 16, out_channels=32 * 16, kernel_size=3, stride=1,
                                       padding=1)  # (batch_size, 32 * 16, 16, 16)
        self.resblock2d_4 = ResBlock2d(in_channels=32 * 16, out_channels=32 * 16, kernel_size=3, stride=1,
                                       padding=1)  # (batch_size, 32 * 16, 16, 16)

        self.upconv2d_1 = nn.ConvTranspose2d(in_channels=32 * 16, out_channels=32 * 8, kernel_size=4, stride=2,
                                             padding=1)  # (batch_size, 32 * 8, 32, 32)
        self.instance_norm1 = nn.InstanceNorm2d(num_features=32 * 8)
        self.upconv2d_2 = nn.ConvTranspose2d(in_channels=32 * 8, out_channels=32 * 4, kernel_size=4, stride=2,
                                             padding=1)  # (batch_size, 32 * 4, 64, 64)
        self.instance_norm2 = nn.InstanceNorm2d(num_features=32 * 4)
        self.upconv2d_3 = nn.ConvTranspose2d(in_channels=32 * 4, out_channels=32 * 2, kernel_size=4, stride=2,
                                             padding=1)  # (batch_size, 32 * 2, 128, 128)
        self.instance_norm3 = nn.InstanceNorm2d(num_features=32 * 2)
        self.upconv2d_4 = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=32, kernel_size=4, stride=2,
                                             padding=1)  # (batch_size, 32, 256, 256)
        self.instance_norm4 = nn.InstanceNorm2d(num_features=32)
        self.upconv2d_5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2,
                                             padding=1)  # (batch_size, 16, 512, 512)
        self.instance_norm5 = nn.InstanceNorm2d(num_features=16)
        self.upconv2d_6 = nn.ConvTranspose2d(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1,
                                             padding=1)  # (batch_size, 3, 512, 512)

    def forward(self, x):
        x_input = x  # (batch_size, 32 * 16, 16, 16)
        # resblock part
        x = self.resblock2d_1(x)
        x = self.resblock2d_2(x)
        x = self.resblock2d_3(x)
        x = self.resblock2d_4(x)
        x = x + x_input  # skip connection

        # upconvolution part
        x = nn.PReLU()(self.instance_norm1(self.upconv2d_1(x)))  # (batch_size, 32 * 8, 32, 32)
        x = nn.PReLU()(self.instance_norm2(self.upconv2d_2(x)))  # (batch_size, 32 * 4, 64, 64)
        x = nn.PReLU()(self.instance_norm3(self.upconv2d_3(x)))  # (batch_size, 32 * 2, 128, 128)
        x = nn.PReLU()(self.instance_norm4(self.upconv2d_4(x)))  # (batch_size, 32, 256, 256)
        x = nn.PReLU()(self.instance_norm5(self.upconv2d_5(x)))  # (batch_size, 16, 512, 512)
        x = nn.Sigmoid()(self.upconv2d_6(x))  # (batch_size, 3, 512, 512)

        return x

"""
if __name__ == '__main__':
    import torch
    import numpy as np
    batch_size = 2
    input_tensor = torch.from_numpy(np.random.randn(batch_size, 32 * 16, 16, 16)).type(torch.float32)
    print(input_tensor.shape)
    image_decoder = ImageDecoder(in_channels=32 * 16, out_channels=3)

    img_dec_output = image_decoder(input_tensor)
    print(img_dec_output.shape)
"""

