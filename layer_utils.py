import torch.nn as nn
# helper class

class ResBlock2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        self.conv2d_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding)

    def forward(self, x):
        x_input = x
        x = nn.PReLU()(self.conv2d_1(x))
        x = self.conv2d_2(x)
        # skip connection
        x = x + x_input

        return x


class ResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        self.conv3d_2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding)

    def forward(self, x):
        x_input = x
        x = nn.PReLU()(self.conv3d_1(x))
        x = self.conv3d_2(x)
        x = x + x_input  # skip connection

        return x



# helper function
def transform_3d_to_2d_tensor(tensor):
    batch_size, channels, depth, height, width = tensor.shape
    tensor = tensor.reshape(batch_size, channels * depth, height, width)

    return tensor

