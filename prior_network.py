import torch
from torchvision.models import alexnet
import torch.nn as nn

# for training
class TNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.encoder = TLEncoder(in_channels=in_channels, out_channels=hidden_channels)
        self.decoder = TLDecoder(in_channels=hidden_channels, out_channels=out_channels)
        self.image_extractor = ImageExtractor(in_channels=3, out_channels=hidden_channels)

    def autoencoder_parameters(self):

        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def encimage_parameters(self):

        return list(self.encoder.parameters()) + list(self.image_extractor.parameters())

    def joint_parameters(self):

        return list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.image_extractor.parameters())

    def forward(self, voxel, multiview_image):
        encoder_output = self.encoder(voxel)
        decoder_output = self.decoder(encoder_output)
        image_extractor_output = self.image_extractor(multiview_image)

        return encoder_output, decoder_output, image_extractor_output


# for test
class LNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.image_extractor = ImageExtractor(in_channels=in_channels, out_channels=hidden_channels)
        self.decoder = TLDecoder(in_channels=hidden_channels, out_channels=out_channels)

        # load pretrained model

    def forward(self, x):
        x = self.image_extractor(x)
        x = self.decoder(x)  # (1, voxel_size, voxel_size, voxel_size)
        x = x.squeeze(0)  # (voxel_size, voxel_size, voxel_size)

        return x



class TLEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # input -> (batch_size, 1, 20, 20, 20)
        # channel sizes (1, 96, 256, 384, 256 / 64)
        # convolution 3d(stride=1)
        # final layer is fully connected layer
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=96, kernel_size=3, stride=1, padding=1)  # (batch_size, 96, 20, 20, 20)
        self.conv2 = nn.Conv3d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1)  # (batch_size, 256, 20, 20, 20)


    def forward(self, x):

        return x


class TLDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x):

        return x



class ImageExtractor(nn.Module):
    """
    use pretrained alexnet
    # code
    alexnet = alexnet(pretrained=True)
    print(alexnet.features[0])
    print(alexnet.classifier[0])
    exit()

    AlexNet
    input => (batch_size, 3, 224, 224)
    1. Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
       output => (batch_size, 64, 56, 56)
    2. relu()
    3. MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
       output => (batch_size, 64, 28, 28)
    4. Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
       output => (batch_size, 192, 28, 28)
    5. relu()
    6. MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
       output => (batch_size, 192, 14, 14)
    7. Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
       output => (batch_size, 384, 14, 14)
    8. relu()
    9. Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
       output => (batch_size, 256, 14, 14)
    10. relu()
    11. Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        output => (batch_size, 256, 14, 14)
    12. relu()
    13. MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
        output => (batch_size, 256, 7, 7)

    14. adaptiveAvgPool2d(output_size=(6, 6))
        output => (batch_size, 256, 6, 6)
    15. flatten()
        output => (batch_size, 256 * 6 * 6)
    # classifier part
    16. DropOut(p=0.5, inplace=False)
        output => (batch_size, 256 * 6 * 6)
    17. Linear(in_features=256 * 6 * 6, out_features=4096)
        output => (batch_size, 4096)
    18.relu()
    19. Dropout(p=0.5, inplace=False)
        output -> (batch_size, 4096)
    20. Linear(in_features=4096, out_features=4096)
        output => (batch_size, 4096)
    21. relu()
    22. Linear(in_features=4096, out_features=64)
        output



    """

    def __init__(self, in_channels=3, out_channels=64):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # base pretrained alexnet model
        pretrained_alexnet_model = alexnet(pretrained=True)

        self.alexnet_model = self.construct_train_alexnet(pretrained_alexnet_model)

    def construct_train_alexnet(self, pretrained_alexnet_model):
        # change classifier
        pretrained_alexnet_model.classifier[6] = nn.Linear(in_features=4096, out_features=self.out_channels)

        return pretrained_alexnet_model

    def forward(self, x):

        x = self.alexnet_model(x)  # (batch_size, out_channels)

        return x



