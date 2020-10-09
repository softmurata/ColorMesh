import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VoxelDataset(Dataset):

    def __init__(self, args, azimuth_bounds=[0, 359], elevation_bounds=[10, 170], scale_bounds=[3, 6.3]):
        super().__init__()

        self.batch_size = args.batch_size
        self.voxel_size = args.voxel_size
        self.image_size = args.image_size
        self.data_num = 20

        self.azimuth_low, self.azimuth_high = azimuth_bounds
        self.elevation_low, self.elevation_high = elevation_bounds
        self.scale_low, self.scale_high = scale_bounds
        # if you want to add more transformations,
        # you should use transforms.Compose([ToTensor(), Normalize([0.5, 0.5, 0.5])])

        self.real_images = []
        self.real_params = []
        self.real_voxel_model = []

    def create_dummy_data(self):
        print('create dummy dataset')
        self.real_images = np.random.rand(self.data_num, 3, self.image_size, self.image_size)  # already normalize
        # print('image ok', self.real_images.shape)
        real_params = [[np.random.randint(self.azimuth_low, self.azimuth_high) * np.pi / 180.0, np.random.randint(self.elevation_low, self.elevation_high) * np.pi / 180.0, random.sample(list(np.arange(self.scale_low, self.scale_high)), 1)[0]] for _ in range(self.data_num)]
        self.real_params = np.array(real_params, dtype=np.float32)
        # print('params ok', self.real_params.shape)
        real_voxel_model = [[[[np.random.randint(0, 2) for _ in range(self.voxel_size)] for _ in range(self.voxel_size)] for _ in range(self.voxel_size)] for _ in range(self.data_num)]
        self.real_voxel_model = np.array(real_voxel_model, dtype=np.float32)
        # print('voxel ok', self.real_voxel_model.shape)

    def __len__(self):

        return len(self.real_images)

    def __getitem__(self, item):

        real_image = torch.from_numpy(self.real_images[item])
        real_voxel_model = torch.from_numpy(self.real_voxel_model[item])
        real_voxel_model = real_voxel_model.reshape(1, self.voxel_size, self.voxel_size, self.voxel_size)
        real_param = torch.from_numpy(self.real_params[item])

        real_image = real_image.type(torch.float32)
        real_voxel_model = real_voxel_model.type(torch.float32)
        real_param = real_param.type(torch.float32)

        # print('image shape: {}  voxel shape: {} param shape:{}'.format(real_image.shape, real_voxel_model.shape, real_param.shape))
        return real_image, real_voxel_model, real_param


"""
# test code
if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--voxel_size', type=int, default=64)
    args = parser.parse_args()

    dataset = VoxelDataset(args)
    dataset.create_dummy_data()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print('build dataloader ok')

    for n, data in enumerate(dataloader):
        real_image, real_voxel_model, real_param = data
        print(n, real_image.shape, real_voxel_model.shape, real_param.shape)
"""



