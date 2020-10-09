from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TLNetSampleDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.dataset_dir = args.dataset_dir
        self.voxel_dir = self.dataset_dir + 'voxels/'  # maybe .npy file
        self.multiview_dir = self.dataset_dir + 'multiviews/'  # maybe {i}.png or {i}.jpg

        self.voxel_transform = transforms.ToTensor()
        self.multiview_images = transforms.ToTensor()

        self.voxels = []  # (n_voxel, np.array((voxel_size, voxel_size, voxel_size))
        self.multiview_images = []  # (n_voxel, np.array((n_view, channels, image_height, image_width))


    def load_images(self):
        pass


    def __getitem__(self, idx):
        voxel = self.voxels[idx]
        multiview_image = self.multiview_images[idx]

        # transform

        # maybe convert format for pytorch

        return voxel, multiview_image


