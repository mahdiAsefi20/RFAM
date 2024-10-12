import os

from torch.utils.data import Dataset
from PIL import Image 
from src.frequency_aware_cue import frequency_aware_cue


class BaseDataset(Dataset):
    def __init__(self, fake_root, original_root, transform=None, num_classes=2):
        super(BaseDataset,self).__init__()
        self.roots = [original_root, fake_root]
        self.transform = transform
        self.num_classes = num_classes
        assert transform is not None, "transform is None"

    def __getitem__(self,idx):
        img_path = self.imgs[idx][0]
        label = self.imgs[idx][1]
        img_name = img_path.split("/")[-1]
        for pair_img_name in os.listdir(self.roots[1 - label]):
            if pair_img_name == img_name:
                pair_img_path = os.path.join(self.roots[1 - label], pair_img_name)
                break

        # TODO Start:
        # Make similarity map using a pair of fake and original image.
        # TODO End.

        image = Image.open(img_path).convert('RGB')
        rgb_image = self.transform(image)
        freq_image = frequency_aware_cue(rgb_image)

        return rgb_image, freq_image, label

    def __len__(self):
        return len(self.imgs)