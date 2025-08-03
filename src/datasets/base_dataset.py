import os

import cv2
import numpy as np
import torchvision.transforms as transforms
from GT_generator import make_mask, make_similarity_map
from torch.utils.data import Dataset
from PIL import Image
from frequency_aware_cue import frequency_aware_cue


class BaseDataset(Dataset):
    def __init__(self, fake_root, original_root, transform=None, train_type="train",num_classes=2, alpha=0.33):
        super(BaseDataset,self).__init__()
        self.roots = [original_root, fake_root]
        self.transform = transform
        self.num_classes = num_classes
        self.train_type = train_type
        self.alpha = alpha
        assert transform is not None, "transform is None"

    def __getitem__(self,idx):
        img_path = self.imgs[idx][0]
        label = self.imgs[idx][1]
        img_name = img_path.split("/")[-1]
        for pair_img_name in os.listdir(os.path.join(self.roots[1 - label], self.train_type)):
            if pair_img_name == img_name:
                pair_img_path = os.path.join(self.roots[1 - label], self.train_type, pair_img_name)
                break
        image = Image.open(img_path).convert('RGB')
        np_image = np.array(image)
        pair_img = Image.open(pair_img_path).convert('RGB')
        np_pair_img = np.array(pair_img)
        # Make similarity map using a pair of fake and original image.
        binary_mask = make_mask(np_image, np_pair_img, label, img_name+str(idx))
        similarity_map = make_similarity_map(binary_mask, 5)
        tensor = transforms.ToTensor()
        tensor_similarity_map = tensor(similarity_map)
        tensor_similarity_map = tensor_similarity_map.view(similarity_map.shape[0], similarity_map.shape[1])
        rgb_image = self.transform(image)
        freq_image = frequency_aware_cue(rgb_image, self.alpha)

        return rgb_image, freq_image, float(label), tensor_similarity_map, img_path

    def __len__(self):
        return len(self.imgs)