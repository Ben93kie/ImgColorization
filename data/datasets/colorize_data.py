from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import cv2
import os
from PIL import Image
import imagesize


class ColorizeData(Dataset):
    def __init__(self, root, cfg, transforms=None, train=True):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        normalize_mean = cfg.INPUT.PIXEL_MEAN
        normalize_std = cfg.INPUT.PIXEL_STD
        normalize_mean_g = cfg.INPUT.PIXEL_MEAN_G
        normalize_std_g = cfg.INPUT.PIXEL_STD_G
        self.color_space = cfg.INPUT.COLOR_SPACE
        self.input_transform = T.Compose([T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.ToTensor(),
                                          T.Normalize(normalize_mean_g, normalize_std_g),
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.Resize(size=(256,256)),
                                           T.ToTensor(),
                                           T.Normalize(normalize_mean, normalize_std),
                                           ])
        self.target_transform_g = T.Compose([T.Resize(size=(256,256)),
                                             T.ToTensor(),
                                             T.Normalize(normalize_mean_g, normalize_std_g),
                                             ])
        #TODO use other transforms
        self.other_transforms = transforms
        self.is_train = train
        self.root_dir = root
        self.image_list = os.listdir(root)
        self.image_ids = dict(enumerate(self.image_list, start=0))
        self.ids = self.image_ids.keys()

    
    def __len__(self) -> int:
        # return Length of dataset
        return len(self.image_ids)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        img_name = self.image_ids[index]
        img_path = os.path.join(self.root_dir,img_name)
        if self.color_space == 'RGB':
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            gray = self.input_transform(img)
            target_img = self.target_transform(img)
        elif self.color_space == 'LAB':
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img = cv2.resize(img, (256,256))
            img = torch.tensor(img)
            img = img / 255.
            gray = img[:,:,0]
            target_img = img[:,:,1:]
            gray = (gray - 0.5)/0.5
            target_img = (target_img - 0.5)/0.5
            gray = torch.unsqueeze(gray,0)
            target_img = target_img.permute(2,0,1)
        else:
            raise ValueError("Color space not yet implemented: ", self.color_space)

        return (gray,target_img)

    def get_img_info(self, index):
        img_name = self.image_ids[index]
        width, height = imagesize.get(os.path.join(self.root_dir,img_name))
        img_data = {}
        img_data['width'] = width
        img_data['height'] = height
        return img_data






        