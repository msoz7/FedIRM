# encoding: utf-8
"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from options import args_parser
import torchvision.transforms as transforms
args = args_parser()

class CheXpertDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None,is_labeled=True,):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(CheXpertDataset, self).__init__()
        file = pd.read_csv(csv_file)
        self.is_labeled = is_labeled
        self.root_dir = root_dir
        self.images = file["ImageID"].values
        self.labels = file.iloc[:, 1:].values.astype(int)
        self.transform = transform
        self.resize = transforms.Compose([transforms.Resize((224, 224))])
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.weak_trans = transforms.Compose([
                transforms.RandomCrop(size=(224, 224)),
                #transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                self.normalize
            ])
        self.strong_trans = transforms.Compose([
                transforms.RandomCrop(size=(224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                self.normalize
            ])

        print("Total # images:{}, labels:{}".format(len(self.images), len(self.labels)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        items = self.images[index]  # .split('/')
        # study = items[2] + '/' + items[3]
        img_name = self.images[index]
        root_path = self.root_dir
        # Default path if no "dect" prefix
        image_name = os.path.join(root_path, img_name)
        image = Image.open(image_name).convert("RGB")
        image_resized = self.resize(image)
        label = self.labels[index]
        # print(label)
        if self.transform is not None:
            image = self.transform(image)
        #if not self.is_labeled:
            
        weak_aug = self.weak_trans(image_resized)
        #strong_aug = self.strong_trans(image_resized)
        #idx_in_all = self.data_idxs[index]

        # for idx in range(len(weak_aug)):
        #     weak_aug[idx] = weak_aug[idx].squeeze()
        #     strong_aug[idx] = strong_aug[idx].squeeze()
        return items, index, image,weak_aug, torch.FloatTensor(label)
        
        
        
        
       # return items, index, image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.images)
    
    def getLabel(self,index):
        return torch.argmax(torch.from_numpy(self.labels[index]),dim=0).item()
    def getImage(self,index):
        img_name = self.images[index]
        # Determine the correct root path based on the image name prefix
        root_path = self.root_dir
        image_name = os.path.join(root_path, img_name)
        image = Image.open(image_name).convert("RGB")
        return image   


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
