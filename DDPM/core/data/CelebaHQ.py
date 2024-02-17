import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

def get_celebahq(root):
    loader_train=DataLoader(CelebAHQTrain(root),batch_size=1)
    loader_validation=DataLoader(CelebAHQValidation(root),batch_size=1)
    return loader_train,loader_validation


class CelebAHQTrain(Dataset):
    def __init__(self, dataroot,listroot):
        super().__init__()
        self.img_path = dataroot 
        # 默认含有
        # celebahq: folder
        # celebahqtrain.txt
        # celebahqvalidation.txt
        
        # read train list
        listpath=os.path.join(listroot,'celebahqtrain.txt')
        datapath=dataroot
        
        with open(listpath, "r") as f:
            paths = f.read().splitlines()
            
        self.paths = [os.path.join(datapath, relpath) for relpath in paths]
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        this_path=self.paths[i]
        this_data=torch.from_numpy(np.load(this_path)).to(torch.float32)
        this_data=this_data.squeeze()
        return this_data
        
        
class CelebAHQValidation(Dataset):
    def __init__(self, dataroot,listroot):
        super().__init__()
        self.img_path = dataroot 
        # 默认含有
        # celebahq: folder
        # celebahqtrain.txt
        # celebahqvalidation.txt
        
        # read train list
        listpath=os.path.join(listroot,'celebahqvalidation.txt')
        datapath=dataroot
        
        with open(listpath, "r") as f:
            paths = f.read().splitlines()
            
        self.paths = [os.path.join(datapath, relpath) for relpath in paths]
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        this_path=self.paths[i]
        this_data=torch.from_numpy(np.load(this_path)).to(dtype=torch.float32)
        this_data=this_data.squeeze()
        return this_data


# test1,test2=get_celebahq('/home/jian/git_all/latent-diffusion/data/')

# for x in test1:
#     print(type(x))
#     print(x.shape)
#     break


def get_ldmcelebahq(dataroot,listroot,batch_size=5):
    trainloader=DataLoader(CelebAHQTrain(dataroot,listroot,64),batch_size=batch_size)
    valloader=DataLoader(CelebAHQValidation(dataroot,listroot,64),batch_size=batch_size)
    return trainloader,valloader


import os
import numpy as np
import albumentations
from torch.utils.data import Dataset


class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex


class CelebAHQTrain(FacesBase):
    def __init__(self,dataroot,listroot, size, keys=None):
        super().__init__()
        root = dataroot
        with open(os.path.join(listroot,"celebahqtrain.txt"),"r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class CelebAHQValidation(FacesBase):
    def __init__(self,dataroot,listroot, size, keys=None):
        super().__init__()
        root = dataroot
        with open(os.path.join(listroot,"celebahqvalidation.txt"), "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/home/jian/git_all/latent_diffusion/data/ffhq"
        with open("/home/jian/git_all/latent_diffusion/data/ffhqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FacesHQTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQTrain(size=size, keys=keys)
        d2 = FFHQTrain(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


class FacesHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQValidation(size=size, keys=keys)
        d2 = FFHQValidation(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex



import bisect
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
