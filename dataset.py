import torch 
import torchvision
import os
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, path:str, transforms:torchvision.transforms):
        self.path=path
        self.image_list=os.listdir(path)
        self.transform=transforms
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,x):
        from PIL import Image
        return self.transform(Image.open(os.path.join(self.path,self.image_list[x])))