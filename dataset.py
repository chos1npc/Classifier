import torch
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
class AICUPdataset(torch.utils.data.Dataset):
    def __init__(self,csv_file,img_dir,transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        with open("categories.csv",'r') as f:
            self.label_index = f.readlines()
            self.label_index = [i.strip('\n') for i in self.label_index]
        # print("suc")
        # print(img_dir)
        #print(len(img_dir))

        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self,index):
        # 取得train.csv之category
        label_index = self.annotations.iloc[index,1]
        label = [0 for i in self.label_index]
        label[label_index] = 1

        # label to category name
        category_name = self.label_index[label_index]
        # print(category_name)

        img_path = os.path.join(self.img_dir, category_name, self.annotations.iloc[index,0])

        image = Image.open(img_path)
        # image to array
        image = np.asarray(image)
        
        w,h,c =image.shape
        if w!=h:
            if w>h:
                pad = np.zeros((w,(w-h)//2,c)).astype(np.uint8)
                image = np.concatenate((pad,image),axis = 1)
                image = np.concatenate((image,pad),axis =1)
            else:
                pad = np.zeros(((h-w)//2, h, c)).astype(np.uint8)
                image = np.concatenate((pad,image),axis = 0)
                image = np.concatenate((image,pad),axis =0)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        
        

        return image ,torch.Tensor(label)

# aicup_dataset = AICUPdataset("train.csv", img_dir="/media/mmlab206/YT8M-4TB/aicup_data/aicup")

# for img_name in aicup_dataset.annotations.iloc[:,0]:
#     print(img_name)
