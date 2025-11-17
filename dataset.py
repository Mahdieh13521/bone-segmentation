from torch.utils.data import Dataset
import cv2
from torch.utils.tensorboard.summary import image
import numpy as np

def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # H.W
    xray = xray.astype(np.float32)/255.0
    #Note
    xray = xray.reshape((1, *xray.shape)) #1.H.W
    return xray


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #Note
    mask = (mask>0).astype(np.float32)
    mask = mask.reshape((1, *mask.shape)) #1.H.W
    return mask

class Knee_dataset(Dataset):
    def __init__(self, df, test=False):
        self.df = df
        self.test = test

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        image = read_xray(self.df['xrays'].iloc[index]) # self.df.xray
        
        if self.test: 
            res = {'image':image}
        else:
            mask = read_mask(self.df['masks'].iloc[index])
            res = {'image':image, 'mask': mask}
        return res