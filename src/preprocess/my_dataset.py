from torch.utils.data import Dataset
import pandas as pd
import torch

class MyDataset(Dataset):
 
  def __init__(self, data_frame):
    df = data_frame
 
    x = df['src'].values
    y = df['tgt'].values
 
    self.x_train = torch.tensor(x,dtype=torch.float32)
    self.y_train = torch.tensor(y,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self, index):
    return self.x_train[index], self.y_train[index]