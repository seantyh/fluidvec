import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(xs):
    return xs

def get_dataloader(fpath, batch_size=8):
    ds = TrainDataset(fpath)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    return loader

class TrainDataset(Dataset):
    def __init__(self, fpath):
        fin = open(fpath, "rb")
        self.data = pickle.load(fin)
        fin.close()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        