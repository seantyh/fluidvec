import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def get_dataloader(fpath, batch_size=8):
    ds = TrainDataset(fpath)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    return loader

def collate_fn(data_list):
    x0 = data_list[0]
    n_batch = len(data_list)
    collated = {}
    
    for k in x0.keys():
        seqs = [x[k].permute(1, 0) for x in data_list]        
        collated[k] = pad_sequence(seqs, batch_first=True, padding_value=1).permute(0, 2, 1)        
    return collated

class TrainDataset(Dataset):
    def __init__(self, fpath):
        fin = open(fpath, "rb")
        self.data = pickle.load(fin)
        fin.close()
        
    def __len__(self):
        return len(self.data)
    
    def pad_tensor(self, values):
        n = max(len(x) for x in values)
        w = len(values)
        t = np.zeros((w, n))
        for i, v in enumerate(values):
            t[i, :len(v)] = v
        return torch.tensor(t, dtype=torch.int32)
    
    def __getitem__(self, idx):
        tgt, ctx = self.data[idx]
        return {
            "tgt_word": torch.tensor(tgt["word"]).reshape(1, 1),
            "tgt_chars": torch.tensor(tgt["chars"]).reshape(1, -1),
            "tgt_compos": torch.tensor(tgt["compos"]).reshape(1, -1),
            "ctx_word": torch.tensor([x["word"] for x in ctx]).reshape(len(ctx), -1),
            "ctx_chars": self.pad_tensor([x["chars"] for x in ctx]),
            "ctx_compos": self.pad_tensor([x["compos"] for x in ctx])
        }