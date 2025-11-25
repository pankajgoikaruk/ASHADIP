import os, json
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader


class LogMelDataset(Dataset):
    def __init__(self, segments_csv, features_root, split='train'):
        df = pd.read_csv(segments_csv)
        self.df = df[df['split']==split].reset_index(drop=True)
        self.features_root = features_root
        self.label2id = {l:i for i,l in enumerate(sorted(df['label'].unique()))}
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.features_root, row['feat_relpath'])
        S = np.load(path) # (n_mels, T)
        x = torch.tensor(S).unsqueeze(0) # (1, M, T)
        y = torch.tensor(self.label2id[row['label']], dtype=torch.long)
        return x, y


def make_loaders(segments_csv, features_root, batch_size=64, num_workers=4):
    ds_tr = LogMelDataset(segments_csv, features_root, 'train')
    ds_va = LogMelDataset(segments_csv, features_root, 'val')
    ds_te = LogMelDataset(segments_csv, features_root, 'test')
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl_tr, dl_va, dl_te, ds_tr.label2id