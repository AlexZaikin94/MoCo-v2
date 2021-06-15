#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision

from tqdm import tqdm

from src import arch
from src import utils
from config import cfg
from src import pytorch_utils as ptu

import warnings
warnings.filterwarnings("ignore")

# assert torch.cuda.is_available(), "no CUDA"


# In[2]:


cfg.seed = 42
random.seed(cfg.seed)
torch.random.manual_seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.backends.cudnn.deterministic = True
cfg.num_workers = 0


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_properties(device))


# In[4]:


print('clf.version\n', cfg.clf.version)


# In[5]:


cfg.clf.load = 'best'
cfg.preload_data = False
# cfg.prints = 'display'
# cfg.tqdm_bar = True


# In[6]:


if cfg.clf.load is not None and os.path.exists(os.path.join(cfg.models_dir, cfg.clf.version, ptu.naming_scheme(cfg.clf.version, epoch=cfg.clf.load)) + '.pth'):
    checkpoint = ptu.load_model(version=cfg.clf.version, models_dir=cfg.models_dir, epoch=cfg.clf.load)


# In[7]:


train_dataset = utils.Dataset(os.path.join(cfg.data_path, 'train'), cfg.clf.train_transforms, preload_data=cfg.preload_data, tqdm_bar=cfg.tqdm_bar)
val_dataset = utils.Dataset(os.path.join(cfg.data_path, 'val'), cfg.clf.val_transforms, preload_data=cfg.preload_data, tqdm_bar=cfg.tqdm_bar)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=checkpoint.model.batch_size,
                                           num_workers=cfg.num_workers,
                                           drop_last=True, shuffle=True, pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=checkpoint.model.batch_size,
                                         num_workers=cfg.num_workers,
                                         drop_last=True, shuffle=False, pin_memory=True)


# In[8]:


train_loss, train_score, train_results = checkpoint.evaluate(train_loader,
                                                             device=device,
                                                             tqdm_bar=cfg.tqdm_bar)


# In[9]:


val_loss, val_score, val_results = checkpoint.evaluate(val_loader,
                                                       device=device,
                                                       tqdm_bar=cfg.tqdm_bar)


# In[10]:


print(f'train | loss: {train_loss:.4f} | top-1 acc: {train_score:.6f}')
print(f'val   | loss: {val_loss:.4f} | top-1 acc: {val_score:.6f}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




