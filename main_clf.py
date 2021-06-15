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


print(cfg())


# In[3]:


if cfg.seed is not None:
    random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_properties(device))


# In[4]:


print(cfg.moco.version)


# In[5]:


print(cfg.clf.version)


# In[6]:


if cfg.clf.load is not None and os.path.exists(os.path.join(cfg.models_dir, cfg.clf.version, ptu.naming_scheme(cfg.clf.version, epoch=cfg.clf.load)) + '.pth'):
    checkpoint = ptu.load_model(version=cfg.clf.version, models_dir=cfg.models_dir, epoch=cfg.clf.load)
    if cfg.prints == 'display':
        display(checkpoint.log.sort_index(ascending=False).head(20))
    elif cfg.prints == 'print':
        print(checkpoint.log.sort_index(ascending=False).head(20))
else:
    moco_checkpoint = ptu.load_model(version=cfg.moco.version, models_dir=cfg.models_dir, epoch=cfg.clf.moco_epoch)
    model = moco_checkpoint.model
    model.end_moco_phase()
    if cfg.prints == 'display':
        display(moco_checkpoint.log.sort_index(ascending=False).head(5))
    elif cfg.prints == 'print':
        print(moco_checkpoint.log.sort_index(ascending=False).head(5))

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=cfg.clf.lr,
                                momentum=cfg.clf.optimizer_momentum,
                                weight_decay=cfg.clf.wd)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=cfg.clf.epochs,
                                                              eta_min=cfg.clf.min_lr) if cfg.clf.cos else None

    checkpoint = utils.MyCheckpoint(version=cfg.clf.version,
                                    model=model,
                                    optimizer=optimizer,
                                    criterion=nn.CrossEntropyLoss().to(device),
                                    score=utils.accuracy_score,
                                    lr_scheduler=lr_scheduler,
                                    models_dir=cfg.models_dir,
                                    seed=cfg.seed,
                                    best_policy=cfg.clf.best_policy,
                                    save=cfg.save,
                                   )

    checkpoint.moco_log = moco_checkpoint.log
    checkpoint.moco_train_batch_log = moco_checkpoint.train_loss_log

    if cfg.save:
        with open(os.path.join(checkpoint.version_dir, 'config.txt'), 'w') as f:
            f.writelines(str(cfg))

ptu.params(checkpoint.model)
ptu.params(checkpoint.model.q_encoder.fc)


# In[8]:


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


# In[ ]:


checkpoint.train(train_loader=train_loader,
                 val_loader=val_loader,
                 train_epochs=int(max(0, cfg.clf.epochs - checkpoint.get_log())),
                 optimizer_params=cfg.clf.optimizer_params,
                 prints=cfg.prints,
                 epochs_save=cfg.epochs_save,
                 epochs_evaluate_train=cfg.epochs_evaluate_train,
                 epochs_evaluate_validation=cfg.epochs_evaluate_validation,
                 device=device,
                 tqdm_bar=cfg.tqdm_bar,
                 save=cfg.save,
                 save_log=cfg.save_log,
                )


# In[ ]:


# import torchviz
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=0, drop_last=True, shuffle=True, pin_memory=True)
# for batch in train_loader:
#     img, labels = batch
#     img = img.to(device)
#     labels = labels.to(device)
#     model = model.to(device)
#     out = model(img, prints=True)
#     print('img',  img.shape)
#     print('labels', labels.shape)
#     print('out', out.shape)
#     loss = nn.functional.cross_entropy(out.float(), labels.long())
#     print('loss',  loss)
#     break
# torchviz.make_dot(out, params=dict(model.named_parameters()))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




