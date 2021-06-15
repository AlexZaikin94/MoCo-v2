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


if cfg.moco.load is not None and os.path.exists(os.path.join(cfg.models_dir, cfg.moco.version, ptu.naming_scheme(cfg.moco.version, epoch=cfg.moco.load)) + '.pth'):
    checkpoint = ptu.load_model(version=cfg.moco.version, models_dir=cfg.models_dir, epoch=cfg.moco.load)
    if cfg.prints == 'display':
        display(checkpoint.log.sort_index(ascending=False).head(20))
    elif cfg.prints == 'print':
        print(checkpoint.log.sort_index(ascending=False).head(20))
else:
    model = arch.MoCo_v2(backbone=cfg.moco.backbone,
                         dim=cfg.moco.dim,
                         queue_size=cfg.moco.queue_size,
                         batch_size=cfg.moco.bs,
                         momentum=cfg.moco.model_momentum,
                         temperature=cfg.moco.temperature,
                         bias=cfg.moco.bias,
                         moco=True,
                         clf_hyperparams=cfg.moco.clf_kwargs,
                         seed=cfg.seed,
                         mlp=cfg.moco.mlp,
                        )

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=cfg.moco.lr,
                                momentum=cfg.moco.optimizer_momentum,
                                weight_decay=cfg.moco.wd)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=cfg.moco.epochs,
                                                              eta_min=cfg.moco.min_lr) if cfg.moco.cos else None

    checkpoint = utils.MyCheckpoint(version=cfg.moco.version,
                                    model=model,
                                    optimizer=optimizer,
                                    criterion=nn.CrossEntropyLoss().to(device),
                                    score=utils.accuracy_score,
                                    lr_scheduler=lr_scheduler,
                                    models_dir=cfg.models_dir,
                                    seed=cfg.seed,
                                    best_policy=cfg.moco.best_policy,
                                    save=cfg.save,
                                   )
    if cfg.save:
        with open(os.path.join(checkpoint.version_dir, 'config.txt'), 'w') as f:
            f.writelines(str(cfg))

ptu.params(checkpoint.model)


# In[7]:


train_dataset = utils.Dataset(os.path.join(cfg.data_path, 'train'), cfg.moco.train_transforms, preload_data=cfg.preload_data, tqdm_bar=cfg.tqdm_bar)
train_eval_dataset = utils.Dataset(os.path.join(cfg.data_path, 'train'), cfg.moco.train_eval_transforms, preload_data=cfg.preload_data, tqdm_bar=cfg.tqdm_bar)
val_dataset = utils.Dataset(os.path.join(cfg.data_path, 'val'), cfg.moco.val_eval_transforms, preload_data=cfg.preload_data, tqdm_bar=cfg.tqdm_bar)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=checkpoint.model.batch_size,
                                           num_workers=cfg.num_workers,
                                           drop_last=True, shuffle=True, pin_memory=True)

train_eval_loader = torch.utils.data.DataLoader(train_eval_dataset,
                                                batch_size=checkpoint.model.batch_size,
                                                num_workers=cfg.num_workers,
                                                drop_last=True, shuffle=True, pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=checkpoint.model.batch_size,
                                         num_workers=cfg.num_workers,
                                         drop_last=True, shuffle=False, pin_memory=True)


# In[ ]:


checkpoint.train(train_loader=train_loader,
                 train_eval_loader=train_eval_loader,
                 val_loader=val_loader,
                 train_epochs=int(max(0, cfg.moco.epochs - checkpoint.get_log())),
                 optimizer_params=cfg.moco.optimizer_params,
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
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, num_workers=0, drop_last=True, shuffle=True, pin_memory=True)
# for batch in train_loader:
#     (q_img, k_img), labels = batch
#     q_img = q_img.to(device)
#     k_img = k_img.to(device)
#     labels = labels.to(device)
#     model = checkpoint.model.to(device)
    
#     out = checkpoint.model(q_img, k_img, prints=True)
#     q, logits, zeros = out
#     print('q_img',  q_img.shape)
#     print('k_img',  k_img.shape)
#     print('labels', labels.shape)
#     print('logits', logits.shape)
#     print('zeros',  zeros.shape)
#     loss = nn.functional.cross_entropy(logits.float(), zeros.long())
#     print('loss',  loss)
#     break
# torchviz.make_dot(out, params=dict(model.named_parameters()))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




