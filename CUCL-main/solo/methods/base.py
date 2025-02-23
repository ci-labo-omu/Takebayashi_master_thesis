# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import logging
from argparse import ArgumentParser
from functools import partial
from re import S
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pytorch_lightning as pl
import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.classification_dataloader import prepare_datasets as prepare_datasets_classification
from solo.data.classification_dataloader import prepare_transforms as prepare_transforms_classification
from solo.data.classification_dataloader import prepare_dataloaders as prepare_dataloaders_classification
from solo.data.pretrain_dataloader import (
    prepare_dataloader,
    prepare_cl_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
    dataset_with_index,
)
from solo.utils.knn import extract_features, run_knn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.backbones import (
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
    poolformer_m36,
    poolformer_m48,
    poolformer_s12,
    poolformer_s24,
    poolformer_s36,
    resnet18,
    resnet50,
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
    wide_resnet28w2,
    wide_resnet28w8,
)
from solo.utils.knn import WeightedKNNClassifier
from solo.utils.lars import LARS
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.utils.misc import remove_bias_and_norm_from_weight_decay
from solo.utils.momentum import MomentumUpdater, initialize_momentum_params
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.cluster import KMeans
# for xmeans
from pyclustering.utils import metric
from pyclustering.utils.metric import distance_metric, type_metric
from scipy.spatial import distance
from sklearn import cluster, preprocessing
import warnings

#clustering process-------------------------------------------
from pytorch_lightning.callbacks import Callback
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import DBSCAN
import os
import csv

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from CAE.ca_plus import ClusterCAplus

import pandas as pd


def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs

def squared_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    # print(f"X shape: {x.shape}")  # 特徴量Xの形状
    # print(f"C shape: {y.shape}")  # コードブックCの形状
    # diff = x.unsqueeze(1) - y.unsqueeze(0)
    # # return torch.sum(diff * diff, -1)
    # return diff.pow(2).sum(dim=2)


    x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
    C = y.unsqueeze(0)  # [1, num_codewords, feature_dim]
    # デバッグ用: x と C の型と内容を確認

    # print("x:", x.device)
    # print("C:", C.device)
    diff = x - C  # [batch_size, num_codewords, feature_dim] ブロードキャストによる引き算
    return diff.pow(2).sum(dim=2)
class CQCLoss(pl.LightningModule):

    def __init__(self, batch_size, tau_cqc):
        super(CQCLoss, self).__init__()
        self.batch_size = batch_size
        self.tau_cqc = tau_cqc
        self.COSSIM = torch.nn.CosineSimilarity(dim=-1)
        self.CE = torch.nn.CrossEntropyLoss(reduction="sum")
        # self.get_corr_mask = self._get_correlated_mask().type(torch.bool)
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        
        self.masks = {batch_size:mask}

    def _get_correlated_mask(self,batch_size):
        if batch_size in self.masks.keys():
            return self.masks[batch_size]
        else:
            diag = np.eye(2 * batch_size)
            l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
            l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
            mask = torch.from_numpy((diag + l1 + l2))
            mask = (1 - mask).type(torch.bool)
            self.masks.update({batch_size:mask})
            return mask
    

    def forward(self, Xa, Xb, Za, Zb, targets):
        batch_size = Xa.shape[0]
        #-------------------------------original--------------------------------------
        XaZb = torch.cat([Xa, Zb], dim=0)
        XbZa = torch.cat([Xb, Za], dim=0)

        Cossim_ab = self.COSSIM(XaZb.unsqueeze(1), XaZb.unsqueeze(0))
        Rab = torch.diag(Cossim_ab, batch_size)
        Lab = torch.diag(Cossim_ab, -batch_size)
        Pos_ab = torch.cat([Rab, Lab]).view(2 * batch_size, 1)
        Neg_ab = Cossim_ab[self._get_correlated_mask(batch_size)].view(2 * batch_size, -1)

        Cossim_ba = self.COSSIM(XbZa.unsqueeze(1), XbZa.unsqueeze(0))
        Rba = torch.diag(Cossim_ba, batch_size)
        Lba = torch.diag(Cossim_ba, -batch_size)
        Pos_ba = torch.cat([Rba, Lba]).view(2 * batch_size, 1)
        Neg_ba = Cossim_ba[self._get_correlated_mask(batch_size)].view(2 * batch_size, -1)

        logits_ab = torch.cat((Pos_ab, Neg_ab), dim=1)
        logits_ab /= self.tau_cqc

        logits_ba = torch.cat((Pos_ba, Neg_ba), dim=1)
        logits_ba /= self.tau_cqc
        labels = torch.zeros(2 * batch_size, device=Xa.device).long()

        loss = self.CE(logits_ab, labels) + self.CE(logits_ba, labels)
        return loss / (2 * batch_size)
        #----------------------original------------------------------

        #---------------------------------using labels---------------------------------------
        # Combine features for pairwise similarity calculation
        # XaZb = torch.cat([Xa, Zb], dim=0)  # (2 * batch_size, feature_dim)
        # XbZa = torch.cat([Xb, Za], dim=0)  # (2 * batch_size, feature_dim)
        #
        # # Cosine similarity matrices
        # Cossim_ab = self.COSSIM(XaZb.unsqueeze(1), XaZb.unsqueeze(0))  # (2 * batch_size, 2 * batch_size)
        # Cossim_ba = self.COSSIM(XbZa.unsqueeze(1), XbZa.unsqueeze(0))  # (2 * batch_size, 2 * batch_size)
        #
        # # Extend targets to handle both views
        # extended_targets = torch.cat([targets, targets], dim=0)  # (2 * batch_size)
        #
        # # Create label masks for positive pairs (ビュー間かつラベル一致)
        # # batch_size = targets.shape[0]
        # pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=targets.device)
        #
        # for i in range(batch_size):
        #     # ビュー間ペアのみに限定
        #     if extended_targets[i] == extended_targets[i + batch_size]:  # ラベルが一致している場合
        #         pos_mask[i, i + batch_size] = True  # Xa[i] <-> Zb[i]
        #         pos_mask[i + batch_size, i] = True  # Xb[i] <-> Za[i]
        #
        # # Negative mask: ポジティブ以外のすべてのペアをネガティブに
        # neg_mask = ~pos_mask
        #
        # # Extract positive and negative scores
        # pos_scores_ab = Cossim_ab[pos_mask].view(batch_size, -1)  # ビュー間ポジティブペア
        # neg_scores_ab = Cossim_ab[neg_mask].view(batch_size, -1)  # その他はネガティブ
        #
        # pos_scores_ba = Cossim_ba[pos_mask].view(batch_size, -1)  # ビュー間ポジティブペア
        # neg_scores_ba = Cossim_ba[neg_mask].view(batch_size, -1)  # その他はネガティブ
        #
        # # Logits for contrastive loss
        # logits_ab = torch.cat([pos_scores_ab, neg_scores_ab], dim=1)  # ポジティブとネガティブを結合
        # logits_ab /= self.tau_cqc
        #
        # logits_ba = torch.cat([pos_scores_ba, neg_scores_ba], dim=1)
        # logits_ba /= self.tau_cqc
        #
        # # Labels for contrastive loss (0 for positives, rest are negatives)
        # contrastive_labels = torch.zeros(logits_ab.size(0), device=targets.device, dtype=torch.long)
        #
        # # Compute cross-entropy loss
        # loss = self.CE(logits_ab, contrastive_labels) + self.CE(logits_ba, contrastive_labels)
        # return loss / (2 * batch_size)
        #-------------------------------using labels--------------------------------------


class Quantization_Head(pl.LightningModule):
        # def __init__(self, N_words, N_books, L_word, tau_q, task_len, proj_output_dim):
        def __init__(self, N_books, L_word, tau_q, task_len, proj_output_dim):

            super(Quantization_Head, self).__init__()
            self.fc = nn.Linear(proj_output_dim, N_books * L_word)
            nn.init.kaiming_normal_(self.fc.weight, a=math.sqrt(5))
            # Codebooks
            N_words = 50
            # self.C = torch.nn.Parameter(Variable((torch.randn(task_len, N_books, N_words, L_word)).type(torch.float32), requires_grad=True)) #多分ここにx_meansを用いることでCを決める．
            self.shared_codebook = {}
            # word_list = [[74, 87, 40, 70, 130, 64, 66, 16],
            # [132, 124, 145, 137, 124, 146, 162, 134],
            # [103, 135, 112, 179, 119, 128, 137, 156],
            # [183, 147, 184, 123, 140, 141, 138, 159],
            # [102, 129, 150, 147, 168, 144, 130, 117],
            # [116, 73, 101, 117, 90, 98, 101, 96],
            # [167, 102, 124, 130, 91, 155, 142, 155],
            # [188, 80, 147, 146, 121, 157, 129, 102],
            # [120, 187, 195, 134, 147, 169, 133, 142],
            # [122, 136, 132, 95, 117, 116, 113, 136]]
            # word_list = [68, 138, 134, 152, 136, 99, 133, 134, 153, 121] #avaregeOftask

            # word_list = [131,120,133,128,125,132,125,121] #averageOfcodeword

            #original------------------------------------------------
            # for i in range(task_len):
            #     self.C[str(i)] = {}
            #     for j in range(N_books):
            #         self.C[str(i)][str(j)] = nn.Parameter(
            #             Variable(torch.randn(N_words, L_word).type(torch.float32), requires_grad=True)
            #         )
            #         # self.C[str(i)][str(j)] = nn.Parameter(
            #         #     Variable(torch.randn(word_list[j], L_word).type(torch.float32), requires_grad=True)
            #         # )
            #         nn.init.kaiming_normal_(self.C[str(i)][str(j)], a=math.sqrt(5))
            #original-----------------------------------------------------------------------------------

            #shared codebook-------------------------------------------------
            # self.shared_codebook = nn.Parameter(
            #     torch.randn(N_books, N_words, L_word).type(torch.float32), requires_grad=True
            # )
            # nn.init.kaiming_normal_(self.shared_codebook, a=math.sqrt(5))

            for i in range(N_books):
                self.shared_codebook[str(i)] = nn.Parameter(
                    Variable(torch.randn(N_words, L_word).type(torch.float32), requires_grad=True)
                )
                nn.init.kaiming_normal_(self.shared_codebook[str(i)], a=math.sqrt(5))

            #shared codebook-------------------------------------------------

            self.N_books = N_books
            self.N_words = N_words
            self.L_word = L_word
            self.tau_q = tau_q

        #original-----------------------
        # def forward(self, input,shared_codebook,codeword_dict=None):
        #     if self.N_books==0:
        #         return input,None
        #     X = self.fc(input)
        #     # print(len(self.C[str(task_id)][0]))
        #     #original----------------
        #     # Z = self.Soft_Quantization(X, self.C[str(task_id)], self.N_books, self.tau_q)
        #     #original----------------
        #     return X, Z

        # def Soft_Quantization(self, X, C, N_books, tau_q):
        #     device = self.device  # モデルが配置されているデバイス
        #     X = X.to(device)
        #     x = torch.split(X, self.L_word, dim=1)
        #
        #     for i in range(N_books):
        #         Ci = C[str(i)].to(device)
        #         # print(len(Ci))
        #         # distance = squared_distances(x[i], C[i,:,:]
        #         distance = squared_distances(x[i], Ci)#前からi番目同士で計算させる．
        #         # arg = torch.argmin(distance, dim=1)
        #         # min_idx = torch.reshape(arg, [-1, 1])
        #         soft_c = F.softmax(distance * (-tau_q), dim=-1)
        #         if i==0:
        #             # Z = soft_c @ C[i,:,:]
        #             Z = soft_c @ Ci
        #             # quant_idx = min_idx
        #         else:
        #             # Z = torch.cat((Z, soft_c @ C[i,:,:]), dim=1)
        #             Z = torch.cat((Z, soft_c @ Ci), dim=1)
        #             # quant_idx = torch.cat((quant_idx, min_idx), dim=1)
        #     return Z
        #original--------------------

        #shared-------------------------------------------------

        def expand_codebook(self, additional_words=4):
            new_codewords = torch.randn(self.N_books, additional_words, self.L_word).type(torch.float32).to(self.device)
            nn.init.kaiming_normal_(new_codewords, a=math.sqrt(5))

            for i in range(len(self.shared_codebook)):
                self.shared_codebook[str(i)] = nn.Parameter(
                torch.cat([self.shared_codebook[str(i)], new_codewords[i]], dim=0), requires_grad=True
            )
                # print(self.shared_codebook[str(i)].shape)



            # print("the number of CW:", self.shared_codebook.size())
        # shared-------------------------------------------------



        #shared-------------------------------------------------
        def forward(self, input, codeword_dict=None):
            # print("forward in quantization")
            if self.N_books==0:
                return input,None
            X = self.fc(input)
            # print(len(self.C[str(task_id)][0]))
            # print("Quantization_head::::", self)
            #shared_codebook-----------------------
            Z = self.Soft_Quantization(X, self.shared_codebook, self.N_books, self.tau_q)
            # print(num_epoch)
            # shared_codebook----------------------------------------------------------
            return X, Z

        def Soft_Quantization(self, X, shared_codebook, N_books, tau_q):
            # device = X.device
            # print(self.device)
            device = self.device
            X = X.to(device)
            x = torch.split(X, self.L_word, dim=1)
            for i in range(N_books):

                Ci = shared_codebook[str(i)].to(device)
                # print(x[i].shape, Ci.shape)
                distance = squared_distances(x[i], Ci)
                soft_c = F.softmax(distance * (-tau_q), dim=-1)
                if i == 0:
                    Z = soft_c @ Ci
                else:
                    Z = torch.cat((Z, soft_c @ Ci), dim=1)
            return Z
        #shared---------------------------------

class FowardModuleList(nn.ModuleList):
    def __init__(self):
        super(FowardModuleList, self).__init__()

    def forward(self, x):
        output = [module(x) for module in self]
        return output


# class ClusteringCallback(Callback):
#     def __init__(self,current_loop):
#         super().__init__()
#         log_file_path = 'num_centers_log_10percentDBSCAN_cifar100_epoch100_buffer20_used1aug_all' + str(current_loop) + '.csv'
#         self.log_file_path = log_file_path
#         # CSVファイルのヘッダーを作成
#         if not os.path.exists(self.log_file_path):
#             with open(self.log_file_path, mode='w', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerow(["epoch", "codebook", "num_center"])
#     def on_train_epoch_start(self, trainer, pl_module):
#         percent = 0.1
#         if trainer.current_epoch == trainer.max_epochs*percent-1:
#             model = pl_module
#             train_dataset = trainer.train_dataloader
#             task_features = model.prepare_data_and_features(train_dataset)
#             data = model.quanti_Model.fc(task_features)
#             data = torch.split(data, model.L_word, dim=1)
#             data = [x.cpu().detach().numpy() for x in data]  # このデータをfor x in data:xmeansを行っていくべき
#             # print(model.quanti_Model.C)
#             # centers_list = [[] for _ in range(model.N_books)]
#             for i, x in enumerate(data):
#                 # codeword_centers, len_center = model.initialize_codebooks_xmeans(x, model.N_books, max_clusters=100) #1codebook分
#                 codeword_centers, len_center = model.initialize_codebooks_dbscan(x)
#                 model.quanti_Model.C[str(model.curr_task)][str(i)] = nn.Parameter(codeword_centers, requires_grad=True)
#                 nn.init.kaiming_normal_(model.quanti_Model.C[str(model.curr_task)][str(i)], a=math.sqrt(5))
#                 # print("codebook{0}: {1}".format(i+1, len(codeword_centers)))
#                 with open(self.log_file_path, mode='a', newline='') as file:
#                     writer = csv.writer(file)
#                     writer.writerow([trainer.current_epoch+1, i+1, len_center])
# ---------------------------------------------------------------

class BaseMethod(pl.LightningModule):
    _BACKBONES = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
        "poolformer_s12": poolformer_s12,
        "poolformer_s24": poolformer_s24,
        "poolformer_s36": poolformer_s36,
        "poolformer_m36": poolformer_m36,
        "poolformer_m48": poolformer_m48,
        "convnext_tiny": convnext_tiny,
        "convnext_small": convnext_small,
        "convnext_base": convnext_base,
        "convnext_large": convnext_large,
        "wide_resnet28w2": wide_resnet28w2,
        "wide_resnet28w8": wide_resnet28w8,
    }
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        backbone_args: dict,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lr: float,
        weight_decay: float,
        classifier_lr: float,
        accumulate_grad_batches: Union[int, None],
        extra_optimizer_args: Dict,
        scheduler: str,
        num_large_crops: int,
        num_small_crops: int,
        min_lr: float = 0.0,
        warmup_start_lr: float = 0.00003,
        warmup_epochs: float = 10,
        scheduler_interval: str = "step",
        lr_decay_steps: Sequence = None,
        knn_eval: bool = False,
        knn_k: int = 20,
        no_channel_last: bool = False,
        num_run: int = 1,
        **kwargs,
    ):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.

        Args:
            backbone (str): architecture of the base backbone.
            num_classes (int): number of classes.
            backbone_params (dict): dict containing extra backbone args, namely:
                #! optional, if it's not present, it is considered as False
                cifar (bool): flag indicating if cifar is being used.
                #! only for resnet
                zero_init_residual (bool): change the initialization of the resnet backbone.
                #! only for vit
                patch_size (int): size of the patches for ViT.
            max_epochs (int): number of training epochs.
            batch_size (int): number of samples in the batch.
            optimizer (str): name of the optimizer.
            lr (float): learning rate.
            weight_decay (float): weight decay for optimizer.
            classifier_lr (float): learning rate for the online linear classifier.
            accumulate_grad_batches (Union[int, None]): number of batches for gradient accumulation.
            extra_optimizer_args (Dict): extra named arguments for the optimizer.
            scheduler (str): name of the scheduler.
            num_large_crops (int): number of big crops.
            num_small_crops (int): number of small crops .
            min_lr (float): minimum learning rate for warmup scheduler. Defaults to 0.0.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
                Defaults to 0.00003.
            warmup_epochs (float): number of warmup epochs. Defaults to 10.
            scheduler_interval (str): interval to update the lr scheduler. Defaults to 'step'.
            lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                step. Defaults to None.
            knn_eval (bool): enables online knn evaluation while training.
            knn_k (int): the number of neighbors to use for knn.
            no_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models

        .. note::
            When using distributed data parallel, the batch size and the number of workers are
            specified on a per process basis. Therefore, the total batch size (number of workers)
            is calculated as the product of the number of GPUs with the batch size (number of
            workers).

        .. note::
            The learning rate (base, min and warmup) is automatically scaled linearly based on the
            batch size and gradient accumulation.

        .. note::
            For CIFAR10/100, the first convolutional and maxpooling layers of the ResNet backbone
            are slightly adjusted to handle lower resolution images (32x32 instead of 224x224).

        """

        super().__init__()

        # resnet backbone related
        self.backbone_args = backbone_args

        # training related
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.classifier_lr = classifier_lr
        self.accumulate_grad_batches = accumulate_grad_batches
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        assert scheduler_interval in ["step", "epoch"]
        self.scheduler_interval = scheduler_interval
        self.num_large_crops = num_large_crops
        self.num_small_crops = num_small_crops
        self.knn_eval = knn_eval
        self.knn_k = knn_k
        self.no_channel_last = no_channel_last

        self.task_num = kwargs['Task']
        self.class_per_task = num_classes // self.task_num
        self.acc_matrix=np.zeros((self.task_num,self.task_num))
        self.epoch_per_task = max_epochs
        self.curr_task = 0
        self.buffers = {}
        # self.train_task = kwargs['train_task']
        self.train_task = kwargs['Task']

        self.train_from_task = kwargs['train_from_task']

        self.CodeWord_Feature = []

        self.num_run = num_run

        #Codeword
        # self.C = []
        # self.dimension_reducer = nn.Linear(512, 16)
        # モデル初期化前にデバイスを確認
        #for CAplus
        self.ca_cluster = None
        self.winning_counts = None



        # multicrop
        self.num_crops = self.num_large_crops + self.num_small_crops

        # all the other parameters
        self.extra_args = kwargs

        # turn on multicrop if there are small crops
        self.multicrop = self.num_small_crops != 0

        # if accumulating gradient then scale lr
        if self.accumulate_grad_batches:
            self.lr = self.lr * self.accumulate_grad_batches
            self.classifier_lr = self.classifier_lr * self.accumulate_grad_batches
            self.min_lr = self.min_lr * self.accumulate_grad_batches
            self.warmup_start_lr = self.warmup_start_lr * self.accumulate_grad_batches

        assert backbone in BaseMethod._BACKBONES
        self.base_model = self._BACKBONES[backbone]

        self.backbone_name = backbone

        # initialize backbone
        kwargs = self.backbone_args.copy()
        cifar = kwargs.pop("cifar", False)
        # swin specific
        if "swin" in self.backbone_name and cifar:
            kwargs["window_size"] = 4

        method = self.extra_args.get("method", None)
        self.backbone = self.base_model(method, **kwargs)
        if self.backbone_name.startswith("resnet"):
            self.features_dim = self.backbone.inplanes
            # remove fc layer
            self.backbone.fc = nn.Identity()
            if cifar:
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.backbone.maxpool = nn.Identity()
        else:
            self.features_dim = self.backbone.num_features

        if method in ['agem','gss','si','der','su_Finetune']:
            classifiers = FowardModuleList()
            for i in range(self.task_num):
                classifiers.append(nn.Linear(self.features_dim, self.class_per_task))
            self.classifier = classifiers

        # self.classifier = nn.Linear(self.features_dim, num_classes)

        if self.knn_eval:
            self.knn = WeightedKNNClassifier(k=self.knn_k, distance_fx="euclidean")

        if scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )
        self.run_count = 1
        # CUCL
        self.CUCL = self.extra_args["CUCL"]
        self.LUMP = self.extra_args["LUMP"]
        if self.CUCL or self.LUMP:
            self.buffer_size = self.extra_args["buffer_size"]
            self.LUMP_size = self.extra_args["LUMP_size"]
            self.LUMP_lambda = self.extra_args["LUMP_lambda"]
        if self.CUCL:
            tau_cqc = 0.5
            tau_q = 5
            self.N_books = self.extra_args["N_books"]
            # self.N_words = self.extra_args["N_words"]
            self.N_words = [[] for _ in range(self.task_num)]
            self.L_word = self.extra_args["L_word"]
            self.cqc_criterion = CQCLoss(self.batch_size, tau_cqc)
            self.CUCL_lr = self.extra_args["CUCL_lr"]
            self.sample_type = self.extra_args["sample_type"]
            self.CUCL_cosine = self.extra_args["CUCL_cosine"]
            self.CUCL_lambda = self.extra_args["CUCL_lambda"]
            self.CUCL_epoch = self.extra_args["CUCL_epoch"]
            self.CUCL_type = self.extra_args["CUCL_type"]
            self.CUCL_for_Loss = self.extra_args["CUCL_for_Loss"]
            self.CUCL_loadPath = self.extra_args["CUCL_loadPath"]
            self.codeword_dict = {}
            self.quanti_Model = Quantization_Head(self.N_books, self.L_word, tau_q, self.task_num, self.extra_args["proj_output_dim"])
            # self.quanti_Model = Quantization_Head(self.N_words, self.N_books, self.L_word, tau_q, self.task_num, self.extra_args["proj_output_dim"])

    # x_meansのためのコード -----------------------------------------------------------------------------
    def prepare_data_and_features(self, data_loader):
        features = []
        # self.eval()  # モデルを評価モードに設定
        with torch.no_grad():
            for batch in data_loader:
                # 2番目の要素から画像リストを取得
                image_list = batch[1]
                # 最初と2番目の画像のみを処理する1,2がaugmentationされたもの
                for i, image_tensor in enumerate(image_list):
                    if i > 0:  # 2番目の画像まで処理したらループを終了
                        break
                    if image_tensor.ndim == 3:
                        image_tensor = image_tensor.unsqueeze(0)  # バッチ次元を追加

                    image_tensor = image_tensor.to(self.device)  # デバイスに送る
                    # image_tensor = image_tensor.to("cuda:0")  # デバイスに送る

                    features.append(self.backbone(image_tensor))  # 特徴を抽出してリストに追加
        return torch.cat(features, dim=0)  # 特徴リストを連結して返す

    # def initialize_codebooks_xmeans(self, task_features, N_books, max_clusters, random_state=0):
    #     # TensorをNumPy配列に変換
    #     data = self.quanti_Model.fc(task_features)
    #     data = torch.split(data, self.L_word, dim=1)
    #     data = [x.cpu().detach().numpy() for x in data]#このデータをfor x in data:xmeansを行っていくべき
    #
    #     # self.N_words[self.curr_task] = [] #ここは要確認常に更新されてしまうからね
    #
    #     from pyclustering.cluster.xmeans import xmeans
    #     from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
    #     centers_list = [[] for _ in range(self.N_books)]
    #     for i,x in enumerate(data):
    #         xm_c = kmeans_plusplus_initializer(x, 2).initialize()
    #         xm_i = xmeans(data=x, initial_centers=xm_c, kmax=20, ccore=True)
    #         xm_i.process()
    #         centers = xm_i.get_centers()
    #         centers_list[i] = centers
    #         print("codeword:",i+1)
    #         print("Number of codewords:", len(centers))
    #
    #     # centers_tensor = torch.tensor(centers_list, dtype=torch.float32)
    #     centers_tensors = [torch.tensor(centers, dtype=torch.float32) for centers in centers_list]
    #     #ここはいまなくても問題ない------------------------------
    #     # self.N_words[self.curr_task].append(len(centers))
    #     # print("------")
    #     # print(self.N_words[self.curr_task])
    #     #---------------------------
    #     return centers_tensors

    def initialize_codebooks_xmeans(self, data, N_books, max_clusters, random_state=0):

        xm_c = kmeans_plusplus_initializer(data, 2).initialize()
        xm_i = xmeans(data=data, initial_centers=xm_c, kmax=max_clusters, ccore=True)
        xm_i.process()
        centers = xm_i.get_centers()
        len_center = len(centers)
        centers_tensor = torch.tensor(centers, dtype=torch.float32)
        return centers_tensor, len_center

    def initialize_codebooks_dbscan(self, data, eps=0.5, min_samples=5):
        # DBSCANクラスタリングを実行
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        # クラスタのラベルを取得
        labels = db.labels_

        # ノイズデータを除外
        core_samples_mask = labels != -1

        # クラスタの中心を計算（各クラスタの平均点）
        unique_labels = set(labels) - {-1}
        centers = []
        for k in unique_labels:
            class_member_mask = (labels == k)
            xy = data[class_member_mask & core_samples_mask]
            center = xy.mean(axis=0)
            centers.append(center)

        # 中心点の数を計算
        len_center = len(centers)
        centers_tensor = torch.tensor(centers, dtype=torch.float32)
        return centers_tensor, len_center

    # def initialize_task_codebook(self, train_dataset, task_id):
    #     # データローダーを準備（バッチサイズをデータセットの全サイズに設定）
    #     # data_loader = DataLoader(train_dataset, batch_size=total_size, shuffle=False)
    #     # 特徴抽出
    #     task_features = self.prepare_data_and_features(train_dataset)
    #
    #     # X-meansクラスタリングとself.Cの初期化
    #     centers_tensor = self.initialize_codebooks_xmeans(task_features, self.N_books, max_clusters=20)
    #
    #     self.quanti_Model.C[str(task_id)]=nn.Parameter(centers_tensor, requires_grad=True)
    #
    # # x_meansのためのコード -----------------------------------------------------------------------------

    #original ----------------------
    # def indexing(self, C, X):
    #     device = self.device  # モデルが配置されているデバイス
    #     X = X.to(device)
    #     # C = C.to(device)
    #     x = torch.split(X, self.L_word, 1)
    #     # print(X.device)
    #     # print(C.device)
    #     # print(self.device)
    #
    #     for i in range(self.N_books):
    #         Ci = C[str(i)].to(device)
    #         # diff = squared_distances(x[i], C[i,:,:])
    #         diff = squared_distances(x[i], Ci)
    #         print("###################")
    #         print(Ci.shape)
    #
    #         # diff = squared_distances(x[i], C[str(i)])
    #         arg = torch.argmin(diff, dim=1)
    #         min_idx = torch.reshape(arg, [-1, 1])
    #         if i == 0:
    #             quant_idx = min_idx
    #         else:
    #             quant_idx = torch.cat((quant_idx, min_idx), dim=1)
    #     return quant_idx
    #original ----------------------------------------------------

    #shared_codebook-------------------
    def indexing(self, shared_codebook, X):
        device = self.device  # モデルが配置されているデバイス
        X = X.to(device)
        x = torch.split(X, self.L_word, 1)  # Xを各コードブックに分割 changed

        for i in range(self.N_books):
            # shared_codebookから対応するコードブック（Ci）を取得
            Ci = shared_codebook[str(i)].to(device)  # ここで渡されたshared_codebookを使用

            # x[i]（各データ部分）とCi（対応するコードブック）との距離を計算
            diff = squared_distances(x[i], Ci)
            arg = torch.argmin(diff, dim=1)  # 各データ点に対する最も近いコードワードを取得
            min_idx = torch.reshape(arg, [-1, 1])
            # 結果をquant_idxに連結して蓄積
            if i == 0:
                quant_idx = min_idx
            else:
                quant_idx = torch.cat((quant_idx, min_idx), dim=1)
        return quant_idx
    #shared_codebook-------------------------------------------------------------------
    @torch.no_grad()
    def save_samples(self):
        self.eval()
        images1_bank = []
        images2_bank = []
        notaug_images_bank = []
        quant_idx_bank = []
        dis_bank = []
        target_bank = []
        index_bank = []
        quanti_bank = []
        if self.extra_args['data_format'] == "dali":
            for index, (images1, images2, notaug_images), target in tqdm(self.trainer.datamodule.train_dataloader()):
                notaug_images = notaug_images.to(self.device, non_blocking=True)
                quant_idx,Feature,Quanti_Feature = self.foward_CUCL(notaug_images,self.curr_task)
                if self.sample_type == "Old":
                    x = torch.split(Quanti_Feature, self.L_word, 1)
                    all_dis = 0.0
                    for i in range(self.N_books):
                        # dis = F.pairwise_distance(x[i], self.quanti_Model.C[self.curr_task][i,quant_idx[:,i],:])
                        dis = F.pairwise_distance(x[i], self.quanti_Model.C[str(self.curr_task)][i,quant_idx[:,i],:])

                        # dis_notebook_bank[i] += dis
                        all_dis += dis
                    dis_bank.append(all_dis.cpu())
                else:
                    images1 = images1.to(self.device, non_blocking=True)
                    images2 = images2.to(self.device, non_blocking=True)
                    quant_idx,_,Quanti_Feature_1 = self.foward_CUCL(images1,self.curr_task)
                    quant_idx,_,Quanti_Feature_2 = self.foward_CUCL(images2,self.curr_task)
                    dis = F.pairwise_distance(Quanti_Feature_1, Quanti_Feature_2)
                    dis_bank.append(dis.cpu())

                quant_idx_bank.append(quant_idx.cpu())
                images1_bank.append(images1.cpu())
                images2_bank.append(images2.cpu())
                notaug_images_bank.append(notaug_images.cpu())
                target_bank.append(target.cpu())
                index_bank.append(index.cpu())
                quanti_bank.append(Quanti_Feature.cpu())
        else:
            for index, (images1, images2, notaug_images), target in tqdm(self.trainer.train_dataloader):
                notaug_images = notaug_images.to(self.device, non_blocking=True)
                #check---------------------------------------------------
                quant_idx,Feature,Quanti_Feature = self.foward_CUCL(notaug_images,self.curr_task)
                if self.sample_type == "Old":
                    x = torch.split(Quanti_Feature, self.L_word, 1)
                    all_dis = 0.0
                    for i in range(self.N_books):
                        #check--------------------------------------------------------------------------------------------------
                        dis = F.pairwise_distance(x[i], self.quanti_Model.C[str(self.curr_task)][i,quant_idx[:,i],:])
                        # dis_notebook_bank[i] += dis
                        all_dis += dis
                    dis_bank.append(all_dis)
                else:
                    images1 = images1.to(self.device, non_blocking=True)
                    images2 = images2.to(self.device, non_blocking=True)
                    #chdeck-----------------------------------
                    quant_idx,_,Quanti_Feature_1 = self.foward_CUCL(images1,self.curr_task)
                    quant_idx,_,Quanti_Feature_2 = self.foward_CUCL(images2,self.curr_task)
                    dis = F.pairwise_distance(Quanti_Feature_1, Quanti_Feature_2)
                    dis_bank.append(dis)

                quant_idx_bank.append(quant_idx.cpu())
                images1_bank.append(images1.cpu())
                images2_bank.append(images2.cpu())
                notaug_images_bank.append(notaug_images.cpu())
                target_bank.append(target.cpu())
                index_bank.append(index.cpu())
                quanti_bank.append(Quanti_Feature.cpu())
        # [D, N]
        images1_bank = torch.cat(images1_bank, dim=0).contiguous()
        images2_bank = torch.cat(images2_bank, dim=0).contiguous()
        notaug_images_bank = torch.cat(notaug_images_bank, dim=0).contiguous()
        quant_idx_bank = torch.cat(quant_idx_bank, dim=0).contiguous()
        dis_bank = torch.cat(dis_bank, dim=0).contiguous()
        target_bank = torch.cat(target_bank, dim=0).contiguous()
        index_bank = torch.cat(index_bank, dim=0).contiguous()
        quanti_bank = torch.cat(quanti_bank, dim=0).contiguous()

        if self.sample_type == "Old":
            sor_dis_bank = torch.sort(dis_bank,dim=0,descending=True)
        else:
            sor_dis_bank = torch.sort(dis_bank,dim=0,descending=True)
        if self.sample_type == "random":
            sor_dis_bank_index = torch.randperm(dis_bank.shape[0])[:self.extra_args["buffer_size"]]
        else:
            sor_dis_bank_index = sor_dis_bank[1][:self.extra_args["buffer_size"]].to(images1_bank.device)
        images1_data = images1_bank[sor_dis_bank_index].numpy()
        images2_data = images2_bank[sor_dis_bank_index].numpy()
        notaug_images_data = notaug_images_bank[sor_dis_bank_index].numpy()
        memory_quant_idx = quant_idx_bank[sor_dis_bank_index]
        target_bank = target_bank[sor_dis_bank_index]
        index_bank = index_bank[sor_dis_bank_index]
        quanti_bank = quanti_bank[sor_dis_bank_index].numpy()
        self.buffers[self.curr_task] = ([images1_data,images2_data,notaug_images_data],memory_quant_idx,target_bank,index_bank,quanti_bank)

        if hasattr(self, 'inputs1'):
            del self.inputs1
            del self.inputs2
            del self.notaug_inputs
            del self.quant_idxs
            del self.target_idxs
            del self.index_idxs

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds shared basic arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("base")

        # backbone args
        BACKBONES = BaseMethod._BACKBONES

        parser.add_argument("--backbone", choices=BACKBONES, type=str)
        # extra args for resnet
        parser.add_argument("--zero_init_residual", action="store_true")
        # extra args for ViT
        parser.add_argument("--patch_size", type=int, default=16)

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)

        parser.add_argument("--train_task", type=int, default=10)
        parser.add_argument("--train_from_task", type=int, default=0)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        parser.add_argument(
            "--optimizer", choices=BaseMethod._OPTIMIZERS.keys(), type=str, required=True
        )
        parser.add_argument("--exclude_bias_n_norm_wd", action="store_true")
        # lars args
        parser.add_argument("--grad_clip_lars", action="store_true")
        parser.add_argument("--eta_lars", default=1e-3, type=float)
        parser.add_argument("--exclude_bias_n_norm_lars", action="store_true")
        # adamw args
        parser.add_argument("--adamw_beta1", default=0.9, type=float)
        parser.add_argument("--adamw_beta2", default=0.999, type=float)

        parser.add_argument(
            "--scheduler", choices=BaseMethod._SCHEDULERS, type=str, default="reduce"
        )
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.00003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)
        parser.add_argument(
            "--scheduler_interval", choices=["step", "epoch"], default="step", type=str
        )

        # online knn eval
        parser.add_argument("--knn_eval", action="store_true")
        parser.add_argument("--knn_k", default=100, type=int)
        parser.add_argument("--knn_temperature", default=0.1, type=float)
        parser.add_argument("--knn_distance_function", default="cosine", type=str)
        parser.add_argument("--knn_feature_type", default="backbone", type=str)

        # disables channel last optimization
        parser.add_argument("--no_channel_last", action="store_true")

        # CUCL
        parser.add_argument("--CUCL", action="store_true")
        # backbone using the feature from backbone
        # projector using the feature from projector
        parser.add_argument("--CUCL_type", choices=["backbone", "projector"], default="backbone", type=str)
        parser.add_argument("--CUCL_for_Loss", action="store_true")
        parser.add_argument("--Task", default=10, type=int)
        parser.add_argument('--N_books', default=8, type=int, help="""The number of the codebooks.""")
        # parser.add_argument('--N_words', default=16, type=int, help="""The number of the codewords. It should be a power of two.""")
        parser.add_argument('--L_word', default=16, type=int, help="""Dimensionality of the codeword.""")
        parser.add_argument('--buffer_size', default=256, type=int)
        parser.add_argument('--LUMP_size', default=256, type=int)
        parser.add_argument("--CUCL_lr", type=float, default=0.1)
        parser.add_argument("--sample_type", type=str, default='Old')
        parser.add_argument("--CUCL_cosine", action="store_true")
        parser.add_argument("--CUCL_lambda", type=float, default=1.0)
        parser.add_argument("--CUCL_epoch", type=int, default=1)
        parser.add_argument("--CUCL_loadPath", type=str, default='')
        parser.add_argument("--LUMP", action="store_true")
        parser.add_argument("--LUMP_lambda", type=float, default=0.1)

        return parent_parser

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            # {
            #     "name": "classifier",
            #     "params": self.classifier.parameters(),
            #     "lr": self.classifier_lr,
            #     "weight_decay": 0,
            # },
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """
        learnable_params = self.learnable_params
        if self.CUCL:
            cucl_index = len(learnable_params)
            paras = {"name": "quanti_Model",
                "params": [param for name, param in self.quanti_Model.named_parameters()],
                # "eta":1e-3,
                # "exclude_bias_n_norm":False,
                # "momentum":0,
                # "weight_decay":0,
                # "clip_lars_lr":False,
                "lr":self.CUCL_lr}
            if self.extra_args["method"] == "simclr":
                if not self.CUCL_for_Loss:
                    paras.update({"clip_lars_lr":False,"exclude_bias_n_norm":False})
            elif self.extra_args["method"] == "swav":
                paras.update({"clip_lars_lr":False,"exclude_bias_n_norm":False,})
            elif self.extra_args["method"] == "mocov2plus":
                paras.update({"clip_lars_lr":False,"exclude_bias_n_norm":False,})
            learnable_params.extend([paras])

        # exclude bias and norm from weight decay
        if self.extra_args.get("exclude_bias_n_norm_wd", False):
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        # indexes of parameters without lr scheduler
        idxs_no_scheduler = [i for i, m in enumerate(learnable_params) if m.pop("static_lr", False)]

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        # create optimizer
        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        if self.scheduler.lower() == "none":
            return [optimizer]

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

        if idxs_no_scheduler:
            lrs_to_replace = [self.lr]*len(idxs_no_scheduler)
            if self.CUCL:
                for i,value in enumerate(idxs_no_scheduler):
                    if value == cucl_index:
                        lrs_to_replace[i] = self.CUCL_lr
            partial_fn = partial(
                static_lr,
                get_lr=scheduler["scheduler"].get_lr
                if isinstance(scheduler, dict)
                else scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=lrs_to_replace,
            )
            if isinstance(scheduler, dict):
                scheduler["scheduler"].get_lr = partial_fn
            else:
                scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        """
        This improves performance marginally. It should be fine
        since we are not affected by any of the downsides descrited in
        https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad

        Implemented as in here
        https://pytorch-lightning.readthedocs.io/en/1.5.10/guides/speed.html#set-grads-to-none
        """
        try:
            optimizer.zero_grad(set_to_none=True)
        except:
            optimizer.zero_grad()

        #BaseMethodの中身
    def forward(self, X) -> Dict:
        """Basic forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """
        # print("forward_BaseMethod")
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        # logits = self.classifier(feats.detach())
        return {"logits": None, "feats": feats}

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Basic multicrop forward method that performs the forward pass
        for the multicrop views. Children classes can override this method to
        add new outputs but should still call this function. Make sure
        that this method and its overrides always return a dict.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of features.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        return {"feats": feats}

    def _base_shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        out = self(X)
        # logits = out["logits"]

        # loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # handle when the number of classes is smaller than 5
        # top_k_max = min(5, logits.size(1))
        # acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))

        out.update({"loss": 0, "acc1": 0, "acc5": 0})
        return out

    def base_training_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Allows user to re-write how the forward step behaves for the training_step.
        Should always return a dict containing, at least, "loss", "acc1" and "acc5".
        Defaults to _base_shared_step
        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        return self._base_shared_step(X, targets)

    @staticmethod
    def distance(A, B):
        squareA = (A ** 2).sum(-1, keepdim=True)
        squareB = (B ** 2).sum(-1)
        return squareA + squareB - 2 * A @ B.T

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits.
        """
        # print("training_step_Base")
        _, X, targets = batch

        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.num_crops + 1

        outs = [self.base_training_step(x, targets) for x in X[: self.num_large_crops]]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        if self.multicrop:
            multicrop_outs = [self.multicrop_forward(x) for x in X[self.num_large_crops :]]
            for k in multicrop_outs[0].keys():
                outs[k] = outs.get(k, []) + [out[k] for out in multicrop_outs]

        # loss and stats
        outs["loss"] = sum(outs["loss"]) / self.num_large_crops
        outs["acc1"] = sum(outs["acc1"]) / self.num_large_crops
        outs["acc5"] = sum(outs["acc5"]) / self.num_large_crops

        # metrics = {
        #     "train_class_loss": outs["loss"],
        #     "train_acc1": outs["acc1"],
        #     "train_acc5": outs["acc5"],
        # }

        # self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if self.knn_eval:
            targets = targets.repeat(self.num_large_crops)
            mask = targets != -1
            self.knn(
                train_features=torch.cat(outs["feats"][: self.num_large_crops])[mask].detach(),
                train_targets=targets[mask],
            )

        return outs


    # def save_visualization_per_task(self, shared_codebook, n_books, task_id, run_count):
    #     """
    #     Save visualizations for each task's codebook using t-SNE, with folder organized by run_count.
    #
    #     Args:
    #         shared_codebook (dict): A dictionary where each key is a codebook index and value is a PyTorch tensor.
    #         n_books (int): Number of codebooks.
    #         task_id (int): Current task ID.
    #         run_count (int): Current run number (used to organize folders).
    #     """
    #     # Set the directory to save visualizations
    #     save_dir = f"with_visualization_200epochs_10task_tinyimagenet_10CwforEachtask_CAplus_simclr_euclidean/run_{run_count}/task_{task_id}"
    #     # save_dir = f"visualization_200epochs_10task_tinyimagenet_100CwforEachtask_CAplusMerge_euclidean/run_{run_count}/task_{task_id}"
    #
    #     # save_dir = f"withCAplus_visualization_TinyImageNet/run_{run_count}/task_{task_id}"
    #
    #     os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    #
    #     # Iterate over each codebook
    #     for book_idx in range(n_books):
    #         # Extract the current codebook from the dictionary
    #         codebook = shared_codebook[str(book_idx)].cpu().detach().numpy()
    #
    #         # Determine the appropriate perplexity for t-SNE
    #         n_samples = codebook.shape[0]
    #         perplexity = min(30, max(1, n_samples // 3))
    #
    #         # Apply t-SNE
    #         tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    #         tsne_results = tsne.fit_transform(codebook)
    #
    #         # Plot and save the visualization
    #         plt.figure(figsize=(8, 6))
    #         plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c='blue', alpha=0.7, label=f'Codebook {book_idx}')
    #         plt.title(f"t-SNE Visualization of Codebook {book_idx} for Task {task_id}")
    #         plt.xlabel("t-SNE Dimension 1")
    #         plt.ylabel("t-SNE Dimension 2")
    #         plt.legend()
    #
    #         # Save the figure to the appropriate task folder
    #         save_path = os.path.join(save_dir, f"codebook_{book_idx}.png")
    #         plt.savefig(save_path)
    #         plt.close()

    def save_visualization_per_task(self, shared_codebook, n_books, task_id, run_count):
        """
        Save visualizations for each task's codebook using t-SNE, with folder organized by run_count.

        Args:
            shared_codebook (dict): A dictionary where each key is a codebook index and value is a PyTorch tensor.
            n_books (int): Number of codebooks.
            task_id (int): Current task ID.
            run_count (int): Current run number (used to organize folders).
        """
        import os
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        # Set the directory to save visualizations
        save_dir = f"with_visualization_50CWforeach_CAplus_simsiam_tinyimagenet_200epoch_10tasks/run_{run_count}/task_{task_id}"
        os.makedirs(save_dir, exist_ok=True)

        for book_idx in range(n_books):
            codebook = shared_codebook[str(book_idx)].cpu().detach().numpy()
            n_samples = codebook.shape[0]
            perplexity = min(30, max(1, n_samples // 3))

            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_results = tsne.fit_transform(codebook)

            plt.figure(figsize=(8, 6))
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                        c='blue', alpha=0.7, label=f'Codebook {book_idx}')
            plt.title(f"t-SNE Visualization of Codebook {book_idx} for Task {task_id}")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.legend()

            save_path = os.path.join(save_dir, f"codebook_{book_idx}.png")
            plt.savefig(save_path)
            plt.close()

    def save_codeword_count_to_excel(self, shared_codebook, task_id, run_count, output_file):
        """
        Save the codeword counts for each task and codebook to an Excel spreadsheet, organized by run_count.

        Args:
            shared_codebook (dict): A dictionary where keys are codebook indices (as strings) and
                                    values are tensors representing the codebooks.
            task_id (int): Current task ID to label the row.
            run_count (int): Current run number (used to organize rows in the Excel file).
            output_file (str): Path to the output Excel file.
        """
        # Prepare the data for the spreadsheet
        codeword_counts = {}
        for book_idx, codebook in shared_codebook.items():
            # Count the number of codewords (first dimension of the tensor)
            codeword_counts[f"Codebook_{book_idx}"] = codebook.size(0)

        # Add run count and task information
        codeword_counts["Run"] = run_count
        codeword_counts["Task"] = task_id

        # Convert the data to a DataFrame
        df = pd.DataFrame([codeword_counts])

        try:
            # If the file already exists, append data
            existing_df = pd.read_excel(output_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            # If the file does not exist, create a new one
            combined_df = df

        # Save the DataFrame to an Excel file
        combined_df.to_excel(output_file, index=False)
        # print(f"Codeword counts saved to {output_file}")

    def quantization(self, feature_1, feature_2, metrics, out, targets, batch_idx):
        #original----------------------------------------------------
        # Xa,Za = self.quanti_Model(feature_1,self.curr_task)
        # Xb,Zb = self.quanti_Model(feature_2,self.curr_task)
        #original----------------------------------------------------


        Xa, Za = self.quanti_Model(feature_1) #this is going to forward in Quantization_Head
        Xb, Zb = self.quanti_Model(feature_2)

        out["X_Feat"] = [Xa,Xb]
        out["Z_Feat"] = [Za,Zb]
        quanti_loss = 0.0
        sample_loss = 0.0
        #------calculating quanti_loss from here--------
        if self.curr_task > 0 and self.buffer_size > 0:
            if not hasattr(self, 'inputs1'):
                inputs1 = []
                inputs2 = []
                notaug_inputs = []
                quant_idxs = []
                target_idxs = []
                index_idxs = []
                quanti_feats = []
                for t in range(self.curr_task):
                    (memory_inputs1,memory_inputs2,memory_notaug_inputs),memory_quant_idx, memory_target, memory_index, memory_quanti = self.buffers[t]
                    memory_inputs1 = torch.from_numpy(memory_inputs1).to(self.device,non_blocking=True)
                    memory_inputs2 = torch.from_numpy(memory_inputs2).to(self.device,non_blocking=True)
                    memory_notaug_inputs = torch.from_numpy(memory_notaug_inputs).to(self.device,non_blocking=True)
                    memory_quant_idx = memory_quant_idx.to(self.device,non_blocking=True)
                    memory_target = memory_target.to(self.device,non_blocking=True)
                    memory_index = memory_index.to(self.device,non_blocking=True)
                    memory_quanti = torch.from_numpy(memory_quanti).to(self.device,non_blocking=True)
                    #listを作成している
                    inputs1.append(memory_inputs1)
                    inputs2.append(memory_inputs2)
                    notaug_inputs.append(memory_notaug_inputs)
                    quant_idxs.append(memory_quant_idx)
                    target_idxs.append(memory_target)
                    index_idxs.append(memory_index)
                    quanti_feats.append(memory_quanti)
                self.inputs1 = torch.cat(inputs1,dim=0).contiguous()
                self.inputs2 = torch.cat(inputs2,dim=0).contiguous()
                self.notaug_inputs = torch.cat(notaug_inputs,dim=0).contiguous()
                self.quant_idxs = torch.cat(quant_idxs,dim=0).contiguous()
                self.target_idxs = torch.cat(target_idxs,dim=0).contiguous()
                self.index_idxs = torch.cat(index_idxs,dim=0).contiguous()
                self.quanti_feats = torch.cat(quanti_feats,dim=0).contiguous()

            #simsiamの損失値を計算 複数のクラスがあるけどどうしてるん？
            outputs1 = self.base_training_step(self.inputs1,self.target_idxs)
            outputs2 = self.base_training_step(self.inputs2,self.target_idxs)
            outputs1_Xs = []
            outputs2_Xs = []
            outputs1_Zs = []
            outputs2_Zs = []

            #手法によるがbufferも量子化が行われる
            for t in range(self.curr_task):
                if self.extra_args["method"] == "simsiam":
                    #original----------------------------------------------------
                    outputs1_X,outputs1_Z = self.quanti_Model(outputs1['p'], t)
                    outputs2_X,outputs2_Z = self.quanti_Model(outputs2['p'], t)
                    #original------------------------------------------------------
                    # outputs1_X, outputs1_Z = self.quanti_Model(outputs1['p'], self.shared_codebook)
                    # outputs2_X, outputs2_Z = self.quanti_Model(outputs2['p'], self.shared_codebook)
                elif self.extra_args["method"] == "barlow_twins":
                    outputs1_X,outputs1_Z = self.quanti_Model(outputs1['z'],t)
                    outputs2_X,outputs2_Z = self.quanti_Model(outputs2['z'],t)
                elif self.extra_args["method"] == "simclr":
                    out["memory_index"] = self.index_idxs
                    outputs1_X,outputs1_Z = self.quanti_Model(outputs1['z'],t)
                    outputs2_X,outputs2_Z = self.quanti_Model(outputs2['z'],t)
                # elif self.extra_args["method"] == "byol":
                #     outputs1_X,outputs1_Z = self.quanti_Model(outputs1['p'],t)
                #     outputs2_X,outputs2_Z = self.quanti_Model(outputs2['p'],t)
                #     momentum_1 = self.momentum_forward(self.inputs1)
                #     momentum_2 = self.momentum_forward(self.inputs2)
                #     outputs1["momentum_z"] = momentum_1["z"]
                #     outputs2["momentum_z"] = momentum_2["z"]
                # elif self.extra_args["method"] == "mocov2plus":
                #     outputs1_X,outputs1_Z = self.quanti_Model(outputs1['z'],t)
                #     outputs2_X,outputs2_Z = self.quanti_Model(outputs2['z'],t)
                #     momentum_1 = self.momentum_forward(self.inputs1)
                #     momentum_2 = self.momentum_forward(self.inputs2)
                #     outputs1["momentum_z"] = momentum_1["z"]
                #     outputs2["momentum_z"] = momentum_2["z"]
                # elif self.extra_args["method"] == "swav":
                #     outputs1_X,outputs1_Z = self.quanti_Model(outputs1['z'],t)
                #     outputs2_X,outputs2_Z = self.quanti_Model(outputs2['z'],t)
                
                outputs1_Xs.append(outputs1_X)
                outputs2_Xs.append(outputs2_X)
                outputs1_Zs.append(outputs1_Z)
                outputs2_Zs.append(outputs2_Z)

            final_Xa = torch.cat((Xa,outputs1_X), dim=0).contiguous()
            final_Xb = torch.cat((Xb,outputs2_X), dim=0).contiguous()
            final_Za = torch.cat((Za,outputs1_Z), dim=0).contiguous()
            final_Zb = torch.cat((Zb,outputs2_Z), dim=0).contiguous()
            targets = torch.cat((targets, self.target_idxs), dim=0)


            quanti_loss += self.cqc_criterion(final_Xa, final_Xb, final_Za, final_Zb, targets)

            out["memory_out1"] = outputs1
            out["memory_out2"] = outputs2
        else:
            quanti_loss += self.cqc_criterion(Xa, Xb, Za, Zb, targets)
        #----------to here---------------

        metrics['quanti_loss'+f'_{self.curr_task}'] = quanti_loss
        if sample_loss > 0.0:
            metrics['sample_loss'+f'_{self.curr_task}'] = sample_loss

        cosine_loss = 0.0
        if self.CUCL_cosine:
            Za_s = torch.split(Xa, self.L_word, dim=1)
            Zb_s = torch.split(Xb, self.L_word, dim=1)
            for i in range(self.N_books):
                Za_sn = F.normalize(Za_s[i],p=2,dim=1)
                dots = torch.mm(Za_sn.data, Za_sn.data.t())
                n = Za_sn.data.shape[0]
                dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
                _, I = torch.max(dots, 1)  # max inner prod -> min distance
                distances = F.pairwise_distance(Za_sn, Za_sn[I])
                diss = - torch.log(n * distances).mean()
                cosine_loss += diss*0.01

                Zb_sn = F.normalize(Zb_s[i],p=2,dim=1)
                dots = torch.mm(Zb_sn.data, Zb_sn.data.t())
                n = Zb_sn.data.shape[0]
                dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
                _, I = torch.max(dots, 1)  # max inner prod -> min distance
                distances = F.pairwise_distance(Zb_sn, Zb_sn[I])
                diss = - torch.log(n * distances).mean()
                cosine_loss += diss*0.01

            if cosine_loss > 0.0:
                metrics['cosine_loss'+f'_{self.curr_task}'] = cosine_loss
        # print(f'cosine_loss:{cosine_loss}')
        out["cosine_loss"] = cosine_loss
        out["quanti_loss"] = quanti_loss
        out["sample_loss"] = sample_loss
        current_task = self.curr_task
        # print(batch_idx)

        # Below is for Tsne for CW and merge with CA+
        if self.trainer.current_epoch == 199 and batch_idx == 18:
        # if self.trainer.current_epoch == 99 and batch_idx == 8:

            #
            # 元の shared_codebook を辞書として取得
            shared_codebook = self.quanti_Model.shared_codebook

            # 新しいコードブックを格納するリスト
            new_codebook = []
            #
            # clustering_flag = False
            clustering_flag = True

        # # クラスタリング適用

            if clustering_flag:

                for book_idx in range(len(shared_codebook)):  # N_books 分ループ
                    # 現在のコードブックを NumPy 配列に変換（CPU 上で操作）


                    current_codebook = shared_codebook[str(book_idx)].cpu().detach().numpy()
                    # クラスタリングの適用
                    self.ca_cluster = ClusterCAplus()

                    self.ca_cluster.fit(current_codebook)
                    centroids = list(self.ca_cluster._CAplus__get_node_attributes_from('weight', list(self.ca_cluster.G_.nodes)))
                    new_codebook.append(np.array(centroids))



            else:
                for book_idx in range(len(shared_codebook)):  # N_books 分ループ
                    # 現在のコードブックを NumPy 配列に変換（CPU 上で操作）
                    current_codebook = shared_codebook[str(book_idx)].cpu().detach().numpy()
                    centroids = current_codebook
                # 新しいリスト構造に追加（NumPy 配列として）
                    new_codebook.append(np.array(centroids))

            # GPU デバイスを明示的に指定
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 新しいコードブックを辞書型に再設定し GPU に戻す
            for book_idx, centroids_np in enumerate(new_codebook):
                # NumPy 配列から PyTorch テンソルを作成し、明示的に GPU に送る
                tensor_codebook = torch.from_numpy(centroids_np).float().to(device)

                # shared_codebook に戻す
                self.quanti_Model.shared_codebook[str(book_idx)] = torch.nn.Parameter(
                    tensor_codebook,
                    requires_grad=True  # 学習可能に設定
                )

            # # タスク ID と共有コードブックを指定して可視化を保存
            self.save_visualization_per_task(self.quanti_Model.shared_codebook, self.quanti_Model.N_books, current_task, self.num_run)

            # コードワード数を Excel に保存
            self.save_codeword_count_to_excel(self.quanti_Model.shared_codebook, current_task, self.num_run, output_file="codewords_counts_200epochs_10task_tinyimagenet_50CWforeachTask_CAplus_euclidean_simsiam.xlsx")

            # self.save_codeword_count_to_excel(self.quanti_Model.shared_codebook, current_task, self.num_run, output_file="codeword_counts_200epochs_10task_cifar100_10CWforallTask_simsiam_euclidean.xlsx")



class BaseMomentumMethod(BaseMethod):
    def __init__(
        self,
        base_tau_momentum: float,
        final_tau_momentum: float,
        momentum_classifier: bool,
        **kwargs,
    ):
        """Base momentum model that implements all basic operations for all self-supervised methods
        that use a momentum backbone. It adds shared momentum arguments, adds basic learnable
        parameters, implements basic training and validation steps for the momentum backbone and
        classifier. Also implements momentum update using exponential moving average and cosine
        annealing of the weighting decrease coefficient.

        Args:
            base_tau_momentum (float): base value of the weighting decrease coefficient (should be
                in [0,1]).
            final_tau_momentum (float): final value of the weighting decrease coefficient (should be
                in [0,1]).
            momentum_classifier (bool): whether or not to train a classifier on top of the momentum
                backbone.
        """

        super().__init__(**kwargs)

        # momentum backbone
        kwargs = self.backbone_args.copy()
        cifar = kwargs.pop("cifar", False)
        # swin specific
        if "swin" in self.backbone_name and cifar:
            kwargs["window_size"] = 4

        method = self.extra_args.get("method", None)
        self.momentum_backbone = self.base_model(method, **kwargs)
        if self.backbone_name.startswith("resnet"):
            # remove fc layer
            self.momentum_backbone.fc = nn.Identity()
            if cifar:
                self.momentum_backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.momentum_backbone.maxpool = nn.Identity()
        else:
            self.features_dim = self.momentum_backbone.num_features

        initialize_momentum_params(self.backbone, self.momentum_backbone)

        # momentum classifier
        if momentum_classifier:
            self.momentum_classifier: Any = nn.Linear(self.features_dim, self.num_classes)
        else:
            self.momentum_classifier = None

        # momentum updater
        self.momentum_updater = MomentumUpdater(base_tau_momentum, final_tau_momentum)

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Adds momentum classifier parameters to the parameters of the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        momentum_learnable_parameters = []
        if self.momentum_classifier is not None:
            momentum_learnable_parameters.append(
                {
                    "name": "momentum_classifier",
                    "params": self.momentum_classifier.parameters(),
                    "lr": self.classifier_lr,
                    "weight_decay": 0,
                }
            )
        return super().learnable_params + momentum_learnable_parameters

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Defines base momentum pairs that will be updated using exponential moving average.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """

        return [(self.backbone, self.momentum_backbone)]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic momentum arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parent_parser = super(BaseMomentumMethod, BaseMomentumMethod).add_model_specific_args(
            parent_parser
        )
        parser = parent_parser.add_argument_group("base")

        # momentum settings
        parser.add_argument("--base_tau_momentum", default=0.99, type=float)
        parser.add_argument("--final_tau_momentum", default=1.0, type=float)
        parser.add_argument("--momentum_classifier", action="store_true")

        return parent_parser

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Momentum forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.momentum_backbone(X)
        return {"feats": feats}

    def _shared_step_momentum(self, X: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Forwards a batch of images X in the momentum backbone and optionally computes the
        classification loss, the logits, the features, acc@1 and acc@5 for of momentum classifier.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict[str, Any]:
                a dict containing the classification loss, logits, features, acc@1 and
                acc@5 of the momentum backbone / classifier.
        """

        out = self.momentum_forward(X)

        if self.momentum_classifier is not None:
            feats = out["feats"]
            logits = self.momentum_classifier(feats)

            loss = F.cross_entropy(logits, targets, ignore_index=-1)
            acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
            out.update({"logits": logits, "loss": loss, "acc1": acc1, "acc5": acc5})

        return out

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding the crops in the momentum backbone
        and classifier, and computing statistics.
        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: a dict with the features of the momentum backbone and the classification
                loss and logits of the momentum classifier.
        """

        outs = super().training_step(batch, batch_idx)

        _, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X

        # remove small crops
        X = X[: self.num_large_crops]

        momentum_outs = [self._shared_step_momentum(x, targets) for x in X]
        momentum_outs = {
            "momentum_" + k: [out[k] for out in momentum_outs] for k in momentum_outs[0].keys()
        }

        if self.momentum_classifier is not None:
            # momentum loss and stats
            momentum_outs["momentum_loss"] = (
                sum(momentum_outs["momentum_loss"]) / self.num_large_crops
            )
            momentum_outs["momentum_acc1"] = (
                sum(momentum_outs["momentum_acc1"]) / self.num_large_crops
            )
            momentum_outs["momentum_acc5"] = (
                sum(momentum_outs["momentum_acc5"]) / self.num_large_crops
            )

            metrics = {
                "train_momentum_class_loss": momentum_outs["momentum_loss"],
                "train_momentum_acc1": momentum_outs["momentum_acc1"],
                "train_momentum_acc5": momentum_outs["momentum_acc5"],
            }
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            # adds the momentum classifier loss together with the general loss
            outs["loss"] += momentum_outs["momentum_loss"]

        outs.update(momentum_outs)
        return outs

    def on_train_batch_end(self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.

        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.
        """

        if self.trainer.global_step > self.last_step:
            # update momentum backbone and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # log tau momentum
            self.log("tau", self.momentum_updater.cur_tau)
            # update tau
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step,
                max_steps=self.trainer.estimated_stepping_batches,
            )
        self.last_step = self.trainer.global_step
