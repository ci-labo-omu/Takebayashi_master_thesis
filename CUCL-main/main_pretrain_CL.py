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
# PURP AOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# # FORNY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from ast import arg
from http.client import MOVED_PERMANENTLY
from lib2to3.pgen2.token import N_TOKENS
import os

# ###
# import sys
# import os
#
# # プロジェクトのルートディレクトリを取得
# project_root = os.path.abspath(os.path.dirname(__file__))
#
# # ルートディレクトリを sys.path に追加
# if project_root not in sys.path:
#     sys.path.append(project_root)
# ###

import pandas as pd
from pprint import pprint
from random import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch,copy

# print(torch.cuda.is_available())
import torchvision
import torchvision.transforms as transforms ##add for downloading data
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from main_knn import extract_features, run_knn
from solo.args.setup import parse_args_pretrain
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
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous
import torch.nn.functional as F 
import torchvision.transforms as T
from solo.utils.plot_codebook import plot_codebook
from prepareTinyimagenet import SplitTinyImageNet

import wandb

try:
    from solo.data.dali_dataloader import PretrainDALIDataModule
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

#clustering process
# from solo.methods.base import ClusteringCallback

def main(num_run):
    seed_everything(num_run+1)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

    #------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
    #                                          download=True, transform=transform)
    # testset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                         download=True, transform=transform)
    #---------
    # train_dataset = SplitTinyImageNet(root='./data/tiny-imagenet-200', train=True, transform=transform)
    # test_dataset = SplitTinyImageNet(root='./data/tiny-imagenet-200', train=False, transform=transform)



    args = parse_args_pretrain()

    args.num_run = num_run

    ##for tsne-----------------------------------------------------
    all_dataset = prepare_datasets(
        args.dataset,
        transform,
        train_data_path=args.train_data_path,
        data_format=args.data_format,
        no_labels=args.no_labels,
        data_fraction=args.data_fraction,
    )

    all_data_loader = prepare_dataloader(
        all_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


    ##for tsne-----------------------------------------------------

    assert args.method in METHODS, f"Choose from {METHODS.keys()}"

    if args.num_large_crops != 2:
        assert args.method in ["wmse", "mae"]


    #data formation

    model = METHODS[args.method](**args.__dict__)

    # print("Model:", model)
    make_contiguous(model)
    # can provide up to ~20% speed up
    if not args.no_channel_last:
        model = model.to(memory_format=torch.channels_last)


    # pretrain dataloader
    if args.data_format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."

        dali_datamodule = PretrainDALIDataModule(
            dataset=args.dataset,
            train_data_path=args.train_data_path,
            unique_augs=args.unique_augs,
            transform_kwargs=args.transform_kwargs,
            num_crops_per_aug=args.num_crops_per_aug,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            no_labels=args.no_labels,
            data_fraction=args.data_fraction,
            dali_device=args.dali_device,
            encode_indexes_into_labels=args.encode_indexes_into_labels,
        )
        # dali_datamodule.val_dataloader = lambda: val_loader
    else:
        transform_kwargs = (
            args.transform_kwargs if args.unique_augs > 1 else [args.transform_kwargs]
        )
        transform = prepare_n_crop_transform(
            [prepare_transform(args.dataset, **kwargs) for kwargs in transform_kwargs],
            num_crops_per_aug=args.num_crops_per_aug,
        )
        transform_mixup = copy.deepcopy(transform.transforms[0].transform.transform.transforms)
        transform_mixup.insert(0,torchvision.transforms.ToPILImage())
        model.transform = torchvision.transforms.Compose(transform_mixup)
        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)
        
    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    if args.auto_resume and args.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(args.checkpoint_dir, args.method),
            max_hours=args.auto_resumer_max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(args)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    # callbacks = [ClusteringCallback(num_run)]
    callbacks = []


    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, args.method),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    if args.auto_umap:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            args,
            logdir=os.path.join(args.auto_umap_dir, args.method),
            frequency=args.auto_umap_frequency,
        )
        callbacks.append(auto_umap)

    # wandb logging
    # wandb by myself
    wandb_logger = WandbLogger(
        # name="LUMP_cifar100_epoch200_buffer256_seed" + str(num_run),
        # name="CUCL_20tasks_tiyimagenet_5CWforEachTask_CAplusMerge_buffer20_cosine_random_epoch100_seed"+str(num_run),
        name="CUCL_10tasks_tinyimagenet_50CWforeach_CAplus_buffer20_random_epoch200_simsiam_seed" + str(num_run),
        # name="CUCL_10tasks_tinyImagenet_100CWforEachTask_CAplusMerge_buffer20_random_epoch200_seed" + str(num_run),
        # name="take2_CUCL_10tasks_cifar100_10CWforAllTask_buffer20_random_epoch200_seed" + str(num_run),
    #     --LUMP - -LUMP_lambda
    # 0.1 - -LUMP_size
    # 256

        # name="CUCL_10tasks_tinyimagenet_10CWforEachTask_buffer20_random_epoch200_simclr_seed" + str(num_run),
        # project="CUCL_SimSiamOnly",#nameより規模が大きい
        # project="CUCL_fixed",  # nameより規模が大きい
        project="CUCL_additional_experiment",
        # project = "check",
        entity="takanori_takebayashi",
        offline=False,
        resume="allow",
        id=wandb.util.generate_id()
    )

    """data name
    cifar100
    tinyimagenet
    """

    # wandb_logger = None
    # if args.wandb:
    #     wandb_logger = WandbLogger(
    #         name=args.name,
    #         project=args.project,
    #         entity=args.entity,
    #         offline=args.offline,
    #         resume="allow" if wandb_run_id else None,
    #         id=wandb_run_id,
    #     )
    #     wandb_logger.watch(model, log="gradients", log_freq=100)
    #     wandb_logger.log_hyperparams(args)
    #
    #     os.system(f'cp -r solo/* {wandb_logger.experiment.dir}')
    #     os.system(f'cp -r main_pretrain_CL.py {wandb_logger.experiment.dir}')
    #
    #     # lr logging
    #     lr_monitor = LearningRateMonitor(logging_interval="epoch")
    #     callbacks.append(lr_monitor)

    class_per_task = model.class_per_task #タスクごとにいくつクラスあるか
    acc_matrix=np.zeros((model.task_num,model.task_num)) #[*range(5)]だと[0,1,2,3,4]となる．

    sample_orders = [*range(model.task_num)]
    for i in range(model.task_num):
        if i > model.train_task-1:
            break
        model.train()
        trainer = Trainer.from_argparse_args(
            args, 
            # logger=wandb_logger if args.wandb else None,
            logger = wandb_logger,
            callbacks=callbacks,
            enable_checkpointing=False,
            detect_anomaly=True,
            strategy=DDPStrategy(find_unused_parameters=False)
            if args.strategy == "ddp"
            else args.strategy,
            check_val_every_n_epoch = model.max_epochs + 1,
        )
        train = True
        if i < args.train_from_task:
            # model_path
            path = os.path.join(model.CUCL_loadPath, f"{i}.pth")
            # if i == 0:
            #     path = os.path.join('./checkpoints/cifar100_results/byol_CUCL_testp-3p2edh4k', f"{i}.pth")
            save_dict = torch.load(path, map_location='cpu')
            buffer = model.buffers
            msg = model.load_state_dict(save_dict['state_dict'], strict=True)
            model.buffers = buffer
            make_contiguous(model)
            if not args.no_channel_last:
                model = model.to(memory_format=torch.channels_last)
            train = False
        model.curr_task = i
        ##add codeword for each task-------------------------
        if i != 0:
            model.quanti_Model.expand_codebook(additional_words=50)
        #------------------------------------------------------------
        if args.data_format == "dali":
            order = sample_orders[i]
            dali_datamodule.class_order = (order*class_per_task,(order+1)*class_per_task)
            trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
        else:

            train_dataset = prepare_datasets(args.dataset,transform,train_data_path=args.train_data_path,data_format=args.data_format,\
                no_labels=args.no_labels,data_fraction=args.data_fraction,)
            #which task samples to use
            order = sample_orders[i]
            train_mask = np.logical_and(np.array(train_dataset.targets) >= order*class_per_task,
                np.array(train_dataset.targets) < (order+1)*class_per_task)
            if args.dataset == 'imagenet100':
                train_dataset.samples = np.array(train_dataset.samples)[train_mask]
                train_dataset.imgs = np.array(train_dataset.imgs)[train_mask]
            else:
                train_dataset.data = np.array(train_dataset.data)[train_mask]
                train_dataset.targets = np.array(train_dataset.targets)[train_mask]
            train_loader = prepare_dataloader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

            if train:   #training starts from here   ここの部分が大事
                print(f'Training on task {i} samples {order}')
                model.train_loader = train_loader
                model.train_dataset = train_dataset #train_dataset is already transformed
                # model.initialize_and_use_quantization(train_dataset, i) #i is task_id
                trainer.fit(model, train_loader, ckpt_path=ckpt_path)
            else:
                trainer.train_dataloader = train_loader
                model.trainer = trainer

        if args.method in ['agem','gss','su_Finetune']:
            evaluate(args, model, i, sample_orders[:i+1], class_per_task, acc_matrix, wandb_logger)
        else:
            if args.method in ['si','der'] and args.supervised:
                evaluate(args, model, i, sample_orders[:i+1], class_per_task, acc_matrix, wandb_logger)
            else:
                test_model(args, model, i, sample_orders[:i+1], class_per_task, acc_matrix, num_run,  wandb_logger)

        # 学習した全データの特徴分布を確認するコード-----------------------------------------------------------------------
        # フィルタリングされたデータセットを作成
        # type = "only_current"
        # if type == "only_current":
        #     current_order = sample_orders[i]  # 現在のタスクiのクラスのみを取得
        #     current_task_mask = np.array(all_dataset.targets) // class_per_task == current_order  # 現在のタスクのクラスに対応するデータだけを取得
        # elif type == "to_current":
        #     current_order = sample_orders[:i + 1]
        #     current_task_mask = np.isin(np.array(all_dataset.targets) // class_per_task, current_order)

        # filtered_data = np.array(all_dataset.data)[current_task_mask]
        # filtered_targets = np.array(all_dataset.targets)[current_task_mask]
        #
        # # フィルタしたデータセットを基にデータローダーを再作成
        # filtered_dataset = copy.deepcopy(all_dataset)
        # filtered_dataset.data = filtered_data
        # filtered_dataset.targets = filtered_targets

#---------------------------------------------------------------------------
        # current_order = sample_orders[i]  # 現在のタスクiのクラスのみを取得
        #
        # # テストデータセットをフィルタリング（current_order に対応するクラスのみ）
        # current_test_mask = np.array(test_dataset.targets) // class_per_task == current_order
        #
        # filtered_test_data = np.array(test_dataset.data)[current_test_mask]
        # filtered_test_targets = np.array(test_dataset.targets)[current_test_mask]
        #
        # filtered_test_dataset = copy.deepcopy(test_dataset)
        # filtered_test_dataset.data = filtered_test_data
        # filtered_test_dataset.targets = filtered_test_targets
        # ---------------------------------------------------------------------------

        # DataLoaderの作成
        # current_data_loader = DataLoader(
        #     filtered_dataset,
        #     batch_size=args.batch_size,
        #     shuffle=False,
        #     num_workers=args.num_workers,
        #     pin_memory=True
        # )

        # _, test_dataset = prepare_datasets_classification(
        #     args.dataset,
        #     T_train=transform,  # あるいは T_train=T, など
        #     T_val=transform,
        #     train_data_path=args.train_data_path,
        #     val_data_path=args.val_data_path,
        #     data_format=args.data_format,
        # )
        #
        # # 2) テストデータに対する DataLoader を用意
        # test_data_loader = DataLoader(
        #     test_dataset,
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     shuffle=False,
        #     pin_memory=True
        # )
        #
        #
        # # t-SNEでの可視化: データローダーを直接使った方法
        # print(f"Visualizing feature distribution using current_data_loader after task {i}...")
        #
        #
        # # 4) t-SNE の可視化関数に渡す
        # visualize_feature_distribution(model, test_data_loader, save_path="test_data_simsiam_CUCL_10CWforeach_CAplus_cifar100_10tasks.png")
        #
        #
        #
        #
        # # 学習した全データの特徴分布を確認するコード-----------------------------------------------------------------------
        #
        if model.CUCL:
            if not args.data_format == "dali":
                calculate_Codeword(args, model, train_loader, i)
            if model.buffer_size > 0 and i < model.task_num - 1:
                model.save_samples()

        if train:
            print("Debug: Saving model...")
            print("Checkpoint directory:", args.ckpt_dir)
            print("Experiment name:", wandb_logger.name)
            print("Wandb version:", wandb_logger.version)
            print("Task index:", i)

            if wandb_logger.version is not None:
                model_path = os.path.join(args.ckpt_dir, wandb_logger.name + '-' + wandb_logger.version, f"{i}.pth")
            else:
                model_path = os.path.join(args.ckpt_dir, wandb_logger.name, f"{i}.pth")

            directory_path = os.path.dirname(model_path)
            os.makedirs(directory_path, exist_ok=True)
            dict = {'state_dict': model.state_dict()}
            torch.save(dict, model_path)
            print(f"Task Model saved to {model_path}")

#### for tsne
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import torch

def visualize_feature_distribution(test_features, test_targets, save_path=None):
    """
    t-SNE を使って特徴量分布を可視化する関数
    Args:
        test_features (dict): {"backbone": Tensor, "projector": Tensor} の辞書
        test_targets (Tensor): ラベルの Tensor
        save_path (str): 画像の保存パス
    """
    # `backbone` の特徴量を使う
    features = test_features["backbone"].cpu().numpy()
    labels = test_targets.cpu().numpy()

    # t-SNE で 2次元に可視化
    tsne = TSNE(n_components=2, random_state=0)
    transformed = tsne.fit_transform(features)

    # プロット
    plt.figure(figsize=(10, 8))
    plt.scatter(transformed[:, 0], transformed[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar()
    plt.title("Feature distribution across tasks")

    # save_path が指定されていたらファイル保存
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

def visualize_task_distribution(test_features, test_targets, class_per_task, save_path=None):
    """
    t-SNE を使ってタスク単位の特徴量分布を可視化する関数
    Args:
        test_features (dict): {"backbone": Tensor, "projector": Tensor} の辞書
        test_targets (Tensor): クラスのラベル (例: CIFAR-100なら 0~99)
        class_per_task (int): 1タスクあたりのクラス数
        save_path (str): 画像の保存パス
    """
    # `backbone` の特徴量を使う
    features = test_features["backbone"].cpu().numpy()

    # クラスIDからタスクIDに変換 (例: CIFAR-100で10クラス/タスクなら、0~9 → タスク0, 10~19 → タスク1 ...)
    task_labels = (test_targets.cpu().numpy() // class_per_task).astype(int)
    print(task_labels)
    # t-SNE で 2次元に可視化
    tsne = TSNE(n_components=2, random_state=0)
    transformed = tsne.fit_transform(features)

    # プロット
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(transformed[:, 0], transformed[:, 1], c=task_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label="Task ID")
    plt.title("Feature distribution across tasks")

    # save_path が指定されていたらファイル保存
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()




# def visualize_feature_distribution(model, data_loader, save_path=None):
#     model.eval()
#     features = []
#     labels = []
#
#     with torch.no_grad():
#         for batch in data_loader:
#             x = batch[0].to(model.device)  # 画像データ
#             y = batch[1]  # ラベルデータ
#
#             # エンコーダから特徴量を取得
#             feature = model.backbone(x)   # ここでは backbone を呼び出している
#             features.append(feature.cpu())
#             labels.append(y.cpu())
#
#     # データが空でないかチェック
#     if len(features) == 0:
#         print("Error: No features extracted! Check the data loader.")
#         return
#
#     # データをプロット用に準備
#     features = torch.cat(features).numpy()
#     labels = torch.cat(labels).numpy()
#
#     # t-SNE で 2次元に可視化
#     tsne = TSNE(n_components=2, random_state=0)
#     transformed = tsne.fit_transform(features)
#
#     # プロット
#     plt.figure(figsize=(10, 8))
#     plt.scatter(transformed[:, 0], transformed[:, 1], c=labels, cmap='tab10', alpha=0.7)
#     plt.colorbar()
#     plt.title("Feature distribution across tasks")
#
#     # save_path が指定されていたらファイル保存
#     if save_path is not None:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Figure saved to {save_path}")
#
#     plt.show()


    # if train:
        #     # model_path = os.path.join(args.ckpt_dir, args.name + '-' + wandb_logger.version if wandb_logger.version else '', f"{i}.pth")
        #     if wandb_logger.version is not None:
        #         model_path = os.path.join(args.ckpt_dir, args.name + '-' + wandb_logger.version, f"{i}.pth")
        #     else:
        #         model_path = os.path.join(args.ckpt_dir, args.name, f"{i}.pth")
        #
        #     os.makedirs(os.path.join(args.ckpt_dir, args.name + '-' + wandb_logger.version if wandb_logger.version else ''),exist_ok=True)
        #     dict = {'state_dict':model.state_dict(),}
        #     torch.save(dict, model_path)
        #     print(f"Task Model saved to {model_path}")

######################################################################################
#-CUCL version? to lump?---------------------------------------------------------------------------------
# @torch.no_grad()
# def test_model(args, model, task_id, sample_orders, N_CLASSES_PER_TASK, acc_matrix, wandb_logger):
#     model.eval()
#     model = model.cuda()
#
#     # 全タスクの特徴を保存するためのリスト
#     all_train_features = {"backbone": [], "projector": []}
#     all_train_targets = []
#
#
#     for i in range(task_id + 1):
#         # prepare data
#         _, T = prepare_transforms_classification(args.dataset)
#         # print("1:", T)
#         train_dataset, _ = prepare_datasets_classification(
#             args.dataset,
#             T_train=T,
#             T_val=T,
#             train_data_path=args.train_data_path,
#             val_data_path=args.val_data_path,
#             data_format=args.data_format,
#         )
#
#         order = sample_orders[i]
#
#         # タスクごとのデータセットをフィルタリング
#         train_mask = np.logical_and(np.array(train_dataset.targets) >= order * N_CLASSES_PER_TASK,
#                                     np.array(train_dataset.targets) < (order + 1) * N_CLASSES_PER_TASK)
#
#
#         # タスクごとのデータを設定
#         train_dataset.data = np.array(train_dataset.data)[train_mask]
#         train_dataset.targets = np.array(train_dataset.targets)[train_mask]
#         # train_dataset.targetsのクラス（ユニークな値）を取得
#         unique_classes = np.unique(train_dataset.targets)
#
#         # クラスを出力
#         # print("Train Dataset Targets Classes:", unique_classes)
#         # train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order * N_CLASSES_PER_TASK
#
#
#         train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
#                                   num_workers=args.num_workers, pin_memory=True)
#
#         # 現在のタスクの特徴を抽出
#         train_features_bb, train_features_proj, train_targets = extract_features(train_loader, model)
#         train_features = {"backbone": train_features_bb, "projector": train_features_proj}
#
#         # 全タスクの特徴ベクトルに蓄積
#         all_train_features["backbone"].append(train_features["backbone"])
#         all_train_features["projector"].append(train_features["projector"])
#         all_train_targets.append(train_targets)
#     # 全てのタスクの特徴を連結
#     all_train_features["backbone"] = torch.cat(all_train_features["backbone"])
#     all_train_features["projector"] = torch.cat(all_train_features["projector"])
#     all_train_targets = torch.cat(all_train_targets)
#
#     # val_num = len(val_dataset.targets)
#     # 各タスクごとに評価を実施
#     for i in range(task_id + 1):
#         _, val_dataset = prepare_datasets_classification(
#             args.dataset,
#             T_train=T,
#             T_val=T,
#             train_data_path=args.train_data_path,
#             val_data_path=args.val_data_path,
#             data_format=args.data_format,
#         )
#         # タスクごとのテストデータを設定
#         order = sample_orders[i]
#
#         val_mask = np.logical_and(np.array(val_dataset.targets) >= order * N_CLASSES_PER_TASK,
#                                   np.array(val_dataset.targets) < (order + 1) * N_CLASSES_PER_TASK)
#         val_dataset.data = np.array(val_dataset.data)[val_mask]
#         val_dataset.targets = np.array(val_dataset.targets)[val_mask]
#
#         # val_dataset.targets = np.array(val_dataset.targets)[val_mask] - order * N_CLASSES_PER_TASK
#
#         val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
#                                 pin_memory=True)
#
#         # 現在のタスクのテスト特徴を抽出
#         test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model)
#         test_features = {"backbone": test_features_bb, "projector": test_features_proj}
#         # k-NNを実行（全ての蓄積された特徴を使う）
#         feat_type = args.knn_feature_type
#         print(f"\n### {feat_type.upper()} ###")
#         distance_fx = args.knn_distance_function
#         print("---")
#         print(f"Running k-NN with params: distance_fx={distance_fx}, k={args.knn_k}, T={args.knn_temperature}...")
#
#         # print("train_features shape:", all_train_features[feat_type].shape)
#         # print("test_features shape:", test_features[feat_type].shape)
#         print("the number of k:", min(args.knn_k, len([j for j in range(task_id+1)])))
#         acc1, acc5 = run_knn(
#             train_features=all_train_features[feat_type],
#             train_targets=all_train_targets,
#             test_features=test_features[feat_type],  # 各タスクごとのテストデータを使用
#             test_targets=test_targets,
#             # k=args.knn_k,
#             # k=20,
#             k=min(args.knn_k, len([j for j in range(task_id+1)])),
#             T=args.knn_temperature,
#             distance_fx=distance_fx,
#             # distance_fx="euclidean",
#             current_task = task_id+1,
#             N_CLASSES_PER_TASK = N_CLASSES_PER_TASK
#         )
#         print(f"Task: {i} Result: acc@1={acc1}, acc@5={acc5}")
#         acc_matrix[task_id, i] = acc1  # 各タスクの結果を保存
#
#     # 結果を表示
#     print(f'Task:{task_id} Accuracies =')
#     average = []
#     for i_a in range(task_id + 1):
#         print('\t', end='')
#         for j_a in range(acc_matrix.shape[1]):
#             print('{:5.1f}% '.format(acc_matrix[i_a, j_a]), end='')
#         print()
#         average.append(acc_matrix[i_a][:i_a + 1].mean())
#     print('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[task_id].mean()))
#     bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1])
#     print('Backward transfer: {:5.2f}%'.format(bwt))
#     print('Mean Avg Accuracy: {:5.2f}%'.format(np.mean(average)))
#
#     # WandBに結果をログ
#     wandb_logger.log_table("AA" + str(task_id), columns=[("Task" + str(i)) for i in range(model.task_num)],
#                            data=acc_matrix)
#     if task_id == 20:
#         wandb.finish()
    # ------------------CUCL version? to lump?

# original (solo)
@torch.no_grad()
def test_model(args, model, task_id, sample_orders, N_CLASSES_PER_TASK, acc_matrix, num_run, wandb_logger):
    model.eval()
    model = model.cuda()

    # tsne save_dir
    save_dir = f"CUCL_simasiam_10CWforeach_CAplus_cifar100/{num_run}"
    os.makedirs(save_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成

    # # 【変更点】タスクをまたいで特徴量を保存するリスト
    # all_train_features_bb = []
    # all_train_features_proj = []
    # all_train_targets = []


    for i in range(task_id + 1):
        # prepare data
        _, T = prepare_transforms_classification(args.dataset)
        train_dataset, val_dataset = prepare_datasets_classification(
            args.dataset,
            T_train=T,
            T_val=T,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            data_format=args.data_format,
        )

        order = sample_orders[i]
        train_mask = np.logical_and(np.array(train_dataset.targets) >= order * N_CLASSES_PER_TASK,
                                    np.array(train_dataset.targets) < (order + 1) * N_CLASSES_PER_TASK)
        if args.dataset == 'imagenet100':
            train_dataset.samples = np.array(train_dataset.samples)[train_mask]
            train_dataset.imgs = np.array(train_dataset.imgs)[train_mask]
            train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order * N_CLASSES_PER_TASK
        else:
            train_dataset.data = np.array(train_dataset.data)[train_mask]
            train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order * N_CLASSES_PER_TASK

        test_mask = np.logical_and(np.array(val_dataset.targets) >= order * N_CLASSES_PER_TASK,
                                   np.array(val_dataset.targets) < (order + 1) * N_CLASSES_PER_TASK)
        if args.dataset == 'imagenet100':
            val_dataset.samples = np.array(val_dataset.samples)[test_mask]
            val_dataset.imgs = np.array(val_dataset.imgs)[test_mask]
            val_dataset.targets = np.array(val_dataset.targets)[test_mask] - order * N_CLASSES_PER_TASK
        else:
            val_dataset.data = np.array(val_dataset.data)[test_mask]
            val_dataset.targets = np.array(val_dataset.targets)[test_mask] - order * N_CLASSES_PER_TASK

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=False, )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True, drop_last=False, )


        # extract train features
        train_features_bb, train_features_proj, train_targets = extract_features(train_loader, model)
        train_features = {"backbone": train_features_bb, "projector": train_features_proj}

        # extract test features
        test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model)
        test_features = {"backbone": test_features_bb, "projector": test_features_proj}

        # # # 【変更点】タスクごとの特徴を統合
        # all_train_features_bb.append(train_features_bb)
        # all_train_features_proj.append(train_features_proj)
        # all_train_targets.append(train_targets+i*10)  # タスク番号をラベルとして加算



        # タスクごとの t-SNE 可視化（オプション）
        # visualize_feature_distribution(test_features, test_targets,
        #                                save_path=f"{save_dir}/task_{i}_features_tsne.png")
        if args.dataset == 'imagenet100':
            train_targets = train_targets - order * N_CLASSES_PER_TASK
            test_targets = test_targets - order * N_CLASSES_PER_TASK

        # run k-nn for all possible combinations of parameters
        feat_type = args.knn_feature_type
        print(f"\n### {feat_type.upper()} ###")
        distance_fx = args.knn_distance_function
        print("---")
        print(f"Running k-NN with params: distance_fx={distance_fx}, k={args.knn_k}, T={args.knn_temperature}...")
        # acc1, acc5, precision, recall, f1 = run_knn(

        acc1, acc5 = run_knn(
            train_features=train_features[feat_type],
            train_targets=train_targets,
            test_features=test_features[feat_type],
            test_targets=test_targets,
            k=args.knn_k,
            T=args.knn_temperature,
            distance_fx=distance_fx,
        )
        print(f"Task: {i} Result: acc@1={acc1}, acc@5={acc5}")
        acc_matrix[task_id, i] = acc1

    # # 【変更点】すべてのタスクの特徴を結合して可視化
    # all_train_features_bb = torch.cat(all_train_features_bb, dim=0)
    # all_train_targets = torch.cat(all_train_features_bb, dim=0)
    #
    # train_features_combined = {"backbone": all_train_features_bb}
    #
    # visualize_task_distribution(train_features_combined, all_train_targets,
    #                             class_per_task=N_CLASSES_PER_TASK,
    #                             save_path=f"{save_dir}/all_tasks_tsne_until_task{task_id}.png")

    print(f'Task:{task_id} Accuracies =')
    average = []
    for i_a in range(task_id + 1):
        print('\t', end='')
        for j_a in range(acc_matrix.shape[1]):
            print('{:5.1f}% '.format(acc_matrix[i_a, j_a]), end='')
        print()
        average.append(acc_matrix[i_a][:i_a + 1].mean())
    print('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[task_id].mean()))
    bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1])
    print('Backward transfer: {:5.2f}%'.format(bwt))
    print('Mean Avg Accuracy: {:5.2f}%'.format(np.mean(average)))
    wandb_logger.log_table("AA" + str(task_id), columns=[("Task" + str(i)) for i in range(model.task_num)],
                           data=acc_matrix)


@torch.no_grad()
def evaluate(args,model,task_id, sample_orders,N_CLASSES_PER_TASK, acc_matrix, wandb_logger):
    model.eval()
    model = model.cuda()
    for i in range(task_id+1):
        _, T = prepare_transforms_classification(args.dataset)
        train_dataset, val_dataset = prepare_datasets_classification(
            args.dataset,
            T_train=T,
            T_val=T,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            data_format=args.data_format,
        )
        order = sample_orders[i]
        train_mask = np.logical_and(np.array(train_dataset.targets) >= order*N_CLASSES_PER_TASK,
        np.array(train_dataset.targets) < (order+1)*N_CLASSES_PER_TASK)
        if args.dataset == 'imagenet100':
            train_dataset.samples = np.array(train_dataset.samples)[train_mask]
            train_dataset.imgs = np.array(train_dataset.imgs)[train_mask]
            train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order*N_CLASSES_PER_TASK
        else:
            train_dataset.data = np.array(train_dataset.data)[train_mask]
            train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order*N_CLASSES_PER_TASK

        test_mask = np.logical_and(np.array(val_dataset.targets) >= order*N_CLASSES_PER_TASK,
        np.array(val_dataset.targets) < (order+1)*N_CLASSES_PER_TASK)
        if args.dataset == 'imagenet100':
            val_dataset.samples = np.array(val_dataset.samples)[test_mask]
            val_dataset.imgs = np.array(val_dataset.imgs)[test_mask]
            val_dataset.targets = np.array(val_dataset.targets)[test_mask] - order*N_CLASSES_PER_TASK
        else:
            val_dataset.data = np.array(val_dataset.data)[test_mask]
            val_dataset.targets = np.array(val_dataset.targets)[test_mask] - order*N_CLASSES_PER_TASK

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,drop_last=False,)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,drop_last=False,)

        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            if args.multitask:
                outputs = model.eval_forward(inputs)
            else:
                outputs = model.eval_forward(inputs)[order]

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
        
        acc1 = correct / total * 100
        print(f"Task: {i} Result: acc@1={acc1}")
        acc_matrix[task_id,i] = acc1
    
    print(f'Task:{task_id} Accuracies =')
    average = []
    for i_a in range(task_id+1):
        print('\t',end='')
        for j_a in range(acc_matrix.shape[1]):
            print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
        print()
        average.append(acc_matrix[i_a][:i_a+1].mean())
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[task_id].mean()))
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print ('Mean Avg Accuracy: {:5.2f}%'.format(np.mean(average)))
    wandb_logger.log_table("AA"+str(task_id), columns=[("Task" + str(i)) for i in range(model.task_num)], data=acc_matrix)


@torch.no_grad()
def calculate_Codeword(args, net, train_loader, task_id):
    net.eval()
    Quant_idx_bank = []
    # print("calculate_codeword:args", net.shared_codebook)
    with torch.no_grad():
        for index, (images1, images2, notaug_images), target in tqdm(train_loader, desc='Feature extracting', leave=False, disable=True):
            quant_idx,Feature,_ = net.foward_CUCL(notaug_images.cuda(non_blocking=True),task_id) #256データのそれぞれの最近傍コードワードを8コードブック分． quanti_idx.Size() => torch([256,8]) FeatureはXaでtorch.Size([256,128])
            Quant_idx_bank.append(quant_idx)
        Quant_idx_bank = torch.cat(Quant_idx_bank, dim=0).contiguous() #Quant_idx_bankにindexを保存していく
        codeword_dict = {}
        for i in range(net.N_books):
            total_length = len(Quant_idx_bank[:,i])
            valid_length = total_length*args.CUCL_lambda
            id_list = torch.sort(torch.bincount(Quant_idx_bank[:,i]),descending=True)
            r = np.where((np.cumsum(np.array(id_list[0].cpu()))>=valid_length) == True)[0][0]
            codeword_dict[i] = id_list[1][:r+1].tolist()
            print('Task: {}, Codeword: {} for CodeBook: {}'.format(task_id,id_list[1][:r+1].tolist(),i))
            print()
        net.codeword_dict[task_id] = codeword_dict
        for task, dict in net.codeword_dict.items():
            print('Task: {}, Codeword: {}'.format(task,dict))


if __name__ == "__main__":
    num_runs = 3
    # num_runs = 2
    for i in range(1, num_runs+1):  # 1から開始するカウント
        main(i)
        wandb.finish()
    # main()