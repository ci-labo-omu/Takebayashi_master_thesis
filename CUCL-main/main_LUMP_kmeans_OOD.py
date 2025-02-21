
import wandb

import os


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


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print("Checkpoint Keys:", checkpoint['state_dict'].keys())  # どんなキーが含まれているか確認
    print("Model Keys:", model.state_dict().keys())  # モデルのキーと比較
    model.load_state_dict(checkpoint['state_dict'])
    return model

from sklearn.metrics import roc_auc_score
import numpy as np

def main(num_run):
    seed_everything(num_run + 1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # transform = transforms.Compose([
    #     transforms.Resize((64, 64)),  # TinyImageNetサイズにリサイズ
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    ###########fmnist
    from torchvision.transforms import Lambda, Compose, ToTensor, Normalize
    #
    # transform = Compose([
    #     ToTensor(),
    #     # Lambda(lambda x: x.repeat(3, 1, 1)),  # グレースケール画像を3チャンネルに変換
    #     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化
    # ])
    ###################

    args = parse_args_pretrain()

    args.dataset = "svhn"
    # args.dataset = "tinyimagenet"     --CUCL --CUCL_lr 0.03 --CUCL_lambda 1.0   barlow --CUCL --CUCL_lr 0.03 --CUCL_lambda 1.0

    if args.dataset == "caltech101":
        args.train_data_path = "./data/caltech101/101_ObjectCategories"
        args.val_data_path = "./data/caltech101/101_ObjectCategories"

    # args.dataset = "mnist"

    # args.dataset = "cifar100"

    args.num_run = num_run


    # checkpoint_path = os.path.join(
    #     "checkpoints",
    #     "tinyimagenet_results",
    #     # f"CUCL_shared_CAplus_10CW_seed1{num_run}",
    #     f"CUCL_shared_10CW_seed{num_run}",
    #     f"9.pth"
    # )



    assert args.method in METHODS, f"Choose from {METHODS.keys()}"

    if args.num_large_crops != 2:
        assert args.method in ["wmse", "mae"]

    # data formation

    model = METHODS[args.method](**args.__dict__)
    # print(model)

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
        transform_mixup.insert(0, torchvision.transforms.ToPILImage())
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
        name="CUCL_origin_Barlowtwins_svhn_seed" + str(num_run),
        # name="CUCL_original_stl10_simsiam_seed" + str(num_run),
        # name="CUCL_10CWforeach_cifar100_simsiam_seed" + str(num_run),
        # project="CUCL_shared",
        # project="OOD_test",
        project="check",
        entity="takanori_takebayashi",
        offline=False,
        resume="allow",
        id=wandb.util.generate_id()
    )

    """data name
    cifar100
    tinyimagenet
    """


    class_per_task = model.class_per_task  # タスクごとにいくつクラスあるか
    print(class_per_task)
    print("task_num", model.task_num)
    acc_matrix = np.zeros((model.task_num, model.task_num))  # [*range(5)]だと[0,1,2,3,4]となる．
    sample_orders = [*range(model.task_num)]
    print("sample order", sample_orders)

    # OODモデルのチェックポイントパスを構築
    # checkpoint_path = os.path.join(
    #     "checkpoints",
    #     "ood_results",
    #     f"CUCL_original_seed{num_run}",
    #     f"9.pth"
    # )
    checkpoint_path = os.path.join(
        "checkpoints",
        "tinyimagenet_results",
        # f"CUCL_original_seed{num_run}",
        f"CUCL_original_barlow_seed{num_run}",

        # f"CUCL_8CWforeach_simsiam_seed{num_run}",
        # f"CUCL_50CWforeach_CAplus_barlow_seed{num_run}",
        f"9.pth"
    )
    print("os_path")
    print(checkpoint_path)
    print(os.path.exists(checkpoint_path))

    # チェックポイントが存在する場合のみロード
    if os.path.exists(checkpoint_path):
        model = load_checkpoint(model, checkpoint_path)
        print(f"Loaded OOD model checkpoint: {checkpoint_path}")
    # else:
    #     print(f"Checkpoint not found for task {i}: {checkpoint_path}")

    # モデルのメモリフォーマットとバッファを調整
    make_contiguous(model)
    if not args.no_channel_last:
        model = model.to(memory_format=torch.channels_last)

    # for i in range(model.task_num):
    i = model.task_num-1
    # if i > model.train_task - 1:
    #     break
    # print(args.train_from_task)
    # if i < args.train_from_task:
    #     # model_path
    #     path = os.path.join(model.CUCL_loadPath, f"{i}.pth")
    #     # if i == 0:
    #     #     path = os.path.join('./checkpoints/cifar100_results/byol_CUCL_testp-3p2edh4k', f"{i}.pth")
    #     save_dict = torch.load(path, map_location='cpu')
    #     buffer = model.buffers
    #     # msg = model.load_state_dict(save_dict['state_dict'], strict=True)
    #     model = load_checkpoint(model, checkpoint_path)
    #     model.buffers = buffer
    #     make_contiguous(model)
    #     if not args.no_channel_last:
    #         model = model.to(memory_format=torch.channels_last)
    #     train = False
    model.curr_task = i

    # model = load_checkpoint(model, checkpoint_path)
    print("dataset", args.dataset)

    train_dataset = prepare_datasets(args.dataset, transform, train_data_path=args.train_data_path,
                                     data_format=args.data_format, \
                                     no_labels=args.no_labels, data_fraction=args.data_fraction, )
    # データセットのサンプル確認
    # print(f"Dataset length: {len(train_dataset)}")
    # for idx, (sample_idx, img, target) in enumerate(train_dataset):
    #     print(f"Sample {idx}: index={sample_idx}, target={target}, img_shape={img.shape}")
    #     if idx >= 2:  # 3サンプル確認したら終了
    #         break

    # which task samples to use
    order = sample_orders[i]
    # print("order:", order)
    if args.dataset in ["cifar10", "stl10"]:
        class_per_task = 2
    elif args.dataset == "svhn" or args.dataset == "fmnist":
        class_per_task = 10
    if args.dataset == "svhn" or args.dataset == "stl10":
        train_mask = np.logical_and(np.array(train_dataset.labels) >= order * class_per_task,
                                    np.array(train_dataset.labels) < (order + 1) * class_per_task)
        train_dataset.data = np.array(train_dataset.data)[train_mask]
        # print(len(train_dataset.data))
        train_dataset.targets = np.array(train_dataset.labels)[train_mask]
    elif args.dataset == "caltech101":


        train_mask = np.logical_and(
            np.array(train_dataset.targets) >= order * class_per_task,
            np.array(train_dataset.targets) < (order + 1) * class_per_task
        )

        # 適切な属性にアクセス（例: samples または imgs）
        train_dataset.samples = np.array(train_dataset.samples)[train_mask]  # 'samples' を利用
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]  # targets をフィルタリング

    else:
        train_mask = np.logical_and(np.array(train_dataset.targets) >= order * class_per_task,
                                np.array(train_dataset.targets) < (order + 1) * class_per_task)

        train_dataset.data = np.array(train_dataset.data)[train_mask]
        # print(len(train_dataset.data))
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    # train_loader = prepare_dataloader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print("class per task:", class_per_task)


    test_model(args, model, i, sample_orders[:i + 1], class_per_task, acc_matrix, num_run, wandb_logger)


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean

def analyze_task_distributions(task_features):
    """
    各タスクの特徴ベクトルの分散・共分散を計算し、タスク間のコサイン類似度とユークリッド距離を測定
    """
    num_tasks = len(task_features)

    # 分散・共分散を保存するリスト
    task_variances = []
    task_covariances = []

    # タスク間のコサイン類似度とユークリッド距離を保存する行列
    task_cosine_similarity = np.zeros((num_tasks, num_tasks))
    task_euclidean_distance = np.zeros((num_tasks, num_tasks))

    # 各タスクの特徴ベクトルの分散・共分散を計算
    for task_id, features in task_features.items():
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        variance = np.var(features, axis=0).mean()  # 特徴ベクトル次元ごとの分散の平均
        covariance = np.cov(features, rowvar=False).mean()  # 共分散行列の平均

        task_variances.append(variance)
        task_covariances.append(covariance)

    # タスク間のコサイン類似度とユークリッド距離を計算
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                task_cosine_similarity[i, j] = 1.0
                task_euclidean_distance[i, j] = 0.0
            else:
                task_cosine_similarity[i, j] = 1 - cosine(task_features[i].mean(axis=0), task_features[j].mean(axis=0))
                task_euclidean_distance[i, j] = euclidean(task_features[i].mean(axis=0), task_features[j].mean(axis=0))

    # データを pandas.DataFrame に整理
    variance_df = pd.DataFrame({'Task': range(num_tasks), 'Variance': task_variances, 'Covariance': task_covariances})
    cosine_df = pd.DataFrame(task_cosine_similarity, index=range(num_tasks), columns=range(num_tasks))
    euclidean_df = pd.DataFrame(task_euclidean_distance, index=range(num_tasks), columns=range(num_tasks))

    return variance_df, cosine_df, euclidean_df

def analyze_task_feature_distribution(all_task_features, all_task_targets, task_id, save_dir):
    """
    タスクごとの特徴ベクトルの分布を分析し、タスク間の統計的距離（分散、コサイン類似度、ユークリッド距離）を計算し、可視化する。

    Args:
        all_task_features (dict): {task_id: 特徴ベクトル} の辞書
        all_task_targets (dict): {task_id: ラベル} の辞書
        task_id (int): 現在の最大タスクID
        save_dir (str): 保存フォルダ
    """
    # 保存フォルダを作成
    os.makedirs(save_dir, exist_ok=True)

    task_variances = {}
    task_means = {}

    for i in range(task_id + 1):
        task_variances[i] = np.var(all_task_features[i], axis=0)  # 各タスクの特徴ベクトルの分散
        task_means[i] = np.mean(all_task_features[i], axis=0)  # 平均特徴ベクトル

    # タスク間のコサイン類似度とユークリッド距離を計算
    task_cosine_similarity = np.zeros((task_id + 1, task_id + 1))
    task_euclidean_distance = np.zeros((task_id + 1, task_id + 1))

    for i in range(task_id + 1):
        for j in range(task_id + 1):
            if i != j:
                task_cosine_similarity[i, j] = 1 - cosine(task_means[i], task_means[j])
                task_euclidean_distance[i, j] = euclidean(task_means[i], task_means[j])

    # 結果をプロット
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    def plot_heatmap(matrix, title, ax):
        im = ax.imshow(matrix, cmap="coolwarm", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Task ID")
        ax.set_ylabel("Task ID")
        ax.set_xticks(range(task_id + 1))
        ax.set_yticks(range(task_id + 1))

        # 各セルに数値を表示
        for i in range(task_id + 1):
            for j in range(task_id + 1):
                text = f"{matrix[i, j]:.3f}"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)

        # カラーバーを追加
        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.set_ylabel(title, rotation=-90, va="bottom")

    plot_heatmap(task_cosine_similarity, "Task-wise Cosine Similarity", ax[0])
    plot_heatmap(task_euclidean_distance, "Task-wise Euclidean Distance", ax[1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "task_feature_similarity.png"), dpi=300)
    plt.show()

    # 数値データを保存
    np.save(os.path.join(save_dir, "task_variances.npy"), task_variances)
    np.save(os.path.join(save_dir, "task_means.npy"), task_means)
    np.save(os.path.join(save_dir, "task_cosine_similarity.npy"), task_cosine_similarity)
    np.save(os.path.join(save_dir, "task_euclidean_distance.npy"), task_euclidean_distance)

    print("分析結果を保存しました。")


def compute_cluster_stats(features, labels):
    unique_labels = np.unique(labels)
    print("unique_labels", unique_labels)
    centroids = {}
    scatter = {}

    # 各クラスの中心と散らばりを計算
    for label in unique_labels:
        cluster_data = features[labels == label]
        centroid = np.mean(cluster_data, axis=0)
        centroids[label] = centroid
        # クラス内の散らばり（例：平均ユークリッド距離）
        scatter[label] = np.mean(np.linalg.norm(cluster_data - centroid, axis=1))

    # 各クラス間の距離を計算
    distances = {}
    for i in unique_labels:
        for j in unique_labels:
            if i < j:
                distances[(i, j)] = np.linalg.norm(centroids[i] - centroids[j])

    return centroids, scatter, distances





from sklearn.metrics import davies_bouldin_score


@torch.no_grad()
def test_model(args, model, task_id, sample_orders, N_CLASSES_PER_TASK, acc_matrix,num_run,  wandb_logger):
    model.eval()
    model = model.cuda()

    # すべてのタスクの特徴を保存するための辞書
    all_task_features = {}
    all_task_targets = {}

    # 各タスクごとのDBIを保存する辞書（タスク内でのクラスタリング評価）
    dbi_values = {}

    # WandB用テーブルの準備（全タスクまとめて1つのテーブル）
    # columns = ["Task", "acc@1", "acc@5", "DBI", "Scatter", "Distances"]
    columns = ["Task", "acc@1", "acc@5", "DBI"]

    results_table = wandb.Table(columns=columns)

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



        if args.dataset == "svhn" or args.dataset == "stl10":
            train_mask = np.logical_and(
                np.array(train_dataset.labels) >= order * N_CLASSES_PER_TASK,
                np.array(train_dataset.labels) < (order + 1) * N_CLASSES_PER_TASK,
            )
            train_dataset.data = np.array(train_dataset.data)[train_mask]
            train_dataset.labels = np.array(train_dataset.labels)[train_mask] - order * N_CLASSES_PER_TASK

            test_mask = np.logical_and(
                np.array(val_dataset.labels) >= order * N_CLASSES_PER_TASK,
                np.array(val_dataset.labels) < (order + 1) * N_CLASSES_PER_TASK,
            )
            val_dataset.data = np.array(val_dataset.data)[test_mask]
            val_dataset.labels = np.array(val_dataset.labels)[test_mask] - order * N_CLASSES_PER_TASK

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        elif args.dataset == "caltech101":
            # print(f"Train targets before masking: {np.unique(train_dataset.targets)}")
            # print(f"Test targets before masking: {np.unique(val_dataset.targets)}")



            train_mask = np.logical_and(
                np.array(train_dataset.targets) >= order * N_CLASSES_PER_TASK,
                np.array(train_dataset.targets) < (order + 1) * N_CLASSES_PER_TASK,
            )

            # train_mask = np.logical_and(
            #     train_dataset.targets >= order * N_CLASSES_PER_TASK,
            #     train_dataset.targets < min((order + 1) * N_CLASSES_PER_TASK, 102),  # 最大クラス数を考慮
            # )

            # Train mask の内容を確認
            # print(f"Train mask (True count): {np.sum(train_mask)} / {len(train_mask)}")
            # print(f"Train targets masked: {np.unique(np.array(train_dataset.targets)[train_mask])}")

            train_dataset.samples = np.array(train_dataset.samples)[train_mask]
            train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order * N_CLASSES_PER_TASK

            test_mask = np.logical_and(
                np.array(val_dataset.targets) >= order * N_CLASSES_PER_TASK,
                np.array(val_dataset.targets) < min((order + 1) * N_CLASSES_PER_TASK, 102),
            )


            # test_mask = np.logical_and(
            #     val_dataset.targets >= order * N_CLASSES_PER_TASK,
            #     val_dataset.targets < min((order + 1) * N_CLASSES_PER_TASK, 102),  # 最大クラス数を考慮
            # )

            # Test mask の内容を確認
            # print(f"Test mask (True count): {np.sum(test_mask)} / {len(test_mask)}")
            # print(f"Test targets masked: {np.unique(np.array(val_dataset.targets)[test_mask])}")

            val_dataset.samples = np.array(val_dataset.samples)[test_mask]
            val_dataset.targets = np.array(val_dataset.targets)[test_mask] - order * N_CLASSES_PER_TASK


            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            train_mask = np.logical_and(
                np.array(train_dataset.targets) >= order * N_CLASSES_PER_TASK,
                np.array(train_dataset.targets) < (order + 1) * N_CLASSES_PER_TASK,
            )
            train_dataset.data = np.array(train_dataset.data)[train_mask]
            train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order * N_CLASSES_PER_TASK

            test_mask = np.logical_and(
                np.array(val_dataset.targets) >= order * N_CLASSES_PER_TASK,
                np.array(val_dataset.targets) < (order + 1) * N_CLASSES_PER_TASK,
            )
            val_dataset.data = np.array(val_dataset.data)[test_mask]
            val_dataset.targets = np.array(val_dataset.targets)[test_mask] - order * N_CLASSES_PER_TASK

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )


        # extract train features
        train_features_bb, train_features_proj, train_targets = extract_features(train_loader, model)
        train_features = {"backbone": train_features_bb, "projector": train_features_proj}

        # extract test features
        test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model)
        test_features = {"backbone": test_features_bb, "projector": test_features_proj}

        # 特徴ベクトルを保存
        all_task_features[i] = test_features_bb.cpu().numpy()
        all_task_targets[i] = test_targets.cpu().numpy()

        # ------------------------------
        # DBIの計算（タスク内のみ）
        # ------------------------------

        centroids, scatter, distances = compute_cluster_stats(test_features_bb.cpu().numpy(), test_targets.cpu().numpy())
        dbi = davies_bouldin_score(test_features_bb.cpu().numpy(), test_targets.cpu().numpy())


        # # scatter のキーを int にキャスト
        # scatter = {int(k): v for k, v in scatter.items()}
        #
        # # distances のキーを "i-j" 形式の文字列に変換
        # distances = {f"{int(k[0])}-{int(k[1])}": v for k, v in distances.items()}

        # クラス内散らばり（intra-class scatter）の平均値を計算
        # avg_scatter = sum(scatter.values()) / len(scatter) if len(scatter) > 0 else 0
        #
        # # クラス間距離（inter-class distance）の平均値を計算
        # avg_distance = sum(distances.values()) / len(distances) if len(distances) > 0 else 0

        dbi_values[i] = dbi

        print(f"Task {i} DBI: {dbi:.4f}")
        # wandb_logger.log_metrics({"Task": i, "DBI": dbi})
        wandb_logger.log_metrics({
            "Task": i,
            "DBI": dbi
            # "Scatter": avg_scatter,
            # "Distances": avg_distance
        })


        # run k-nn for all possible combinations of parameters
        feat_type = args.knn_feature_type
        print(f"\n### {feat_type.upper()} ###")
        distance_fx = args.knn_distance_function
        print("---")
        print(f"Running k-NN with params: distance_fx={distance_fx}, k={args.knn_k}, T={args.knn_temperature}...")
        # acc1, acc5 = run_knn(
        #     train_features=train_features[feat_type],
        #     train_targets=train_targets,
        #     test_features=test_features[feat_type],
        #     test_targets=test_targets,
        #     k=min(args.knn_k, model.task_num),
        #     T=args.knn_temperature,
        #     distance_fx=distance_fx,
        # )

        acc1, acc5= run_knn(
            train_features=train_features[feat_type],
            train_targets=train_targets,
            test_features=test_features[feat_type],
            test_targets=test_targets,
            k=min(args.knn_k, model.task_num),
            T=args.knn_temperature,
            distance_fx=distance_fx,
        )

        print(f"Top-1 Accuracy: {acc1:.2f}%")
        print(f"Top-5 Accuracy: {acc5:.2f}%")
        print(f"Task: {i} Result: acc@1={acc1}, acc@5={acc5}")


        print(f"Result: acc@1={acc1:.4f}, acc@5={acc5:.4f}")

        # 結果を記録
        wandb_logger.log_metrics(
            {
                "Task": i,
                "acc@1": acc1,
                "acc@5": acc5,
            }
        )
        acc_matrix[task_id, i] = acc1

        # ここで各タスクの結果をテーブルに追加
        # results_table.add_data(i, acc1, acc5, dbi, avg_scatter, avg_distance)
        results_table.add_data(i, acc1, acc5, dbi)


    # save_dir = "save_feature_vector_analysis_CUCL_10CW_simsiam_seed" + str(num_run)
    # # すべてのタスクの特徴を解析
    # analyze_task_feature_distribution(all_task_features, all_task_targets, task_id, save_dir)
    #
    # print("タスク間の特徴ベクトル分析が完了しました。")
    # # 実行
    # variance_df, cosine_df, euclidean_df = analyze_task_distributions(all_task_features)
    #
    # # 可視化
    # # save_dir = "task_feature_analysis_results"
    # os.makedirs(save_dir, exist_ok=True)
    #
    # # CSV ファイルとして保存
    # variance_df.to_csv(os.path.join(save_dir, "task_variances.csv"))
    # cosine_df.to_csv(os.path.join(save_dir, "task_cosine_similarity.csv"))
    # euclidean_df.to_csv(os.path.join(save_dir, "task_euclidean_distance.csv"))

    # print(f"Task-wise feature analysis results saved in {save_dir}")

    # 最終タスク時に全タスクの平均DBIを計算してprint出力
    avg_dbi = sum(dbi_values.values()) / len(dbi_values)
    print(f"Average DBI over tasks: {avg_dbi:.4f}")

    # タスク全体の集計と出力
    print(f"Task {task_id} Accuracies:")
    average = []
    for i_a in range(task_id + 1):
        print('\t', end='')
        for j_a in range(acc_matrix.shape[1]):
            print('{:5.1f}% '.format(acc_matrix[i_a, j_a]), end='')
        print()
        average.append(acc_matrix[i_a][:i_a + 1].mean())
    print('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[task_id].mean()))
    bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1])
    # print('Backward transfer: {:5.2f}%'.format(bwt))
    print('Mean Avg Accuracy: {:5.2f}%'.format(np.mean(average)))
    print(f"Total Tasks: {model.task_num}")


##############
    # WandB用テーブルの準備
    # columns = ["Task", "acc@1", "acc@5", "DBI"]
    # table = wandb.Table(columns=columns)

    # 最後のタスクの結果をテーブルに追加（例として最後のタスクのみ追加しています）
    # table.add_data(task_id, acc1, acc5, dbi_values[task_id])
    wandb.log({f"Evaluation Results Task {task_id}": results_table})

    # Wandb用テーブルの準備
    # columns = ["Task", "acc@1", "acc@5"]

    # table = wandb.Table(columns=columns)
    #
    # # テーブルに各タスクの評価指標を追加
    # # table.add_data(i, acc1, acc5, precision, recall, f1)
    # table.add_data(i, acc1, acc5)
    #
    # # テーブルをWandbに記録
    # wandb.log({f"Evaluation Results Task {task_id}": table})


    # # 出力と集計
    # print(f"Task:{task_id} Accuracies =")
    # average = []
    # for i_a in range(task_id + 1):
    #     print('\t', end='')
    #     for j_a in range(acc_matrix.shape[1]):
    #         print('{:5.1f}% '.format(acc_matrix[i_a, j_a]), end='')
    #     print()
    #     average.append(acc_matrix[i_a][:i_a + 1].mean())
    # print('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[task_id].mean()))
    # bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1])
    # print('Backward transfer: {:5.2f}%'.format(bwt))
    # print('Mean Avg Accuracy: {:5.2f}%'.format(np.mean(average)))
    # print(model.task_num)
    #
    # # テーブルに記録するデータの準備
    # columns = ["Task", "acc@1", "acc@5"]
    # table = wandb.Table(columns=columns)
    #
    # # 各タスクの評価指標をテーブルに追加
    # table.add_data(i, acc1, acc5)
    #
    # # テーブルをWandbに記録
    # wandb.log({f"Evaluation Results Task {task_id}": table})


# @torch.no_grad()
# def test_model(args, model, task_id, sample_orders, N_CLASSES_PER_TASK, acc_matrix, wandb_logger):
#     model.eval()
#     model = model.cuda()
#     for i in range(task_id + 1):
#         # prepare data
#         _, T = prepare_transforms_classification(args.dataset)
#         train_dataset, val_dataset = prepare_datasets_classification(
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
#         if args.dataset == "svhn":
#             train_mask = np.logical_and(np.array(train_dataset.labels) >= order * N_CLASSES_PER_TASK,
#                                         np.array(train_dataset.labels) < (order + 1) * N_CLASSES_PER_TASK)
#             train_dataset.data = np.array(train_dataset.data)[train_mask]
#             train_dataset.labels = np.array(train_dataset.labels)[train_mask] - order * N_CLASSES_PER_TASK
#
#             test_mask = np.logical_and(np.array(val_dataset.labels) >= order * N_CLASSES_PER_TASK,
#                                        np.array(val_dataset.labels) < (order + 1) * N_CLASSES_PER_TASK)
#
#             # if args.dataset == 'imagenet100':
#             #     val_dataset.samples = np.array(val_dataset.samples)[test_mask]
#             #     val_dataset.imgs = np.array(val_dataset.imgs)[test_mask]
#             #     val_dataset.targets = np.array(val_dataset.targets)[test_mask] - order * N_CLASSES_PER_TASK
#             # else:
#             val_dataset.data = np.array(val_dataset.data)[test_mask]
#             val_dataset.labels = np.array(val_dataset.labels)[test_mask] - order * N_CLASSES_PER_TASK
#
#             train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
#                                       num_workers=args.num_workers, pin_memory=True, drop_last=False, )
#             val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
#                                     num_workers=args.num_workers,
#                                     pin_memory=True, drop_last=False, )
#
#             # extract train features
#             train_features_bb, train_features_proj, train_targets = extract_features(train_loader, model)
#             train_features = {"backbone": train_features_bb, "projector": train_features_proj}
#
#             # extract test features
#             test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model)
#             test_features = {"backbone": test_features_bb, "projector": test_features_proj}
#
#             # if args.dataset == 'imagenet100':
#             #     train_targets = train_targets - order * N_CLASSES_PER_TASK
#             #     test_targets = test_targets - order * N_CLASSES_PER_TASK
#
#             # run k-nn for all possible combinations of parameters
#             feat_type = args.knn_feature_type
#             print(f"\n### {feat_type.upper()} ###")
#             distance_fx = args.knn_distance_function
#             print("---")
#             print(f"Running k-NN with params: distance_fx={distance_fx}, k={args.knn_k}, T={args.knn_temperature}...")
#
#             acc1, acc5 = run_knn(
#                 train_features=train_features[feat_type],
#                 train_targets=train_targets,
#                 test_features=test_features[feat_type],
#                 test_targets=test_targets,
#                 k=min(args.knn_k, model.task_num),
#                 T=args.knn_temperature,
#                 distance_fx=distance_fx,
#             )
#         elif args.dataset == "caltech101":
#             train_mask = np.logical_and(
#                 np.array(train_dataset.targets) >= order * N_CLASSES_PER_TASK,
#                 np.array(train_dataset.targets) < (order + 1) * N_CLASSES_PER_TASK
#             )
#             # train_dataset のサンプルとターゲットをフィルタリング
#             train_dataset.samples = np.array(train_dataset.samples)[train_mask]
#             train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order * N_CLASSES_PER_TASK
#
#             test_mask = np.logical_and(
#                 np.array(val_dataset.targets) >= order * N_CLASSES_PER_TASK,
#                 np.array(val_dataset.targets) < (order + 1) * N_CLASSES_PER_TASK
#             )
#             # val_dataset のサンプルとターゲットをフィルタリング
#             val_dataset.samples = np.array(val_dataset.samples)[test_mask]
#             val_dataset.targets = np.array(val_dataset.targets)[test_mask] - order * N_CLASSES_PER_TASK
#
#             # DataLoader の設定
#             train_loader = DataLoader(
#                 train_dataset, batch_size=args.batch_size, shuffle=False,
#                 num_workers=args.num_workers, pin_memory=True, drop_last=False
#             )
#             val_loader = DataLoader(
#                 val_dataset, batch_size=args.batch_size, shuffle=False,
#                 num_workers=args.num_workers, pin_memory=True, drop_last=False
#             )
#
#             # 訓練用特徴量の抽出
#             train_features_bb, train_features_proj, train_targets = extract_features(train_loader, model)
#             train_features = {"backbone": train_features_bb, "projector": train_features_proj}
#
#             # テスト用特徴量の抽出
#             test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model)
#             test_features = {"backbone": test_features_bb, "projector": test_features_proj}
#
#             # k-NN の実行
#             feat_type = args.knn_feature_type
#             print(f"\n### {feat_type.upper()} ###")
#             distance_fx = args.knn_distance_function
#             print("---")
#             print(f"Running k-NN with params: distance_fx={distance_fx}, k={args.knn_k}, T={args.knn_temperature}...")
#             print(test_targets.unique())
#             acc1, acc5 = run_knn(
#                 train_features=train_features[feat_type],
#                 train_targets=train_targets,
#                 test_features=test_features[feat_type],
#                 test_targets=test_targets,
#                 k=min(args.knn_k, model.task_num),
#                 T=args.knn_temperature,
#                 distance_fx=distance_fx,
#             )
#
#         else:
#             train_mask = np.logical_and(np.array(train_dataset.targets) >= order * N_CLASSES_PER_TASK,
#                                     np.array(train_dataset.targets) < (order + 1) * N_CLASSES_PER_TASK)
#         # if args.dataset == 'imagenet100':
#         #     train_dataset.samples = np.array(train_dataset.samples)[train_mask]
#         #     train_dataset.imgs = np.array(train_dataset.imgs)[train_mask]
#         #     train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order * N_CLASSES_PER_TASK
#         # else:
#             train_dataset.data = np.array(train_dataset.data)[train_mask]
#             train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order * N_CLASSES_PER_TASK
#
#             test_mask = np.logical_and(np.array(val_dataset.targets) >= order * N_CLASSES_PER_TASK,
#                                        np.array(val_dataset.targets) < (order + 1) * N_CLASSES_PER_TASK)
#             # if args.dataset == 'imagenet100':
#             #     val_dataset.samples = np.array(val_dataset.samples)[test_mask]
#             #     val_dataset.imgs = np.array(val_dataset.imgs)[test_mask]
#             #     val_dataset.targets = np.array(val_dataset.targets)[test_mask] - order * N_CLASSES_PER_TASK
#             # else:
#             val_dataset.data = np.array(val_dataset.data)[test_mask]
#             val_dataset.targets = np.array(val_dataset.targets)[test_mask] - order * N_CLASSES_PER_TASK
#
#             train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
#                                       num_workers=args.num_workers, pin_memory=True, drop_last=False, )
#             val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
#                                     pin_memory=True, drop_last=False, )
#
#             # extract train features
#             train_features_bb, train_features_proj, train_targets = extract_features(train_loader, model)
#             train_features = {"backbone": train_features_bb, "projector": train_features_proj}
#
#             # extract test features
#             test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model)
#             test_features = {"backbone": test_features_bb, "projector": test_features_proj}
#
#             # if args.dataset == 'imagenet100':
#             #     train_targets = train_targets - order * N_CLASSES_PER_TASK
#             #     test_targets = test_targets - order * N_CLASSES_PER_TASK
#
#             # run k-nn for all possible combinations of parameters
#             feat_type = args.knn_feature_type
#             print(f"\n### {feat_type.upper()} ###")
#             distance_fx = args.knn_distance_function
#             print("---")
#             print(f"Running k-NN with params: distance_fx={distance_fx}, k={args.knn_k}, T={args.knn_temperature}...")
#             print(test_targets.unique())
#             acc1, acc5 = run_knn(
#                 train_features=train_features[feat_type],
#                 train_targets=train_targets,
#                 test_features=test_features[feat_type],
#                 test_targets=test_targets,
#                 k=min(args.knn_k, model.task_num),
#                 T=args.knn_temperature,
#                 distance_fx=distance_fx,
#             )
#         print(f"Task: {i} Result: acc@1={acc1}, acc@5={acc5}")
#         acc_matrix[task_id, i] = acc1
#
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
#     print(model.task_num)
#     wandb_logger.log_table("AA" + str(task_id), columns=[("Task" + str(i)) for i in range(model.task_num)],
#                            data=acc_matrix)



if __name__ == "__main__":
    num_runs = 3
    for run_idx in range(1, num_runs + 1):
        main(run_idx)
        wandb.finish()


