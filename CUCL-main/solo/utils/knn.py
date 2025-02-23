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

from typing import Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics.metric import Metric
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module) -> Tuple[torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """

    model.eval()
    backbone_features, proj_features, labels = [], [], []
    for im, lab in tqdm(loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        outs = model(im)
        backbone_features.append(outs["feats"].detach())
        # backbone_features = F.normalize(backbone_features, dim=1)
        proj_features.append(outs["z"])
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    proj_features = torch.cat(proj_features)
    labels = torch.cat(labels)
    return backbone_features, proj_features, labels


@torch.no_grad()
def run_knn(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    k: int,
    T: float,
    distance_fx: str,
) -> Tuple[float]:
    """Runs offline knn on a train and a test dataset.

    Args:
        train_features (torch.Tensor, optional): train features.
        train_targets (torch.Tensor, optional): train targets.
        test_features (torch.Tensor, optional): test features.
        test_targets (torch.Tensor, optional): test targets.
        k (int): number of neighbors.
        T (float): temperature for the exponential. Only used with cosine
            distance.
        distance_fx (str): distance function.

    Returns:
        Tuple[float]: tuple containing the the knn acc@1 and acc@5 for the model.
    """

    # build knn
    knn = WeightedKNNClassifier(
        k=k,
        T=T,
        distance_fx=distance_fx,
    )

    # add features
    knn(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
    )

    # compute
    acc1, acc5 = knn.compute()

    # free up memory
    del knn

    return acc1, acc5

class WeightedKNNClassifier(Metric):
    def __init__(
        self,
        k: int = 20,
        T: float = 0.07,
        max_distance_matrix_size: int = int(5e6),
        distance_fx: str = "cosine",
        epsilon: float = 0.00001,
        dist_sync_on_step: bool = False,
    ):
        """Implements the weighted k-NN classifier used for evaluation.

        Args:
            k (int, optional): number of neighbors. Defaults to 20.
            T (float, optional): temperature for the exponential. Only used with cosine
                distance. Defaults to 0.07.
            max_distance_matrix_size (int, optional): maximum number of elements in the
                distance matrix. Defaults to 5e6.
            distance_fx (str, optional): Distance function. Accepted arguments: "cosine" or
                "euclidean". Defaults to "cosine".
            epsilon (float, optional): Small value for numerical stability. Only used with
                euclidean distance. Defaults to 0.00001.
            dist_sync_on_step (bool, optional): whether to sync distributed values at every
                step. Defaults to False.
        """

        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)

        self.k = k
        self.T = T
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon


        self.add_state("train_features", default=[], persistent=False)
        self.add_state("train_targets", default=[], persistent=False)
        self.add_state("test_features", default=[], persistent=False)
        self.add_state("test_targets", default=[], persistent=False)

    def update(
        self,
        train_features: torch.Tensor = None,
        train_targets: torch.Tensor = None,
        test_features: torch.Tensor = None,
        test_targets: torch.Tensor = None,
    ):
        """Updates the memory banks. If train (test) features are passed as input, the
        corresponding train (test) targets must be passed as well.

        Args:
            train_features (torch.Tensor, optional): a batch of train features. Defaults to None.
            train_targets (torch.Tensor, optional): a batch of train targets. Defaults to None.
            test_features (torch.Tensor, optional): a batch of test features. Defaults to None.
            test_targets (torch.Tensor, optional): a batch of test targets. Defaults to None.
        """
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)

        if train_features is not None:
            assert train_features.size(0) == train_targets.size(0)
            self.train_features.append(train_features.detach())
            self.train_targets.append(train_targets.detach())

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())

    # @torch.no_grad()
    # def compute(self) -> Tuple[float]:
    #     """Computes weighted k-NN accuracy @1 and @5 and stores scores for evaluation."""
    #
    #     # デバッグ情報を出力
    #     # print(f"Train features: {len(self.train_features)}")
    #     # print(f"Train targets: {len(self.train_targets)}")
    #     # print(f"Test features: {len(self.test_features)}")
    #     # print(f"Test targets: {len(self.test_targets)}")
    #     #
    #     # print(f"Train features shape: {[f.shape for f in self.train_features]}")
    #     # print(f"Test features shape: {[f.shape for f in self.test_features]}")
    #
    #     train_features = torch.cat(self.train_features)
    #     train_targets = torch.cat(self.train_targets)
    #     test_features = torch.cat(self.test_features)
    #     test_targets = torch.cat(self.test_targets)
    #
    #
    #
    #     if self.distance_fx == "cosine":
    #         train_features = F.normalize(train_features)
    #         test_features = F.normalize(test_features)
    #
    #     num_classes = torch.unique(test_targets).numel()
    #     num_train_images = train_targets.size(0)
    #     num_test_images = test_targets.size(0)
    #
    #     # print(f"Num classes: {num_classes}")
    #     # print(f"Train targets min: {train_targets.min()}, max: {train_targets.max()}")
    #     # print(f"Test targets min: {test_targets.min()}, max: {test_targets.max()}")
    #
    #     chunk_size = min(
    #         max(1, self.max_distance_matrix_size // num_train_images),
    #         num_test_images,
    #     )
    #     k = min(self.k, num_train_images)
    #
    #     top1, top5, total = 0.0, 0.0, 0
    #     retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    #     # print(f"Retrieval one-hot size: {retrieval_one_hot.size()}")
    #
    #     all_predictions = []  # 予測結果を保持
    #     all_targets = []  # 真のラベルを保持
    #
    #     for idx in range(0, num_test_images, chunk_size):
    #         # Get the features for test images
    #         features = test_features[idx: min((idx + chunk_size), num_test_images), :]
    #         targets = test_targets[idx: min((idx + chunk_size), num_test_images)]
    #         batch_size = targets.size(0)
    #
    #         # Calculate the dot product and compute top-k neighbors
    #         if self.distance_fx == "cosine":
    #             similarities = torch.mm(features, train_features.t())
    #         elif self.distance_fx == "euclidean":
    #             similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
    #         else:
    #             raise NotImplementedError
    #
    #         similarities, indices = similarities.topk(k, largest=True, sorted=True)
    #         candidates = train_targets.view(1, -1).expand(batch_size, -1)
    #         retrieved_neighbors = torch.gather(candidates, 1, indices)
    #         # リマップする
    #         # retrieved_neighbors = retrieved_neighbors - train_targets.min()
    #
    #
    #         retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
    #
    #         # リマップ
    #         retrieved_neighbors = retrieved_neighbors - train_targets.min()
    #
    #
    #
    #         retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
    #         print(f"Remapped retrieved neighbors unique values: {torch.unique(retrieved_neighbors)}")
    #         print(f"Retrieval one-hot sum per class: {retrieval_one_hot.sum(dim=0)}")
    #
    #         if self.distance_fx == "cosine":
    #             similarities = similarities.clone().div_(self.T).exp_()
    #
    #         probs = torch.sum(
    #             torch.mul(
    #                 retrieval_one_hot.view(batch_size, -1, num_classes),
    #                 similarities.view(batch_size, -1, 1),
    #             ),
    #             1,
    #         )
    #         _, predictions = probs.sort(1, True)
    #
    #         # 予測クラスと真のラベルを保存
    #         all_predictions.append(predictions[:, 0].cpu())  # top-1予測
    #         all_targets.append(targets.cpu())
    #
    #         # Find the predictions that match the target
    #         correct = predictions.eq(targets.data.view(-1, 1))
    #         top1 += correct.narrow(1, 0, 1).sum().item()
    #         top5 += correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
    #         total += targets.size(0)
    #
    #     top1 = top1 * 100.0 / total
    #     top5 = top5 * 100.0 / total
    #
    #     # 全クラススコアを結合して 2 次元配列として保持
    #
    #     self.reset()
    #
    #     return top1, top5, all_predictions, all_targets

    @torch.no_grad()
    def compute(self) -> Tuple[float]:
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """

        train_features = torch.cat(self.train_features)
        train_targets = torch.cat(self.train_targets)
        test_features = torch.cat(self.test_features)
        test_targets = torch.cat(self.test_targets)

        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)
            test_features = F.normalize(test_features)

        num_classes = torch.unique(test_targets).numel()
        num_train_images = train_targets.size(0)
        num_test_images = test_targets.size(0)
        num_train_images = train_targets.size(0)
        chunk_size = min(
            max(1, self.max_distance_matrix_size // num_train_images),
            num_test_images,
        )
        k = min(self.k, num_train_images)

        top1, top5, total = 0.0, 0.0, 0
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)

        all_scores = []  # スコアを保持するリスト


        for idx in range(0, num_test_images, chunk_size):
            # get the features for test images
            features = test_features[idx : min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
            batch_size = targets.size(0)

            # calculate the dot product and compute top-k neighbors
            if self.distance_fx == "cosine":
                similarities = torch.mm(features, train_features.t())
            elif self.distance_fx == "euclidean":
                # cdist を実行
                # similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)

                similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
            else:
                raise NotImplementedError

            similarities, indices = similarities.topk(k, largest=True, sorted=True)
            candidates = train_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            if self.distance_fx == "cosine":
                similarities = similarities.clone().div_(self.T).exp_()

            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    similarities.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            all_scores.append(probs.max(dim=1).values)


            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = (
                top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
            )  # top5 does not make sense if k < 5
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total

        self.reset()

        return top1, top5