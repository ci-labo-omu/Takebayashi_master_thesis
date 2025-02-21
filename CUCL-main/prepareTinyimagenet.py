import os
import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms
from PIL import Image
import random

class SplitTinyImageNet(VisionDataset):
    def __init__(self, root, num_tasks=20, num_classes_per_task=10, train=True, transform=None):
        """
        Args:
            root (string): データセットのルートディレクトリ。
            num_tasks (int): タスクの総数。
            num_classes_per_task (int): 各タスクに割り当てるクラスの数。
            train (bool): Trueなら訓練データセット、Falseならテストデータセットをロード。
            transform (callable, optional): 画像に適用する変換。
        """
        super(SplitTinyImageNet, self).__init__(root, transform=transform)
        self.train = train

        # データセットのパスを設定
        self.data_path = os.path.join(root, 'train' if self.train else 'val')

        # 全クラスのリストを取得
        self.all_classes = sorted(os.listdir(self.data_path))
        random.shuffle(self.all_classes)  # クラスをランダムにシャッフル

        # タスクごとのクラス分割
        self.classes_by_task = [self.all_classes[i:i + num_classes_per_task] for i in range(0, len(self.all_classes), num_classes_per_task)]

    def __len__(self):
        return len(self.classes_by_task)

    def __getitem__(self, index):
        """
        タスク番号に基づいてデータをロードします。
        Args:
            index (int): タスクインデックス。
        Returns:
            tuple: (images, labels) タスクに含まれる画像とラベルのリスト。
        """
        task_classes = self.classes_by_task[index]
        images = []
        labels = []

        for class_index, class_name in enumerate(task_classes):
            class_dir = os.path.join(self.data_path, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = Image.open(image_path)
                if self.transform:
                    image = self.transform(image)
                images.append(image)
                labels.append(class_index)

        return torch.stack(images), torch.tensor(labels)