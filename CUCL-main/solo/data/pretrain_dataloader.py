# # Copyright 2022 solo-learn development team.
#
# # Permission is hereby granted, free of charge, to any person obtaining a copy of
# # this software and associated documentation files (the "Software"), to deal in
# # the Software without restriction, including without limitation the rights to use,
# # copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# # Software, and to permit persons to whom the Software is furnished to do so,
# # subject to the following conditions:
#
# # The above copyright notice and this permission notice shall be included in all copies
# # or substantial portions of the Software.
#
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# # INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# # PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# # FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# # OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# # DEALINGS IN THE SOFTWARE.
#
# import os
# import random
# from pathlib import Path
# from typing import Any, Callable, List, Optional, Sequence, Type, Union
# import numpy as np
# import torch
# import torchvision
# from PIL import Image, ImageFilter, ImageOps
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from torch.utils.data import DataLoader
# from torch.utils.data.dataset import Dataset
# from torchvision import transforms
# from torchvision.datasets import STL10, ImageFolder
# from torch.utils.data.sampler import SubsetRandomSampler
# try:
#     from solo.data.h5_dataset import H5Dataset
#     from solo.data.tinyimagenet import TinyImagenet, MiniImageNet
# except ImportError:
#     _h5_available = False
# else:
#     _h5_available = True
#
#
#
#
# def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
#     """Factory for datasets that also returns the data index.
#
#     Args:
#         DatasetClass (Type[Dataset]): Dataset class to be wrapped.
#
#     Returns:
#         Type[Dataset]: dataset with index.
#     """
#
#     class DatasetWithIndex(DatasetClass):
#         def __getitem__(self, index):
#             data = super().__getitem__(index)
#             return (index, *data)
#
#     return DatasetWithIndex
#
#
# class CustomDatasetWithoutLabels(Dataset):
#     def __init__(self, root, transform=None):
#         self.root = Path(root)
#         self.transform = transform
#         self.images = os.listdir(root)
#
#     def __getitem__(self, index):
#         path = self.root / self.images[index]
#         x = Image.open(path).convert("RGB")
#         if self.transform is not None:
#             x = self.transform(x)
#         return x, -1
#
#     def __len__(self):
#         return len(self.images)
#
#
# class GaussianBlur:
#     def __init__(self, sigma: Sequence[float] = None):
#         """Gaussian blur as a callable object.
#
#         Args:
#             sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
#                 Defaults to [0.1, 2.0].
#         """
#
#         if sigma is None:
#             sigma = [0.1, 2.0]
#
#         self.sigma = sigma
#
#     def __call__(self, img: Image) -> Image:
#         """Applies gaussian blur to an input image.
#
#         Args:
#             img (Image): an image in the PIL.Image format.
#
#         Returns:
#             Image: blurred image.
#         """
#
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return img
#
#
# class Solarization:
#     """Solarization as a callable object."""
#
#     def __call__(self, img: Image) -> Image:
#         """Applies solarization to an input image.
#
#         Args:
#             img (Image): an image in the PIL.Image format.
#
#         Returns:
#             Image: solarized image.
#         """
#
#         return ImageOps.solarize(img)
#
#
# class Equalization:
#     def __call__(self, img: Image) -> Image:
#         return ImageOps.equalize(img)
#
#
# class NCropAugmentation:
#     def __init__(self, transform: Callable, num_crops: int):
#         """Creates a pipeline that apply a transformation pipeline multiple times.
#
#         Args:
#             transform (Callable): transformation pipeline.
#             num_crops (int): number of crops to create from the transformation pipeline.
#         """
#
#         self.transform = transform
#         self.num_crops = num_crops
#
#     def __call__(self, x: Image) -> List[torch.Tensor]:
#         """Applies transforms n times to generate n crops.
#
#         Args:
#             x (Image): an image in the PIL.Image format.
#
#         Returns:
#             List[torch.Tensor]: an image in the tensor format.
#         """
#
#         return [self.transform(x) for _ in range(self.num_crops)]
#
#     def __repr__(self) -> str:
#         return f"{self.num_crops} x [{self.transform}]"
#
#
# class FullTransformPipeline:
#     def __init__(self, transforms: Callable, dali=False,clip=None,coin05=None) -> None:
#         self.transforms = transforms
#         self.dali = dali
#         if not dali:
#             self.not_aug_transform = torchvision.transforms.Compose([transforms[0].transform.transform.transforms[0]]+
#                 transforms[0].transform.transform.transforms[-2:])
#         else:
#             self.cmn = transforms[0].transform.cmn
#             self.coin05 = coin05
#             self.clip = clip
#     def __call__(self, x: Image) -> List[torch.Tensor]:
#         """Applies transforms n times to generate n crops.
#
#         Args:
#             x (Image): an image in the PIL.Image format.
#
#         Returns:
#             List[torch.Tensor]: an image in the tensor format.
#         """
#
#         out = []
#         for transform in self.transforms:
#             out.extend(transform(x))
#         if not self.dali:
#             out.extend([self.not_aug_transform(x)])
#         else:
#             out.extend([self.cmn(self.clip(x),mirror=self.coin05())])
#         return out
#
#     def __repr__(self) -> str:
#         return "\n".join([str(transform) for transform in self.transforms])
#
#
# class BaseTransform:
#     """Adds callable base class to implement different transformation pipelines."""
#
#     def __call__(self, x: Image) -> torch.Tensor:
#         return self.transform(x)
#
#     def __repr__(self) -> str:
#         return str(self.transform)
#
#
# class CifarTransform(BaseTransform):
#     def __init__(
#         self,
#         cifar: str,
#         brightness: float,
#         contrast: float,
#         saturation: float,
#         hue: float,
#         color_jitter_prob: float = 0.8,
#         gray_scale_prob: float = 0.2,
#         horizontal_flip_prob: float = 0.5,
#         gaussian_prob: float = 0.5,
#         solarization_prob: float = 0.0,
#         equalization_prob: float = 0.0,
#         min_scale: float = 0.08,
#         max_scale: float = 1.0,
#         crop_size: int = 32,
#     ):
#         """Class that applies Cifar10/Cifar100 transformations.
#
#         Args:
#             cifar (str): type of cifar, either cifar10 or cifar100.
#             brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
#             contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
#             saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
#             hue (float): sampled uniformly in [-hue, hue].
#             color_jitter_prob (float, optional): probability of applying color jitter.
#                 Defaults to 0.8.
#             gray_scale_prob (float, optional): probability of converting to gray scale.
#                 Defaults to 0.2.
#             horizontal_flip_prob (float, optional): probability of flipping horizontally.
#                 Defaults to 0.5.
#             gaussian_prob (float, optional): probability of applying gaussian blur.
#                 Defaults to 0.0.
#             solarization_prob (float, optional): probability of applying solarization.
#                 Defaults to 0.0.
#             equalization_prob (float, optional): probability of applying equalization.
#                 Defaults to 0.0.
#             min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
#             max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
#             crop_size (int, optional): size of the crop. Defaults to 32.
#         """
#
#         super().__init__()
#
#         if cifar == "cifar10":
#             mean = (0.4914, 0.4822, 0.4465)
#             std = (0.2470, 0.2435, 0.2615)
#         else:
#             mean = (0.5071, 0.4865, 0.4409)
#             std = (0.2673, 0.2564, 0.2762)
#
#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     (crop_size, crop_size),
#                     scale=(min_scale, max_scale),
#                     interpolation=transforms.InterpolationMode.BICUBIC,
#                 ),
#                 transforms.RandomApply(
#                     [transforms.ColorJitter(brightness, contrast, saturation, hue)],
#                     p=color_jitter_prob,
#                 ),
#                 transforms.RandomGrayscale(p=gray_scale_prob),
#                 transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
#                 transforms.RandomApply([Solarization()], p=solarization_prob),
#                 transforms.RandomApply([Equalization()], p=equalization_prob),
#                 transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std),
#             ]
#         )
#
#
# class STLTransform(BaseTransform):
#     def __init__(
#         self,
#         brightness: float,
#         contrast: float,
#         saturation: float,
#         hue: float,
#         color_jitter_prob: float = 0.8,
#         gray_scale_prob: float = 0.2,
#         horizontal_flip_prob: float = 0.5,
#         gaussian_prob: float = 0.5,
#         solarization_prob: float = 0.0,
#         equalization_prob: float = 0.0,
#         min_scale: float = 0.08,
#         max_scale: float = 1.0,
#         crop_size: int = 96,
#     ):
#         """Class that applies STL10 transformations.
#
#         Args:
#             brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
#             contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
#             saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
#             hue (float): sampled uniformly in [-hue, hue].
#             color_jitter_prob (float, optional): probability of applying color jitter.
#                 Defaults to 0.8.
#             gray_scale_prob (float, optional): probability of converting to gray scale.
#                 Defaults to 0.2.
#             horizontal_flip_prob (float, optional): probability of flipping horizontally.
#                 Defaults to 0.5.
#             gaussian_prob (float, optional): probability of applying gaussian blur.
#                 Defaults to 0.0.
#             solarization_prob (float, optional): probability of applying solarization.
#                 Defaults to 0.0.
#             equalization_prob (float, optional): probability of applying equalization.
#                 Defaults to 0.0.
#             min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
#             max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
#             crop_size (int, optional): size of the crop. Defaults to 96.
#         """
#
#         super().__init__()
#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     (crop_size, crop_size),
#                     scale=(min_scale, max_scale),
#                     interpolation=transforms.InterpolationMode.BICUBIC,
#                 ),
#                 transforms.RandomApply(
#                     [transforms.ColorJitter(brightness, contrast, saturation, hue)],
#                     p=color_jitter_prob,
#                 ),
#                 transforms.RandomGrayscale(p=gray_scale_prob),
#                 transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
#                 transforms.RandomApply([Solarization()], p=solarization_prob),
#                 transforms.RandomApply([Equalization()], p=equalization_prob),
#                 transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
#             ]
#         )
#
#
# class ImagenetTransform(BaseTransform):
#     def __init__(
#         self,
#         brightness: float,
#         contrast: float,
#         saturation: float,
#         hue: float,
#         color_jitter_prob: float = 0.8,
#         gray_scale_prob: float = 0.2,
#         horizontal_flip_prob: float = 0.5,
#         gaussian_prob: float = 0.5,
#         solarization_prob: float = 0.0,
#         equalization_prob: float = 0.0,
#         min_scale: float = 0.08,
#         max_scale: float = 1.0,
#         crop_size: int = 224,
#     ):
#         """Class that applies Imagenet transformations.
#
#         Args:
#             brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
#             contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
#             saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
#             hue (float): sampled uniformly in [-hue, hue].
#             color_jitter_prob (float, optional): probability of applying color jitter.
#                 Defaults to 0.8.
#             gray_scale_prob (float, optional): probability of converting to gray scale.
#                 Defaults to 0.2.
#             horizontal_flip_prob (float, optional): probability of flipping horizontally.
#                 Defaults to 0.5.
#             gaussian_prob (float, optional): probability of applying gaussian blur.
#                 Defaults to 0.0.
#             solarization_prob (float, optional): probability of applying solarization.
#                 Defaults to 0.0.
#             equalization_prob (float, optional): probability of applying equalization.
#                 Defaults to 0.0.
#             min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
#             max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
#             crop_size (int, optional): size of the crop. Defaults to 224.
#         """
#
#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     crop_size,
#                     scale=(min_scale, max_scale),
#                     interpolation=transforms.InterpolationMode.BICUBIC,
#                 ),
#                 transforms.RandomApply(
#                     [transforms.ColorJitter(brightness, contrast, saturation, hue)],
#                     p=color_jitter_prob,
#                 ),
#                 transforms.RandomGrayscale(p=gray_scale_prob),
#                 transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
#                 transforms.RandomApply([Solarization()], p=solarization_prob),
#                 transforms.RandomApply([Equalization()], p=equalization_prob),
#                 transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
#             ]
#         )
#
#
# class CustomTransform(BaseTransform):
#     def __init__(
#         self,
#         brightness: float,
#         contrast: float,
#         saturation: float,
#         hue: float,
#         color_jitter_prob: float = 0.8,
#         gray_scale_prob: float = 0.2,
#         horizontal_flip_prob: float = 0.5,
#         gaussian_prob: float = 0.5,
#         solarization_prob: float = 0.0,
#         equalization_prob: float = 0.0,
#         min_scale: float = 0.08,
#         max_scale: float = 1.0,
#         crop_size: int = 224,
#         mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
#         std: Sequence[float] = IMAGENET_DEFAULT_STD,
#     ):
#         """Class that applies Custom transformations.
#         If you want to do exoteric augmentations, you can just re-write this class.
#
#         Args:
#             brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
#             contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
#             saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
#             hue (float): sampled uniformly in [-hue, hue].
#             color_jitter_prob (float, optional): probability of applying color jitter.
#                 Defaults to 0.8.
#             gray_scale_prob (float, optional): probability of converting to gray scale.
#                 Defaults to 0.2.
#             horizontal_flip_prob (float, optional): probability of flipping horizontally.
#                 Defaults to 0.5.
#             gaussian_prob (float, optional): probability of applying gaussian blur.
#                 Defaults to 0.0.
#             solarization_prob (float, optional): probability of applying solarization.
#                 Defaults to 0.0.
#             equalization_prob (float, optional): probability of applying equalization.
#                 Defaults to 0.0.
#             min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
#             max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
#             crop_size (int, optional): size of the crop. Defaults to 224.
#             mean (Sequence[float], optional): mean values for normalization.
#                 Defaults to IMAGENET_DEFAULT_MEAN.
#             std (Sequence[float], optional): std values for normalization.
#                 Defaults to IMAGENET_DEFAULT_STD.
#         """
#
#         super().__init__()
#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     crop_size,
#                     scale=(min_scale, max_scale),
#                     interpolation=transforms.InterpolationMode.BICUBIC,
#                 ),
#                 transforms.RandomApply(
#                     [transforms.ColorJitter(brightness, contrast, saturation, hue)],
#                     p=color_jitter_prob,
#                 ),
#                 transforms.RandomGrayscale(p=gray_scale_prob),
#                 transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
#                 transforms.RandomApply([Solarization()], p=solarization_prob),
#                 transforms.RandomApply([Equalization()], p=equalization_prob),
#                 transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std),
#             ]
#         )
#
#
# class TinyImagenetTransform(BaseTransform):
#     def __init__(
#         self,
#         brightness: float,
#         contrast: float,
#         saturation: float,
#         hue: float,
#         color_jitter_prob: float = 0.8,
#         gray_scale_prob: float = 0.2,
#         horizontal_flip_prob: float = 0.5,
#         gaussian_prob: float = 0.5,
#         solarization_prob: float = 0.0,
#         equalization_prob: float = 0.0,
#         min_scale: float = 0.08,
#         max_scale: float = 1.0,
#         crop_size: int = 64,
#         mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
#         std: Sequence[float] = IMAGENET_DEFAULT_STD,
#     ):
#         """Class that applies Custom transformations.
#         If you want to do exoteric augmentations, you can just re-write this class.
#
#         Args:
#             brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
#             contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
#             saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
#             hue (float): sampled uniformly in [-hue, hue].
#             color_jitter_prob (float, optional): probability of applying color jitter.
#                 Defaults to 0.8.
#             gray_scale_prob (float, optional): probability of converting to gray scale.
#                 Defaults to 0.2.
#             horizontal_flip_prob (float, optional): probability of flipping horizontally.
#                 Defaults to 0.5.
#             gaussian_prob (float, optional): probability of applying gaussian blur.
#                 Defaults to 0.0.
#             solarization_prob (float, optional): probability of applying solarization.
#                 Defaults to 0.0.
#             equalization_prob (float, optional): probability of applying equalization.
#                 Defaults to 0.0.
#             min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
#             max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
#             crop_size (int, optional): size of the crop. Defaults to 224.
#             mean (Sequence[float], optional): mean values for normalization.
#                 Defaults to IMAGENET_DEFAULT_MEAN.
#             std (Sequence[float], optional): std values for normalization.
#                 Defaults to IMAGENET_DEFAULT_STD.
#         """
#
#         super().__init__()
#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     crop_size,
#                     scale=(min_scale, max_scale),
#                     interpolation=transforms.InterpolationMode.BICUBIC,
#                 ),
#                 transforms.RandomApply(
#                     [transforms.ColorJitter(brightness, contrast, saturation, hue)],
#                     p=color_jitter_prob,
#                 ),
#                 transforms.RandomGrayscale(p=gray_scale_prob),
#                 transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
#                 transforms.RandomApply([Solarization()], p=solarization_prob),
#                 transforms.RandomApply([Equalization()], p=equalization_prob),
#                 transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std),
#             ]
#         )
#
#
# class MiniImagenetTransform(BaseTransform):
#     def __init__(
#         self,
#         brightness: float,
#         contrast: float,
#         saturation: float,
#         hue: float,
#         color_jitter_prob: float = 0.8,
#         gray_scale_prob: float = 0.2,
#         horizontal_flip_prob: float = 0.5,
#         gaussian_prob: float = 0.5,
#         solarization_prob: float = 0.0,
#         equalization_prob: float = 0.0,
#         min_scale: float = 0.08,
#         max_scale: float = 1.0,
#         crop_size: int = 84,
#         mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
#         std: Sequence[float] = IMAGENET_DEFAULT_STD,
#     ):
#         """Class that applies Custom transformations.
#         If you want to do exoteric augmentations, you can just re-write this class.
#
#         Args:
#             brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
#             contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
#             saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
#             hue (float): sampled uniformly in [-hue, hue].
#             color_jitter_prob (float, optional): probability of applying color jitter.
#                 Defaults to 0.8.
#             gray_scale_prob (float, optional): probability of converting to gray scale.
#                 Defaults to 0.2.
#             horizontal_flip_prob (float, optional): probability of flipping horizontally.
#                 Defaults to 0.5.
#             gaussian_prob (float, optional): probability of applying gaussian blur.
#                 Defaults to 0.0.
#             solarization_prob (float, optional): probability of applying solarization.
#                 Defaults to 0.0.
#             equalization_prob (float, optional): probability of applying equalization.
#                 Defaults to 0.0.
#             min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
#             max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
#             crop_size (int, optional): size of the crop. Defaults to 224.
#             mean (Sequence[float], optional): mean values for normalization.
#                 Defaults to IMAGENET_DEFAULT_MEAN.
#             std (Sequence[float], optional): std values for normalization.
#                 Defaults to IMAGENET_DEFAULT_STD.
#         """
#
#         super().__init__()
#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     crop_size,
#                     scale=(min_scale, max_scale),
#                     interpolation=transforms.InterpolationMode.BICUBIC,
#                 ),
#                 transforms.RandomApply(
#                     [transforms.ColorJitter(brightness, contrast, saturation, hue)],
#                     p=color_jitter_prob,
#                 ),
#                 transforms.RandomGrayscale(p=gray_scale_prob),
#                 transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
#                 transforms.RandomApply([Solarization()], p=solarization_prob),
#                 transforms.RandomApply([Equalization()], p=equalization_prob),
#                 transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std),
#             ]
#         )
#
# def prepare_transform(dataset: str, **kwargs) -> Any:
#     """Prepares transforms for a specific dataset. Optionally uses multi crop.
#
#     Args:
#         dataset (str): name of the dataset.
#
#     Returns:
#         Any: a transformation for a specific dataset.
#     """
#
#     if dataset in ["cifar10", "cifar100"]:
#         return CifarTransform(cifar=dataset, **kwargs)
#     elif dataset == "stl10":
#         return STLTransform(**kwargs)
#     elif dataset == "tinyimagenet":
#         return TinyImagenetTransform(**kwargs)
#     elif dataset == "miniimagenet":
#         return MiniImagenetTransform(**kwargs)
#     elif dataset in ["imagenet", "imagenet100"]:
#         return ImagenetTransform(**kwargs)
#     elif dataset == "custom":
#         return CustomTransform(**kwargs)
#     else:
#         raise ValueError(f"{dataset} is not currently supported.")
#
#
# def prepare_n_crop_transform(
#     transforms: List[Callable], num_crops_per_aug: List[int]
# ) -> NCropAugmentation:
#     """Turns a single crop transformation to an N crops transformation.
#
#     Args:
#         transforms (List[Callable]): list of transformations.
#         num_crops_per_aug (List[int]): number of crops per pipeline.
#
#     Returns:
#         NCropAugmentation: an N crop transformation.
#     """
#
#     assert len(transforms) == len(num_crops_per_aug)
#
#     T = []
#     for transform, num_crops in zip(transforms, num_crops_per_aug):
#         T.append(NCropAugmentation(transform, num_crops))
#     return FullTransformPipeline(T)
#
#
# def prepare_datasets(
#     dataset: str,
#     transform: Callable,
#     train_data_path: Optional[Union[str, Path]] = None,
#     data_format: Optional[str] = "image_folder",
#     no_labels: Optional[Union[str, Path]] = False,
#     download: bool = True,
#     data_fraction: float = -1.0,
# ) -> Dataset:
#     """Prepares the desired dataset.
#
#     Args:
#         dataset (str): the name of the dataset.
#         transform (Callable): a transformation.
#         train_dir (Optional[Union[str, Path]]): training data path. Defaults to None.
#         data_format (Optional[str]): format of the data. Defaults to "image_folder".
#             Possible values are "image_folder" and "h5".
#         no_labels (Optional[bool]): if the custom dataset has no labels.
#         data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
#             Defaults to -1.0.
#     Returns:
#         Dataset: the desired dataset with transformations.
#     """
#
#     if train_data_path is None:
#         sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#         train_data_path = sandbox_folder / "datasets"
#
#     if dataset in ["cifar10", "cifar100"]:
#         DatasetClass = vars(torchvision.datasets)[dataset.upper()]
#         train_dataset = dataset_with_index(DatasetClass)(
#             train_data_path,
#             train=True,
#             download=download,
#             transform=transform,
#         )
#
#     elif dataset == "stl10":
#         train_dataset = dataset_with_index(STL10)(
#             train_data_path,
#             split="train+unlabeled",
#             download=download,
#             transform=transform,
#         )
#
#     elif dataset == "tinyimagenet":
#         train_dataset = dataset_with_index(TinyImagenet)(
#             train_data_path,
#             train=True,
#             transform=transform,
#         )
#     elif dataset == "miniimagenet":
#         train_dataset = dataset_with_index(MiniImageNet)(
#             train_data_path,
#             train=True,
#             transform=transform,
#         )
#     elif dataset in ["imagenet", "imagenet100"]:
#         if data_format == "h5":
#             assert _h5_available
#             train_dataset = dataset_with_index(H5Dataset)(dataset, train_data_path, transform)
#         else:
#             train_dataset = dataset_with_index(ImageFolder)(train_data_path, transform)
#
#     elif dataset == "custom":
#         if no_labels:
#             dataset_class = CustomDatasetWithoutLabels
#         else:
#             dataset_class = ImageFolder
#
#         train_dataset = dataset_with_index(dataset_class)(train_data_path, transform)
#
#     if data_fraction > 0:
#         assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
#         data = train_dataset.samples
#         files = [f for f, _ in data]
#         labels = [l for _, l in data]
#
#         from sklearn.model_selection import train_test_split
#
#         files, _, labels, _ = train_test_split(
#             files, labels, train_size=data_fraction, stratify=labels, random_state=42
#         )
#         train_dataset.samples = [tuple(p) for p in zip(files, labels)]
#
#     return train_dataset
#
#
# def prepare_dataloader(
#     train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
# ) -> DataLoader:
#     """Prepares the training dataloader for pretraining.
#     Args:
#         train_dataset (Dataset): the name of the dataset.
#         batch_size (int, optional): batch size. Defaults to 64.
#         num_workers (int, optional): number of workers. Defaults to 4.
#     Returns:
#         DataLoader: the training dataloader with the desired dataset.
#     """
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )
#     return train_loader
#
# def prepare_cl_dataloader(
#     train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4, task_id: int = 0, class_per_task: int = 10,
# ) -> DataLoader:
#     """Prepares the training dataloader for pretraining.
#     Args:
#         train_dataset (Dataset): the name of the dataset.
#         batch_size (int, optional): batch size. Defaults to 64.
#         num_workers (int, optional): number of workers. Defaults to 4.
#     Returns:
#         DataLoader: the training dataloader with the desired dataset.
#     """
#     train_indice1 = np.where(np.array(train_dataset.targets) >= task_id*class_per_task)[0]
#     train_indice2 = np.where(np.array(train_dataset.targets) < (task_id+1) *class_per_task)[0]
#     train_indice = np.intersect1d(train_indice1,train_indice2)
#
#     test_indice1 = np.where(np.array(val_dataset.targets) >= task_id*class_per_task)[0]
#     test_indice2 = np.where(np.array(val_dataset.targets) < (task_id+1) *class_per_task)[0]
#     test_indice = np.intersect1d(test_indice1,test_indice2)
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True,
#         sampler=SubsetRandomSampler(train_indice)
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=False,
#         sampler=SubsetRandomSampler(test_indice)
#     )
#     return train_loader, val_loader


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

import os
import random
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Type, Union
import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
try:
    from solo.data.h5_dataset import H5Dataset
    from solo.data.tinyimagenet import TinyImagenet, MiniImageNet
except ImportError:
    _h5_available = False
else:
    _h5_available = True




def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex


class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = os.listdir(root)

    def __getitem__(self, index):
        path = self.root / self.images[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, -1

    def __len__(self):
        return len(self.images)


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: solarized image.
        """

        return ImageOps.solarize(img)


class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)


class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"


class FullTransformPipeline:
    def __init__(self, transforms: Callable, dali=False,clip=None,coin05=None) -> None:
        self.transforms = transforms
        self.dali = dali
        if not dali:
            self.not_aug_transform = torchvision.transforms.Compose([transforms[0].transform.transform.transforms[0]]+
                transforms[0].transform.transform.transforms[-2:])
        else:
            self.cmn = transforms[0].transform.cmn
            self.coin05 = coin05
            self.clip = clip
    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for transform in self.transforms:
            out.extend(transform(x))
        if not self.dali:
            out.extend([self.not_aug_transform(x)])
        else:
            out.extend([self.cmn(self.clip(x),mirror=self.coin05())])
        return out

    def __repr__(self) -> str:
        return "\n".join([str(transform) for transform in self.transforms])


class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

    def __repr__(self) -> str:
        return str(self.transform)


class CifarTransform(BaseTransform):
    def __init__(
        self,
        cifar: str,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 32,
    ):
        """Class that applies Cifar10/Cifar100 transformations.

        Args:
            cifar (str): type of cifar, either cifar10 or cifar100.
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            equalization_prob (float, optional): probability of applying equalization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 32.
        """

        super().__init__()

        if cifar == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2615)
        else:
            mean = (0.5071, 0.4865, 0.4409)
            std = (0.2673, 0.2564, 0.2762)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


class STLTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 96,
    ):
        """Class that applies STL10 transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            equalization_prob (float, optional): probability of applying equalization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 96.
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )


class ImagenetTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 224,
    ):
        """Class that applies Imagenet transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            equalization_prob (float, optional): probability of applying equalization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
        """

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )


class CustomTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 224,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
    ):
        """Class that applies Custom transformations.
        If you want to do exoteric augmentations, you can just re-write this class.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            equalization_prob (float, optional): probability of applying equalization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
            mean (Sequence[float], optional): mean values for normalization.
                Defaults to IMAGENET_DEFAULT_MEAN.
            std (Sequence[float], optional): std values for normalization.
                Defaults to IMAGENET_DEFAULT_STD.
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


class TinyImagenetTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 64,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
    ):
        """Class that applies Custom transformations.
        If you want to do exoteric augmentations, you can just re-write this class.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            equalization_prob (float, optional): probability of applying equalization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
            mean (Sequence[float], optional): mean values for normalization.
                Defaults to IMAGENET_DEFAULT_MEAN.
            std (Sequence[float], optional): std values for normalization.
                Defaults to IMAGENET_DEFAULT_STD.
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


class MiniImagenetTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 84,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
    ):
        """Class that applies Custom transformations.
        If you want to do exoteric augmentations, you can just re-write this class.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            equalization_prob (float, optional): probability of applying equalization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
            mean (Sequence[float], optional): mean values for normalization.
                Defaults to IMAGENET_DEFAULT_MEAN.
            std (Sequence[float], optional): std values for normalization.
                Defaults to IMAGENET_DEFAULT_STD.
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

class SVHNTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 32,
    ):
        """
        Transformation class for the SVHN dataset.

        Args:
            brightness, contrast, saturation, hue: Parameters for color jitter.
            color_jitter_prob, gray_scale_prob, horizontal_flip_prob, gaussian_prob:
                Probabilities for applying augmentations.
            solarization_prob, equalization_prob: Probabilities for additional augmentations.
            min_scale, max_scale: Minimum and maximum scale for resizing.
            crop_size: Size of the crop.
        """
        super().__init__()

        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

class FMNISTTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 28,  # FMNIST's default size is 28x28
    ):
        """
        Transformation class for the FashionMNIST dataset.

        Args:
            brightness, contrast, saturation, hue: Parameters for color jitter.
            color_jitter_prob, gray_scale_prob, horizontal_flip_prob, gaussian_prob:
                Probabilities for applying augmentations.
            solarization_prob, equalization_prob: Probabilities for additional augmentations.
            min_scale, max_scale: Minimum and maximum scale for resizing.
            crop_size: Size of the crop.
        """
        super().__init__()

        mean = (0.2860,)
        std = (0.3530,)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

class CaltechTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 224,
    ):
        """
        Transformation class for the Caltech-101/256 dataset.

        Args:
            brightness, contrast, saturation, hue: Parameters for color jitter.
            color_jitter_prob, gray_scale_prob, horizontal_flip_prob, gaussian_prob:
                Probabilities for applying augmentations.
            solarization_prob, equalization_prob: Probabilities for additional augmentations.
            min_scale, max_scale: Minimum and maximum scale for resizing.
            crop_size: Size of the crop.
        """
        super().__init__()

        # ImageNet-like normalization
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )



def prepare_transform(dataset: str, **kwargs) -> Any:
    """Prepares transforms for a specific dataset. Optionally uses multi crop.

    Args:
        dataset (str): name of the dataset.

    Returns:
        Any: a transformation for a specific dataset.
    """

    if dataset in ["cifar10", "cifar100"]:
        return CifarTransform(cifar=dataset, **kwargs)
    elif dataset == "stl10":
        return STLTransform(**kwargs)
    elif dataset == "tinyimagenet":
        return TinyImagenetTransform(**kwargs)
    elif dataset == "miniimagenet":
        return MiniImagenetTransform(**kwargs)
    elif dataset == "svhn":
        return SVHNTransform(**kwargs)
    elif dataset == "fmnist":
        return FMNISTTransform(**kwargs)
    elif dataset in ["imagenet", "imagenet100"]:
        return ImagenetTransform(**kwargs)
    elif dataset == "caltech101" or dataset == "caltech256":
        return CaltechTransform(**kwargs)
    elif dataset == "custom":
        return CustomTransform(**kwargs)
    else:
        raise ValueError(f"{dataset} is not currently supported.")


def prepare_n_crop_transform(
    transforms: List[Callable], num_crops_per_aug: List[int]
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    assert len(transforms) == len(num_crops_per_aug)

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))
    return FullTransformPipeline(T)


def prepare_datasets(
    dataset: str,
    transform: Callable,
    train_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    no_labels: Optional[Union[str, Path]] = False,
    download: bool = True,
    data_fraction: float = -1.0,
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        train_dir (Optional[Union[str, Path]]): training data path. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        no_labels (Optional[bool]): if the custom dataset has no labels.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
    Returns:
        Dataset: the desired dataset with transformations.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = dataset_with_index(DatasetClass)(
            train_data_path,
            train=True,
            download=download,
            transform=transform,
        )

    elif dataset == "stl10":
        train_dataset = dataset_with_index(STL10)(
            train_data_path,
            split="train+unlabeled",
            download=download,
            transform=transform,
        )

    elif dataset == "tinyimagenet":
        train_dataset = dataset_with_index(TinyImagenet)(
            train_data_path,
            train=True,
            transform=transform,
        )
    elif dataset == "miniimagenet":
        train_dataset = dataset_with_index(MiniImageNet)(
            train_data_path,
            train=True,
            transform=transform,
        )
    elif dataset == "svhn":
        DatasetClass = torchvision.datasets.SVHN
        train_dataset = dataset_with_index(DatasetClass)(
            train_data_path,
            split="train",
            download=download,
            transform=transform,
        )

    # elif dataset == "fmnist":
    #     DatasetClass = torchvision.datasets.FashionMNIST
    #     train_dataset = dataset_with_index(DatasetClass)(
    #         train_data_path,
    #         train=True,
    #         download=download,
    #         transform=transform,
    #     )
    elif dataset == "fmnist":
        class FashionMNISTWithIndex(torchvision.datasets.FashionMNIST):
            def __getitem__(self, index):
                img, target = super().__getitem__(index)  # すでに transform が適用されている
                if not isinstance(img, torch.Tensor):  # img が Tensor でない場合に限り PIL に変換
                    img = Image.fromarray(img, mode="L")
                return index, img, target

        DatasetClass = FashionMNISTWithIndex
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = DatasetClass(
            train_data_path,
            train=True,
            download=True,
            transform=transform,
        )
    elif dataset == "caltech101" or dataset == "caltech256":


        DatasetClass = torchvision.datasets.ImageFolder
        train_dataset = dataset_with_index(DatasetClass)(
            train_data_path,
            transform=transform,
        )
        # val_dataset = dataset_with_index(DatasetClass)(
        #     val_data_path,
        #     transform=transform,
        # )

    elif dataset in ["imagenet", "imagenet100"]:
        if data_format == "h5":
            assert _h5_available
            train_dataset = dataset_with_index(H5Dataset)(dataset, train_data_path, transform)
        else:
            train_dataset = dataset_with_index(ImageFolder)(train_data_path, transform)

    elif dataset == "custom":
        if no_labels:
            dataset_class = CustomDatasetWithoutLabels
        else:
            dataset_class = ImageFolder

        train_dataset = dataset_with_index(dataset_class)(train_data_path, transform)



    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        data = train_dataset.samples
        files = [f for f, _ in data]
        labels = [l for _, l in data]

        from sklearn.model_selection import train_test_split

        files, _, labels, _ = train_test_split(
            files, labels, train_size=data_fraction, stratify=labels, random_state=42
        )
        train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset


def prepare_dataloader(
    train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader

def prepare_cl_dataloader(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4, task_id: int = 0, class_per_task: int = 10,
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """
    train_indice1 = np.where(np.array(train_dataset.targets) >= task_id*class_per_task)[0]
    train_indice2 = np.where(np.array(train_dataset.targets) < (task_id+1) *class_per_task)[0]
    train_indice = np.intersect1d(train_indice1,train_indice2)

    test_indice1 = np.where(np.array(val_dataset.targets) >= task_id*class_per_task)[0]
    test_indice2 = np.where(np.array(val_dataset.targets) < (task_id+1) *class_per_task)[0]
    test_indice = np.intersect1d(test_indice1,test_indice2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=SubsetRandomSampler(train_indice)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=SubsetRandomSampler(test_indice)
    )
    return train_loader, val_loader
