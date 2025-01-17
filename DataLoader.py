import copy
import os
import json
import math
import random
import cv2
import tqdm
import glob
import torch
import numpy as np
import albumentations as A

from typing import List
from albumentations.pytorch import ToTensorV2
from utils import get_boxes_from_mask, init_point_sampling, \
    get_edge_points_from_mask, get_transform
from torch.utils.data import Dataset, DataLoader, Sampler
from preprocess.split_dataset import split_dataset


class TestingDataset(Dataset):
    
    def __init__(self, split_paths, requires_name: bool = True,
                 point_num: int = 1, edge_point_num: int = 3,
                 return_ori_mask: bool = True, prompt_path: str = None,
                 sample_rate: float = 1.0):
        """
        Initializes a TestingDataset object.
        Args:
            requires_name (bool, optional): Indicates whether the dataset
                requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve.
                Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return
                the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file.
                Defaults to None.
        """
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None \
            else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num
        self.edge_point_num = edge_point_num
        self.split_paths = split_paths
        self.image_paths, self.label_paths = self.read_data()

        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        self.sample_rate = sample_rate
        self.sample_indices = random.choices(
            range(len(self.image_paths)),
            k=int(len(self.image_paths) * self.sample_rate)
        )

    def read_data(self):
        image_paths = []
        label_paths = []

        for i, split_path in enumerate(self.split_paths):
            with open(split_path, "r") as f:
                split_data = json.load(f)

            data_root = os.path.dirname(split_path)
            # print(f"Read test data from: {data_root}")
            for data_path, label_path in split_data["test"]:
                data_path = os.path.join(data_root, data_path)
                label_path = os.path.join(data_root, label_path)
                label = np.load(label_path)
                vals = sorted([val for val in np.unique(label) if val != 0])
                for val in vals:
                    image_paths.append(data_path)
                    label_paths.append((val, label_path))

        return image_paths, label_paths
    
    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and
                associated information.
        """
        index = self.sample_indices[index]

        image_input = {}
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = (image - self.pixel_mean) / self.pixel_std

        mask_val, mask_path = self.label_paths[index]
        ori_np_mask = np.load(mask_path).astype(np.float32)
        binary_mask = ori_np_mask.copy()
        binary_mask[ori_np_mask != mask_val] = 0
        binary_mask[ori_np_mask == mask_val] = 1

        assert np.array_equal(binary_mask, binary_mask.astype(bool)), \
            (f"Mask should only contain binary values 0 and 1. "
             f"{self.label_paths[index]}")

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(binary_mask).unsqueeze(0) # ori_mask = torch.tensor(cv2.resize(ori_np_mask, (self.image_size, self.image_size))).unsqueeze(0)

        transforms = A.Compose([ToTensorV2(p=1.0)], p=1.)
        augments = transforms(image=image, mask=binary_mask)
        image, mask = augments['image'], augments['mask'].to(torch.int64)

        edge_point_coords = None
        if self.prompt_path is None:
            boxes = get_boxes_from_mask(mask, max_pixel=0)
            point_coords, point_labels = init_point_sampling(mask, self.point_num)
            if self.edge_point_num:
                edge_point_coords = get_edge_points_from_mask(mask_val, ori_np_mask,
                                                              self.edge_point_num)
        else:
            prompt_key = mask_path.split('/')[-1]
            boxes = torch.as_tensor(
                self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(
                self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(
                self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)
            if self.edge_point_num:
                edge_point_coords = torch.as_tensor(
                    self.prompt_list[prompt_key]["edges"], dtype=torch.float)

        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        if edge_point_coords is not None:
            image_input["edges"] = edge_point_coords
        image_input["original_size"] = (h, w)
        image_input["image_path"] = image_path
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])
        image_input["dataset_name"] = os.path.basename(
            os.path.dirname(
                os.path.dirname(image_path)
            )
        )

        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask
     
        image_name = self.label_paths[index][1].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.sample_indices)


class TrainingDataset(Dataset):

    def __init__(self, split_paths, requires_name: bool = True,
                 point_num: int = 1, mask_num: int = 5,
                 edge_point_num: int = 3):

        """
        Initializes a training dataset.
        Args:
            requires_name (bool, optional): Indicates whether to include
             image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.requires_name = requires_name
        self.point_num = point_num
        self.edge_point_num = edge_point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        self.image_paths = []
        self.label_paths = []

        split_paths = [split_paths] if isinstance(split_paths, str) else split_paths

        for i, split_path in enumerate(split_paths):
            with open(split_path, "r") as f:
                split_data = json.load(f)

            data_root = os.path.dirname(split_path)
            # print(f"Read train data from: {data_root}")
            for data_path, label_path in split_data["train"]:
                data_path = os.path.join(data_root, data_path)
                label_path = os.path.join(data_root, label_path)
                self.image_paths.append(data_path)
                self.label_paths.append(label_path)


    def __add__(self, other):
        instance = copy.deepcopy(self)
        instance.image_paths += other.image_paths
        instance.label_paths += other.label_paths

        return instance
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """
        while True:
            image_input = {}
            try:
                image_path = self.image_paths[index]
                image = cv2.imread(image_path)
                image = (image - self.pixel_mean) / self.pixel_std
            except:
                # print("read image error: ", self.image_paths[index])
                index = random.choice(range(self.__len__()))
                continue

            h, w, _ = image.shape
            transforms = A.Compose([ToTensorV2(p=1.0)], p=1.)

            masks_list = []
            boxes_list = []
            point_coords_list, point_labels_list = [], []
            edge_point_coords_list = []
            # mask_path = random.choices(self.label_paths[index], k=self.mask_num)

            mask_path = self.label_paths[index]
            if not os.path.exists(mask_path):
                old_index = index
                index = random.choice(range(self.__len__()))
                # print(f"mask {old_index} not exists, randon select {index}")
                continue
            original_mask = np.load(mask_path).astype(np.float32)
            mask_vals = [_ for _ in np.unique(original_mask) if _ != 0]
            if not mask_vals:
                old_index = index
                index = random.choice(range(self.__len__()))
                # print(f"mask {old_index} is empty, randon choose {index}")
                continue
            choices_nuclei = random.choices(mask_vals, k=self.mask_num)

            for mask_val in choices_nuclei:
                pre_mask = original_mask.copy()

                # 只考虑当前细胞核编号，其它编号细胞核都按照背景处理
                pre_mask[pre_mask != mask_val] = 0.0
                pre_mask[pre_mask == mask_val] = 1.0

                augments = transforms(image=image, mask=pre_mask)
                image_tensor, mask_tensor = (augments['image'],
                                             augments['mask'].to(torch.int64))

                boxes = get_boxes_from_mask(mask_tensor)
                point_coords, point_label = init_point_sampling(
                    mask_tensor,  self.point_num)

                masks_list.append(mask_tensor)
                boxes_list.append(boxes)
                point_coords_list.append(point_coords)
                point_labels_list.append(point_label)
                if self.edge_point_num:
                    edge_point_coords = get_edge_points_from_mask(
                        mask_val, original_mask, self.edge_point_num)
                    edge_point_coords_list.append(edge_point_coords)

            mask = torch.stack(masks_list, dim=0)
            boxes = torch.stack(boxes_list, dim=0)
            point_coords = torch.stack(point_coords_list, dim=0)
            point_labels = torch.stack(point_labels_list, dim=0)
            if edge_point_coords_list:
                edges = torch.stack(edge_point_coords_list, dim=0)
            else:
                edges = None

            image_input["image"] = image_tensor.unsqueeze(0)
            image_input["label"] = mask.unsqueeze(1)
            image_input["boxes"] = boxes
            image_input["point_coords"] = point_coords
            image_input["point_labels"] = point_labels
            if edges is not None:
                image_input["edges"] = edges
            image_input["image_path"] = image_path

            image_input["dataset_name"] = os.path.basename(
                os.path.dirname(
                    os.path.dirname(image_path)
                )
            )

            image_name = self.image_paths[index].split('/')[-1]
            if self.requires_name:
                image_input["name"] = image_name
                return image_input
            else:
                return image_input

    def __len__(self):
        return len(self.image_paths)


def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict


class DatasetFolderMixin:

    def is_dataset_dir(self, dir_path: str) -> bool:
        return os.path.exists(os.path.join(dir_path, "data")) and \
               os.path.exists(os.path.join(dir_path, "label"))

    def get_split_path(self, dir_path: str) -> str:
        return os.path.join(
            dir_path,
            f"split_seed-{self.random_seed}_test_size-{self.test_size}.json"
        )

    def get_split_paths(self):
        split_paths = []
        if self.is_dataset_dir(self.data_root):
            split_path = self.get_split_path(self.data_root)
            # if not os.path.exists(split_path):
            if True:
                split_dataset(
                    self.data_root,
                    ext="png",
                    test_size=self.test_size,
                    seed=self.random_seed,
                    split_path=split_path
                )
            split_paths.append(split_path)
        else:
            for name in os.listdir(self.data_root):
                sub_dir_path = os.path.join(self.data_root, name)
                if os.path.isdir(sub_dir_path):
                    split_path = self.get_split_path(sub_dir_path)
                    # if not os.path.exists(split_path):
                    if True:
                        split_dataset(
                            sub_dir_path,
                            ext="png",
                            test_size=self.test_size,
                            seed=self.random_seed,
                            split_path=split_path
                        )
                    split_paths.append(split_path)

        return split_paths


class TestingDatasetFolder(TestingDataset, DatasetFolderMixin):

    def __init__(self, data_root: str, test_size: float, requires_name=True,
                 point_num=1, edge_point_num=3, return_ori_mask=True, prompt_path=None,
                 random_seed: int = 42, sample_rate: float = 1.0):

        self.data_root = data_root
        self.test_size = test_size
        self.random_seed = random_seed

        super().__init__(
            split_paths=self.get_split_paths(), requires_name=requires_name,
            point_num=point_num, edge_point_num=edge_point_num, return_ori_mask=return_ori_mask,
            prompt_path=prompt_path, sample_rate=sample_rate
        )


class TrainingDatasetFolder(TrainingDataset, DatasetFolderMixin):

    def __init__(self, data_root: str, train_size: float,
                 requires_name: bool = True, point_num: int = 1,
                 mask_num: int = 5, edge_point_num: int = 3,
                 random_seed: int = 42):

        self.data_root = data_root
        self.test_size = 1 - train_size
        self.random_seed = random_seed

        super().__init__(
            split_paths=self.get_split_paths(), requires_name=requires_name,
            point_num=point_num, mask_num=mask_num,
            edge_point_num=edge_point_num
        )




