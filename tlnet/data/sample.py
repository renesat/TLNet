from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(
        self,
        path: Union[Path, str],
        balance: Union[bool, None, Tuple[float, float]] = None,
        transforms=None,
    ):
        if isinstance(path, str):
            path = Path(path)

        if balance is None:
            self.balance = None
        elif isinstance(balance, bool):
            self.balance = (0.5, 0.5)
        else:
            self.balance = (
                balance[0] / (balance[0] + balance[1]),
                balance[1] / (balance[0] + balance[1]),
            )

        self.transforms = transforms

        self.path = path
        neg_imgs = list((self.path / "negative").iterdir())
        pos_imgs = list((self.path / "positive").iterdir())

        if balance is not None:
            min_class = np.sign(len(neg_imgs) - len(pos_imgs))

            if min_class > 0:
                neg_imgs = self._balance(
                    neg_imgs,
                    int(self.balance[0] / self.balance[1] * len(pos_imgs)),
                )
            else:
                pos_imgs = self._balance(
                    pos_imgs,
                    int(self.balance[1] / self.balance[0] * len(neg_imgs)),
                )

        self.imgs = neg_imgs + pos_imgs
        self.cls = [0 for _ in range(len(neg_imgs))] + [1 for _ in range(len(pos_imgs))]
        self.annotation = pd.read_csv(path / "annotation.csv")

    def __getitem__(self, idx):
        img_path = self.imgs[idx]

        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)

        if self.cls[idx] == 0:
            res = {
                "image": img,
                "class": 0.0,
                "box": torch.Tensor((0, 0, 0, 0)).float(),
            }
        else:
            img_rel_path = str(Path(img_path.parent.name) / img_path.name)
            box = self.annotation.loc[self.annotation["file"] == img_rel_path, :].iloc[
                0, 1:
            ]
            res = {
                "image": img,
                "class": 1.0,
                "box": torch.Tensor(tuple(box)).float(),
            }
        return res

    def __len__(self):
        return len(self.imgs)

    def _balance(self, max_class: List[str], size: int):
        return list(
            np.random.choice(
                max_class,
                size=size,
                replace=False,
            )
        )
