import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from tlnet.model.roi import ROISelector

DEFAULT_BOX_SIZE = (128, 128)


def target_in_region(target_box, region):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(target_box[0], region[0])
    yA = max(target_box[1], region[1])
    xB = min(target_box[2], region[2])
    yB = min(target_box[3], region[3])

    target_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA) / target_area
    return interArea


def __get_random_region(
    img: torch.Tensor,
    point: Tuple[int, int],
    size: Tuple[int, int] = DEFAULT_BOX_SIZE,
):
    x_r = size[0] // 2
    y_r = size[1] // 2

    x_shift = (
        (
            torch.sign(torch.rand(1) * 2 - 1)
            * torch.randint(
                int(x_r * 0.1),
                x_r,
                (1,),
            )
        )
        .int()
        .item()
    )
    y_shift = (
        (
            torch.sign(torch.rand(1) * 2 - 1)
            * torch.randint(
                int(y_r * 0.1),
                x_r,
                (1,),
            )
        )
        .int()
        .item()
    )

    new_point = [
        point[0] + x_shift,
        point[1] + y_shift,
    ]

    if new_point[0] - x_r < 0:
        new_point[0] += x_r - new_point[0]
    if new_point[0] + x_r >= img.shape[1]:
        new_point[0] -= x_r - (img.shape[1] - new_point[0])
    if new_point[1] - y_r < 0:
        new_point[1] += y_r - new_point[1]
    if new_point[1] + y_r >= img.shape[2]:
        new_point[1] -= y_r - (img.shape[2] - new_point[1])

    box = (
        int(new_point[1] - y_r),
        int(new_point[0] - x_r),
        int(new_point[1] + y_r) - 1,
        int(new_point[0] + x_r) - 1,
    )
    return (
        img[
            :,
            int(new_point[0] - x_r) : int(new_point[0] + x_r),
            int(new_point[1] - y_r) : int(new_point[1] + y_r),
        ],
        box,
    )


def sample_generator(
    annotations: Dict[str, List[Tuple[int, ...]]],
    out_path: Union[Path, str],
    shuffle_size: Union[None, int] = None,
):

    if isinstance(out_path, str):
        out_path = Path(out_path)

    selector = ROISelector(train=True)

    negative_path = out_path / "negative"
    positive_path = out_path / "positive"

    out_path.mkdir(parents=True, exist_ok=True)
    negative_path.mkdir(parents=True, exist_ok=True)
    positive_path.mkdir(parents=True, exist_ok=True)

    sample_annotation = {
        "file": [],
        "up_x": [],
        "up_y": [],
        "down_x": [],
        "down_y": [],
    }
    index = 0
    files = list(annotations.items())
    if shuffle_size is not None:
        files = random.sample(
            files,
            shuffle_size,
        )

    for file_path, boxes in tqdm(files):
        pil_rgb_img = Image.open(file_path)
        pil_hsv_img = pil_rgb_img.convert("HSV")

        rgb_img = transforms.ToTensor()(pil_rgb_img)
        hsv_img = transforms.ToTensor()(pil_hsv_img)

        mask, rois = selector(rgb_img.unsqueeze(0), hsv_img.unsqueeze(0))
        mask = mask[0]
        rois = rois[0].detach().int().numpy()

        for i, roi in enumerate(rois):
            roi: Tuple[int, int] = tuple(roi[0])
            region, box = __get_random_region(rgb_img, roi)

            positive = False
            rbox = None
            if box[0] < roi[1] < box[2] and box[1] < roi[0] < box[3]:
                for real_box in boxes:
                    inter_area = target_in_region(real_box, box)
                    if inter_area > 0.5:
                        positive = True
                        rbox = real_box
                        break
            if positive:
                rel_file_path = (
                    Path(positive_path.name) / f"img_{Path(file_path).name}_{index}.png"
                )
            else:
                rel_file_path = (
                    Path(negative_path.name) / f"img_{Path(file_path).name}_{index}.png"
                )

            save_image(region, out_path / rel_file_path)
            if rbox is not None:
                sample_annotation["file"].append(str(rel_file_path))
                sample_annotation["up_x"].append(rbox[0] - box[0])
                sample_annotation["up_y"].append(rbox[1] - box[1])
                sample_annotation["down_x"].append(rbox[2] - box[0])
                sample_annotation["down_y"].append(rbox[3] - box[1])
            index += 1

            if positive:
                rel_file_path = (
                    Path(positive_path.name) / f"img_{Path(file_path).name}_{index}.png"
                )

                inter_area = 0
                while not (inter_area > 0.5):
                    region, box = __get_random_region(rgb_img, roi)
                    inter_area = target_in_region(rbox, box)

                save_image(region, out_path / rel_file_path)

                if rbox is not None:
                    sample_annotation["file"].append(str(rel_file_path))
                    sample_annotation["up_x"].append(rbox[0] - box[0])
                    sample_annotation["up_y"].append(rbox[1] - box[1])
                    sample_annotation["down_x"].append(rbox[2] - box[0])
                    sample_annotation["down_y"].append(rbox[3] - box[1])
                index += 1

    return pd.DataFrame(sample_annotation)
