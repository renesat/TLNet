from typing import Tuple

import cv2
import numpy as np
import torch

DEFAULT_RGB_THRESHOLDS = (0.2, 0.2, 0.5)
DEFAULT_HSV_THRESHOLDS = (0.5, 0.7)
DEFAULT_BRIGHT_THRESHOLD = 0.3
DEFAULT_WINDOWS_SIZE = (64, 64)
DEFAULT_WINDOWS_STEP = (32, 32)


class ROISelector:

    # TODO
    def __init__(
        self,
        train: bool = False,
        rgb_thresholds: Tuple[float, float, float] = DEFAULT_RGB_THRESHOLDS,
        hsv_thresholds: Tuple[float, float] = DEFAULT_HSV_THRESHOLDS,
        bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD,
    ):
        self.train = train
        self.rgb_thresholds = rgb_thresholds
        self.hsv_thresholds = hsv_thresholds
        self.bright_threshold = bright_threshold

    def _get_mask(self, region: torch.Tensor, threshold: float):
        mean_value = region.mean(dim=(1, 2))
        mask = region > (mean_value + threshold).unsqueeze(1).unsqueeze(1)
        return mask

    def _global_mask(self, values: torch.Tensor, thresholds: Tuple[float, ...]):
        masks = torch.ones(
            (
                values.shape[0],
                values.shape[2],
                values.shape[3],
            ),
            dtype=bool,
        )
        for chanel in range(values.shape[1]):
            region = values[:, chanel, :, :]
            mask = self._get_mask(
                region,
                thresholds[chanel],
            )
            masks = masks & mask
        return masks.int()

    def _sliding_window_mask(
        self,
        values: torch.Tensor,
        thresholds: Tuple[float, ...],
        window_size: Tuple[int, int] = DEFAULT_WINDOWS_SIZE,
        window_step: Tuple[int, int] = DEFAULT_WINDOWS_STEP,
    ):
        masks = torch.zeros(
            (
                values.shape[0],
                values.shape[2],
                values.shape[3],
            ),
            dtype=bool,
        )
        for chanel in range(values.shape[1]):
            mask = torch.ones(
                (
                    values.shape[0],
                    values.shape[2],
                    values.shape[3],
                ),
                dtype=bool,
            )
            for i in range(0, values.shape[2] + window_size[0], window_step[0]):
                for j in range(0, values.shape[3] + window_size[1], window_step[1]):
                    region = values[
                        :, chanel, i : i + window_size[0], j : j + window_size[0]
                    ]
                    rmask = self._get_mask(
                        region,
                        thresholds[chanel],
                    )
                    mask[:, i : i + window_size[0], j : j + window_size[0]] &= rmask
            masks = masks | mask

        return masks.int()

    def _get_roi(self, mask: torch.Tensor):
        cv_mask = np.array(mask * 255).astype(np.uint8)
        # apply morphology open then close
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blob = cv2.morphologyEx(cv_mask, cv2.MORPH_OPEN, kernel)

        # invert blob
        blob = 255 - blob

        # Get contours
        cnts, _ = cv2.findContours(blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Фильтрация контуров
        good_cnts = []
        for cnt in cnts:
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)

            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            if 0.5 < circularity < 20.0:
                good_cnts.append(cnt)

        centers = torch.as_tensor([cnt.mean(axis=0) for cnt in good_cnts])
        return centers

    def __call__(
        self,
        rgb_img: torch.Tensor,
        hsv_img: torch.Tensor,
        window_size: Tuple[int, int] = DEFAULT_WINDOWS_SIZE,
        window_step: Tuple[int, int] = DEFAULT_WINDOWS_STEP,
    ):
        mask = torch.zeros(
            (
                rgb_img.shape[0],
                rgb_img.shape[2],
                rgb_img.shape[3],
            ),
            dtype=int,
        )

        if self.train:
            bright_mask = self._global_mask(
                hsv_img[:, [2], :, :],
                (self.bright_threshold,),
            )
            mask |= bright_mask

        rgb_mask = self._sliding_window_mask(rgb_img, self.rgb_thresholds)
        mask |= rgb_mask

        hsv_mask = self._sliding_window_mask(hsv_img[:, 1:], self.hsv_thresholds)
        mask |= hsv_mask

        roi = [self._get_roi(mask[i]) for i in range(mask.shape[0])]

        return mask, roi
