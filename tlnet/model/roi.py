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
    """Generate region of interest by image with RGB and HSV colorspaces."""

    def __init__(
        self,
        train: bool = False,
        rgb_thresholds: Tuple[float, float, float] = DEFAULT_RGB_THRESHOLDS,
        hsv_thresholds: Tuple[float, float] = DEFAULT_HSV_THRESHOLDS,
        bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD,
    ):
        """Create ROI selector.

        Parameters
        ----------
        train
            Using global bright mask if True.
        rgb_thresholds
            Thresholds for R, G and B layers in RGB image.
        hsv_thresholds
            Thresholds for S and V layers in HSV image.
        bright_thresholds
            Threshold for brightnes layer.
        """

        self.train = train
        self.rgb_thresholds = rgb_thresholds
        self.hsv_thresholds = hsv_thresholds
        self.bright_threshold = bright_threshold

    @staticmethod
    def _get_mask(region: torch.Tensor, threshold: float) -> torch.Tensor:
        """Generate mask for image region. Based on fixed threshold and
        mean value in region.

        Parameters
        ----------
        region
            Part of image with shape B x H x W where B is batch size.
        threshold
            Fixed threshold for region.

        Returns
        -------
        mask
            Binary mask with shape B x H x W.
        """
        mean_value = region.mean(dim=(1, 2))
        mask = region > (mean_value + threshold).unsqueeze(1).unsqueeze(1)
        return mask

    def _global_mask(
        self, values: torch.Tensor, thresholds: Tuple[float, ...]
    ) -> torch.Tensor:
        """Create mask for all image threshold.

        Parameters
        ----------
        values
            Tensor with shape B x C x H x W where B is batch size and
            C is number of chanels.
        thresholds
            thresholds for all chanels.

        Returns
        -------
        masks
            Binary mask with shape B x H x W.
        """
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
    ) -> torch.Tensor:
        """Create mask using sliding window.

        Parameters
        ----------
        values
            Tensor with shape B x C x H x W where B is batch size and
            C is number of chanels.
        thresholds
            Thresholds for all chanels.
        window_size
            Size of sliding window ( (64, 64) by default )
        window_step
            Step of sliding window by x and y ( (32, 32) by default )


        Returns
        -------
        masks
            Binary masks with shape B x C x H x W.
        """
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

    @staticmethod
    def _get_roi(mask: torch.Tensor) -> torch.Tensor:
        """Find blobs using binary mask.

        Parameters
        ----------
        mask
            Tensor binary mask with shape B x H x W where B is batch size.

        Returns
        -------
        centers
            Tensor with centers of ROIs. Shape B x 2.
        """
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
