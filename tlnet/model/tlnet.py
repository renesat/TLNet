import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class TLNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Sequential(
            nn.Linear(64 * 30 * 30, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 5),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        pred = torch.sigmoid(x[:, 0])
        box = x[:, 1:]
        return pred, box

    def training_step(self, batch, _batch_idx):
        img = batch["image"]
        cls = batch["class"]
        box = batch["box"]

        cls_pred, box_pred = self.forward(img)

        bse_loss = F.binary_cross_entropy(
            cls_pred.float(),
            cls.float(),
        )
        reg_loss = 0
        for i, p in enumerate(cls):
            if p > 0.5:
                reg_loss += 2 * ((box_pred[i] - box[i]) ** 2).sum().sqrt()
        loss = reg_loss + bse_loss

        self.log("train/loss", loss)
        self.log("train/reg_loss", reg_loss)
        self.log("train/bse_loss", bse_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["image"]
        cls = batch["class"]
        box = batch["box"]

        cls_pred, box_pred = self.forward(img)

        bse_loss = F.binary_cross_entropy(
            cls_pred.float(),
            cls.float(),
        )
        reg_loss = 0
        for i, p in enumerate(cls):
            if p > 0.5:
                reg_loss += 2 * ((box_pred[i] - box[i]) ** 2).sum().sqrt()
        loss = reg_loss + bse_loss

        accuracy = (cls_pred > 0.5) == (cls > 0.5)

        self.log("val/loss", loss)
        self.log("val/reg_loss", reg_loss)
        self.log("val/bse_loss", bse_loss)

        return {
            "loss": loss,
            "accuracy": accuracy,
        }

    def validation_epoch_end(self, outputs):
        accuracy = []
        for out in outputs:
            accuracy += list(out["accuracy"].detach().cpu().numpy())
        self.log("val/accuracy", np.mean(accuracy))

    def test_step(self, batch, batch_idx):
        img = batch["image"]
        cls = batch["class"]
        box = batch["box"]

        cls_pred, box_pred = self.forward(img)

        accuracy = (cls_pred > 0.5) == (cls > 0.5)
        tp = ((cls_pred[cls > 0.5] > 0.5) == (cls[cls > 0.5] > 0.5)).sum()
        fp = ((cls_pred[cls > 0.5] > 0.5) != (cls[cls > 0.5] > 0.5)).sum()
        fn = ((cls_pred[cls <= 0.5] > 0.5) != (cls[cls <= 0.5] > 0.5)).sum()

        return {
            "count": len(cls),
            "fn": fn,
            "fp": fp,
            "tp": tp,
            "accuracy": accuracy,
        }

    def test_epoch_end(self, outputs):
        accuracy = []
        tp = 0
        fp = 0
        fn = 0
        count = 0
        for out in outputs:
            accuracy += list(out["accuracy"].detach().cpu().numpy())
            tp += out["tp"].detach().cpu().item()
            fp += out["fp"].detach().cpu().item()
            fn += out["fn"].detach().cpu().item()
            count += out["count"]
        tp_rate = tp / (tp + fp)
        print("TP Rate:", tp_rate)
        print("FN:", fn / count)

    def test_dataloader(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
