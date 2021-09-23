import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from torch.utils.data import Dataset

from .utils import download_and_extract_data

# Logger
logger = logging.getLogger("tlnet.data.lisa")


class LISA(Dataset):
    def __init__(
        self,
        root: Union[Path, str],
        train: bool = False,
    ):
        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.train = train

        self.annotation_path = self.root / "Annotations" / "Annotations"
        if self.train:
            self.seq_names: List[str] = []
            self.seq_paths: List[Path] = []
            for name in ["nightTrain", "dayTrain"]:
                dir_files = list((self.annotation_path / name).iterdir())
                self.seq_names += [name for _ in range(len(dir_files))]
                self.seq_paths += [Path(name) / d.name for d in dir_files]
        else:
            self.seq_names = [
                "daySequence1",
                "daySequence2",
                "nightSequence1",
                "nightSequence2",
            ]
            self.seq_paths: List[Path] = list(map(Path, self.seq_names))

        self.annotations = self._get_annotations()

    def _get_annotations(self) -> Dict:
        annotations = {}
        for i, path in enumerate(self.seq_paths):
            df = pd.read_csv(
                self.annotation_path / path / "frameAnnotationsBOX.csv",
                sep=";",
            )
            for name, group in df.groupby("Filename"):
                full_path = (
                    self.root / self.seq_names[i] / path / "frames" / Path(name).name
                )
                annotations[str(full_path)] = [
                    tuple(x)
                    for _, x in group[
                        [
                            "Upper left corner X",
                            "Upper left corner Y",
                            "Lower right corner X",
                            "Lower right corner Y",
                        ]
                    ].iterrows()
                ]
        return annotations

    # TODO
    def __getitem__():
        pass

    # TODO
    def __len__():
        pass
