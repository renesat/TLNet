import logging
from os import remove
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve
from zipfile import ZipFile

from tqdm import tqdm

# Logger
logger = logging.getLogger("tlnet.data.utils")


def _create_urlretrieve_hook(pbar: tqdm):
    def hook(block: int, block_size: int, total_size: int):
        if pbar.total is None:
            pbar.total = int(total_size / 1024 / 1024)
        pbar.update(int((block * block_size) / 1024 / 1024) - pbar.n)

    return hook


def download_and_extract_data(url: str, path: Union[Path, str], verbosity: bool = True):
    if isinstance(path, str):
        path = Path(path)

    path.mkdir(exist_ok=True, parents=True)
    archive_path = path / "lisa.zip"

    # Download
    if verbosity:
        pbar = tqdm(
            unit="Mb",
        )
        bar_hook = _create_urlretrieve_hook(pbar)
    else:
        bar_hook = None
    urlretrieve(
        url,
        archive_path,
        reporthook=bar_hook,
    )

    # Extract
    if verbosity:
        logger.warning("Extracting...")
    with ZipFile(archive_path) as archive:
        archive.extractall(path)

    # Remove archive
    # archive_path.unlink()
