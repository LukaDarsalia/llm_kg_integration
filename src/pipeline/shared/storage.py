"""Storage + run helpers shared by every pipeline stage.

Mirrors the convention from the reference pipelines: data physically lives in S3, and
W&B artifacts hold only a lightweight ``s3://`` reference (see each stage's runner.py).
``.env`` is loaded here so AWS/W&B/provider keys are available process-wide.
"""

import datetime
import os
from pathlib import Path
from typing import Union

import boto3
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env at import time (AWS creds, WANDB_API_KEY, keys).
load_dotenv(".env")


def generate_folder_name() -> str:
    """Timestamped folder name, e.g. ``2026-06-18_14-30-45`` — one per stage run."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_s3_loader(bucket_name: str) -> "S3DataLoader":
    """Build an :class:`S3DataLoader` from ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``."""
    return S3DataLoader(
        bucket=bucket_name,
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


class S3DataLoader:
    """Upload/download files or folders to/from S3, mirroring the local path as the key.

    A local path like ``artifacts/index_lightrag/<ts>/medical/...`` is stored at
    ``s3://<bucket>/artifacts/index_lightrag/<ts>/medical/...`` so ``add_reference`` +
    ``use_artifact().download()`` round-trips cleanly.
    """

    def __init__(self, bucket: str, access_key: str, secret_key: str):
        self.bucket = bucket
        self.s3_client = boto3.client(
            "s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )

    # --- upload ---------------------------------------------------------------------
    def upload(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if path.is_dir():
            self._upload_folder(str(path))
        else:
            self._upload_file(path)

    def _upload_file(self, filepath: Union[str, Path]) -> None:
        self.s3_client.upload_file(str(filepath), self.bucket, str(filepath))

    def _upload_folder(self, folder_path: str) -> None:
        all_files = list(Path(folder_path).rglob("*"))
        for file_path in tqdm(all_files, desc=f"Uploading {folder_path} -> s3://{self.bucket}"):
            if not file_path.is_dir():
                self._upload_file(file_path)

    # --- download -------------------------------------------------------------------
    def download(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if "." in path.name:
            self._download_file(path)
        else:
            self._download_folder(str(path))

    def _download_file(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(self.bucket, str(filepath), str(filepath))

    def _download_folder(self, remote_dir_name: str) -> None:
        resource = boto3.resource(
            "s3",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        bucket = resource.Bucket(self.bucket)
        for obj in tqdm(
            list(bucket.objects.filter(Prefix=remote_dir_name)),
            desc=f"Downloading s3://{self.bucket}/{remote_dir_name}",
        ):
            if obj.key.endswith("/"):
                continue
            local_dir = os.path.dirname(obj.key)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
            bucket.download_file(obj.key, obj.key)
