from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ModelS3Gateway:
    """Encapsulate all interactions with S3 for cached models."""

    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def download(self, model_id: str, destination: Path) -> Dict[str, Any]:
        destination.mkdir(parents=True, exist_ok=True)
        metadata_path = destination / "metadata.json"

        self._client.download_file(
            self._bucket,
            f"models/{model_id}/metadata.json",
            str(metadata_path)
        )

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        artifacts = metadata.get("artifacts", {})
        if isinstance(artifacts, dict):
            for artifact in artifacts.values():
                path_hint = artifact.get("path") if isinstance(artifact, dict) else None
                if not path_hint:
                    continue
                filename = Path(path_hint).name
                local_path = destination / filename
                self._client.download_file(
                    self._bucket,
                    f"models/{model_id}/{filename}",
                    str(local_path)
                )

        if "path" in metadata:
            filename = Path(metadata["path"]).name
            local_path = destination / filename
            self._client.download_file(
                self._bucket,
                f"models/{model_id}/{filename}",
                str(local_path)
            )

        return metadata

    def upload(self, model_id: str, source_dir: Path) -> None:
        for file_path in source_dir.rglob("*"):
            if not file_path.is_file():
                continue
            key = f"models/{model_id}/{file_path.name}"
            self._client.upload_file(str(file_path), self._bucket, key)

    def delete(self, model_id: str) -> None:
        response = self._client.list_objects_v2(
            Bucket=self._bucket,
            Prefix=f"models/{model_id}/"
        )
        for entry in response.get("Contents", []):
            key = entry.get("Key")
            if key:
                self._client.delete_object(Bucket=self._bucket, Key=key)


class NullModelS3Gateway(ModelS3Gateway):
    """No-op gateway used when S3 is not configured."""

    def __init__(self) -> None:  # type: ignore[super-init-not-called]
        pass

    def download(self, model_id: str, destination: Path) -> Dict[str, Any]:  # pragma: no cover - simple passthrough
        raise ValueError("S3 storage is not configured")

    def upload(self, model_id: str, source_dir: Path) -> None:  # pragma: no cover - simple passthrough
        return

    def delete(self, model_id: str) -> None:  # pragma: no cover - simple passthrough
        return