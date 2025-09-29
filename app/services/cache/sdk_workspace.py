from __future__ import annotations

import shutil
import uuid
from pathlib import Path


class SDKWorkspaceManager:
    """Manage temporary SDK-compatible workspaces for serving models."""

    def __init__(self, sdk_root: Path) -> None:
        self.sdk_root = sdk_root
        self.sdk_root.mkdir(parents=True, exist_ok=True)

    def create_workspace(self) -> Path:
        workspace = self.sdk_root / str(uuid.uuid4())
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "models").mkdir(parents=True, exist_ok=True)
        return workspace

    def cleanup(self, workspace: Path) -> None:
        if workspace.exists() and workspace.is_relative_to(self.sdk_root):
            shutil.rmtree(workspace, ignore_errors=True)