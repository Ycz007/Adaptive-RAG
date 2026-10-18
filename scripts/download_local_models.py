"""Download the two local retrieval models into an explicit directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    from huggingface_hub import HfApi, snapshot_download

    destination = args.directory.resolve()
    api = HfApi(endpoint=args.endpoint, token=False)
    for repository in ("BAAI/bge-small-zh-v1.5", "BAAI/bge-reranker-base"):
        print(f"Checking {repository}", flush=True)
        info = api.model_info(repository, files_metadata=True, timeout=20)
        names = [item.rfilename for item in info.siblings]
        weights = [name for name in names if name.endswith(".safetensors") and "/" not in name]
        if not weights:
            weights = [name for name in names if name == "pytorch_model.bin"]
        if not weights:
            raise RuntimeError(f"No supported PyTorch weights found: {repository}")
        files = weights + [
            name
            for name in names
            if name.endswith((".json", ".txt", ".model"))
            and not name.startswith(("onnx/", "openvino/"))
        ]
        size = sum(item.size or 0 for item in info.siblings if item.rfilename in files)
        target = destination / repository.rsplit("/", 1)[-1]
        print(f"Downloading {size / 1024**2:.1f} MiB to {target}", flush=True)
        snapshot_download(
            repo_id=repository,
            revision=info.sha,
            local_dir=target,
            allow_patterns=files,
            endpoint=args.endpoint,
            token=False,
            max_workers=2,
        )
        if not (target / "config.json").is_file():
            raise RuntimeError(f"Missing config.json in {target}")
        if any(not (target / name).is_file() for name in weights):
            raise RuntimeError(f"Missing model weights in {target}")
        print(f"Completed {repository} at revision {info.sha}", flush=True)


if __name__ == "__main__":
    main()
