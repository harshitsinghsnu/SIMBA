"""Environment setup shared by every script that touches HuggingFace/transformers.

Call `setup()` before importing `transformers` in any entry point. All PLM
checkpoints SIMBA uses (ESM2 family) are already cached locally, so we force
offline mode -- this also sidesteps a broken/empty `SSL_CERT_FILE` inherited
from the shell that otherwise crashes huggingface_hub's newer "list repo
templates" network call on tokenizer load.
"""

from __future__ import annotations

import os
from pathlib import Path


def setup() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # conda's activation script points SSL_CERT_FILE at <env>/ssl/cacert.pem,
    # which only exists if the `ca-certificates` conda package was installed.
    # This env was built with pip, so that path is dangling and crashes any
    # httpx/huggingface_hub SSLContext construction (even under offline mode,
    # the client session is still constructed eagerly). Fall back to
    # certifi's bundled CA file whenever the current value isn't a real file.
    cert_file = os.environ.get("SSL_CERT_FILE")
    if not cert_file or not Path(cert_file).is_file():
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            os.environ.pop("SSL_CERT_FILE", None)
