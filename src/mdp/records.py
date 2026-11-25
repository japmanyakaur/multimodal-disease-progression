"""Experiment records (Appendix A): one JSON line per training run."""
import datetime as dt
import json
import platform
import uuid

import numpy as np
import sklearn
import torch


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "sklearn": sklearn.__version__, "platform": platform.platform(), "processor": platform.machine(),
            "accelerator": "none (CPU)"}


def write_record(path, **fields):
    record = {"run_id": uuid.uuid4().hex[:12], "date": dt.datetime.now().isoformat(timespec="seconds"),
              "environment": environment(), **fields}
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record["run_id"]
