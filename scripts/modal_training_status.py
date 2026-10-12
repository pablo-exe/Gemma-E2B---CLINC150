"""Read student progress from Modal without starting a GPU container."""

import json
from pathlib import Path

import modal

root = Path(__file__).resolve().parents[1]
receipt = json.loads((root / "artifacts/distillation/full_gpu_bf16/modal_launch.json").read_text())
volume = modal.Volume.from_name(receipt["volume"])
payload = b"".join(volume.read_file(f"{receipt['run']}/status.json"))
status = json.loads(payload)
print(json.dumps(status, indent=2))
(root / "artifacts/distillation/full_gpu_bf16/modal_status.json").write_bytes(payload)
