"""
Downloads the RadAI WM-811K wafer defect model weights from Hugging Face
into the same folder as app.py, so app.py can find them.

Run this once before starting app.py:
    python download_model.py
"""

import os

from huggingface_hub import hf_hub_download

REPO_ID = "radai-agent/radai-wm811k-defect-detection"
DEST_DIR = os.path.dirname(__file__)


def find_weight_filename():
    """The model card references best_radai_resnet.pt; fall back to listing
    the repo's files in case the exact filename differs."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(REPO_ID)
    candidates = [f for f in files if f.endswith(".pt") or f.endswith(".pth")]
    if not candidates:
        raise RuntimeError(
            f"在 {REPO_ID} 找不到 .pt / .pth 權重檔，請至 "
            f"https://huggingface.co/{REPO_ID}/tree/main 手動確認檔名"
        )
    # Prefer the documented filename if present.
    for name in candidates:
        if "best_radai_resnet" in name:
            return name
    return candidates[0]


if __name__ == "__main__":
    filename = find_weight_filename()
    print(f"下載 {filename} ...")
    path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=DEST_DIR)

    target = os.path.join(DEST_DIR, "best_radai_resnet.pt")
    if os.path.abspath(path) != os.path.abspath(target):
        os.replace(path, target)

    print(f"完成！模型已存至：{target}")
