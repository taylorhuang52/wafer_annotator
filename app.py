"""
Wafer Defect Detection Demo - Flask backend
Wraps the RadAI WM-811K ResNet34 model (Hugging Face) for browser-based inference.

Model: https://huggingface.co/radai-agent/radai-wm811k-defect-detection
"""

import io
import os

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from scipy.ndimage import zoom
from torchvision import models

from wafer_log_parser import WaferLogParseError, parse_dlog_csv

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_radai_resnet.pt")

CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Random", "Scratch", "Near-full",
]

# Short, plain-language description of each pattern for the UI.
CLASS_INFO = {
    "Center":    "缺陷集中在晶圓中心區域",
    "Donut":     "缺陷呈環狀分佈，中心與邊緣正常",
    "Edge-Loc":  "缺陷集中在邊緣的局部區域",
    "Edge-Ring": "缺陷沿整個邊緣呈環狀分佈",
    "Loc":       "缺陷集中在某個局部區域（非邊緣）",
    "Random":    "缺陷隨機散佈，無明顯規律",
    "Scratch":   "缺陷呈線狀，類似刮痕",
    "Near-full": "近乎整片晶圓都被判定為缺陷",
}


class RadAI_ResNet(nn.Module):
    """Architecture must match the checkpoint exactly (see model card)."""

    def __init__(self, num_classes=8):
        super().__init__()
        self.base = models.resnet34(weights=None)
        self.base.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base.fc.in_features, num_classes),
        )

    def forward(self, x):
        return self.base(x)


app = Flask(__name__)

_model = None


def get_model():
    """Lazy-load the model once and cache it."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"找不到模型權重檔：{MODEL_PATH}\n"
                "請先執行 download_model.py 下載 best_radai_resnet.pt"
            )
        model = RadAI_ResNet(num_classes=8)
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        _model = model
    return _model


def preprocess_and_predict(wafer_map: np.ndarray):
    """wafer_map: 2D numpy array (any size, any grayscale range)."""
    h, w = wafer_map.shape
    resized = zoom(wafer_map, (64 / h, 64 / w), order=1)[:64, :64]

    if resized.max() > 0:
        resized = resized / resized.max()

    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)

    model = get_model()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].numpy()

    ranked = sorted(
        (
            {"label": CLASSES[i], "desc": CLASS_INFO[CLASSES[i]], "prob": float(probs[i])}
            for i in range(len(CLASSES))
        ),
        key=lambda item: item["prob"],
        reverse=True,
    )
    return ranked, resized


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "沒有收到檔案"}), 400

    file = request.files["file"]
    try:
        image = Image.open(file.stream).convert("L")  # grayscale
    except Exception:
        return jsonify({"error": "無法讀取這個檔案，請確認是圖片格式"}), 400

    wafer_map = np.array(image, dtype=np.float32)

    try:
        ranked, resized = preprocess_and_predict(wafer_map)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"推論失敗：{e}"}), 500

    # Small 64x64 preview (normalized 0-255) so the UI can show what the model saw.
    preview = (resized * 255).astype(np.uint8).tolist()

    return jsonify({"predictions": ranked, "preview": preview})


def downsample_preview(arr: np.ndarray, size: int = 48):
    """Small preview grid for the results table (values already 0/1/2)."""
    h, w = arr.shape
    small = zoom(arr, (size / h, size / w), order=0)[:size, :size]
    if small.max() > 0:
        small = small / small.max()
    return (small * 255).astype(np.uint8).tolist()


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "沒有收到任何檔案"}), 400

    results = []
    errors = []

    for f in files:
        filename = f.filename or "unknown.csv"
        if not filename.lower().endswith(".csv"):
            continue
        try:
            raw = f.read()
            parsed = parse_dlog_csv(raw, filename)
            ranked, resized = preprocess_and_predict(parsed.bin_map)
            results.append({
                "filename": filename,
                "lot_id": parsed.lot_id,
                "wafer_id": parsed.wafer_id,
                "n_die": parsed.n_die,
                "n_pass": parsed.n_pass,
                "n_fail": parsed.n_fail,
                "yield_pct": round(100 * parsed.n_pass / parsed.n_die, 2) if parsed.n_die else 0,
                "predictions": ranked,
                "preview": downsample_preview(parsed.bin_map),
            })
        except WaferLogParseError as e:
            errors.append({"filename": filename, "error": str(e)})
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            errors.append({"filename": filename, "error": f"處理失敗：{e}"})

    if not results and not errors:
        return jsonify({"error": "沒有找到任何 .csv 檔案"}), 400

    # Sort wafers by their numeric suffix when possible, else by name.
    def sort_key(r):
        import re
        m = re.search(r"(\d+)\s*$", r["wafer_id"])
        return (0, int(m.group(1))) if m else (1, r["wafer_id"])

    results.sort(key=sort_key)

    # Lot-level pattern distribution summary.
    from collections import Counter
    pattern_counts = Counter(r["predictions"][0]["label"] for r in results)
    lot_id = results[0]["lot_id"] if results else (errors[0]["filename"] if errors else "")

    return jsonify({
        "lot_id": lot_id,
        "wafer_count": len(results),
        "pattern_summary": [
            {"label": k, "count": v} for k, v in pattern_counts.most_common()
        ],
        "results": results,
        "errors": errors,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
