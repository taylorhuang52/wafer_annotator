"""
Wafer Defect Detection Demo - Flask backend
Wraps the RadAI WM-811K ResNet34 model (Hugging Face) for browser-based inference.

Model: https://huggingface.co/radai-agent/radai-wm811k-defect-detection
"""

import csv
import io
import os
import re
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from scipy.ndimage import zoom
from torchvision import models

from wafer_log_parser import WaferLogParseError, parse_dlog_csv

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_radai_resnet.pt")

# Wafers at or above this yield are classified as "None" (no defect pattern)
# by a simple rule instead of running them through the model. This avoids
# training-data imbalance: the model was never trained on a "None" class,
# and yield alone is a reliable, deterministic signal for "this wafer is fine".
YIELD_NONE_THRESHOLD = 95.0

CORRECTIONS_DIR = os.path.join(os.path.dirname(__file__), "corrected_labels")
BINMAPS_DIR = os.path.join(CORRECTIONS_DIR, "binmaps")
LABELS_CSV = os.path.join(CORRECTIONS_DIR, "labels.csv")
LABELS_CSV_HEADER = [
    "timestamp", "lot_id", "wafer_id", "original_pred", "original_conf",
    "corrected_label", "dominant_fail_bin", "dominant_fail_test",
    "n_die", "n_pass", "n_fail", "binmap_file",
]


def ensure_correction_storage():
    os.makedirs(BINMAPS_DIR, exist_ok=True)
    if not os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(LABELS_CSV_HEADER)


def dominant_fail(raw_bin_map: np.ndarray, bin_test_map: dict):
    """Most common fail Bin# on this wafer (excludes background=0, pass=1)."""
    fails = raw_bin_map[(raw_bin_map != 0) & (raw_bin_map != 1)]
    if fails.size == 0:
        return None, ""
    vals, counts = np.unique(fails, return_counts=True)
    dominant_bin = int(vals[np.argmax(counts)])
    dominant_test = bin_test_map.get(dominant_bin, f"Bin{dominant_bin}")
    return dominant_bin, dominant_test


def build_hotspot_grid(parsed_list):
    """Aligns one or more wafers' raw bin maps into a single shared
    absolute-coordinate grid, and for each (x,y) die position counts how
    many of the wafers failed there. Works for a single wafer (n=1, just
    that wafer's own fail map) or many (cross-wafer hotspot detection)."""
    x_maxs = [p.x_min + p.raw_bin_map.shape[1] - 1 for p in parsed_list]
    y_maxs = [p.y_min + p.raw_bin_map.shape[0] - 1 for p in parsed_list]
    gx_min = min(p.x_min for p in parsed_list)
    gy_min = min(p.y_min for p in parsed_list)
    gx_max = max(x_maxs)
    gy_max = max(y_maxs)
    w = gx_max - gx_min + 1
    h = gy_max - gy_min + 1

    fail_count = np.zeros((h, w), dtype=np.int32)
    tested_count = np.zeros((h, w), dtype=np.int32)

    for p in parsed_list:
        r0 = p.y_min - gy_min
        c0 = p.x_min - gx_min
        sub = p.raw_bin_map
        sh, sw = sub.shape
        is_tested = sub != 0
        is_fail = is_tested & (sub != 1)
        tested_count[r0:r0 + sh, c0:c0 + sw] += is_tested
        fail_count[r0:r0 + sh, c0:c0 + sw] += is_fail

    return fail_count, tested_count, gx_min, gy_min


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
    return render_template("index.html", yield_none_threshold=YIELD_NONE_THRESHOLD)


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

    yield_threshold = YIELD_NONE_THRESHOLD
    raw_threshold = request.form.get("yield_threshold")
    if raw_threshold is not None and raw_threshold != "":
        try:
            candidate = float(raw_threshold)
            if 0 <= candidate <= 100:
                yield_threshold = candidate
        except ValueError:
            pass  # keep default on bad input rather than failing the whole batch

    results = []
    errors = []

    for f in files:
        filename = f.filename or "unknown.csv"
        if not filename.lower().endswith(".csv"):
            continue
        try:
            raw = f.read()
            parsed = parse_dlog_csv(raw, filename)
            yield_pct = round(100 * parsed.n_pass / parsed.n_die, 2) if parsed.n_die else 0
            dom_bin, dom_test = dominant_fail(parsed.raw_bin_map, parsed.bin_test_map)

            if yield_pct >= yield_threshold:
                # Rule-based call: no defect pattern, model is not run.
                rule_based = True
                ranked = [{
                    "label": "None",
                    "desc": f"良率 {yield_pct}% ≥ {yield_threshold}%，規則判定為無特定圖案（未執行 AI 模型）",
                    "prob": None,
                }]
            else:
                rule_based = False
                ranked, _ = preprocess_and_predict(parsed.bin_map)

            results.append({
                "filename": filename,
                "lot_id": parsed.lot_id,
                "wafer_id": parsed.wafer_id,
                "n_die": parsed.n_die,
                "n_pass": parsed.n_pass,
                "n_fail": parsed.n_fail,
                "yield_pct": yield_pct,
                "predictions": ranked,
                "rule_based": rule_based,
                "preview": downsample_preview(parsed.bin_map),
                "dominant_fail_bin": dom_bin,
                "dominant_fail_test": dom_test,
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
        "yield_threshold_used": yield_threshold,
        "pattern_summary": [
            {"label": k, "count": v} for k, v in pattern_counts.most_common()
        ],
        "results": results,
        "errors": errors,
    })


@app.route("/save_correction", methods=["POST"])
def save_correction():
    file = request.files.get("file")
    corrected_label = request.form.get("corrected_label", "").strip()
    original_pred = request.form.get("original_pred", "")
    original_conf = request.form.get("original_conf", "")

    if not file or not corrected_label:
        return jsonify({"error": "缺少檔案或修正標籤"}), 400

    try:
        raw = file.read()
        parsed = parse_dlog_csv(raw, file.filename)
    except WaferLogParseError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"解析失敗：{e}"}), 500

    ensure_correction_storage()

    dom_bin, dom_test = dominant_fail(parsed.raw_bin_map, parsed.bin_test_map)

    safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", parsed.wafer_id) or "wafer"
    npy_path = os.path.join(BINMAPS_DIR, f"{safe_name}.npy")
    np.save(npy_path, parsed.raw_bin_map)

    with open(LABELS_CSV, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            parsed.lot_id,
            parsed.wafer_id,
            original_pred,
            original_conf,
            corrected_label,
            dom_bin if dom_bin is not None else "",
            dom_test,
            parsed.n_die,
            parsed.n_pass,
            parsed.n_fail,
            os.path.relpath(npy_path, CORRECTIONS_DIR),
        ])

    return jsonify({
        "status": "ok",
        "wafer_id": parsed.wafer_id,
        "dominant_fail_bin": dom_bin,
        "dominant_fail_test": dom_test,
    })


@app.route("/hotspot_batch", methods=["POST"])
def hotspot_batch():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "沒有收到任何檔案"}), 400

    try:
        top_n = int(request.form.get("top_n", 15))
    except ValueError:
        top_n = 15

    parsed_list = []
    errors = []
    for f in files:
        filename = f.filename or "unknown.csv"
        if not filename.lower().endswith(".csv"):
            continue
        try:
            raw = f.read()
            parsed_list.append(parse_dlog_csv(raw, filename))
        except WaferLogParseError as e:
            errors.append({"filename": filename, "error": str(e)})
        except Exception as e:
            errors.append({"filename": filename, "error": f"處理失敗：{e}"})

    if not parsed_list:
        return jsonify({"error": "沒有可用的 wafer 資料，請確認上傳的是有效的 test log CSV"}), 400

    fail_count, tested_count, gx_min, gy_min = build_hotspot_grid(parsed_list)
    max_count = int(fail_count.max()) if fail_count.size else 0
    n_wafers = len(parsed_list)

    order = np.argsort(fail_count, axis=None)[::-1]
    top_coords = []
    for idx in order:
        if len(top_coords) >= top_n:
            break
        r, c = np.unravel_index(idx, fail_count.shape)
        cnt = int(fail_count[r, c])
        if cnt <= 0:
            break
        top_coords.append({
            "x": int(c + gx_min),
            "y": int(r + gy_min),
            "count": cnt,
            "pct": round(100 * cnt / n_wafers, 1),
        })

    return jsonify({
        "lot_id": parsed_list[0].lot_id,
        "n_wafers": n_wafers,
        "grid": {
            "width": int(fail_count.shape[1]),
            "height": int(fail_count.shape[0]),
            "x_min": int(gx_min),
            "y_min": int(gy_min),
        },
        "fail_count": fail_count.tolist(),
        "tested_count": tested_count.tolist(),
        "max_count": max_count,
        "top_coords": top_coords,
        "errors": errors,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
