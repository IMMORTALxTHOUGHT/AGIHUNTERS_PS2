import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.patchcore import PatchCore
from models.vit_classifier import load_model
from models.embedder import Embedder


def run_visual_test(
    image_path: str,
    mem_path: str = "models/weights/mem_bank.npy",
    model_path: str = "models/weights/vit_defect.pt",
    out_dir: str = "outputs/visual_tests",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pc = PatchCore()
    pc.load_memory(mem_path)
    model, label_map = load_model(model_path)
    emb = Embedder(model)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_img = Image.fromarray(img_rgb)

    result = pc.predict(pil_img)
    cls = model.predict(pil_img, label_map)
    vec = emb.encode(pil_img)

    anomaly = result["anomaly_score"]
    heatmap = (result["heatmap"] * 255).astype(np.uint8)
    overlay = result["heatmap_overlay"]
    roi = result["roi"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Anomaly Heatmap\n(score={anomaly:.2f})", fontsize=14)
    axes[1].axis("off")

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    axes[2].imshow(overlay_rgb)
    axes[2].set_title(
        f"Overlay\nPred: {cls['label']} ({cls['confidence']:.2f})", fontsize=14
    )
    axes[2].axis("off")

    plt.suptitle(f"File: {Path(image_path).name}", fontsize=16)
    plt.tight_layout()

    out_path = out_dir / f"{Path(image_path).stem}_test.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visual test: {out_path}")
    print(f"Anomaly score : {anomaly:.3f}")
    print(f"Prediction    : {cls['label']} (conf={cls['confidence']:.3f})")
    print(f"Embedding dim : {vec.shape}")
    print(f"ROI           : {roi}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/visual_test.py <image_path>")
        sys.exit(1)
    run_visual_test(sys.argv[1])
