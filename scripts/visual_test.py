import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.inference import infer_one


def run_visual_test(image_path: str, out_dir: str = "outputs/visual_tests"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = infer_one(image_path)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    anomaly = result["anomaly_score"]
    heatmap = (result["heatmap"] * 255).astype(np.uint8)
    overlay = result["heatmap_overlay"]
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    similar = result["similar_cases"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(heatmap, cmap="jet")
    axes[0, 1].set_title(f"Anomaly Heatmap\n(score={anomaly:.2f})", fontsize=14)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(overlay_rgb)
    axes[0, 2].set_title(
        f"Overlay\nPred: {result['defect']} ({result['vit_confidence']:.2f})",
        fontsize=14,
    )
    axes[0, 2].axis("off")

    for i in range(3):
        ax = axes[1, i]
        if i < len(similar):
            sim = similar[i]
            sim_img = cv2.imread(sim["path"])
            if sim_img is not None:
                sim_img = cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB)
                ax.imshow(sim_img)
                ax.set_title(
                    f"Similar {i+1}: {sim['label']}\n(sim={sim['similarity']:.2f})",
                    fontsize=11,
                )
            else:
                ax.axis("off")
        else:
            ax.axis("off")

    plt.suptitle(f"File: {Path(image_path).name}", fontsize=16)
    plt.tight_layout()

    out_path = out_dir / f"{Path(image_path).stem}_test.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visual test: {out_path}")
    print(f"Anomaly score : {anomaly:.3f}")
    print(f"Prediction    : {result['defect']} (conf={result['vit_confidence']:.3f})")
    print(f"Novel defect  : {result['is_novel_defect']}")
    print(f"Similar cases : {len(similar)}")
    for s in similar:
        print(f"   - {s['label']} (sim={s['similarity']:.3f})")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/visual_test.py <image_path>")
        sys.exit(1)
    run_visual_test(sys.argv[1])
