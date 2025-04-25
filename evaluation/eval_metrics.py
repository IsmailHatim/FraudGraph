"""
Compute metrics & plot ROC / PR curves for a saved model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]   # …/FraudGraph
if ROOT.as_posix() not in sys.path:                  
    sys.path.append(ROOT.as_posix())

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from utils.config import ROOT, CHECKPOINT_DIR
from utils.data_loader import load_elliptic_data
from models.gcn import GCN
from models.gat import GAT
from models.graphsage import GraphSAGE

# ─── where plots will be written ────────────────────────────────────────────
PLOT_DIR = (ROOT / "evaluation" / "plots").resolve()
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ─── registry: add new architectures here once implemented ─────────────────
MODEL_REGISTRY = {
    "gcn": {
        "class": GCN,
        "default_ckpt": CHECKPOINT_DIR / "gcn_elliptic.pt",
    },
    "gat": {
        "class": GAT,
        "default_ckpt": CHECKPOINT_DIR / "gat_elliptic.pt",
    },
    "graphsage": {
        "class": GraphSAGE,
        "default_ckpt": CHECKPOINT_DIR / "graphsage_elliptic.pt",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# evaluation logic
# ════════════════════════════════════════════════════════════════════════════
def evaluate(model_name: str, ckpt_path: Path | None = None, device: str = "cpu") -> None:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Choose from {list(MODEL_REGISTRY)}")

    entry = MODEL_REGISTRY[model_name]
    ckpt_path = ckpt_path or entry["default_ckpt"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── data & model ────────────────────────────────────────────────────────
    data = load_elliptic_data(device=device)
    model_cls = entry["class"]
    model = model_cls(in_channels=data.x.size(1)).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = logits.softmax(dim=1)[:, 1]          # P(fraud)
        preds = logits.argmax(dim=1)

    mask = data.y != -1                              # exclude unknown labels
    y_true = data.y[mask].cpu().numpy()
    y_prob = probs[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()

    # ── metrics ─────────────────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred, target_names=["Legit", "Fraud"], digits=4
    )
    roc_auc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    print(report)
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"Plots stored in: {PLOT_DIR.as_posix()}")

    # ── ROC curve ───────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – {model_name.upper()}")
    plt.legend(loc="lower right")
    roc_path = PLOT_DIR / f"roc_{model_name}.png"
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ── Precision-Recall curve ──────────────────────────────────────────────
    plt.figure()
    plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall – {model_name.upper()}")
    plt.legend(loc="lower left")
    pr_path = PLOT_DIR / f"pr_{model_name}.png"
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
def main() -> None:     # pragma: no cover
    parser = argparse.ArgumentParser(description="Evaluate saved models & draw curves")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY),
                        help="Model architecture to evaluate")
    parser.add_argument("--ckpt", type=Path, help="Custom checkpoint path "
                                                  "(defaults to checkpoints/<model>_elliptic.pt)")
    args = parser.parse_args()
    evaluate(args.model, args.ckpt)


if __name__ == "__main__":
    main()