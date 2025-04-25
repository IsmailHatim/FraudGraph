"""Train a GAT on the Elliptic dataset."""
import argparse
import sys, pathlib

import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import classification_report

ROOT = pathlib.Path(__file__).resolve().parents[1]   # …/FraudGraph
if ROOT.as_posix() not in sys.path:                  
    sys.path.append(ROOT.as_posix())

from utils.data_loader import load_elliptic_data
from models.gat import GAT
from utils.config import CHECKPOINT_DIR

def train(data,
          hidden_channels=64,
          heads=8,
          dropout=0.6,
          lr=0.005,
          weight_decay=5e-4,
          epochs=700,
          patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAT(in_channels=data.x.size(1),
                hidden_channels=hidden_channels,
                heads=heads,
                dropout=dropout,
                out_channels=2).to(device)
    data = data.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, patience_ctr, best_state = float("inf"), 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            pred = out.argmax(dim=1)
            acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Acc {acc:.4f}")

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch:03d}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    mask = data.y != -1
    report = classification_report(data.y[mask].cpu(),
                                   pred[mask].cpu(),
                                   target_names=["Legit", "Fraud"])
    print(report)
    return report


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Train a GAT on Elliptic")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--save", action="store_true", help="Persist best model to disk")
    args = parser.parse_args()

    data = load_elliptic_data()
    model = train(data,
                  hidden_channels=args.hidden_channels,
                  heads=args.heads,
                  dropout=args.dropout,
                  lr=args.lr,
                  epochs=args.epochs)
    evaluate(model, data)

    if args.save:
        ckpt_path = (CHECKPOINT_DIR / "gat_elliptic.pt").as_posix()
        torch.save(model.state_dict(), ckpt_path)
        print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()