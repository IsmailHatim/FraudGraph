"""Minimal entry‑point to train & evaluate GCN quickly."""
import argparse
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]   # …/FraudGraph
if ROOT.as_posix() not in sys.path:                  # idempotent
    sys.path.append(ROOT.as_posix())

from utils.data_loader import load_elliptic_data
from models.train_gcn import train, evaluate


def cli():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden_channels", type=int, default=64)
    args = parser.parse_args()

    data = load_elliptic_data()
    model = train(data, hidden_channels=args.hidden_channels, lr=args.lr, epochs=args.epochs)
    evaluate(model, data)


if __name__ == "__main__":
    cli()