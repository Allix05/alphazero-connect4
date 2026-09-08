#!/usr/bin/env python
"""Export a trained checkpoint to ONNX so the browser demo (docs/) can run
inference client-side via onnxruntime-web, with no backend server needed.

    python scripts/export_onnx.py --checkpoint checkpoints/best.pt --out docs/model.onnx
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx
import torch

from alphazero.config import Config
from alphazero.game import COLS, ROWS
from alphazero.model import AlphaZeroNet


class InferenceWrapper(torch.nn.Module):
    """Wraps the dual-head net so ONNX exports policy *probabilities*
    (softmax already applied) instead of raw logits -- simpler for the JS
    side, which just needs to mask illegal moves and renormalize.
    """

    def __init__(self, model: AlphaZeroNet):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, value = self.model(x)
        return torch.softmax(logits, dim=1), value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out", default="docs/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    cfg = Config()
    model = AlphaZeroNet(cfg.channels, cfg.num_res_blocks)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    wrapper = InferenceWrapper(model)
    dummy = torch.zeros(1, 2, ROWS, COLS, dtype=torch.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={"board": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=args.opset,
    )

    # The dynamo exporter may split large weights into a sidecar ".data"
    # file; re-save inline so docs/model.onnx is the single file the
    # browser demo needs to fetch.
    data_file = args.out + ".data"
    if os.path.exists(data_file):
        onnx_model = onnx.load(args.out)
        onnx.save(onnx_model, args.out, save_as_external_data=False)
        os.remove(data_file)

    size_kb = os.path.getsize(args.out) / 1024
    print(f"Exported {args.out} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
