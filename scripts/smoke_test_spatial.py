#!/usr/bin/env python3
"""Small CPU check for SpatialRXNN's Gaussian layer and spatial attention."""

from types import SimpleNamespace


def main() -> int:
    try:
        import torch
        from chemprop.models.transformer import GaussianLayer, attention
    except ImportError as error:
        raise SystemExit("Install the SpatialRXNN environment before running this smoke test.") from error

    args = SimpleNamespace(num_kernels=8, edge_types=16)
    gaussian = GaussianLayer(args)
    distances = torch.randn(2, 4, 4)
    edge_types = torch.randint(0, args.edge_types, (2, 4, 4))
    encoded = gaussian(distances, edge_types)
    assert encoded.shape == (2, 4, 4, args.num_kernels)
    assert torch.isfinite(encoded).all()

    query = key = value = torch.randn(2, 2, 4, 3)
    bias = torch.randn(2, 4, 4, 2)
    mask = torch.zeros(2, 1, 1, 4, dtype=torch.bool)
    mask[:, :, :, -1] = True
    output, scores = attention(query, key, value, mask=mask, attn_bias=bias)
    assert output.shape == (2, 2, 4, 3)
    assert scores.shape == (2, 2, 4, 4)
    assert torch.isfinite(output).all()
    print("Spatial component smoke test passed on CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
