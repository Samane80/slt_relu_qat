"""Regression tests for the FP32/QAT and inference bugs fixed in this branch."""

import torch
import torch.nn as nn

from quantization_utils.quant_modules import QuantAct, IntLayerNorm
from signjoey.initialization import initialize_model
from signjoey.relu_attention import MinMaxObserver


class _NormOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = IntLayerNorm(4)
        self.sgn_embed = nn.Module()
        self.sgn_embed.norm = IntLayerNorm(4)
        self.txt_embed = None
        self.encoder = nn.Identity()
        self.decoder = None


def test_quantact_does_not_update_observer_in_eval():
    observer = QuantAct(quantize=True)
    observer.train()
    observer(torch.tensor([[-2.0, 1.0], [0.5, 3.0]]))
    min_before = observer.min_val.detach().clone()
    max_before = observer.max_val.detach().clone()

    observer.eval()
    observer(torch.tensor([[-100.0, 100.0]]))

    assert torch.equal(observer.min_val, min_before)
    assert torch.equal(observer.max_val, max_before)

    observer.set_calibration_mode(True)
    observer(torch.tensor([[-100.0, 100.0]]))
    assert not torch.equal(observer.min_val, min_before)


def test_reluformer_observer_does_not_update_in_eval():
    observer = MinMaxObserver(unsigned=True)
    observer.train()
    observer(torch.tensor([[0.0, 2.0, 4.0]]))
    min_before = observer.min_val.detach().clone()
    max_before = observer.max_val.detach().clone()

    observer.eval()
    observer(torch.tensor([[0.0, 100.0]]))

    assert torch.equal(observer.min_val, min_before)
    assert torch.equal(observer.max_val, max_before)


def test_normalization_scale_is_initialized_to_one():
    model = _NormOnlyModel()
    initialize_model(
        model,
        {
            "initializer": "xavier",
            "bias_initializer": "zeros",
            "embed_initializer": "normal",
            "embed_init_weight": 0.01,
        },
        txt_padding_idx=0,
    )
    assert torch.equal(model.layer_norm.weight, torch.ones_like(model.layer_norm.weight))
    assert torch.equal(model.layer_norm.bias, torch.zeros_like(model.layer_norm.bias))
    assert torch.equal(
        model.sgn_embed.norm.weight,
        torch.ones_like(model.sgn_embed.norm.weight),
    )
