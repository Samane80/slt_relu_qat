import torch.nn as nn
from quantization_utils.model_utils import *


def freeze_model(model):
    """
    Freeze the calibrated quantization ranges: recursively invokes
    `.fix()` on every submodule that has one (QuantAct, QuantMatMul,
    IntLayerNorm, IntGELU, IntSoftmax, IntegerReLUFormerAttention, ...),
    not just QuantAct.

    Uses `model.modules()` (PyTorch's own recursive submodule iterator)
    instead of a hand-rolled walk over `dir()`/Sequential/ModuleList --
    simpler, and it can't silently skip a quant module nested inside a
    custom container.
    """
    for m in model.modules():
        if hasattr(m, "fix") and callable(m.fix):
            m.fix()


def unfreeze_model(model):
    """
    Unfreeze the calibrated quantization ranges: recursively invokes
    `.unfix()` on every submodule that has one. See `freeze_model`.
    """
    for m in model.modules():
        if hasattr(m, "unfix") and callable(m.unfix):
            m.unfix()


def set_quantize_mode(model, quantize: bool):
    """
    Toggle every quantization-aware layer in `model` between FP32 mode
    (quantize=False -- plain nn.Linear/nn.LayerNorm/nn.GELU/nn.Softmax
    behavior, computed on the SAME parameters) and QAT mode
    (quantize=True -- fake-quantized forward pass).

    This is the single switch behind the "one file, one flag" design:
    pretrain with `set_quantize_mode(model, False)`, then continue
    training the exact same model/optimizer with
    `set_quantize_mode(model, True)` for QAT fine-tuning -- no
    state_dict export/import, no strict=False, no separate float model
    class to keep in sync.

    Example
    -------
        model = build_model(cfg, ...)

        set_quantize_mode(model, False)
        train(model, float_cfg)          # FP32 pretraining

        set_quantize_mode(model, True)
        train(model, qat_cfg)            # QAT fine-tuning, same weights
    """
    touched = 0
    for m in model.modules():
        if hasattr(m, "quantize"):
            m.quantize = quantize
            touched += 1
    return touched
