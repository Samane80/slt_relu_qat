import torch

from signjoey.batch import Batch


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


def calibrate_activation_ranges(
    model,
    train_iter,
    txt_pad_index: int,
    sgn_dim: int,
    use_cuda: bool,
    num_batches: int = 200,
) -> int:
    """
    Warm-start every `QuantAct.min_val` / `max_val` running-stat buffer
    before real QAT training begins.

    Why this is needed
    -------------------
    `QuantAct` (and the ReLUFormer output observer) carries a *running*
    statistic across forward calls.  The calibration mode below explicitly
    enables those observers while the model remains in eval mode.  The
    remaining arithmetic modules (`QuantLinear`, `IntLayerNorm`, `IntGELU`,
    `IntSoftmax`, `QuantMatMul`) recompute their scaling factor from the
    current weights/input scale on each forward call.

    While `set_quantize_mode(model, False)` is active, `QuantAct.forward`
    returns in its very first line and never touches `min_val`/`max_val` --
    so after a full FP32 pretraining run, every `QuantAct` in the model
    still has `min_val == max_val == 0` (their `__init__` default).
    Flipping straight to `quantize=True` and starting gradient updates at
    that point forces every `QuantAct` in the network (dozens of them) to
    calibrate its activation range *simultaneously* with backprop, from a
    single noisy batch -- this is what produces the catastrophic PPL and
    degenerate ("bild bild bild ...") outputs seen in the first few QAT
    epochs.

    This function runs a few hundred forward-only passes (no gradient,
    weights untouched) so every `QuantAct.min_val`/`max_val` starts from a
    realistic estimate of the true activation range *before* the first
    real optimizer step.

    Call this once, right after `set_quantize_mode(model, True)` and
    right before the training loop starts.

    :param model: the SignModel, already in quantize=True mode
    :param train_iter: a torchtext-style batch iterator (e.g. output of
        `signjoey.data.make_data_iter` on the training set)
    :param txt_pad_index: same value passed to `Batch(...)` in training.py
    :param sgn_dim: same value passed to `Batch(...)` in training.py
        (== self.feature_size in TrainManager)
    :param use_cuda: same value passed to `Batch(...)` in training.py
    :param num_batches: how many batches to run calibration over
    :return: number of batches actually used (may be < num_batches if
        train_iter is shorter)
    """
    was_training = model.training
    model.eval()

    # Calibration is an explicit observer mode.  It must not be implemented
    # by relying on ``model.train()`` because that would enable dropout and
    # BatchNorm batch statistics and would make the calibration itself noisy.
    observer_states = []
    for module in model.modules():
        if hasattr(module, "set_calibration_mode"):
            old_state = bool(getattr(module, "calibrating", False))
            observer_states.append((module, old_state))
            module.set_calibration_mode(True)

    seen = 0
    try:
        with torch.no_grad():
            for torch_batch in train_iter:
                if seen >= num_batches:
                    break
                batch = Batch(
                    is_train=False,
                    torch_batch=torch_batch,
                    txt_pad_index=txt_pad_index,
                    sgn_dim=sgn_dim,
                    use_cuda=use_cuda,
                )
                # Teacher-forced forward is sufficient to visit the complete
                # encoder/decoder graph.  Calling run_batch here used to run
                # TensorFlow CTC decoding and autoregressive search during
                # calibration, and could leave some decoder activations
                # unobserved or make calibration depend on EOS/search bugs.
                model(
                    sgn=batch.sgn,
                    sgn_mask=batch.sgn_mask,
                    sgn_lengths=batch.sgn_lengths,
                    txt_input=batch.txt_input,
                    txt_mask=batch.txt_mask,
                )
                seen += 1
    finally:
        for module, old_state in observer_states:
            module.set_calibration_mode(old_state)
        if was_training:
            model.train()

    return seen
