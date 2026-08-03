import math
import numpy as np
from torch.autograd import Function, Variable
import torch
import bisect
from fractions import Fraction
import decimal
from decimal import Decimal
import time


def linear_quantize(input, scale, zero_point, is_weight):
    """
    Quantize single-precision input tensor to integers.
    Weight: per-channel  →  scale shape مطابق ابعاد weight
    Activation: per-tensor →  scale یک scalar است
    """
    if is_weight:
        if len(input.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
        elif len(input.shape) == 2:
            scale = scale.view(-1, 1)
            zero_point = zero_point.view(-1, 1)
        else:
            scale = scale.view(-1)
            zero_point = zero_point.view(-1)
    else:
        # per-tensor: scale یک scalar است، نیازی به reshape خاص نیست
        scale = scale.view(1)
        zero_point = zero_point.view(1)

    return torch.round(1. / scale * input + zero_point)


def symmetric_linear_quantization_params(num_bits, min_val, max_val):
    """
    Compute scalar scaling factor for symmetric quantization.
    همیشه یک scalar برمیگردونه (per-tensor).
    """
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1
        eps = torch.finfo(torch.float32).eps

        # اطمینان از scalar بودن
        if isinstance(min_val, torch.Tensor):
            min_val = min_val.min()
        if isinstance(max_val, torch.Tensor):
            max_val = max_val.max()

        max_val = torch.max(-min_val, max_val)
        scale = max_val / float(n)
        scale.clamp_(eps)

    return scale  # scalar tensor, shape=[]


class SymmetricQuantFunction(Function):
    """
    Quantize floating-point values with symmetric quantization.
    Activation: per-tensor (scale = scalar)
    Weight:     per-channel (scale = vector)
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale, is_weight):
        scale = specified_scale
        zero_point = torch.tensor(0.).to(x.device)
        n = 2 ** (k - 1) - 1

        new_quant_x = linear_quantize(x, scale, zero_point, is_weight=is_weight)
        new_quant_x = torch.clamp(new_quant_x, -n - 1, n)

        ctx.scale = scale
        ctx.is_weight = is_weight
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        is_weight = ctx.is_weight
        if is_weight:
            if len(grad_output.shape) == 4:
                scale = scale.view(-1, 1, 1, 1)
            elif len(grad_output.shape) == 2:
                scale = scale.view(-1, 1)
            else:
                scale = scale.view(-1)
        else:
            # per-tensor activation: scale یک scalar است
            scale = scale.view(1)
        return grad_output.clone() / scale, None, None, None


def batch_frexp(inputs, max_bit=31):
    """
    Decompose the scaling factor into mantissa and twos exponent.
    """
    device = inputs.device
    shape_of_input = inputs.size()
    inputs = inputs.view(-1)

    output_m, output_e = np.frexp(inputs.cpu().numpy())
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(
            Decimal(m * (2 ** max_bit)).quantize(
                Decimal('1'), rounding=decimal.ROUND_HALF_UP
            )
        )
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)
    output_e = float(max_bit) - output_e

    return (
        torch.from_numpy(output_m).to(device).view(shape_of_input),
        torch.from_numpy(output_e).to(device).view(shape_of_input),
    )


class floor_ste(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class round_ste(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class fixedpoint_mul(Function):
    """
    Fixed-point arithmetic برای dyadic pipeline I-ViT.

    اصلاحات:
    1. pre_act_scaling_factor همیشه به scalar تبدیل میشه (per-tensor activation)
    2. reshape بر اساس ndim قبل از with torch.no_grad() تعریف میشه
       → جلوگیری از UnboundLocalError
    3. z_scaling_factor هم به scalar تبدیل میشه
    """

    @staticmethod
    def forward(ctx, pre_act, pre_act_scaling_factor,
                bit_num, quant_mode, z_scaling_factor,
                identity=None, identity_scaling_factor=None):

        ctx.identity = identity

        if quant_mode == 'symmetric':
            n = 2 ** (bit_num - 1) - 1
        else:
            n = 2 ** bit_num - 1

        # ── per-tensor: هر sf را به scalar تبدیل کن ──────────────────────
        pre_act_scaling_factor = pre_act_scaling_factor.reshape(-1).mean().view(1)
        z_scaling_factor = z_scaling_factor.reshape(-1).mean().view(1)
        if identity is not None and identity_scaling_factor is not None:
            identity_scaling_factor = identity_scaling_factor.reshape(-1).mean().view(1)

        # ── reshape بر اساس ابعاد pre_act ─────────────────────────────────
        ndim = len(pre_act.shape)
        if ndim == 2:
            reshape = lambda x: x.view(1, 1)
        elif ndim == 3:
            reshape = lambda x: x.view(1, 1, 1)
        elif ndim == 4:
            reshape = lambda x: x.view(1, 1, 1, 1)
        else:
            raise NotImplementedError(
                f"fixedpoint_mul: unsupported pre_act ndim={ndim}, shape={pre_act.shape}"
            )

        ctx.z_scaling_factor = z_scaling_factor

        with torch.no_grad():
            pre_act_scaling_factor_r = reshape(pre_act_scaling_factor)
            z_scaling_factor_r = reshape(z_scaling_factor)

            z_int = torch.round(pre_act / pre_act_scaling_factor_r)

            _A = pre_act_scaling_factor_r.type(torch.double)
            _B = z_scaling_factor_r.type(torch.double)
            new_scale = _A / _B
            # new_scale یک scalar است، reshape مجدد لازم نیست
            m, e = batch_frexp(new_scale.view(1))
            m = m.view(1)
            e = e.view(1)

            # broadcast ایمن چون همه scalar هستند
            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round(output / (2.0 ** e))

            if identity is not None:
                identity_scaling_factor_r = reshape(identity_scaling_factor)
                wx_int = torch.round(identity / identity_scaling_factor_r)

                _A2 = identity_scaling_factor_r.type(torch.double)
                new_scale2 = _A2 / _B
                m1, e1 = batch_frexp(new_scale2.view(1))
                m1 = m1.view(1)
                e1 = e1.view(1)

                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / (2.0 ** e1))
                output = output1 + output

            if bit_num in [4, 8, 16, 32]:
                if quant_mode == 'symmetric':
                    return torch.clamp(output.type(torch.float), -n - 1, n)
                else:
                    return torch.clamp(output.type(torch.float), 0, n)
            else:
                return output.type(torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        identity_grad = None
        z_sf = ctx.z_scaling_factor
        if ctx.identity is not None:
            identity_grad = grad_output.clone() / z_sf
        return grad_output.clone() / z_sf, None, None, None, None, identity_grad, None