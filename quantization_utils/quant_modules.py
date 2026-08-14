import math
import numpy as np
from torch.autograd import Function, Variable
import torch
import bisect
from fractions import Fraction
import decimal
from decimal import Decimal
import time
import torch.nn as nn
import torch.nn.functional as F

from quantization_utils.quant_utils import *


def _dummy_scaling_factor(x):
    """
    Placeholder scaling factor returned in float (quantize=False) mode, so
    that every layer's return signature stays `(tensor, scaling_factor)`
    regardless of mode -- callers never need to branch on `quantize`.
    """
    return torch.ones(1, device=x.device, dtype=torch.float32)


class QuantAct(nn.Module):
    def __init__(self,
                 activation_bit=8,
                 act_range_momentum=0.95,
                 running_stat=True,
                 quant_mode="symmetric",
                 quantize=True):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.quantize = quantize
        # Observers are updated during training (or during the explicit
        # calibration pass).  They must not be updated by validation: doing
        # so makes a dev batch change the quantization grid and makes later
        # training/evaluation results depend on how often validation ran.
        self.calibrating = False

        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.register_buffer('act_scaling_factor', torch.zeros(1))

        if self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def fix(self):
        self.running_stat = False
        self.calibrating = False

    def unfix(self):
        self.running_stat = True
        self.calibrating = False

    def set_calibration_mode(self, enabled: bool = True):
        """Enable/disable observer updates while the model is in eval mode.

        Calibration deliberately runs with dropout and BatchNorm disabled;
        this flag is the explicit exception that still lets the activation
        observer see calibration data without making ordinary validation
        mutate its state.
        """
        self.calibrating = bool(enabled)

    def forward(self, x,
                pre_act_scaling_factor=None,
                identity=None,
                identity_scaling_factor=None):

        if not self.quantize:
            # ---- FP32 path: plain (optionally residual) passthrough ----
            out = x if identity is None else identity + x
            return out, _dummy_scaling_factor(x)

        with torch.no_grad():
            x_act = x if identity is None else identity + x

            if self.running_stat and (self.training or self.calibrating):
                x_flat = x_act.reshape(-1)
                cur_min = x_flat.min()
                cur_max = x_flat.max()

                if torch.eq(self.min_val, self.max_val).all():
                    self.min_val = cur_min
                    self.max_val = cur_max
                else:
                    self.min_val = (self.min_val * self.act_range_momentum
                                   + cur_min * (1 - self.act_range_momentum))
                    self.max_val = (self.max_val * self.act_range_momentum
                                   + cur_max * (1 - self.act_range_momentum))

            # self.act_scaling_factor = symmetric_linear_quantization_params(
            #     self.activation_bit, self.min_val, self.max_val
            # )

            self.act_scaling_factor = symmetric_linear_quantization_params(
                self.activation_bit, self.min_val, self.max_val).clamp(min=1e-4, max=10.0)

        if pre_act_scaling_factor is None:
            quant_act_int = self.act_function(
                x, self.activation_bit, self.act_scaling_factor, False
            )
        else:
            pre_act_scaling_factor = pre_act_scaling_factor.reshape(-1).mean()

            if identity_scaling_factor is not None:
                identity_scaling_factor = identity_scaling_factor.reshape(-1).mean()

            quant_act_int = fixedpoint_mul.apply(
                x, pre_act_scaling_factor,
                self.activation_bit, self.quant_mode,
                self.act_scaling_factor,
                identity, identity_scaling_factor
            )

        correct_output_scale = self.act_scaling_factor.view(1)
        return quant_act_int * correct_output_scale, self.act_scaling_factor


class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 per_channel=True,
                 quant_mode='symmetric',
                 quantize=True):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.weight_bit = weight_bit
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = bias_bit is not None
        self.quant_mode = quant_mode
        self.quantize = quantize

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        else:
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))

        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        if self.bias is not None:
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def forward(self, x, prev_act_scaling_factor=None):
        if not self.quantize:
            # ---- FP32 path: this IS the same self.weight/self.bias the
            # QAT path quantizes -- no separate float weights to keep in
            # sync, no state_dict remapping needed when switching modes.
            return F.linear(x, self.weight, self.bias), _dummy_scaling_factor(x)

        with torch.no_grad():
            w = self.weight
            v = w.reshape(w.shape[0], -1)
            cur_min = v.min(axis=1).values
            cur_max = v.max(axis=1).values
            self.min_val = cur_min
            self.max_val = cur_max

            self.fc_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val
            )

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.fc_scaling_factor, True
        )

        prev_act_scaling_factor = prev_act_scaling_factor.reshape(-1).mean()

        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        if self.bias is not None:
            self.bias_integer = self.weight_function(
                self.bias, self.bias_bit, bias_scaling_factor, True
            )
        else:
            self.bias_integer = None

        x_int = x / prev_act_scaling_factor
        # return (
        #     F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer)
        #     * bias_scaling_factor,
        #     bias_scaling_factor,
        # )
        bias_scaling_factor = bias_scaling_factor.clamp(min=1e-4, max=10.0)
        return (
            F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer)
            * bias_scaling_factor,
            bias_scaling_factor,
        )


class QuantMatMul(nn.Module):
    def __init__(self, quantize=True):
        super(QuantMatMul, self).__init__()
        self.register_buffer('act_scaling_factor', torch.zeros(1))
        self.quantize = quantize

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        if not self.quantize:
            out = A @ B
            return out, _dummy_scaling_factor(out)

        sf_A = pre_act_scaling_factor_A.reshape(-1).mean()
        sf_B = pre_act_scaling_factor_B.reshape(-1).mean()

        A_int = A / sf_A
        B_int = B / sf_B
        act_scaling_factor = sf_A * sf_B
        self.act_scaling_factor = act_scaling_factor
        return (A_int @ B_int) * act_scaling_factor, act_scaling_factor


class QuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 quant_mode="symmetric",
                 per_channel=False, # تغییر به False برای سازگاری با QuantAct
                 weight_percentile=0,
                 quantize=True):
        super(QuantConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias
        )
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = bias_bit is not None
        self.quantize = quantize

        self.register_buffer('conv_scaling_factor', torch.zeros(1)) # تغییر به scalar
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        if bias:
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def forward(self, x, pre_act_scaling_factor=None):
        if not self.quantize:
            out = F.conv2d(x, self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)
            return out, _dummy_scaling_factor(out)

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        else:
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))

        with torch.no_grad():
            w = self.weight
            # اصلاح: استفاده از per-tensor برای وزن تا خروجی کاملاً per-tensor باشد
            cur_min = w.min()
            cur_max = w.max()
            self.conv_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, cur_min, cur_max
            )

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.conv_scaling_factor, True
        )

        pre_act_sf = pre_act_scaling_factor.reshape(-1).mean()
        bias_scaling_factor = self.conv_scaling_factor * pre_act_sf
        
        if self.bias is not None:
            self.bias_integer = self.weight_function(
                self.bias, self.bias_bit, bias_scaling_factor, True
            )
        else:
            self.bias_integer = None

        x_int = x / pre_act_sf
        
        out_sf = bias_scaling_factor.view(1) # خروجی کاملاً اسکالر (per-tensor)
        return (F.conv2d(x_int, self.weight_integer, self.bias_integer,
                         self.stride, self.padding, self.dilation, self.groups) * out_sf, out_sf)


class IntLayerNorm(nn.LayerNorm):
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, quantize=True):
        super(IntLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.register_buffer('norm_scaling_factor', torch.zeros(1))
        self.quantize = quantize

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, scaling_factor=None):
        if not self.quantize:
            out = F.layer_norm(
                x, self.normalized_shape,
                self.weight if self.elementwise_affine else None,
                self.bias if self.elementwise_affine else None,
                self.eps,
            )
            return out, _dummy_scaling_factor(out)

        if scaling_factor is not None:
            scaling_factor = scaling_factor.reshape(-1).mean()
        
        if scaling_factor is not None and scaling_factor > 0:
            x_int = x / scaling_factor
        else:
            x_int = x

        mean_int = round_ste.apply(x_int.mean(dim=-1, keepdim=True))
        y_int = x_int - mean_int
        
        y_sq_int = y_int ** 2
        var_int = y_sq_int.mean(dim=-1, keepdim=True)

        std_int = torch.sqrt(var_int + self.eps)
        
        
        new_scaling_factor = scaling_factor * std_int if scaling_factor is not None else std_int
        
        factor = floor_ste.apply(1.0 / (std_int + self.eps) * (2 ** 15))
        y_int = floor_ste.apply(y_int * factor) / (2 ** 15)


        if self.elementwise_affine:

            y_int = y_int * self.weight + self.bias


        # self.norm_scaling_factor = new_scaling_factor.reshape(-1).mean()
        self.norm_scaling_factor = new_scaling_factor.reshape(-1).mean().clamp(min=1e-4, max=10.0)

        output = y_int * self.norm_scaling_factor

        return output, self.norm_scaling_factor


class IntGELU(nn.Module):
    def __init__(self, output_bit=8, quantize=True):
        super(IntGELU, self).__init__()
        self.output_bit = output_bit
        self.n = 23
        self.register_buffer('act_scaling_factor', torch.zeros(1))
        self.quantize = quantize

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)
        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)
        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        if not self.quantize:
            out = F.gelu(x)
            return out, _dummy_scaling_factor(out)

        scaling_factor = scaling_factor.reshape(-1).mean()

        pre_x_int = x / scaling_factor
        scaling_factor_sig = scaling_factor * 1.702

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig)
        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum.clamp_max_(2 ** 31 - 1)
        factor = floor_ste.apply((2 ** 31 - 1) / exp_int_sum)
        sigmoid_int = floor_ste.apply(
            exp_int * factor / 2 ** (31 - self.output_bit + 1)
        )
        sigmoid_scaling_factor = torch.tensor(
            [1 / 2 ** (self.output_bit - 1)], device=x.device
        )

        x_int = pre_x_int * sigmoid_int
        scaling_factor = scaling_factor * sigmoid_scaling_factor
        self.act_scaling_factor = scaling_factor
        return x_int * scaling_factor, scaling_factor


class IntSoftmax(nn.Module):
    def __init__(self, output_bit=8, quantize=True):
        super(IntSoftmax, self).__init__()
        self.output_bit = output_bit
        self.n = 15
        self.register_buffer('act_scaling_factor', torch.zeros(1))
        self.quantize = quantize

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)
        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)
        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        if not self.quantize:
            out = F.softmax(x, dim=-1)
            return out, _dummy_scaling_factor(out)

        scaling_factor = scaling_factor.reshape(-1).mean()

        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        exp_int_sum.clamp_max_(2 ** 31 - 1)
        factor = floor_ste.apply((2 ** 31 - 1) / exp_int_sum)
        exp_int = floor_ste.apply(
            exp_int * factor / 2 ** (31 - self.output_bit + 1)
        )
        scaling_factor = torch.tensor(
            [1 / 2 ** (self.output_bit - 1)], device=x.device
        )
        self.act_scaling_factor = scaling_factor
        return exp_int * scaling_factor, scaling_factor

class IntBatchNorm1d(nn.Module):
    """
    Quantization-aware, masked BatchNorm1d for the integer-only QAT
    pipeline. Same dual-mode contract as IntLayerNorm in this file:

    - quantize=False: bit-for-bit equivalent to a masked nn.BatchNorm1d
      (per-channel statistics over the batch/time axis).
    - quantize=True: the identical nn.Parameter weight/bias and the
      identical running_mean/running_var buffers are used, but the
      normalization is computed with the same straight-through
      round/floor reciprocal trick IntLayerNorm already uses for 1/std,
      so switching `self.quantize` never requires remapping a
      state_dict -- a model pretrained with quantize=False loads and
      continues training unchanged with quantize=True.

    :param num_features: channel dimension (last dim of the input)
    :param eps: numerical stability constant
    :param momentum: running-stats momentum, same semantics as nn.BatchNorm1d
    :param quantize: initial mode; toggled globally by model_utils.set_quantize_mode
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, quantize=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.quantize = quantize
        self._frozen = False

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('norm_scaling_factor', torch.zeros(1))

    def fix(self):
        """Freeze running stats (eval-mode behaviour) -- same convention
        as QuantAct.fix()/IntLayerNorm; called by model_utils.freeze_model."""
        self._frozen = True

    def unfix(self):
        self._frozen = False

    def _batch_stats(self, x_flat):
        """x_flat: [N, C], already masked/selected to valid positions."""
        if self.training and not self._frozen:
            mean = x_flat.mean(dim=0)
            var = x_flat.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(
                    mean.detach(), alpha=self.momentum
                )
                self.running_var.mul_(1 - self.momentum).add_(
                    var.detach(), alpha=self.momentum
                )
                self.num_batches_tracked += 1
            return mean, var
        return self.running_mean, self.running_var

    def forward(self, x_flat, scaling_factor=None):
        """
        :param x_flat: [N, num_features] -- masking/scatter is handled by
            the caller (MaskedNorm), exactly as it already is for IntLayerNorm.
        :param scaling_factor: incoming activation scale (ignored in FP32 mode)
        :return: (normalized output [N, num_features], output scaling factor)
        """
        if not self.quantize:
            mean, var = self._batch_stats(x_flat)
            out = (x_flat - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias
            return out, _dummy_scaling_factor(out)

        if scaling_factor is not None:
            scaling_factor = scaling_factor.reshape(-1).mean()

        mean, var = self._batch_stats(x_flat)
        mean_int = round_ste.apply(
            mean / scaling_factor if scaling_factor is not None else mean
        )
        x_int = x_flat / scaling_factor if scaling_factor is not None else x_flat
        y_int = x_int - mean_int
        var_int = (y_int ** 2).mean(dim=0)
        std_int = torch.sqrt(var_int + self.eps)

        factor = floor_ste.apply(1.0 / (std_int + self.eps) * (2 ** 15))
        y_int = floor_ste.apply(y_int * factor) / (2 ** 15)
        y_int = y_int * self.weight + self.bias

        new_sf = (
            scaling_factor * std_int if scaling_factor is not None else std_int
        )
        self.norm_scaling_factor = new_sf.reshape(-1).mean().clamp(min=1e-4, max=10.0)
        output = y_int * self.norm_scaling_factor
        return output, self.norm_scaling_factor


class IntSoftsign(nn.Module):
    """
    Quantization-aware Softsign: softsign(x) = x / (1 + |x|).

    Same fake-quant contract as IntGELU/IntSoftmax in this file. Softsign
    doesn't need the exp-shift trick those two use (no exponential
    involved), so the quantized path just dequantizes with the incoming
    scale, applies the exact function, and requantizes to `output_bit` --
    but the SAME contract (x in, (x, scaling_factor) out) means it's a
    drop-in replacement for IntGELU inside get_activation().
    """

    def __init__(self, output_bit=8, quantize=True):
        super().__init__()
        self.output_bit = output_bit
        self.register_buffer('act_scaling_factor', torch.zeros(1))
        self.quantize = quantize

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, scaling_factor=None):
        if not self.quantize:
            out = F.softsign(x)
            return out, _dummy_scaling_factor(out)

        # x already carries dequantized real values (x_int * scale), same
        # convention IntGELU/IntSoftmax rely on.
        out_real = x / (1.0 + x.abs())

        qmax = 2 ** (self.output_bit - 1) - 1
        out_scale = torch.tensor([1.0 / qmax], device=x.device)
        out_int = torch.clamp(round_ste.apply(out_real / out_scale), -qmax - 1, qmax)
        self.act_scaling_factor = out_scale
        return out_int * out_scale, out_scale
