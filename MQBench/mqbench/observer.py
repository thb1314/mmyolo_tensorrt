import math
from typing import Tuple

import torch
from torch.quantization.observer import _ObserverBase

from mqbench.fake_quantize.quantize_base import _version_under_1100 
from mqbench.utils import sync_tensor, pot_quantization, is_symmetric_quant
from mqbench.utils.logger import logger
from mqbench.utils.hook import PerChannelLoadHook
import warnings


class ObserverBase(_ObserverBase):
    '''
        Support per-tensor / per-channel.
        dtype: quant min/max can be infered using dtype, we actually do not need this.
        qscheme: quantization scheme
        reduce_range: special for fbgemm to avoid overflow
        quant_min: fix point value min
        quant_max: fix point value max
        ch_axis: per-channel axis or per-tensor(-1)
        above is similiar to torch observer.
        pot_scale: indecate wheather scale is power of two.
    '''

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False,
                 factory_kwargs=None):
        # Since torch 1.10, function calculate_qmin_qmax is not a member function of observer,
        # but import from utils. It is hard to control. We use try...except here.
        stored_min, sotred_max = quant_min, quant_max
        if quant_max is not None and quant_min is not None and (quant_max - quant_min + 1 > 256):
            quant_min, quant_max = -128, 127
        super(ObserverBase, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max)
        self.quant_min = stored_min
        self.quant_max = sotred_max
        self.quant_min, self.quant_max = self._calculate_qmin_qmax()
        self.ch_axis = ch_axis
        self.pot_scale = pot_scale
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.load_state_dict_hook = PerChannelLoadHook(self)

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        if self.pot_scale:
            scale = pot_quantization(scale)
        return scale, zero_point

    @torch.jit.export
    def _calculate_qmin_qmax(self) -> Tuple[int, int]:
        r"""Calculates actual qmin and qmax based on the quantization range,
        observer datatype and if range is reduced.
        """
        if self.has_customized_qrange:
            quant_min, quant_max = self.quant_min, self.quant_max
        else:
            # Fallback onto default 8-bit qmin and qmax calculation if dynamic range is not used.
            if self.dtype == torch.qint8:
                if self.reduce_range:
                    quant_min, quant_max = -64, 63
                else:
                    quant_min, quant_max = -128, 127
            elif self.dtype == torch.quint8:
                if self.reduce_range:
                    quant_min, quant_max = 0, 127
                else:
                    quant_min, quant_max = 0, 255
            else:
                quant_min, quant_max = 0, 15
        return quant_min, quant_max

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={} ch_axis={} pot={}".format(self.min_val if self.ch_axis == -1 else 'List',
                                                                 self.max_val if self.ch_axis == -1 else 'List',
                                                                 self.ch_axis, self.pot_scale)


class MinMaxObserver(ObserverBase):
    '''
    Calculate minmax of whole calibration dataset.
    '''
    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, 
                 pot_scale=False, residual = None, factory_kwargs=None):
        self.residual = residual
        self.max_topk_list = []   
        self.min_topk_list = []   
        super(MinMaxObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                             ch_axis, pot_scale, factory_kwargs)

        

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        numel = x_orig.numel()
        if self.ch_axis == -1:
            
            if self.residual:
                topk_value = max(1, int(numel * self.residual))
                print('numel', numel, 'topk_value', topk_value)

                max_topk_list = torch.topk(x.view(-1), k=topk_value, largest=True, sorted=True)[0]
                if not self.max_topk_list:
                    self.max_topk_list = max_topk_list
                else:
                    self.max_topk_list = torch.topk(torch.cat([max_topk_list, self.max_topk_list], dim=0), k=topk_value, largest=True, sorted=True)[0]

                min_topk_list = torch.topk(x.view(-1), k=topk_value, largest=False, sorted=True)[0]
                if not self.min_topk_list:
                    self.min_topk_list = min_topk_list
                else:
                    self.min_topk_list = torch.topk(torch.cat([min_topk_list, self.min_topk_list], dim=0), k=topk_value, largest=False, sorted=True)[0]

                self.min_val = min_val_cur = self.min_topk_list[-1]
                self.max_val = max_val_cur = self.max_topk_list[-1]
            else:
                min_val_cur, max_val_cur = torch._aminmax(x)

        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)

            if self.residual:
                topk_value = max(1, int(y.shape[-1] * self.residual))

                # [N, top_k]
                max_topk_list = torch.topk(y, k=topk_value, largest=True, sorted=True, dim=-1)[0]
                if not self.max_topk_list:
                    self.max_topk_list = max_topk_list
                else:
                    self.max_topk_list = torch.topk(torch.cat([max_topk_list, self.max_topk_list], dim=-1), k=topk_value, largest=True, sorted=True, dim=-1)[0]

                min_topk_list = torch.topk(y, k=topk_value, largest=False, sorted=True, dim=-1)[0]
                if not self.min_topk_list:
                    self.min_topk_list = min_topk_list
                else:
                    self.min_topk_list = torch.topk(torch.cat([min_topk_list, self.min_topk_list], dim=-1), k=topk_value, largest=False, sorted=True, dim=-1)[0]

                self.min_val = min_val_cur = self.min_topk_list[:, -1]
                self.max_val = max_val_cur = self.max_topk_list[:, -1]
            else:
                min_val_cur, max_val_cur = torch._aminmax(y, 1)

        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)

        return x

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        if self.pot_scale:
            scale = pot_quantization(scale)
        return scale, zero_point

class MinMaxFloorObserver(ObserverBase):
    '''
    Calculate minmax of whole calibration dataset with floor but round.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False,
                 factory_kwargs=None):
        super(MinMaxFloorObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                                  ch_axis, pot_scale, factory_kwargs)
        '''
        The quant_type could be 'input', 'param', 'tensor', the co-responding
        range is 1, 5, 5,
        mth is 2, 3, 2
        '''
        self.quant_type = None


    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            logger.warn('The per-tensor observer does not support per-channel min-max!')
            min_val_cur, max_val_cur = torch._aminmax(x)

        self.min_val = min_val_cur
        self.max_val = max_val_cur
        self._x = x
        return x

    def calculate_qparams(self):
        if self.quant_type is None:
            raise ValueError('You should set the observer type before forward!')
        else:
            scale_range = 1 if self.quant_type == 'input' else 5
            mth = 3 if self.quant_type == 'param' else 2
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = scale.data * 0 + max(self.min_val / self.quant_min, self.max_val / self.quant_max)
        if scale < 2 ** -15:
            max_scale = 0
        else:
            max_scale = 1 / scale
            max_scale = torch.floor(max_scale.log2())
        min_loss = torch.tensor([float('inf')])
        final_scale = max_scale
        max_scale = int(max_scale)
        for s in range(max_scale, max_scale + scale_range):
            _s = 1 / 2 ** s
            if mth == 3:
                new_x = _s * torch.clamp(torch.round(self._x / _s), self.quant_min, self.quant_max)
            elif mth == 2:
                new_x = torch.clamp(self._x / _s, self.quant_min, self.quant_max)
                new_x = torch.where((new_x < 0) & (new_x - new_x.floor() == 0.5), new_x.ceil(), new_x.round())
                new_x *= _s
            loss = ((new_x - self._x)**2).sum()
            min_loss = min_loss.to(loss.device)
            if loss < min_loss:
                min_loss = loss
                final_scale = s
        final_scale = min(final_scale, 12)
        scale = scale.data * 0 + 1 / (2 ** final_scale)
        zero_point = torch.zeros_like(zero_point)
        if not is_symmetric_quant(self.qscheme):
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        sync_tensor(scale)
        sync_tensor(zero_point)
        return scale, zero_point

    def set_quant_type(self, qtype):
        self.quant_type = qtype


class EMAMinMaxObserver(ObserverBase):
    """Moving average min/max among batches.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, ema_ratio=0.9,
                 factory_kwargs=None):
        super(EMAMinMaxObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                                ch_axis, pot_scale, factory_kwargs)
        self.ema_ratio = ema_ratio

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.ema_ratio + min_val_cur * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + max_val_cur * (1.0 - self.ema_ratio)
        return x


class PoTModeObserver(ObserverBase):
    r"""Records the most frequent Potscale of ``x``."""
    """
    Borrow from vitis
    https://github.com/Xilinx/Vitis-AI/blob/master/tools/Vitis-AI-Quantizer/vai_q_pytorch/pytorch_binding/pytorch_nndct/quantization/torchquantizer.py
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, factory_kwargs=None):
        super(PoTModeObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max, ch_axis, pot_scale, factory_kwargs)
        self.quant_type = None
        self.counter = [0] * 20

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            logger.warn('The per-tensor observer does not support per-channel min-max!')
            min_val_cur, max_val_cur = torch._aminmax(x)

        self.min_val = min_val_cur
        self.max_val = max_val_cur
        self._x = x
        return x

    def calculate_qparams(self):
        if self.quant_type is None:
            raise ValueError('You should set the observer type before forward!')
        else:
            scale_range = 1 if self.quant_type == 'input' else 5
            mth = 3 if self.quant_type == 'param' else 2
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = scale.data * 0 + max(self.min_val / self.quant_min, self.max_val / self.quant_max)
        if scale < 2 ** -15:
            max_scale = 0
        else:
            max_scale = 1 / scale
            max_scale = torch.floor(max_scale.log2())
        min_loss = torch.tensor([float('inf')])
        final_scale = max_scale
        max_scale = int(max_scale)
        for s in range(max_scale, max_scale + scale_range):
            _s = 1 / 2 ** s
            if mth == 3:
                new_x = _s * torch.clamp(torch.round(self._x / _s), self.quant_min, self.quant_max)
            elif mth == 2:
                new_x = torch.clamp(self._x / _s, self.quant_min, self.quant_max)
                new_x = torch.where((new_x < 0) & (new_x - new_x.floor() == 0.5), new_x.ceil(), new_x.round())
                new_x *= _s
            loss = ((new_x - self._x)**2).sum()
            min_loss = min_loss.to(loss.device)
            if loss < min_loss:
                min_loss = loss
                final_scale = s
        final_scale = min(final_scale, 12)
        self.counter[final_scale + 7] += 1
        final_scale = self.counter.index(max(self.counter)) - 7
        scale = scale.data * 0 + 1 / (2 ** final_scale)
        zero_point = torch.zeros_like(zero_point)
        if not is_symmetric_quant(self.qscheme):
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        sync_tensor(scale)
        sync_tensor(zero_point)
        return scale, zero_point

    def set_quant_type(self, qtype):
        self.quant_type = qtype


class EMAQuantileObserver(ObserverBase):
    """Moving average quantile among batches.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, ema_ratio=0.9,
                 threshold=0.99999, bins=2048, factory_kwargs=None):
        super(EMAQuantileObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                                  ch_axis, pot_scale, factory_kwargs)
        assert self.ch_axis == -1, "Quantile observer only support in per-tensor scheme."
        self.ema_ratio = ema_ratio
        self.threshold = threshold
        self.bins = bins

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch._aminmax(x)
        max_hist_range = torch.max(-min_val_cur, max_val_cur)
        hist = torch.histc(torch.abs(x), bins=self.bins, min=0., max=max_hist_range)
        cur_total = 0
        clip_value = max_hist_range
        for i, cnt in enumerate(hist):
            if cur_total + cnt >= self.threshold * x.numel():
                clip_value = (i + 0.5) * (max_hist_range / self.bins)
                break
            cur_total += cnt

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = max(min_val_cur, -clip_value)
            self.max_val = min(max_val_cur, clip_value)
        else:
            self.min_val = self.min_val * self.ema_ratio + max(min_val_cur, -clip_value) * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + min(max_val_cur, clip_value) * (1.0 - self.ema_ratio)
        return x


class ClipStdObserver(ObserverBase):
    """Clip std.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, std_scale=2.6,
                 factory_kwargs=None):
        super(ClipStdObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                              ch_axis, pot_scale, factory_kwargs=None)
        self.std_scale = std_scale

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
            mean = x.mean()
            std = x.std()
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
            mean = y.mean(1)
            std = y.std(1)

        # using statistics to clip min and max
        min_val = torch.minimum(mean - self.std_scale * std, min_val_cur)
        max_val = torch.maximum(mean + self.std_scale * std, max_val_cur)

        self.min_val = min_val
        self.max_val = max_val

        return x


class LSQObserver(ObserverBase):
    '''
    LSQ observer.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, factory_kwargs=None):
        super(LSQObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                          ch_axis, pot_scale, factory_kwargs)
        self.tensor_norm = None

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            self.tensor_norm = x.abs().mean()
            self.min_val, self.max_val = torch._aminmax(x)
        else:
            # compute channel-wise mean
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            self.tensor_norm = y.abs().mean(1)
            self.min_val, self.max_val = torch._aminmax(y, 1)

        return x

    def calculate_qparams(self):
        scale = 2 * self.tensor_norm / math.sqrt(self.quant_max)
        zero_point = torch.zeros_like(self.tensor_norm)
        sync_tensor(scale)
        sync_tensor(zero_point)
        if self.pot_scale:
            scale = pot_quantization(scale)
        if not is_symmetric_quant(self.qscheme):
            zero_point = self.quant_min - torch.round(self.min_val / scale)
        return scale, zero_point


class LSQPlusObserver(ObserverBase):
    '''
    LSQ+ observer.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, factory_kwargs=None):

        super(LSQPlusObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                              ch_axis, pot_scale, factory_kwargs)
        self.mean = None
        self.std = None

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            self.mean = x.mean()
            self.std = x.std()
            self.min_val, self.max_val = torch._aminmax(x)
        else:
            # compute channel-wise mean
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            self.mean = y.mean(1)
            self.std = y.std(1)
            self.min_val, self.max_val = torch._aminmax(y)

        return x

    def calculate_qparams(self):
        scale = torch.maximum((self.mean - 3 * self.std).abs(),
                              (self.mean + 3 * self.std).abs()) / (self.quant_max - self.quant_min + 1)
        sync_tensor(scale)
        sync_tensor(zero_point)
        if self.pot_scale:
            scale = pot_quantization(scale)
        zero_point = torch.zeros_like(self.mean)
        if not is_symmetric_quant(self.qscheme):
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        return scale, zero_point


class MSEObserver(ObserverBase):
    '''
    Calculate mseobserver of whole calibration dataset.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, p=2.0,
                 factory_kwargs=None):
        super(MSEObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                          ch_axis, pot_scale, factory_kwargs)
        self.p = p

    def lp_loss(self, pred, tgt, dim=None):
        """
        loss function measured in L_p Norm
        """
        return (pred - tgt).abs().pow(self.p).mean(dim) if dim else (pred - tgt).abs().pow(self.p).mean()


    def mse(self, x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor, iter=80):
        best_score = 1e+10
        best_min, best_max = torch.tensor([1.0], dtype=torch.float), torch.tensor([1.0], dtype=torch.float)
        best_min.copy_(x_min)
        best_max.copy_(x_max)
        for i in range(iter):
            new_min = x_min * (1.0 - (i * 0.01))
            new_max = x_max * (1.0 - (i * 0.01))
            scale, zero_point = self._calculate_qparams(new_min, new_max)
            x_q = torch.fake_quantize_per_tensor_affine(
                x, scale.item(), int(zero_point.item()),
                self.quant_min, self.quant_max)
            score = self.lp_loss(x_q, x)
            if score < best_score:
                best_score = score
                best_min, best_max = new_min, new_max
        return best_min, best_max

    def mse_perchannel(self, x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor, iter=80, ch_axis=0):
        assert x_min.shape == x_max.shape
        assert ch_axis >= 0, f'{ch_axis}'
        best_score = 1e+10 * torch.ones_like(x_min)
        best_min, best_max = x_min.clone(), x_max.clone()
        reduce_dim = tuple([i for i in range(len(x.shape)) if i != ch_axis])
        for i in range(iter):
            new_min = x_min * (1.0 - (i * 0.01))
            new_max = x_max * (1.0 - (i * 0.01))
            scale, zero_point = self._calculate_qparams(new_min, new_max)
            x_q = torch.fake_quantize_per_channel_affine(
                x, scale, zero_point.long() if _version_under_1100 else zero_point, ch_axis, 
                self.quant_min, self.quant_max)
            score = self.lp_loss(x_q, x, reduce_dim)
            update_idx = (score < best_score)
            best_score[update_idx] = score[update_idx]
            best_min[update_idx] = new_min[update_idx]
            best_max[update_idx] = new_max[update_idx]
        return best_min, best_max

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
            min_val_cur, max_val_cur = self.mse(x, min_val_cur, max_val_cur, iter=95)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            x_channel = x.permute(new_axis_list)
            y = torch.flatten(x_channel, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
            min_val_cur, max_val_cur = self.mse_perchannel(x, min_val_cur, max_val_cur, iter=80, ch_axis=self.ch_axis)

        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)
        return x


class EMAMSEObserver(ObserverBase):
    '''
    Calculate mseobserver of whole calibration dataset.
    '''
    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False,
                 p=2.0, ema_ratio=0.9, factory_kwargs=None):
        super(EMAMSEObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                             ch_axis, pot_scale, factory_kwargs)
        self.ema_ratio = ema_ratio
        self.p = p

    def lp_loss(self, pred, tgt, dim=None):
        """
        loss function measured in L_p Norm
        """
        return (pred - tgt).abs().pow(self.p).mean(dim) if dim else (pred - tgt).abs().pow(self.p).mean()

    def mse(self, x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor, iter=80):
        best_score = 1e+10
        best_min, best_max = torch.tensor([1.0], dtype=torch.float), torch.tensor([1.0], dtype=torch.float)
        best_min.copy_(x_min)
        best_max.copy_(x_max)
        for i in range(iter):
            new_min = x_min * (1.0 - (i * 0.01))
            new_max = x_max * (1.0 - (i * 0.01))
            scale, zero_point = self._calculate_qparams(new_min, new_max)
            x_q = torch.fake_quantize_per_tensor_affine(
                x, scale.item(), int(zero_point.item()),
                self.quant_min, self.quant_max)
            score = self.lp_loss(x_q, x)
            if score < best_score:
                best_score = score
                best_min, best_max = new_min, new_max
        return best_min, best_max

    def mse_perchannel(self, x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor, iter=80, ch_axis=0):
        assert x_min.shape == x_max.shape
        assert ch_axis >= 0, f'{ch_axis}'
        best_score = 1e+10 * torch.ones_like(x_min)
        best_min, best_max = x_min.clone(), x_max.clone()
        reduce_dim = tuple([i for i in range(len(x.shape)) if i != ch_axis])
        for i in range(iter):
            new_min = x_min * (1.0 - (i * 0.01))
            new_max = x_max * (1.0 - (i * 0.01))
            scale, zero_point = self._calculate_qparams(new_min, new_max)
            x_q = torch.fake_quantize_per_channel_affine(
                x, scale, zero_point.long() if _version_under_1100 else zero_point, ch_axis, 
                self.quant_min, self.quant_max)
            score = self.lp_loss(x_q, x, reduce_dim)
            update_idx = (score < best_score)
            best_score[update_idx] = score[update_idx]
            best_min[update_idx] = new_min[update_idx]
            best_max[update_idx] = new_max[update_idx]
        return best_min, best_max

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
            min_val_cur, max_val_cur = self.mse(x, min_val_cur, max_val_cur, iter=95)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            x_channel = x.permute(new_axis_list)
            y = torch.flatten(x_channel, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
            min_val_cur, max_val_cur = self.mse_perchannel(x, min_val_cur, max_val_cur, iter=80, ch_axis=self.ch_axis)

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.ema_ratio + min_val_cur * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + max_val_cur * (1.0 - self.ema_ratio)
        return x


class HistogramObserver(ObserverBase):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.

    Args:
        bins: Number of bins to use for the histogram
        upsample_rate: Factor by which the histograms are upsampled, this is
                       used to interpolate histograms with varying ranges across observations
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
        The histogram is computed continuously, and the ranges per bin change
        with every new tensor observed.
    2. Search the distribution in the histogram for optimal min/max values.
        The search for the min/max values ensures the minimization of the
        quantization error with respect to the floating point model.
    3. Compute the scale and zero point the same way as in the
        :class:`~torch.ao.quantization.MinMaxObserver`
    """
    def __init__(
        self,
        dtype: torch.dtype = torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        ch_axis=-1,
        pot_scale=False,
        bins: int = 2048,
        upsample_rate: int = 128,
        eps=torch.finfo(torch.float32).eps,
        factory_kwargs=None,
    ) -> None:
        # bins: The number of bins used for histogram calculation.
        super(HistogramObserver, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            ch_axis=ch_axis,
            pot_scale=pot_scale,
            factory_kwargs=factory_kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.bins = bins
        self.register_buffer("histogram", torch.zeros(self.bins, **factory_kwargs))
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        self.dst_nbins = 2 ** torch.iinfo(self.dtype).bits
        self.upsample_rate = upsample_rate

    def _get_norm(
        self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        return density * norm

    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins, device=self.histogram.device)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1
        )
        dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width

        density = self.histogram / bin_width

        norm = torch.zeros(self.bins, device=self.histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(delta_begin,
                               torch.ones(self.bins, device=self.histogram.device) * delta_end,
                               density)

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)

        return norm.sum().item()

    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        assert self.histogram.size()[0] == self.bins, "bins mistmatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def _adjust_min_max(
        self, combined_min: torch.Tensor, combined_max: torch.Tensor, upsample_rate: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        # We ensure that:
        # (combined_max - combined_min)/(downsample_rate*Nbins) = (max - min)/(upsample_rate*Nbins)
        # This allows us to have a common grid of resolution s, where we can align
        # the input histogram
        # start_idx maps min_val to the histogram bin index.

        hist_bin_width = (self.max_val - self.min_val) / (self.bins * upsample_rate)
        downsample_rate = int(
            torch.ceil(
                (combined_max - combined_min) / (self.bins * hist_bin_width)
            ).item()
        )
        e = downsample_rate * (self.bins * hist_bin_width) - (
            combined_max - combined_min
        )
        # Relax only the max, not the min, so that for one sided distributions, min stays at zero
        combined_max = combined_max + e
        combined_min = combined_min
        start_idx = int(
            torch.round((self.min_val - combined_min) / hist_bin_width).item()
        )
        return combined_min, combined_max, downsample_rate, start_idx

    def _combine_histograms(
        self,
        orig_hist: torch.Tensor,
        new_hist: torch.Tensor,
        upsample_rate: int,
        downsample_rate: int,
        start_idx: int,
        Nbins: int,
    ) -> torch.Tensor:
        # First up-sample the histogram with new data by a factor of L
        # This creates an approximate probability density thats piecwise constant
        upsampled_histogram = new_hist.repeat_interleave(upsample_rate)
        # Now insert the upsampled histogram into the output
        # histogram, which is initialized with zeros.
        # The offset at which the histogram is introduced is determined
        # by the start index as the output histogram can cover a wider range
        histogram_with_output_range = torch.zeros(
            (Nbins * downsample_rate), device=orig_hist.device
        )
        histogram_with_output_range[
            start_idx : Nbins * upsample_rate + start_idx
        ] = upsampled_histogram
        # Compute integral histogram, double precision is needed to ensure
        # that there are no overflows
        integral_histogram = torch.cumsum(
            histogram_with_output_range, 0, dtype=torch.double
        )[downsample_rate - 1 :: downsample_rate]
        # Finally perform interpolation
        shifted_integral_histogram = torch.zeros((Nbins), device=orig_hist.device)
        shifted_integral_histogram[1:Nbins] = integral_histogram[0:-1]
        interpolated_histogram = (
            integral_histogram - shifted_integral_histogram
        ) / upsample_rate
        orig_hist = orig_hist + interpolated_histogram.to(torch.float)
        return orig_hist

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        min_val = self.min_val
        max_val = self.max_val
        same_values = min_val.item() == max_val.item()
        is_uninitialized = min_val == float("inf") and max_val == float("-inf")
        if is_uninitialized or same_values:
            min_val, max_val = torch.aminmax(x)
            self.min_val.resize_(min_val.shape)
            self.min_val.copy_(min_val)
            self.max_val.resize_(max_val.shape)
            self.max_val.copy_(max_val)
            assert (
                min_val.numel() == 1 and max_val.numel() == 1
            ), "histogram min/max values must be scalar."
            torch.histc(
                x, self.bins, min=int(min_val), max=int(max_val), out=self.histogram
            )
        else:
            new_min, new_max = torch.aminmax(x)
            combined_min = torch.min(new_min, min_val)
            combined_max = torch.max(new_max, max_val)
            # combine the existing histogram and new histogram into 1 histogram
            # We do this by first upsampling the histogram to a dense grid
            # and then downsampling the histogram efficiently
            (
                combined_min,
                combined_max,
                downsample_rate,
                start_idx,
            ) = self._adjust_min_max(combined_min, combined_max, self.upsample_rate)
            assert (
                combined_min.numel() == 1 and combined_max.numel() == 1
            ), "histogram min/max values must be scalar."
            combined_histogram = torch.histc(
                x, self.bins, min=int(combined_min), max=int(combined_max)
            )
            if combined_min == min_val and combined_max == max_val:
                combined_histogram += self.histogram
            else:
                combined_histogram = self._combine_histograms(
                    combined_histogram,
                    self.histogram,
                    self.upsample_rate,
                    downsample_rate,
                    start_idx,
                    self.bins,
                )

            self.histogram.detach_().resize_(combined_histogram.shape)
            self.histogram.copy_(combined_histogram)
            self.min_val.detach_().resize_(combined_min.shape)
            self.min_val.copy_(combined_min)
            self.max_val.detach_().resize_(combined_max.shape)
            self.max_val.copy_(combined_max)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        is_uninitialized = self.min_val == float("inf") and self.max_val == float(
            "-inf"
        )
        if is_uninitialized:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor([0], device=self.min_val.device.type)
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_min, new_max = self._non_linear_param_search()

        return self._calculate_qparams(new_min, new_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(HistogramObserver, self)._save_to_state_dict(
            destination, prefix, keep_vars
        )
        destination[prefix + "min_val"] = self.min_val
        destination[prefix + "max_val"] = self.max_val

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 3:
            # if min_val and max_val are not initialized, update their shape
            # to account for the differences between v2 and v3
            min_val_name, max_val_name = prefix + "min_val", prefix + "max_val"
            if min_val_name in state_dict:
                if state_dict[min_val_name].shape == torch.Size([0]):
                    state_dict[min_val_name] = torch.tensor(float("inf"))
            if max_val_name in state_dict:
                if state_dict[max_val_name].shape == torch.Size([0]):
                    state_dict[max_val_name] = torch.tensor(float("-inf"))

        local_state = ["min_val", "max_val"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(HistogramObserver, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
