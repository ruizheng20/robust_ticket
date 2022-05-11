import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


class L0Mask(nn.Module):
    def __init__(self, mask_dim, mask_p):
        super().__init__()
        self.mask_setting = 'mask'
        self.mask_scores = nn.Parameter(torch.zeros(mask_dim))
        self.mask_p = mask_p
        self.l, self.r, self.b = -0.1, 1.1, 2 / 3
        self.init_weights()

    def init_weights(self):
        p = (self.mask_p - self.l) / (self.r - self.l)
        init.constant_(self.mask_scores, val=np.log(p / (1 - p)))
        # init.normal_(self.mask_scores, mean=0, std=0.01)

    def set_temperature(self, temp):
        self.b = temp
        
    def produce_mask(self):
        if self.training:
            u = torch.zeros_like(self.mask_scores).uniform_().clamp(0.0001, 0.9999)
            s = torch.sigmoid((u.log() - (1 - u).log() + self.mask_scores) / self.b)
        else:
            s = torch.sigmoid(self.mask_scores)
        s_bar = s * (self.r - self.l) + self.l
        mask = s_bar.clamp(min=0.0, max=1.0)
        return mask
    
    def regularizer(self):
        return torch.sum(torch.sigmoid(self.mask_scores - self.b * np.log(-self.l / self.r))) / self.mask_scores.numel()


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 mask_p: float=0.9, out_w_per_mask=1, in_w_per_mask=1, num_heads=12):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.num_heads = num_heads
        self.out_w_per_mask = out_w_per_mask
        self.in_w_per_mask = in_w_per_mask
        
        assert out_features % out_w_per_mask == 0, "{} % {} not 0".format(out_features, out_w_per_mask)
        assert in_features % in_w_per_mask == 0, "{} % {} not 0".format(in_features, in_w_per_mask)
        mask_dim = (1, out_features // out_w_per_mask, 1, in_features // in_w_per_mask)
        self.mask = L0Mask(mask_dim, mask_p)
        
        self.cached_activation = None
        self.do_caching = False

    def produce_mask_reshaped(self):
        mask = self.mask.produce_mask()
        mask = mask.repeat(self.out_w_per_mask, 1, self.in_w_per_mask, 1)
        return mask.reshape(self.out_features, self.in_features)

    def produce_mask(self):
        mask = self.mask.produce_mask()
        return mask

    def forward(self, input: torch.tensor):
        # "masked_weight = self.produce_mask_reshaped() * self.weight" is equivalent but slower.
        masked_weight = self.produce_mask() * self.weight.reshape(
            self.out_w_per_mask, self.out_features // self.out_w_per_mask,
            self.in_w_per_mask, self.in_features // self.in_w_per_mask)
        masked_weight = masked_weight.reshape(self.out_features, self.in_features)
        
        act = F.linear(input, masked_weight, self.bias)
        if self.do_caching:
            if self.cached_activation is None:
                self.cached_activation = act.detach()
            else: # only works if subbatched, since maxlen must be constant
                self.cached_activation = torch.cat((
                    self.cached_activation, act.detach()), dim = 0)
        return act
    
    def activate_caching(self, caching = True):
        self.cached_activation = None
        self.do_caching = caching

    @classmethod
    def from_layer(cls, layer, out_w_per_mask, in_w_per_mask, mask_p):
        assert type(layer) == nn.modules.linear.Linear
        res = cls(mask_p=mask_p, in_features=layer.in_features, out_features=layer.out_features,
                  bias=layer.bias is not None, out_w_per_mask=out_w_per_mask, in_w_per_mask=in_w_per_mask)
        res.weight = layer.weight
        res.bias = layer.bias
        return res # make sure to call cuda

#
# class BinaryL0Mask(nn.Module):
#     def __init__(self, mask_dim, mask_p):
#         super().__init__()
#         self.mask_setting = 'mask'
#         self.mask_scores = nn.Parameter(torch.zeros(mask_dim))
#         self.mask_p = mask_p
#         self.l, self.r, self.b = -0.1, 1.1, 2 / 3
#         self.init_weights()
#
#     def init_weights(self):
#         p = (self.mask_p - self.l) / (self.r - self.l)
#         init.constant_(self.mask_scores, val=np.log(p / (1 - p)))
#         # init.normal_(self.mask_scores, mean=0, std=0.01)
#
#     def set_temperature(self, temp):
#         self.b = temp
#
#     def produce_mask(self):
#         if self.training:
#             u = torch.zeros_like(self.mask_scores).uniform_().clamp(0.0001, 0.9999)
#             s = torch.sigmoid((u.log() - (1 - u).log() + self.mask_scores) / self.b)
#         else:
#             s = torch.sigmoid(self.mask_scores)
#         s_bar = s * (self.r - self.l) + self.l
#         mask = s_bar.clamp(min=0.0, max=1.0)
#         mask = torch.round(mask)
#         return mask
#
#     def regularizer(self):
#         return torch.sum(torch.sigmoid(self.mask_scores - self.b * np.log(-self.l / self.r))) / self.mask_scores.numel()
#
#
# class BinaryMaskedLinear(nn.Linear):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True,
#                  mask_p: float = 0.9, out_w_per_mask=1, in_w_per_mask=1, num_heads=12):
#         super().__init__(in_features=in_features, out_features=out_features, bias=bias)
#         self.num_heads = num_heads
#         self.out_w_per_mask = out_w_per_mask
#         self.in_w_per_mask = in_w_per_mask
#
#         assert out_features % out_w_per_mask == 0, "{} % {} not 0".format(out_features, out_w_per_mask)
#         assert in_features % in_w_per_mask == 0, "{} % {} not 0".format(in_features, in_w_per_mask)
#         mask_dim = (1, out_features // out_w_per_mask, 1, in_features // in_w_per_mask)
#         self.mask = BinaryL0Mask(mask_dim, mask_p)
#
#         self.cached_activation = None
#         self.do_caching = False
#
#     def produce_mask_reshaped(self):
#         mask = self.mask.produce_mask()
#         mask = mask.repeat(self.out_w_per_mask, 1, self.in_w_per_mask, 1)
#         return mask.reshape(self.out_features, self.in_features)
#
#     def produce_mask(self):
#         mask = self.mask.produce_mask()
#         return mask
#
#     def forward(self, input: torch.tensor):
#         # "masked_weight = self.produce_mask_reshaped() * self.weight" is equivalent but slower.
#         masked_weight = self.produce_mask() * self.weight.reshape(
#             self.out_w_per_mask, self.out_features // self.out_w_per_mask,
#             self.in_w_per_mask, self.in_features // self.in_w_per_mask)
#         masked_weight = masked_weight.reshape(self.out_features, self.in_features)
#
#         act = F.linear(input, masked_weight, self.bias)
#         if self.do_caching:
#             if self.cached_activation is None:
#                 self.cached_activation = act.detach()
#             else:  # only works if subbatched, since maxlen must be constant
#                 self.cached_activation = torch.cat((
#                     self.cached_activation, act.detach()), dim=0)
#         return act
#
#     def activate_caching(self, caching=True):
#         self.cached_activation = None
#         self.do_caching = caching
#
#     @classmethod
#     def from_layer(cls, layer, out_w_per_mask, in_w_per_mask, mask_p):
#         assert type(layer) == nn.modules.linear.Linear
#         res = cls(mask_p=mask_p, in_features=layer.in_features, out_features=layer.out_features,
#                   bias=layer.bias is not None, out_w_per_mask=out_w_per_mask, in_w_per_mask=in_w_per_mask)
#         res.weight = layer.weight
#         res.bias = layer.bias
#         return res  # make sure to call cuda