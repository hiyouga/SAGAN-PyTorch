import torch
import torch.nn as nn


class SpectralNorm(nn.Module):

    def __init__(self, module, name='weight'):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        if not hasattr(self.module, self.name + '_bar'):
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        height = w.size(0)
        with torch.no_grad():
            v = self._l2normalize(w.view(height, -1).transpose(0, 1).mv(u))
            u = self._l2normalize(w.view(height, -1).mv(v))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.size(0)
        width = w.view(height, -1).size(1)
        u = nn.Parameter(self._l2normalize(w.new_empty(height).normal_(0, 1)), requires_grad=False)
        v = nn.Parameter(self._l2normalize(w.new_empty(width).normal_(0, 1)), requires_grad=False)
        w_bar = nn.Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args, **kwargs):
        self._update_u_v()
        return self.module.forward(*args, **kwargs)

    @staticmethod
    def _l2normalize(v, eps=1e-12):
        return v / (v.norm() + eps)
