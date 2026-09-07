import torch
from copy import deepcopy


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the unwrapped model for accumulating moving average of weights
        raw_model = getattr(model, "module", model)
        self.module = deepcopy(raw_model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                if torch.is_floating_point(ema_v) or torch.is_complex(ema_v):
                    ema_v.copy_(update_fn(ema_v, model_v))
                else:
                    ema_v.copy_(model_v)

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
