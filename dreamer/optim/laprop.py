# MIT License - Source: https://github.com/Z-T-WANG/LaProp-Optimizer
import torch
from torch.optim import Optimizer


class LaProp(Optimizer):
    def __init__(self, params, lr=4e-4, betas=(0.9, 0.999), eps=1e-15,
                 weight_decay=0, amsgrad=False, centered=False):
        self.steps_before_using_centered = 10
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, centered=centered)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")
                amsgrad = group["amsgrad"]
                centered = group["centered"]
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_lr_1"] = 0.0
                    state["exp_avg_lr_2"] = 0.0
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if centered:
                        state["exp_mean_avg_beta2"] = torch.zeros_like(p.data)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if centered:
                    exp_mean_avg_beta2 = state["exp_mean_avg_beta2"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                state["exp_avg_lr_1"] = state["exp_avg_lr_1"] * beta1 + (1 - beta1) * group["lr"]
                state["exp_avg_lr_2"] = state["exp_avg_lr_2"] * beta2 + (1 - beta2)
                bias_correction1 = (state["exp_avg_lr_1"] / group["lr"] if group["lr"] != 0.0 else 1.0)
                step_size = 1 / bias_correction1
                bias_correction2 = state["exp_avg_lr_2"]
                denom = exp_avg_sq
                if centered:
                    exp_mean_avg_beta2.mul_(beta2).add_(grad, alpha=1 - beta2)
                    if state["step"] > self.steps_before_using_centered:
                        mean = exp_mean_avg_beta2 ** 2
                        denom = denom - mean
                if amsgrad and not (centered and state["step"] <= self.steps_before_using_centered):
                    torch.max(max_exp_avg_sq, denom, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq
                denom = denom.div(bias_correction2).sqrt_().add_(group["eps"])
                step_of_this_grad = grad / denom
                exp_avg.mul_(beta1).add_(step_of_this_grad, alpha=(1 - beta1) * group["lr"])
                p.data.add_(exp_avg, alpha=-step_size)
                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"])
