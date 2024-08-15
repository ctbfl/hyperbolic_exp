import torch
import torch.nn as nn
from typing import Dict, List
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required


class RGrad(torch.autograd.Function):
    """
    For euclidean params who map to hyperbolic and be used to calculate the loss.

    Apply this layer to the params, then in forward period it doesn't do anything, but in backward it will divide the scaler.

    NOTE: when call this function, the params should be on hyperbolic.
    """
    @staticmethod
    def forward(ctx, x, c: float = 1.):
        ctx.save_for_backward(x)
        ctx.c = c
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (x, ) = ctx.saved_tensors
        c = ctx.c
        scale = (1. - c * x.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad_output * scale, None

class RSGD(Optimizer):
    def __init__(self, params, lr=1e-3, poincare_ball=None, burnin=1.0):
        if poincare_ball is None:
            raise ValueError("PoincareBall instance is required")
        defaults = dict(lr=lr, burnin=burnin)
        super(RSGD, self).__init__(params, defaults)
        self.poincare_ball = poincare_ball  

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            burnin = group['burnin']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # print("grad", grad)
                scaling_factor = self.poincare_ball._rgrad_scaling_factor(p.data)
                # print("scaling_factor", scaling_factor)
                rescaled_grad = scaling_factor * grad

                update = -(lr * burnin * rescaled_grad)

                p.data = self.poincare_ball._expmap(p.data, update)

        return loss

class RiemannianSGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = required,
        c: float = 1.,
        bound_eps: float = 1e-5,
        momentum: float = 0.,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False
    ) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super(RiemannianSGD, self).__init__(params=params, defaults=defaults)

        self.poincare_ball = PoincareBall(c, bound_eps)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nestorov = group["nesterov"]
            rgrad = self.poincare_ball._rgrad
            expmap = self.poincare_ball._expmap
            transp = self.poincare_ball._transp

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                if len(state) == 0:
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()

                grad.add_(param.data, alpha=weight_decay)
                grad = rgrad(param.data, grad)
                if momentum > 0:
                    momentum_buffer = rgrad(
                        param.data, state["momentum_buffer"])
                    momentum_buffer.mul_(momentum).add_(
                        grad, alpha=1 - dampening)
                    if nestorov:
                        grad.add_(momentum_buffer, alpha=momentum)
                    else:
                        grad = momentum_buffer
                    grad = -lr * grad
                    new_param = expmap(param.data, grad)
                    momentum_buffer = transp(
                        param.data, new_param, momentum_buffer)
                    param.data.copy_(new_param)
                else:
                    grad = -lr * grad
                    new_param = expmap(param.data, grad)
                    param.data.copy_(new_param)

class PoincareBall:
    def __init__(self,
                 c: float = 1.,
                 bound_eps: float = 1e-5):
        self.c = c
        self.max_norm = (1. - bound_eps) / (c ** 0.5)
        self.eps = bound_eps

    def _rgrad_scaling_factor(self, x):
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        scale = (1. - self.c * x2).pow(2) / 4.
        return scale
    
    def _rgrad(self, x, grad):
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        scale = (1. - self.c * x2).pow(2) / 4.
        return scale * grad

    def _dist(self, x, y):
        """
        calculate the distance on hyperbolic space. With broadcast mechanism, it can handle one-many distances in one call.
        """
        sqrt_c = self.c ** 0.5
        mobius_result = self._mobius_add(-x, y)
        mobius_add_norm = mobius_result.norm(dim=-1, p=2)
        # print("mobius_result:", mobius_result)
        # print("mobius_add_norm:", mobius_add_norm)
        mobius_add_norm = mobius_add_norm.clamp(min=-1.+1e-15, max=1.-1e-15)
        return 2. / sqrt_c * torch.atanh(sqrt_c * mobius_add_norm)

    def _expmap0(self, x):
        """
        map a speed vector on tangent space to a point on manifold 

        specifically, this function handle the condition where the start point is the "origin".
        """
        sqrt_c = self.c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-12)
        return self._project(torch.tanh(sqrt_c * x_norm) * x / (sqrt_c * x_norm))

    def _logmap0(self, x):
        """
        map a point on the poincare ball back to euclidean
        """
        sqrt_c = self.c ** 0.5
        epsilon = 1e-6
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-12)
        scaled_x_norm = sqrt_c * x_norm
        # Clamp the value to be within the valid input range of atanh
        atanh_input = scaled_x_norm.clamp(max=1 - epsilon)
        atanh_result = torch.atanh(atanh_input)
        return atanh_result * x / scaled_x_norm

    def _expmap(self, x, v):
        """
        map a speed vector on tangent space to a point on manifold 

        general case
        """
        sqrt_c = self.c ** 0.5
        v_norm = v.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-12)
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        lambda_x = 2. / (1. - self.c * x2).clamp_min(1e-12)
        y = torch.tanh(sqrt_c * lambda_x * v_norm / 2.) * v / (sqrt_c * v_norm)
        return self._project(self._mobius_add(x, y))

    def _mobius_add(self, x, y):
        """
        mobius addition, act as a component to simplify the calculation on the poincare ball.
        """
        c = self.c
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        numerator = (1. + 2. * c * xy + c * y2) * x + (1. - c * x2) * y
        # print("numerator", numerator)
        denominator = 1. + 2. * c * xy + c ** 2 * x2 * y2
        # print("demoninator", denominator)
        return numerator / denominator.clamp_min(1e-12)

    def _project(self, x):
        """
        if the L2 norm of the point is greater than maxinum, then change its norm to max_norm, keep the angle. 
        
        (Actually this may be inaccurate, as it can't distinguish two points with the same angle but different distances)
        """
        norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-12)
        return torch.where(norm > self.max_norm,   
                           x / norm * self.max_norm,
                           x)

    def _transp(self, x, y, v):
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y = y.pow(2).sum(dim=-1, keepdim=True)
        lambda_x = 2 / (1 - self.c * x2).clamp_min(1e-12)
        lambda_y = 2 / (1 - self.c * y).clamp_min(1e-12)
        return self._gyration(y, -x, v) * lambda_x / lambda_y

    def _gyration(self, u, v, w):
        c = self.c
        c2 = self.c ** 2
        u2 = u.pow(2).sum(dim=-1, keepdim=True)
        v2 = v.pow(2).sum(dim=-1, keepdim=True)
        uv = (u * v).sum(dim=-1, keepdim=True)
        uw = (u * w).sum(dim=-1, keepdim=True)
        vw = (v * w).sum(dim=-1, keepdim=True)
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1. + 2. * c * uv + c2 * u2 * v2
        return w + 2. * (a * u + b * v) / d.clamp_min(1e-12)

    def _inner_product(self, x: torch.Tensor, u:torch.Tensor, v:torch.Tensor):
        """ calculate hyperbolic innerproduct on poincare model.\n 
        because inner product depends on the shared start point of both vector, so we need x.

        Args:
            x (torch.Tensor): hyperbolic embedding ofthe shared start point of both vector.
            u (torch.Tensor): gyro-vector, start from point x, belongs to x's tangent space.
            v (torch.Tensor): gyro-vector, start from point x, belongs to x's tangent space.
        """
        # assert x.dim() == 1, "x should be a 1-dimensional tensor"
        # assert u.size(-1) == x.size(0) and v.size(-1) == x.size(0), "The last dimension of u and v must match the size of x"
        # assert u.size(-1) == v.size(-1), "The last dimensions of u and v must be equal"
        euclidean_inner_product = torch.matmul(u, v.unsqueeze(-1)).squeeze(-1) # use broadcast mechanism to accommodate flexible input formats
        c_x_norm = self.c*torch.norm(x)
        scaler = (2/(1-c_x_norm**2))**2
        return scaler*euclidean_inner_product