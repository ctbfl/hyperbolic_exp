import torch
import torch.nn as nn
from typing import Dict, List
import torch.nn.functional as F

class Lorentz_RGrad(torch.autograd.Function):
    """
    TODO: This is incomplete for lorentz RGrad, it needs a costomized framework.
    For euclidean params who map to hyperbolic and be used to calculate the loss.

    Apply this layer to the params, then in forward period it doesn't do anything, but in backward it will divide the scaler.

    NOTE: when call this function, the params should be on hyperbolic.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (x, ) = ctx.saved_tensors
        x[0] = -1 * x[0]  # left multiple a diagonal matrix whose fisrt element is -1 else is 1(on diagona).
        return x 
    



class Lorentz:
    """
        Lorentz Calculation Class
    """
    @staticmethod
    def dist(x, y):
        """ 
        (N, D+1)&(N,D+1) -> (N,D+1)
        """
        return torch.acosh(-1*Lorentz.inner_product(x,y))
    
    @staticmethod
    def expmap(x, v):
        """
        x:(D+1), v:(N, D+1) -> (N, D+1);
        if v:(D+1), use v[none] to get (1, D+1)
        """
        assert x.dim()==1
        assert v.dim()==2
        ltz_norm = Lorentz.lorentz_norm(v) # (N, D+1) -> (N, )
        first_term = torch.acosh(ltz_norm).unsqueeze(1)*x # (N,1) x (D+1,) -> (N, D+1)
        second_term = (torch.sinh(ltz_norm)/ltz_norm).unsqueeze(1)*v # (N, 1) x (N, D+1) ->(N, D+1)
        return first_term + second_term

    @staticmethod
    def expmap0(v):
        """
        v:(N, D+1) -> (N, D+1);
        """
        assert v.dim()==2
        ltz_norm = Lorentz.lorentz_norm(v) # (N, D+1) -> (N, )
        print("ltz_norm",ltz_norm)
        return (torch.sinh(ltz_norm)/ltz_norm).unsqueeze(1)*v
        
    
    @staticmethod
    def project(x, u):
        """

        Args:
            x (torch.Tensor): len = D+1, the base point on hyperbolic space
            u (torch.Tensor): len = (N, D+1), some vector on tengent space
        """
        return u + Lorentz.inner_product(x,u)*x
    
    @staticmethod
    def lorentz_norm(x):
        """ 
        NOTE: Could cause NaN if the inner product is negative.
        x: (N, D+1) -> (N, ); (D+1) -> single number
        """
        norm = torch.sqrt(Lorentz.inner_product(x,x))
        print("Norm", norm)
        return norm

    @staticmethod
    def inner_product(x, y):
        #  (N, D+1) & (D+1,) -> (N)
        #  (D+1,) & (N, D+1) -> (N)
        # （N, D+1）& (N, D+1) -> (N, )
        #  (D+1, ) & (D+1, ) -> single number
        #  -x0y0 + x1y1 + x2y2+ ... +xd*yd
        full_inner_product = torch.sum(x * y, dim=-1) # (N,) or single number | x0y0 + x1y1 + x2y2+ ... +xd*yd
        print("full_inner_product", full_inner_product)
        adjustment = -2 * x[..., 0] * y[..., 0] # (N,)  | -2*x0y0
        print("adjustment", adjustment)
        if x.dim()==1 and y.dim()==1:
            adjustment = adjustment.squeeze(-1) # (1, ) -> single number
        print("full_inner_product + adjustment", full_inner_product + adjustment)
        return full_inner_product + adjustment
    
    @staticmethod
    def cal_x0(x):
        """given len D vector, calculate its x0

        Args:
            x (torch.Tensor): len = D, not D+1
        """
        return torch.sqrt(1+x.norm(p=2))