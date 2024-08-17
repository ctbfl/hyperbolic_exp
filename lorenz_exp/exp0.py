import torch
from lorenz import Lorentz

x = torch.rand(2)

y = Lorentz.expmap0(x[None])

print(y)