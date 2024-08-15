import torch
from poincare import PoincareBall, RSGD, RiemannianSGD, RGrad

c = 1
eps = 1e-5
poincare_ball = PoincareBall(c, eps)

x = torch.rand(2, requires_grad=True) 
x = torch.nn.Parameter(x) # param

optimizer = torch.optim.SGD([x], lr=0.01)
# optimizer = RSGD([x], 0.1, poincare_ball, 1)
# optimizer = RiemannianSGD([x], 0.1, c, eps)


print(f"x origin: {x.detach().numpy()}")
p = torch.rand(2)
print(f"p: {p.detach().numpy()}")


for step in range(10): 
    optimizer.zero_grad()
    x_mapped = RGrad.apply(poincare_ball._expmap0(x)) # model
    loss = torch.sum(x_mapped*p, dim=-1)
    loss.backward()
    optimizer.step()
    
    print(f"Step {step+1}:")
    print(f"Updated x_mapped: {x_mapped.detach().numpy()}")
    print(f"Updated similarity: {loss.item()}")
