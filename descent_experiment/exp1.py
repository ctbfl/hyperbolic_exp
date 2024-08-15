import torch
from poincare import PoincareBall, RSGD, RiemannianSGD

c = 1
eps = 1e-5
poincare_ball = PoincareBall(c, eps)

x = torch.rand(2, requires_grad=True) 
x = torch.nn.Parameter(x)
p = torch.rand(2)

# x = torch.tensor([0.8308664, 0.947953], requires_grad=True).double()
# x = torch.nn.Parameter(x)

# p = torch.tensor([0.5115065, 0.22072566]).double()


print(f"x: {x.detach().numpy()}")
print(f"p: {p.detach().numpy()}")
print(f"current distance: {poincare_ball._dist(x,p).detach().numpy()}")

# optimizer = torch.optim.AdamW([x], lr=0.01)
# optimizer = RSGD([x], 0.1, poincare_ball, 1)
optimizer = RiemannianSGD([x], 0.1, c, eps)

for step in range(10): 
    optimizer.zero_grad()
    loss = poincare_ball._dist(x, p)  
    loss.backward()
    optimizer.step()
    
    print(f"Step {step+1}:")
    print(f"Updated x: {x.detach().numpy()}")
    print(f"Updated distance: {loss.item()}")

# print("Final x:", x.detach().numpy())
# print("Final distance:", poincare_ball._dist(x, p).item())