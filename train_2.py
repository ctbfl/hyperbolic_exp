import torch
import torch.nn.functional as F
from poincare import PoincareBall, RiemannianSGD, RSGD, RGrad
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from hyptorch.nn import HyperbolicMLR

c = 1
bound_eps = 1e-5
poincare_ball = PoincareBall(c=c, bound_eps=bound_eps)

class HypMLR(nn.Module):
    def __init__(self, 
                 poincare_ball: PoincareBall,
                 class_num=10,
                 hyp_dim=32
                 ):
        super(HypMLR, self).__init__() 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14x32
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # 14x14x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # =>7x7x64
        )

        self.fc = nn.Linear(7 * 7 * 64, hyp_dim) # => hyp_dim

        self.output = nn.Linear(hyp_dim, class_num)

        self.poincare_ball = PoincareBall(c=c, bound_eps=bound_eps)
        self.hyp_mlr = HyperbolicMLR(hyp_dim, class_num, c)
        self.hyperplane_normals = nn.Parameter(torch.randn(class_num, hyp_dim)) # (classnum,hyp_dim) the embedding of all classes.
        self.hyperplane_offsets = nn.Parameter(torch.zeros(class_num, hyp_dim)) 
        nn.init.normal_(self.hyperplane_normals.data, mean=0., std=0.05)
        nn.init.constant_(self.hyperplane_offsets.data, 0.)

    def check_numerics(self, input_tensor, name="NoName"):
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            raise ValueError(f"Numeric issue (NaN or Inf) detected in tensor: {name}")

    def clamp_row_norms(self):
        norms = self.hyperplane_offsets.norm(p=2, dim=1, keepdim=True)
        scale = torch.where(norms > self.poincare_ball.max_norm, self.poincare_ball.max_norm / norms, torch.ones_like(norms))
        self.hyperplane_offsets.data *= scale

        norms = self.hyperplane_normals.norm(p=2, dim=1, keepdim=True)
        scale = torch.where(norms > self.poincare_ball.max_norm, self.poincare_ball.max_norm / norms, torch.ones_like(norms))
        self.hyperplane_normals.data *= scale

    def _hyp_multi_layer_regression_origin(self, x):
        """hyper bolic multiclass logistic regression
        Args:
            x (torch.Tensor): shape=(N, hyp_dim)

        Returns:
            _type_: _description_
        """
        if self.training:
            self.clamp_row_norms() # avoid norm of hyperbolic params to be greater than 1
        c = self.poincare_ball.c
        sqrt_c = c ** 0.5
        eps = self.poincare_ball.eps
        max_norm = self.poincare_ball.max_norm
        w = self.hyperplane_normals # hyperbolic, (classnum=k+1, hyp_dim)
        b = self.hyperplane_offsets # hyperbolic, (classnum=k+1, hyp_dim)
        
        x2 = x.pow(2).sum(dim=-1, keepdim=True) # (N, 1)
        b2 = b.pow(2).sum(dim=-1, keepdim=False) # (K+1, )
        w_norm = w.norm(dim=-1, keepdim=False) # (K+1,)

        # N x (K + 1)
        inner_x_b = x @ (-1. * b.T) # (N, K+1)
        a_numerator = 1. + 2. * c * inner_x_b + c * x2 # (N, K+1)
        b_numerator = 1. - c * b2  # (K+1, )
        denominator = 1. + 2. * c * inner_x_b + c ** 2 * (x2 @ b2[None])  # (N, K+1) + (N,1)@(1,K+1) => (N, K+1)
        alpha = a_numerator / denominator.clamp_min(eps) # (N, K+1)
        beta = b_numerator[None] / denominator # (N, K+1)

        mob_add_2 = (
            alpha ** 2 * b2[None]
            + 2. * alpha * beta * inner_x_b
            + beta ** 2 * x2
        )  # (N, K+1)

        hyperplane_normalizer = torch.where(
            torch.sqrt(mob_add_2) > max_norm,
            max_norm / torch.sqrt(mob_add_2).clamp_min(eps),
            torch.ones_like(mob_add_2)
        )

        mob_add_2 = torch.where(
            torch.sqrt(mob_add_2) < max_norm,
            mob_add_2,
            torch.ones_like(mob_add_2) * max_norm ** 2
        )

        w = F.normalize(w, p=2, dim=-1) # normalize it, let the L2 norm of final dimension to be 1.
        hyperplane = alpha * ((w * -1. * b).sum(dim=-1, keepdim=False)[None]) + beta * (x @ w.T) # (N, K+1)
        hyperplane *= hyperplane_normalizer

        asinh_in = 2. * sqrt_c * hyperplane/ (torch.clamp(1. - c * mob_add_2, min=eps)) # (N, k+1)

        scaler = 2/(1 - self.poincare_ball.c * b.pow(2).sum(dim=-1, keepdim=False)).clamp_min(eps) # (K+1, )
        # scaler = 2
        return (1 / sqrt_c) * (scaler * w_norm)[None] * torch.asinh(asinh_in) # (N, K+1) output logits


    def _hyp_multi_layer_regression(self, x):
        """hyper bolic multiclass logistic regression
        Args:
            x (torch.Tensor): shape=(N, hyp_dim)

        Returns:
            _type_: _description_
        """
        if self.training:
            self.clamp_row_norms() # avoid norm of hyperbolic params to be greater than 1
        c = self.poincare_ball.c
        sqrt_c = c ** 0.5
        eps = self.poincare_ball.eps
        max_norm = self.poincare_ball.max_norm
        w = self.hyperplane_normals # hyperbolic, (classnum=k+1, hyp_dim)
        b = self.hyperplane_offsets # hyperbolic, (classnum=k+1, hyp_dim)
        
        x2 = x.pow(2).sum(dim=-1, keepdim=True) # (N, 1)
        b2 = b.pow(2).sum(dim=-1, keepdim=False) # (K+1, )
        w_norm = w.norm(dim=-1, keepdim=False) # (K+1,)

        # N x (K + 1)
        inner_x_b = x @ (-1. * b.T) # (N, K+1)
        a_numerator = 1. + 2. * c * inner_x_b + c * x2 # (N, K+1)
        b_numerator = 1. - c * b2  # (K+1, )
        denominator = 1. + 2. * c * inner_x_b + c ** 2 * (x2 @ b2[None])  # (N, K+1) + (N,1)@(1,K+1) => (N, K+1)
        alpha = a_numerator / denominator.clamp_min(eps) # (N, K+1)
        beta = b_numerator[None] / denominator # (N, K+1)

        mob_add_2 = (
            alpha ** 2 * b2[None]
            + 2. * alpha * beta * inner_x_b
            + beta ** 2 * x2
        )  # (N, K+1)

        hyperplane_normalizer = torch.where(
            torch.sqrt(mob_add_2) > max_norm,
            max_norm / torch.sqrt(mob_add_2).clamp_min(eps),
            torch.ones_like(mob_add_2)
        )

        mob_add_2 = torch.where(
            torch.sqrt(mob_add_2) < max_norm,
            mob_add_2,
            torch.ones_like(mob_add_2) * max_norm ** 2
        )

        w = F.normalize(w, p=2, dim=-1) # normalize it, let the L2 norm of final dimension to be 1.
        hyperplane = alpha * ((w * -1. * b).sum(dim=-1, keepdim=False)[None]) + beta * (x @ w.T) # (N, K+1)
        hyperplane *= hyperplane_normalizer

        asinh_in = 2. * sqrt_c * hyperplane/ (torch.clamp(1. - c * mob_add_2, min=eps)) # (N, k+1)

        scaler = 2/(1 - self.poincare_ball.c * b.pow(2).sum(dim=-1, keepdim=False)).clamp_min(eps) # (K+1, )
        # scaler = 2
        return (1 / sqrt_c) * (scaler * w_norm)[None] * torch.asinh(asinh_in) # (N, K+1) output logits
    
    def _hyp_multi_layer_regression_copy(self, x):
        """ 
        x: (batch_size, hyp_dim) = (B, D)

        refer to https://github.com/MinaGhadimiAtigh/HyperbolicImageSegmentation/blob/main/hesp/util/layers.py
        """
        c = self.poincare_ball.c
        sqrt_c = c ** 0.5
        eps = self.poincare_ball.eps
        max_norm = self.poincare_ball.max_norm
        A_mlr = self.hyperplane_normals # euclidean, (classnum, hyp_dim) = (C, D)
        P_mlr = self.hyperplane_offsets # hyperbolic, (C, D)        

        xx = x.pow(2).sum(dim=-1, keepdim=True) # (B, 1), |x|^2
        pp = (-P_mlr).pow(2).sum(dim=-1, keepdim=False) # (C,), |-p|^2
        self.check_numerics(pp, 'pp nan')
        px = x@(-P_mlr.T) # (B, D) @ (D, C) => (B, C), -p*x

        # c^2 * | X|^2 * |-P|^2
        sqsq = (c*c)*xx@pp[None] # (B, 1) @ (1, C) => (B, C)

        A_norm = A_mlr.norm(p=2, dim=1) # (C)
        normed_A = F.normalize(A_mlr, p=2, dim=1) # (C, D)
        # TODO: A_kernel = (normed_A.T)[None, None, :, :]

        # rewrite mob add as alpha * p + beta * x
        # alpha = A/D, beta = B/D
        A = 1 + 2*c*px + c*xx # (B, C) + (B, 1) => (B, C)
        self.check_numerics(A, 'A nan')
        B = 1 - c * pp  # (C, )
        self.check_numerics(B, 'B nan')
        D = 1 + 2*c*px + sqsq # (B, C)
        D = torch.clamp(D, min=eps)
        self.check_numerics(D, 'D nan')

        alpha = A / D # (B, C)
        self.check_numerics(alpha, 'alpha nan')

        beta = B[None] / D # (B, 1)/(B, C) => (B, C)
        self.check_numerics(beta, 'beta nan')

        mobaddnorm = (
                (alpha ** 2 * pp[None])     # (B, C)
                + (beta ** 2 * xx)          # (B, C) * (B, 1)-> (B, C)
                + (2 * alpha * beta * px)   # (B, C)
        )

        max_norm = self.poincare_ball.max_norm

        project_normalized = torch.where( # (B, C)
            torch.sqrt(mobaddnorm) > max_norm,
            max_norm / torch.clamp(torch.sqrt(mobaddnorm), min=eps),
            torch.ones_like(mobaddnorm)
        )
        self.check_numerics(mobaddnorm, 'mobaddnorm nan')

        mobaddnormprojected = torch.where( # (B, C)
            torch.sqrt(mobaddnorm) < max_norm,
            mobaddnorm,
            torch.ones_like(mobaddnorm)*max_norm**2
        )
        self.check_numerics(mobaddnormprojected, 'mobaddnormprojected nan')

        xdota = beta * (x@(normed_A.T)) # (B, C) * [(B, D) @ (D, C)] = (B, C)

        pdota_wo_alpha = torch.sum(-P_mlr * normed_A, dim=1, keepdim=False) # (C, D) @ (C, D) => (C,)
        pdota = alpha * pdota_wo_alpha[None]  # (B, C) * (1, C) => (B, C)

        mobdota = xdota + pdota # (B, C)
        mobdota *= project_normalized # (B, C)

        lamb_px = 2.0 / torch.clamp(1 - c * mobaddnormprojected, min=eps) # (B, C)
        self.check_numerics(lamb_px, 'lamb_px nan')

        sineterm = sqrt_c * mobdota * lamb_px # (B, C)
        scaler = 2/(1 - self.poincare_ball.c * P_mlr.pow(2).sum(dim=-1, keepdim=False)).clamp_min(eps) # (C, )
        self.check_numerics(scaler, 'scaler nan')

        return scaler / sqrt_c * A_norm * torch.asinh(sineterm) # (C,) / (C,) * (B, C) -> (B, C)

    def forward(self, x):
        """ 
        x: (N, 28, 28)
        """
        x = self.conv_layers(x) # (N, 7, 7, 64)
        x = x.view(x.size(0), -1) # (N, 3136)
        x = self.fc(x) # (N, hyp_dim)
        # x_hyperbolic = RGrad.apply(self.poincare_ball._expmap0(x))
        # x = self._hyp_multi_layer_regression(x_hyperbolic) # (N, class_num)
        x = self.output(x)
        return x

batch_size = 64
learning_rate = 0.001
num_epochs = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


model = HypMLR(poincare_ball)
criterion = nn.CrossEntropyLoss()

# # adam
# adam_params = []
# for name, param in model.named_parameters():
#     if 'hyperplane_normals' not in name and 'hyperplane_offsets' not in name:
#         adam_params.append(param)
# adam_optimizer = optim.Adam(adam_params, lr=learning_rate)
# # RSGD
# rsgd_params = [model.hyperplane_normals, model.hyperplane_offsets]
# # rsgd_optimizer = RiemannianSGD(rsgd_params,0.001,c,bound_eps) 
# rsgd_optimizer = RSGD(rsgd_params,learning_rate,poincare_ball,1) 

adam_optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

def train(model, device, train_loader, optimizers, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for opti in optimizers:
            opti.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        for opti in optimizers:
            opti.step()

        if batch_idx % 200 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, [adam_optimizer], criterion, epoch)
    evaluate(model, device, test_loader, criterion)