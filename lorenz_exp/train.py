import torch
import torch.nn.functional as F
from lorenz import Lorentz
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ..lib.lorentz.layers.LMLR import LorentzMLR
from ..lib.lorentz.manifold import CustomLorentz

class HypMLR(nn.Module):
    def __init__(self,
                 class_num=10,
                 hyp_dim=32
                 ):
        super(HypMLR, self).__init__() 
        self.lorentz = CustomLorentz(1, False)
        self.LorentzMLR = LorentzMLR(self.lorentz, hyp_dim, class_num)
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

        self.hyperplane_normals = nn.Parameter(torch.randn(class_num, hyp_dim)) # euclidean, (classnum, hyp_dim) = (C, D)
        self.hyperplane_offsets = nn.Parameter(torch.zeros(class_num, hyp_dim)) # hyperbolic, (C, D)
        nn.init.normal_(self.hyperplane_normals.data, mean=0., std=0.05)
        nn.init.constant_(self.hyperplane_offsets.data, 0.)


    def forward(self, x):
        """ 
        x: (N, 28, 28)
        """
        x = self.conv_layers(x) # (N, 7, 7, 64)
        x = x.view(x.size(0), -1) # (N, 3136)
        x = self.fc(x) # (N, hyp_dim)
        x_hyperbolic = self.lorentz.expmap0(x)
        x = self.LorentzMLR(x_hyperbolic) # (N, class_num)
        return x

batch_size = 64
learning_rate = 0.001
num_epochs = 20

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


model = HypMLR()
criterion = nn.CrossEntropyLoss()

# # adam
# adam_params = []
# for name, param in model.named_parameters():
#     if 'hyperplane_offsets' not in name:
#         adam_params.append(param)
# adam_optimizer = optim.Adam(adam_params, lr=learning_rate)
# # RSGD
# rsgd_params = [model.hyperplane_offsets]
# # rsgd_optimizer = RiemannianSGD(rsgd_params,0.001,c,bound_eps) 
# rsgd_optimizer = RSGD(rsgd_params,learning_rate,poincare_ball,1) 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
    train(model, device, train_loader, [optimizer], criterion, epoch)
    evaluate(model, device, test_loader, criterion)