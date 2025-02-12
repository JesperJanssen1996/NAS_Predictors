import copy
import torch
import numpy as np
import torch.nn as nn
from get_correlation import get_correlation


class Net(nn.Module):
    # N-layer MLP
    def __init__(self, n_feature, n_layers, n_hidden, drop):
        n_output = 1
        super(Net, self).__init__()

        self.stem = nn.Sequential(nn.Linear(n_feature, n_hidden), nn.ReLU())

        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.append(nn.Linear(n_hidden, n_hidden))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)

        self.regressor = nn.Linear(n_hidden, n_output)  # output layer
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.stem(x)
        x = self.hidden(x)
        x = self.drop(x)
        x = self.regressor(x)  # linear output
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)


class MLP:
    """ Multi Layer Perceptron """
    def __init__(self, **kwargs):
        #net_params = {key: self.params[key] for key in ['n_layers', 'n_hidden', 'drop'] if key in self.params}
        self.model = Net(**kwargs)
        self.name = 'mlp'

    def fit(self, x, y, x_val=None, y_val=None, **kwargs):
        self.model = train_network(self.model, x, y,x_val,y_val, **kwargs)

    def predict(self, test_data, device='cpu'):
        return predict(self.model, test_data, device=device)


def train_network(net, x_train, y_train, x_val=None, y_val=None, pretrained=None, device='cpu', epochs=2000, verbose=False, **kwargs):
    lr = kwargs.get('lr', 1e-3)
    
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    best_net = copy.deepcopy(net)

    # Convert training data to tensors
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1).to(device)

    if x_val is not None and y_val is not None:
        x_val_tensor = torch.from_numpy(x_val).float().to(device)
        y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(1).to(device)
        validation = True
    else:
        validation = False

    best_loss = float('inf')

    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad()
        outputs = net(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if validation:
            net.eval()
            with torch.no_grad():
                val_outputs = net(x_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            current_loss = val_loss.item()
        else:
            current_loss = loss.item()

        if current_loss < best_loss:
            best_loss = current_loss
            best_net = copy.deepcopy(net)

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss}")

    return best_net.to('cpu')


def train_one_epoch(net, data, target, criterion, optimizer, device):
    net.train()
    optimizer.zero_grad()

    data, target = data.to(device), target.to(device)
    pred = net(data)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

    return loss.item()


def infer(net, data, target, criterion, device):
    net.eval()

    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        pred = net(data)
        loss = criterion(pred, target)

    return loss.item()



def predict(net, query, device):

    if query.ndim < 2:
        data = torch.zeros(1, query.shape[0])
        data[0, :] = torch.from_numpy(query).float()
    else:
        data = torch.from_numpy(query).float()

    net = net.to(device)
    net.eval()
    with torch.no_grad():
        data = data.to(device)
        pred = net(data)

    return pred.cpu().detach().numpy()