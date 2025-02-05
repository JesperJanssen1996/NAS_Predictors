from get_correlation import get_correlation
import torch

def validate(net, data, target, device):
    net.eval()
    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target).float()
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        pred = net(data)
        pred, target = pred.cpu().detach().numpy(), target.cpu().detach().numpy()

        # RMSE, Rho, and Tau are computed here
        rmse, rho, tau = get_correlation(pred, target)

    return rmse, rho, tau, pred, target
