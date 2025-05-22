import torch
import torch.nn as nn

'''
output: (1, num_params)
target: (num_params)
'''

class Loss(nn.Module):
    def __init__(self, loss_type='implicit_likelihood'):
        super().__init__()
        valid_losses = ['implicit_likelihood', 'log_implicit_likelihood', 'mse']
        if loss_type not in valid_losses:
            raise ValueError(f"Invalid loss_type. Choose from {valid_losses}")
        self.loss_type = loss_type

    def forward(self, output, target):
        if self.loss_type == 'implicit_likelihood':
            return self.implicit_likelihood_loss(output, target)
        elif self.loss_type == 'log_implicit_likelihood':
            return self.log_implicit_likelihood_loss(output, target)
        elif self.loss_type == 'mse':
            return self.mse_loss(output, target)

    def implicit_likelihood_loss(self, output, target):
        num_params = len(target)
        y_out, err_out = output[:, :num_params], output[:, num_params:]
        loss_mse = torch.mean(torch.sum((y_out - target) ** 2., dim=1), dim=0)
        loss_ili = torch.mean(torch.sum(((y_out - target) ** 2. - err_out ** 2.) ** 2., dim=1), dim=0)
        return loss_mse + loss_ili

    def log_implicit_likelihood_loss(self, output, target):
        num_params = len(target)
        y_out, err_out = output[:, :num_params], output[:, num_params:]
        loss_mse = torch.mean(torch.sum((y_out - target) ** 2., dim=1), dim=0)
        loss_ili = torch.mean(torch.sum(((y_out - target) ** 2. - err_out ** 2.) ** 2., dim=1), dim=0)
        return torch.log(loss_mse) + torch.log(loss_ili)

    def mse_loss(self, output, target):
        return torch.mean(torch.sum((output - target) ** 2., dim=1), dim=0)

def get_loss_fn(loss_type='implicit_likelihood'):
    return Loss(loss_type)
