import torch


class CustomMultiWrapper:
    def __init__(self):
        self.regression = torch.tensor([True, True, False])
        self.multi_loss = MultiTaskLoss(is_regression=self.regression)
        self.loss_suvr = torch.nn.MSELoss()
        self.loss_age = torch.nn.MSELoss()
        self.loss_apoe = torch.nn.CrossEntropyLoss()

    def __call__(self, net_out, target):
        suvr_net = net_out[:, 0]
        suvr_y = target[:, 0]
        # divide by 100 for similar size to suvr
        age_net = net_out[:, 1]/100
        age_y = target[:, 1]/100
        apoe_net = net_out[:, 2:]
        apoe_y = target[:, 2]
        curr_loss_apoe = self.loss_apoe(apoe_net, apoe_y.long())
        curr_loss_age = self.loss_age(age_net, age_y)
        curr_loss_suvr = self.loss_suvr(suvr_net, suvr_y)
        # suvr loss is generally quite a bit smaller..
        curr_loss_suvr = curr_loss_suvr*100
        curr_loss_age = curr_loss_age/25
        curr_loss_apoe = curr_loss_apoe/50
        comb_loss = torch.stack([curr_loss_suvr, curr_loss_age, curr_loss_apoe])
        # multi_task_loss = self.multi_loss(comb_loss)
        print(comb_loss)
        return (curr_loss_suvr + curr_loss_age + curr_loss_apoe)/3

    def to_train(self):
        self.multi_loss.train()

    def to_eval(self):
        self.multi_loss.eval()

class MultiTaskLoss(torch.nn.Module):
  '''https://arxiv.org/abs/1705.07115'''
  def __init__(self, is_regression, reduction='mean'):
    super(MultiTaskLoss, self).__init__()
    self.is_regression = is_regression
    self.n_tasks = len(is_regression)
    self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
    self.reduction = reduction

  def forward(self, losses):
    dtype = losses.dtype
    device = losses.device
    print(losses)
    stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
    self.is_regression = self.is_regression.to(device).to(dtype)
    coeffs = 1 / ( (self.is_regression+1)*(stds**2) )
    multi_task_losses = coeffs*losses + torch.log(stds)
    print(multi_task_losses)
    # if self.reduction == 'sum':
    #   multi_task_losses = multi_task_losses.sum()
    if self.reduction == 'mean':
      multi_task_losses = multi_task_losses.mean()
    return multi_task_losses

'''
usage
is_regression = torch.Tensor([True, True, False]) # True: Regression/MeanSquaredErrorLoss, False: Classification/CrossEntropyLoss
multitaskloss_instance = MultiTaskLoss(is_regression)
params = list(model.parameters()) + list(multitaskloss_instance.parameters())
torch.optim.Adam(params, lr=1e-3)
model.train()
multitaskloss.train()
losses = torch.stack(loss0, loss1, loss3)
multitaskloss = multitaskloss_instance(losses)
source = https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/multi_task_loss.py
'''