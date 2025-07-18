import torch
import torch.nn as nn


class KLLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(KLLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, true):
        pred = pred / pred.norm(dim=1, keepdim=True)
        true = true/ true.norm(dim=1, keepdim=True)
        pred = nn.functional.softmax(pred, dim=1)
        true = nn.functional.softmax(true, dim=1)
        pred = pred/0.1
        true = true/0.1
        kl = (pred * (pred + self.eps).log() - pred * (true + self.eps).log()).sum(dim=1)
        kl = kl.mean()
        return kl

class ContrastLoss(nn.Module):
# https://github.com/thuiar/MMSA/blob/master/src/MMSA/models/singleTask/MMIM.py#L135
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """

    def __init__(self, x_size, y_size, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.activation = getattr(nn, activation)

    def forward(self, x, y):
        """Calulate the score
        """
        # import ipdb;ipdb.set_trace()
        x_pred = y  # bs, emb_size

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        pos = torch.sum(x * x_pred, dim=-1)  # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)  # bs
        nce = -(pos - neg).mean()
        return nce


class Consine(nn.Module):
    def __init__(self, tau=0.005, eps=1e-5):
        super(Consine, self).__init__()
        self.tau = tau
        self.eps = eps
        self.f1=self.f2=self.mean_f1=self.length_b=self.mean_f2=self.length=None

    def forward(self, feat1, feat2):
        """"
        Parameters
        ----------
        feat1: input features from one stream
        feat2: input features from other stream
        """

        self.f1 = feat1
        self.f2 = feat2

        self.length_b = feat1.size(0)  # batch_size
        self.mean_f1 = torch.reshape(self.f1, (self.length_b, -1))
        self.mean_f2 = torch.reshape(self.f2, (self.length_b, -1))
        self.mean1_f1 = nn.functional.normalize(self.mean_f1, 2, dim=1)
        self.mean1_f2 = nn.functional.normalize(self.mean_f2, 2, dim=1)

        value_f1 = self.mean_f1.norm(dim=-1, keepdim=True)
        value_f2 = self.mean_f2.norm(dim=-1, keepdim=True)

        new = torch.mm(self.mean1_f1, self.mean1_f2.t()) / self.tau
        new = new.sum(dim=-1)
        loss = new/(value_f1*value_f2)

        return loss
