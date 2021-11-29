import torch.nn.functional as F
import torch
import sys
sys.path.append("../../simplex/")
import utils
from utils import *


class PoisonedCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def poisoned_celoss(self, output, target_var):
        """
        logits = torch.log(1 - self.softmax(output) + 1e-12)

        return self.ce(logits, target_var.to(torch.int64))
        """
        logits = torch.log(1 - self.softmax(output) + 1e-12)
        one_hot_y = F.one_hot(target_var.unsqueeze(0).to(torch.int64),
                              num_classes=output.shape[-1])
        return - torch.mean(torch.sum(logits * one_hot_y, axis=-1))

    def clean_celoss(self, output, target_var):
        """
        return self.ce(output, target_var.to(torch.int64))
        """
        logits = torch.log(self.softmax(output) + 1e-12)
        one_hot_y = F.one_hot(target_var.unsqueeze(0).to(torch.int64),
                              num_classes=output.shape[-1])

        return - torch.mean(torch.sum(logits * one_hot_y, axis=-1))

    def forward(self, output, target_var, poison_flag):
        clean_loss = self.clean_celoss(output[poison_flag == 0],
                                       target_var[poison_flag == 0])

        poison_loss = self.poisoned_celoss(output[poison_flag == 1],
                                           target_var[poison_flag == 1])
        return clean_loss, poison_loss


def get_criterion_trainer_columns(poison_factor):
    if poison_factor != 0:
        criterion = PoisonedCriterion()
        trainer = utils.poison_train_epoch_volume
        columns = [
            'ep', 'lr', 'cl_tr_loss', 'cl_tr_acc', 'po_tr_loss',
            'po_tr_acc', 'te_loss', 'te_acc', 'time', 'vol'
        ]
    else:
        criterion = torch.nn.CrossEntropyLoss()
        trainer = utils.train_epoch_volume
        columns = [
            'ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', 'vol'
        ]
    return criterion, trainer, columns
