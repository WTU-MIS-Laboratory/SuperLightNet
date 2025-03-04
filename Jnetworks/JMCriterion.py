"""MCriterion Release
Author: Jiacheng Cao
Date: 2025-3-3
Description: This file contains a Medical Criterion Python code.
"""
from monai.metrics import *
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
import torch
from torch import distributed

GREEN = '\033[92m'
RED = '\033[91m'
END = '\033[0m'


# ================ MCriterion-Class ================
class MCriterionData:
    def __init__(self):
        self.dice_0124_EACH_sum = []
        self.dice_ETTCWT_EACH_sum = []
        self.dice_ETTCWT_MEAN_sum = []
        self.hd95_ETTCWT_EACH_sum = []
        self.sdc_ETTCWT_MEAN_sum = []
        self.dice_EACH_sum = []
        self.dice_MEAN_sum = []
        self.sdc_EACH_sum = []


class MCriterion:
    def __init__(
            self,
            include_background: bool = True,
            reduction: str = "",
            datasets_flag: str = "",
            data: MCriterionData = MCriterionData(),
    ) -> None:
        self.diceMetric = DiceMetric(include_background=include_background)
        self.percentile = 95
        self.hdMetric = HausdorffDistanceMetric(include_background=include_background, percentile=self.percentile)
        self.tolerance = [1]
        self.sdcMetric = SurfaceDiceMetric(include_background=include_background, class_thresholds=self.tolerance)
        self.reduction = reduction
        self.data = data

        self.datasets_flag = datasets_flag
        self.classname = "MCriterion"
        self.output = None
        self.tensorCopy = None

    def __call__(self, output, target):
        if torch.cuda.is_available():
            output = output.to('cuda')
            target = target.to('cuda')
        else:
            print("There is no GPU available, and the target will remain on the CPU")
        if self.datasets_flag == "JCMNet_BraTS_New":
            DSC, SDC = JCMNet_BraTS_New(output, target, self.data)
        else:
            raise TypeError("MCriterion：datasets_flag not existing")
        return DSC, SDC

    def BraTS_124toETTCWT_tensor(self, y_pred_hot, y_hot):
        o, t = y_pred_hot.clone(), y_hot.clone()
        et_o = o[:, 3, :, :, :].unsqueeze(1)
        et_t = t[:, 3, :, :, :].unsqueeze(1)
        tc_o = (o[0, 1, :, :, :] | o[0, 3, :, :, :]).unsqueeze(0).unsqueeze(0)
        tc_t = (t[0, 1, :, :, :] | t[0, 3, :, :, :]).unsqueeze(0).unsqueeze(0)
        wt_o = (o[0, 1, :, :, :] | o[0, 2, :, :, :] | o[0, 3, :, :, :]).unsqueeze(0).unsqueeze(0)
        wt_t = (t[0, 1, :, :, :] | t[0, 2, :, :, :] | t[0, 3, :, :, :]).unsqueeze(0).unsqueeze(0)
        self.tensorCopy = torch.cat((et_o, et_t, tc_o, tc_t, wt_o, wt_t), dim=0).unsqueeze(1)
        return self.tensorCopy

    def BraTS_DSC_SDC(self, tensorcopy: Tensor):
        et_dice = self.diceMetric(tensorcopy[0, ...], tensorcopy[1, ...])
        tc_dice = self.diceMetric(tensorcopy[2, ...], tensorcopy[3, ...])
        wt_dice = self.diceMetric(tensorcopy[4, ...], tensorcopy[5, ...])
        et_dice = torch.where(torch.isnan(et_dice), torch.tensor(0).to(et_dice.device), et_dice)
        tc_dice = torch.where(torch.isnan(tc_dice), torch.tensor(0).to(tc_dice.device), tc_dice)
        wt_dice = torch.where(torch.isnan(wt_dice), torch.tensor(0).to(wt_dice.device), wt_dice)
        DSC = (et_dice + tc_dice + wt_dice) / 3
        et_sdc = self.sdcMetric(tensorcopy[0, ...], tensorcopy[1, ...])
        tc_sdc = self.sdcMetric(tensorcopy[2, ...], tensorcopy[3, ...])
        wt_sdc = self.sdcMetric(tensorcopy[4, ...], tensorcopy[5, ...])
        et_sdc = torch.where(torch.isnan(et_sdc), torch.tensor(0).to(et_sdc.device), et_sdc)
        tc_sdc = torch.where(torch.isnan(tc_sdc), torch.tensor(0).to(tc_sdc.device), tc_sdc)
        wt_sdc = torch.where(torch.isnan(wt_sdc), torch.tensor(0).to(wt_sdc.device), wt_sdc)
        SDC = (et_sdc + tc_sdc + wt_sdc) / 3
        return DSC.cpu().detach().numpy(), SDC.cpu().detach().numpy()

    def calculate_BraTS_list1_averages(self, lists):
        sum1 = 0.0
        error_flag = 0
        for t in lists:
            if np.isnan(t) or np.isinf(t):
                error_flag += 1
                continue
            sum1 += t.item()
        avg = sum1 / len(lists)
        self.output = avg
        return self.output

    def argmax2one_hot(self, ori, classes):
        ori = ori.clone()
        ori[ori == 4] = 3
        new_tensor = torch.zeros(1, classes, *ori.shape[1:], device=ori.device)
        for i in range(classes):
            new_tensor[0][i] = (ori == i)
        self.output = new_tensor.bool()
        return self.output

    def feature_maps2one_hot(self, ori):
        ori = ori.clone()
        _, max_indices = torch.max(ori, dim=1, keepdim=True)
        result_onehot = torch.zeros_like(ori)
        result_onehot.scatter_(1, max_indices, 1)
        self.output = result_onehot.bool()
        return self.output


# ===================== Utils =====================
def all_reduce(tensor, op=torch.distributed.ReduceOp.AVG):
    dist.all_reduce(tensor, op)
    return tensor


def print_if_rank0(*args):
    if distributed.get_rank() == 0:
        print(*args)


# ====================== Loss ======================
class BCEWithDiceLoss(nn.Module):
    def __init__(self):
        super(BCEWithDiceLoss, self).__init__()

    def forward(self, outputs, targets):
        loss1, dice1 = self.BCEDice(outputs[:, 1, ...], (targets == 1).float())
        loss2, dice2 = self.BCEDice(outputs[:, 2, ...], (targets == 2).float())
        loss3, dice3 = self.BCEDice(outputs[:, 3, ...], (targets == 4).float())
        return loss1 + loss2 + loss3, dice1, dice2, dice3

    def BCEDice(self, inputs, targets, smooth=1e-5):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE, 1 - dice_loss


# ==================== Criterion ====================
def JCMNet_BraTS_New(output, target, data):
    H, W, T = 160, 160, 128
    mc_inter = MCriterion()

    output = output[:, :, :H, :W, :T]
    output_tohot = mc_inter.feature_maps2one_hot(output)
    target = target[:, :H, :W, :T]
    target_tohot = mc_inter.argmax2one_hot(target, 4)

    tensorCopy = mc_inter.BraTS_124toETTCWT_tensor(output_tohot, target_tohot)

    item_dsc, item_sdc = mc_inter.BraTS_DSC_SDC(tensorCopy)

    data.dice_ETTCWT_MEAN_sum += item_dsc,
    DSC = mc_inter.calculate_BraTS_list1_averages(data.dice_ETTCWT_MEAN_sum)
    print(GREEN + "As of now DICE_ETTCWT_MEAN_avg(DSC): " + str(DSC), end="\n")

    data.sdc_ETTCWT_MEAN_sum += item_sdc,
    SDC = mc_inter.calculate_BraTS_list1_averages(data.sdc_ETTCWT_MEAN_sum)
    print(GREEN + "As of now SDC_ETTCWT_MEAN_avg: " + str(SDC), end="\n")
    return DSC, SDC


"""MCriterion Release
Author: Jiacheng Cao
Date: 2025-3-3
Description: This file contains a Medical Criterion Python code.
"""
