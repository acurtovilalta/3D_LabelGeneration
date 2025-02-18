import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init
import numpy as np


class Unsupervised_Segmentation_Model(nn.Module):
    def __init__(self, input_dim, nChannel, nConv):
        super(Unsupervised_Segmentation_Model, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm3d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append(nn.Conv3d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append(nn.BatchNorm3d(nChannel))
        self.conv3 = nn.Conv3d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm3d(nChannel)
        self.nConv = nConv

    def forward(self, x1, x2):
        # Modality 1
        x1 = self.conv1(x1)
        x1 = F.relu(x1)
        x1 = self.bn1(x1)
        for i in range(self.nConv-1):
            x1 = self.conv2[i](x1)
            x1 = F.relu(x1)
            x1 = self.bn2[i](x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)

        # Modality 2
        x2 = self.conv1(x2)
        x2 = F.relu(x2)
        x2 = self.bn1(x2)
        for i in range(self.nConv-1):
            x2 = self.conv2[i](x2)
            x2 = F.relu(x2)
            x2 = self.bn2[i](x2)
        x2 = self.conv3(x2)
        x2 = self.bn3(x2)

        # Concatenate feature dimension:
        x = torch.cat((x1, x2), 1)
        torch.cuda.empty_cache()

        return x


def myloss3D_opt(pred, img, nChannel, device):
    stepsize_sim = 1
    stepsize_con = 1
    loss_hpy = torch.nn.L1Loss(reduction="mean")
    loss_hpz = torch.nn.L1Loss(reduction="mean")
    loss_hpx = torch.nn.L1Loss(reduction="mean")
    loss_fn = torch.nn.CrossEntropyLoss()

    output = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, nChannel).to(device)
    outputHP = output.reshape((img.shape[0], img.shape[2], img.shape[3], img.shape[4], nChannel)).to(device)

    HPy = outputHP[:, 1:, :, :, :] - outputHP[:, 0:-1, :, :, :].to(device)
    HPz = outputHP[:, :, 1:, :, :] - outputHP[:, :, 0:-1, :, :].to(device)
    HPx = outputHP[:, :, :, 1:, :] - outputHP[:, :, :, 0:-1, :].to(device)

    lhpy = loss_hpy(HPy, HPy.new_zeros(HPy.shape)).to(device)
    lhpz = loss_hpz(HPz, HPz.new_zeros(HPz.shape)).to(device)
    lhpx = loss_hpx(HPx, HPx.new_zeros(HPx.shape)).to(device)

    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))

    # Loss
    loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz + lhpx).to(device)
    return loss, nLabels


