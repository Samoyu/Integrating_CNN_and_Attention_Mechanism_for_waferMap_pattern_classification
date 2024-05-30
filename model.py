import torch.nn as nn
import torch
from torchvision import models
from torchvision.models import ResNet34_Weights, ResNet50_Weights

class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1, stride=1, padding='valid')
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1, stride=1, padding='valid')
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding='valid')
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        proj_value = proj_value.permute(0, 2, 1)
        out = torch.bmm(attention, proj_value).view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

class CAM_Module(nn.Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = energy.max(dim=-1, keepdim=True)[0]
        energy_new = energy_new - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value).view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

class DANet(nn.Module):
    def __init__(self, in_channels=512, out_channels=128, out_dim=8):
        super(DANet, self).__init__()
        inter_channels = in_channels // 4
        base_model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = torch.nn.Sequential(*list(base_model.children())[:-2])

        self.conv5a = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU()
        self.conv5c = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module()
        self.conv51 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(inter_channels)
        self.bn4 = nn.BatchNorm2d(inter_channels)
        self.dropout = nn.Dropout2d(0.1)
        self.conv6 = nn.Conv2d(inter_channels, out_channels, kernel_size=1)
        self.conv7 = nn.Conv2d(inter_channels, out_channels, kernel_size=1)
        self.conv8 = nn.Conv2d(inter_channels, out_channels, kernel_size=1)

        self.fc = nn.Sequential(
            nn.Linear(6272, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, out_dim)
        )

        self.feat = None
        self.gradients = None
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.backbone(x)
        feat1 = self.relu(self.bn1(self.conv5a(x)))
        sa_feat = self.sa(feat1)
        sa_conv = self.relu(self.bn3(self.conv51(sa_feat)))
        sa_output = self.conv6(self.dropout(sa_conv))

        feat2 = self.relu(self.bn2(self.conv5c(x)))
        sc_feat = self.sc(feat2)
        sc_conv = self.relu(self.bn4(self.conv52(sc_feat)))
        sc_output = self.conv7(self.dropout(sc_conv))

        # feat_sum = sa_conv + sc_conv
        feat_sum = sa_output + sc_output
        sasc_output = self.conv8(self.dropout(feat_sum))
        self.feat = sasc_output
        h = sasc_output.register_hook(self.activations_hook)
        output = nn.Flatten()(sasc_output)
        output = self.fc(output)
        return output
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.feat