import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .backbone import resnet
import pdb
import torchvision.models as models
import numpy as np
import math
import torch.nn.functional as F
from .tricenter_loss import TripletCenterLoss
#from .transformer_focus.pos_Models import Encoder
from .transformer_base.pos_Models import Encoder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], init=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if init:
        init_weights(net, init_type, init_gain=init_gain)
    return net

def define_retrieval_nets(opt, net_option, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    init = True
    if net_option == 'resnet34_pytorch':
        orig_model = models.__dict__['resnet34'](pretrained=True)
        net = ResNetPytorch(orig_model)
        init = False
    elif net_option == 'cate_estimator':
        net = CateEstimation(opt, cate_num=opt.cate_num)
    elif net_option == 'further_conv':
        net = FurtherConv(opt, pose_num=opt.pose_num)
    elif net_option == 'triplet_center':
        net = TripletCenterLoss(margin=5, num_classes=opt.cate_num, feat_dim=opt.input_dim)
    elif net_option == 'attn':
        l_in  =256
        nLayer = 2
        views = 1 
        net = Encoder( d_model=l_in, d_inner=512,
                   len_max_seq = views, d_word_vec = l_in,
                   n_layers=nLayer, n_head=2, d_k=64, d_v=64, dropout=0.1)

    elif net_option == 'bn':
        net = Bottleneck()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net_option)
    return init_net(net, init_type, init_gain, gpu_ids, init=init)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

################################## Classes##############################################################################
class ResNetPytorch(nn.Module):
    def __init__(self, orig_model):
        super(ResNetPytorch, self).__init__()
        self.layer0 = nn.Sequential(orig_model.conv1, orig_model.bn1, orig_model.relu, orig_model.maxpool)
        self.layer1 = orig_model.layer1
        self.layer2 = orig_model.layer2
        self.layer3 = orig_model.layer3
        self.layer4 = orig_model.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.layer0(x)
        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);
        if return_feature_maps:
            return conv_out
        #x = torch.flatten(x, 1)
        #x = self.fc(x)
        return [x]


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):

    expansion = 4
    def __init__(self, inplanes=512, planes=256, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = conv1x1(512, 256)
        self.stride = stride
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(256,256)
    def forward(self, x):
        x =x[-1]
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.avgpool(out).flatten(1)
        #out = F.relu(self.fc(out))
        return out



class FurtherConv(nn.Module):
    def __init__(self, opt, pose_num=12, hidden_dim=512):
        super(FurtherConv, self).__init__()
        self.fpn_last_conv = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
                )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, conv_out, return_feat=False):
        f = self.fpn_last_conv(conv_out[-1])
        f = self.avgpool(f)
        f = f.view(f.size(0), -1)
        return f

class CateEstimation(nn.Module):
    def __init__(self, opt, cate_num=12):
        super(CateEstimation, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(opt.input_dim, cate_num)
        #self.l2norm = Normalize(2)
    
    def forward(self, conv_out, return_feat=False):
        x = self.dropout(conv_out)
        x = self.fc(x)
        if return_feat:
            return 0,x #0, x
        return x

