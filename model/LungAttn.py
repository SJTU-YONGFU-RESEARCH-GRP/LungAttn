
import os
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import time
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--size', type=int, default=224)
parser.add_argument('--mixup', type=eval, default=False, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_bs', type=int, default=64)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--step', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--dropout0', type=float, default=0.2)
parser.add_argument('--dropout1', type=float, default=0.4)
parser.add_argument('--dropout2', type=float, default=0.3)
parser.add_argument('--dk', type=int, default=20)
parser.add_argument('--dv', type=int, default=4)
parser.add_argument('--nh', type=int, default=2)
parser.add_argument('--shape', type=int, default=14)
parser.add_argument('--save', type=str, default='../log/details/')
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--input', '-i',
                    default="../pack/official/tqwt1_4_train.p", type=str,
                    help='path to directory with input data archives')
parser.add_argument('--test', default="../pack/official/tqwt1_4_test.p",
                    type=str, help='path to directory with test data archives')
args = parser.parse_args()
#######################################################################################
# for our model build
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride,
                                  padding=self.padding)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride,
                                  padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_att=False, relatt=False, shape=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.droupout = nn.Dropout(args.dropout0)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = AugmentedConv(inplanes, planes, kernel_size=3, dk=args.dk, dv=args.dv, Nh=args.nh, relative=relatt, stride=stride, shape=shape) if use_att else conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x
        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.droupout(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.droupout(out)
        return out + shortcut
class LungAttn(nn.Module):
    # (self, inplanes, planes, shape, drop_rate, stride=1, v=0.2, k=2, Nh=2, downsample=None, attention=False)
    def __init__(self):
        super(LungAttn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=True),
            norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.ResNet_0_0 = ResBlock(64, 64, stride=2, downsample= conv1x1(64, 64, 2))
        self.ResNet_0_1 = ResBlock(64, 64, stride=2, downsample= conv1x1(64, 64, 2))
        self.ResNet_0 = ResBlock(64, 64)
        self.ResNet_1 = ResBlock(64, 64)
        self.ResNet_2 = ResBlock(64, 64)
        self.ResNet_3 = ResBlock(64, 64)
        self.ResNet_4 = ResBlock(64, 64)
        self.ResNet_5 = ResBlock(64, 64, use_att=True, relatt=True, shape=args.shape)
        self.ResNet_6 = ResBlock(64, 64)
        self.norm0 = norm(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 4)
        self.dropout1 = nn.Dropout(args.dropout1)
        self.dropout2 = nn.Dropout(args.dropout2)
        self.flat = Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.ResNet_0_0(x)
        x = self.ResNet_0_1(x)
        x = self.ResNet_0(x)
        x = self.ResNet_1(x)
        x = self.ResNet_2(x)
        x = self.ResNet_3(x)
        x = self.ResNet_4(x)
        x = self.ResNet_5(x)
        x = self.ResNet_6(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

#############################################################################
# for loss calculation
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  if epoch < 110:
    lr = args.lr * (0.1 ** (epoch // args.step))
    print("Adaptive learning rate: %e" % (lr))
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
  elif epoch == 110:
    for param_group in optimizer.param_groups:
      print("learning rate:", param_group['lr'])

def normalize(x):
    min_s = np.min(x)
    max_s = np.max(x)
    sample_stft = (x - min_s) / (max_s - min_s)
    return sample_stft

class myDataset(data.Dataset):
    def __init__(self, stft, targets):
        self.stft = stft
        self.targets = targets

    def __getitem__(self, index):
        sample_stft = self.stft[index][0]
        sample_stfth = self.stft[index][1]
        sample_stftr = self.stft[index][2]
        #        sample_mfcc = self.mfcc[index]
        target = self.targets[index]
        target = torch.from_numpy(target)

        sample_stft = normalize(sample_stft)
        sample_stfth = normalize(sample_stfth)
        sample_stftr = normalize(sample_stftr)

        output_stft = torch.FloatTensor([sample_stft])
        crop_s = transforms.Resize([args.size, args.size])
        img_s = transforms.ToPILImage()(output_stft)
        croped_img = crop_s(img_s)
        output_stft = transforms.ToTensor()(croped_img)

        output_stfth = torch.FloatTensor([sample_stfth])
        img_s = transforms.ToPILImage()(output_stfth)
        croped_img = crop_s(img_s)
        output_stfth = transforms.ToTensor()(croped_img)

        output_stftr = torch.FloatTensor([sample_stftr])
        img_s = transforms.ToPILImage()(output_stftr)
        croped_img = crop_s(img_s)
        output_stftr = transforms.ToTensor()(croped_img)

        output_stft = torch.cat((output_stft, output_stfth, output_stftr),0)
        return output_stft, target

    def __len__(self):
        return len(self.targets)


def get_mnist_loaders(batch_size=128, test_batch_size = 500, workers = 4, perc=1.0):
    # ori, ck, wh, res, label
    stft, labels, stfth, stftr = joblib.load(open(args.input, mode='rb'))
    stft, labels, stfth, stftr = np.array(stft), one_hot(np.array(labels), 4), np.array(stfth), np.array(stftr)
    stft = np.concatenate((stft[:, np.newaxis], stfth[:, np.newaxis], stftr[:, np.newaxis]), 1)

    stft_test, labels_test, stft_testh, stft_testr = joblib.load(open(args.test, mode='rb'))
    stft_test, labels_test, stft_testh, stft_testr = np.array(stft_test), one_hot(np.array(labels_test), 4), np.array(
        stft_testh), np.array(stft_testr)
    stft_test = np.concatenate((stft_test[:, np.newaxis], stft_testh[:, np.newaxis], stft_testr[:, np.newaxis]), 1)

    train_loader = DataLoader(
        myDataset(stft, labels), batch_size=batch_size,
        shuffle=True, num_workers=workers, drop_last=True
    )
    train_eval_loader = DataLoader(
        myDataset(stft, labels), batch_size=test_batch_size,
        shuffle=False, num_workers=workers, drop_last=True
    )

    test_loader = DataLoader(
        myDataset(stft_test, labels_test),
        batch_size=test_batch_size, shuffle=False, num_workers=workers, drop_last=False
    )

    return train_loader, train_eval_loader, test_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def one_hot(x, K):
    # x is a array from np
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader,criterion):
    total_correct = 0
    targets = []
    outputs = []
    losses = AverageMeter()
    for cat_stft, y in dataset_loader:
        target_class = np.argmax(y.numpy(), axis=1)
        targets = np.append(targets, target_class)
        with torch.no_grad():
            logits = model(cat_stft.to(device))
        y = y.type_as(logits)
        loss = criterion(logits, y)
        losses.update(loss.data, y.size(0))
        predicted_class = np.argmax(logits.cpu().detach().numpy(), axis=1)
        outputs = np.append(outputs, predicted_class)
        total_correct += np.sum(predicted_class == target_class)
    acc = total_correct / len(dataset_loader.dataset)
    Confusion_matrix=sk_confusion_matrix(targets.tolist(), outputs.tolist())
    print('Confusion_matrix:')
    print(Confusion_matrix)
    Sq = Confusion_matrix[0][0]/(sum(Confusion_matrix[0])) #specificity
    Se = (Confusion_matrix[1][1]+Confusion_matrix[2][2]+Confusion_matrix[3][3])/(sum(Confusion_matrix[1])+sum(Confusion_matrix[2])+sum(Confusion_matrix[3])) #sensitivity
    return acc, Se, Sq, (Se+Sq)/2, losses.avg.item(), Confusion_matrix

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def augment(x, y, index1, index2, alpha, target_type, mode=0):
    # index always belong to normal class if target_type!=4
    min_len = min(len(index1), len(index2))
    if target_type == 2:
        min_len = int(min_len*2/3)
        index1 = index1[:min_len]
        index2 = index2[:min_len]
    else:
        if len(index1) < len(index2):
            index2 = index2[:min_len]
        else:
            index1 = index1[:min_len]

    index1 = torch.from_numpy(index1)
    index2 = torch.from_numpy(index2)
    lam = np.random.beta(alpha, alpha, min_len)
    lam_x = lam.reshape(len(lam),1,1,1)
    lam_y = lam.reshape(len(lam),1)
    lam_x = torch.from_numpy(lam_x)
    lam_y = torch.from_numpy(lam_y)
    # lbd = 0.3*np.random.rand()+0.4

    if mode == 0:
        if target_type == 4:
            cat_x = lam_x * x[index1] + (1-lam_x) * x[index2]
            cat_y = [[0, 0, 0, 1] for i in range(min_len)]
            cat_y = torch.tensor(cat_y).to(cat_x.device)

        else:
            lam_x[lam_x < 0.5] = 1 - lam_x[lam_x < 0.5]  # all larger than 0.5
            lam_y[lam_y < 0.5] = 1 - lam_y[lam_y < 0.5]
            cat_x = (1 - lam_x) * x[index1] + lam_x * x[index2]
            cat_y = (1 - lam_y) * y[index1] + lam_y * y[index2]
    else:
        lam_x[lam_x < 0.5] = 1 - lam_x[lam_x < 0.5]  # all larger than 0.5
        lam_y[lam_y < 0.5] = 1 - lam_y[lam_y < 0.5]
        cat_x = lam_x * x[index2] + (1 - lam_x) * x[index1]
        cat_y = lam_y * y[index2] + (1 - lam_y) * y[index1]

    return cat_x, cat_y


def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    ny = np.argmax(y.cpu().numpy(), axis=1)
    first = np.where(ny == 0)[0]  # first is the index that the value equal to [1,0,0,0]
    second = np.where(ny == 1)[0]
    third = np.where(ny == 2)[0]
    fourth = np.where(ny == 3)[0]

    mixed_x1, mixed_y1 = augment(x, y, first, second, alpha, target_type=2)
    mixed_x2, mixed_y2 = augment(x, y, first, third, alpha, target_type=3)
    mixed_x3, mixed_y3 = augment(x, y, second, third, alpha, target_type=4)
    # cycle_ = np.random.randint(2, 4)
    cycle_ = 1
    for i in range(cycle_):
        np.random.shuffle(first)
        if i<2:
            nmixed_x2, nmixed_y2 = augment(x, y, first, third, alpha, target_type=3)
            mixed_x2 = torch.cat((mixed_x2, nmixed_x2), 0)
            mixed_y2 = torch.cat((mixed_y2, nmixed_y2), 0)
        np.random.shuffle(second)
        nmixed_x3, nmixed_y3 = augment(x, y, second, third, alpha, target_type=4)
        # np.random.shuffle(first)
        # nmixed_x3, nmixed_y3 = augment(x, y, first, fourth, alpha, target_type=4,mode=1)
        mixed_x3 = torch.cat((mixed_x3, nmixed_x3), 0)
        mixed_y3 = torch.cat((mixed_y3, nmixed_y3), 0)

    cat_x = torch.cat((x, mixed_x1, mixed_x2, mixed_x3), 0)
    y = y.type_as(mixed_y2)
    mixed_y3 = mixed_y3.type_as(mixed_y2)
    cat_y = torch.cat((y, mixed_y1, mixed_y2, mixed_y3), 0)

    return len(first), len(second)+mixed_x1.size()[0], len(third)+mixed_x2.size()[0], len(fourth)+mixed_x3.size()[0], cat_x, cat_y

if __name__ == '__main__':
    # os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    # memory_gpu = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
    # top_k_idx = memory_gpu.argsort()[::-1][0:args.gpu]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu#','.join(str(i) for i in top_k_idx)
    # os.system('rm tmp')
    pid = os.getpid()
    print('PID:', pid)
    start_t = time.time()
    saved_dir = args.save + 'mix_' if args.mixup else args.save + 'nomix_'
    saved_dir = saved_dir + str(args.size) +'bs' + str(args.batch_size) + 'lr' + str(args.lr) + 'dp' + str(args.dropout0) + str(args.dropout1) + str(args.dropout2)+ \
                'dk'+str(args.dk)+'dv'+str(args.dv)+'nh'+str(args.nh)+'ep' + str(args.nepochs) +'wd' + str(args.weight_decay)
    makedirs(saved_dir)
    logger = get_logger(logpath=os.path.join(saved_dir, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    use_cuda = True
    batch_size = args.batch_size
    net = LungAttn()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    net.to(device)
    if use_cuda:
        # data parallel
        # device = torch.device("cuda:0" if use_cuda else "cpu")
        # net.to(device)
        # net = torch.nn.DataParallel(net, device_ids=list(range(args.gpu)))
        print('Using', torch.cuda.device_count(), 'GPUs.')
        print(torch.cuda.get_device_name(0))

    logger.info(net)
    logger.info('Number of parameters: {}'.format(count_parameters(net)))

    if args.mixup == False:
        pos_weight = torch.tensor([2,3,9,10]).to(device)
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)
    # milestones = [50,70,90,100]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nepochs, eta_min=0, last_epoch=-1)
    train_loader, train_eval_loader, test_loader = get_mnist_loaders(batch_size, args.test_bs, args.workers)

    prepare_t = time.time()
    print('The preparation time:', prepare_t - start_t)
    best_val_score = 0
    with open(saved_dir + "/result.csv", 'w') as f:
        csv_write = csv.writer(f)
        csv_head = ['epoch', 'train_loss', 'train_acc','train_se', 'train_sq','train_score','test_loss', 'test_acc', 'test_se','test_sq', 'test_score']
        csv_write.writerow(csv_head)

    for epoch in range(args.nepochs):

        losses = AverageMeter()
        epoch_start_t = time.time()
        batch_end_t = time.time()
        ################################train##########################################
        net.train()
        temp = 0
        sn0,sn1,sn2,sn3 = 0,0,0,0
        load_t, mix_t = 0, 0
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (cat_stft, labels) in enumerate(train_loader):
            batch_t = time.time()
            load_t += batch_t - batch_end_t

            if args.mixup:
                n0, n1, n2, n3, cat_stft, labels = mixup_data(cat_stft, labels, args.alpha)
                mix_end_t = time.time()
                mix_t += mix_end_t - batch_t
                sn0 = sn0+n0
                sn1 = sn1+n1
                sn2 = sn2+n2
                sn3 = sn3+n3
            # target_class = torch.argmax(labels, dim=1)
            cat_stft, labels = Variable(cat_stft.type(torch.FloatTensor).to(device)), Variable(labels.to(device))

            outputs = net(cat_stft)
            labels = labels.type_as(outputs)
            loss = criterion(outputs, labels)
            losses.update(loss.data, labels.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_end_t = time.time()

        train_t = batch_end_t - epoch_start_t
        if args.mixup == True:
            logger.info("Normal {}  | Crackle {} | Wheeze {} | Both {}".format(sn0,sn1,sn2,sn3))
        print(sn0,sn1,sn2,sn3)
        print('Loading dataset time:', load_t)
        print('Data augmentation time:', mix_t)
        print('Training time per epoch:', train_t)
        scheduler.step()
        #################################test##########################################
        eval_start_t = time.time()
        net.eval()
        train_acc, train_Se, train_Sq, train_Score, _, train_confm = accuracy(net, train_eval_loader, criterion)
        test_acc, test_Se, test_Sq, test_Score, test_loss, test_confm = accuracy(net, test_loader,criterion)
        # scheduler.step()
        eval_end_t = time.time()
        print('Evaluation time:', eval_end_t - eval_start_t)
        ###############################################################################
        with open (saved_dir + "/result.csv",'a+') as f:
            csv_write = csv.writer(f)
            # data_row = [epoch, prec_t, rec_t, f1_t, suc_t]
            data_row = [epoch, losses.avg.item(), train_acc, train_Se, train_Sq, train_Score, test_loss, test_acc, test_Se, test_Sq, test_Score]
            csv_write.writerow(data_row)
        logger.info(
            "Epoch {:04d}  |  "
            "Train Loss {:.4f} |Train Acc {:.4f} |  train Se {:.4f} | train Sq {:.4f} | train Score {:.4f} | Test Loss {:.4f} |Test Acc {:.4f} | test Se {:.4f} | test Sq {:.4f} | test Score {:.4f}".format(
                epoch, losses.avg.item(), train_acc, train_Se, train_Sq, train_Score, test_loss, test_acc, test_Se, test_Sq, test_Score
            )
        )

        if test_Score > best_val_score and test_Score!=0.5:
            best_val_score = test_Score
            logger.info(test_confm)
            print('Saving best model parameters with Val F1 score = %.4f' % (best_val_score))
            torch.save(net.state_dict(), saved_dir + '/saved_model_params')