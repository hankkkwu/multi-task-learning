# !wget https://hydranets-data.s3.eu-west-3.amazonaws.com/hydranets-data-2.zip && unzip -q hydranets-data-2.zip && mv hydranets-data-2/* . && rm hydranets-data-2.zip && rm -rf hydranets-data-2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
from torch.utils.data import Dataset
from utils import Normalise, RandomCrop, ToTensor, RandomMirror, InvHuberLoss, AverageMeter, MeanIoU, RMSE
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model_helpers import Saver, load_state_dict
import operator
import json
from tqdm import tqdm


class HydranetDataset(Dataset):
    """Define the dataset"""
    def __init__(self, data_file, transform=None):
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        self.datalist = [x.decode("utf-8").strip("\n").split("\t") for x in datalist]
        self.root_dir = "nyud"
        self.transform = transform
        self.masks_names = ("segm", "depth")

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        abs_paths = [os.path.join(self.root_dir, rpath) for rpath in self.datalist[idx]] # Will output list of nyud/*/00000.png
        sample = {}
        sample["image"] = np.array(Image.open(abs_paths[0])) #TODO: Copy/Paste your loaded code

        for mask_name, mask_path in zip(self.masks_names, abs_paths[1:]):
            #TODO: Copy/Paste your loaded code
            sample[mask_name] = np.array(Image.open(mask_path))

        if self.transform:
            sample["names"] = self.masks_names
            sample = self.transform(sample)
            # the names key can be removed by the transformation
            if "names" in sample:
                del sample["names"]
        return sample

def conv3x3(in_channels, out_channels, stride=1, dilation=1, groups=1, bias=False):
    """3x3 Convolution: Depthwise:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=bias, groups=groups)

def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False,):
    "1x1 Convolution: Pointwise"
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias, groups=groups)

def batchnorm(num_features):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    """
    return nn.BatchNorm2d(num_features, affine=True, eps=1e-5, momentum=0.1)

def convbnrelu(in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels),
                             nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels))

class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block from https://arxiv.org/abs/1801.04381"""
    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super().__init__() # Python 3
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1) # Boolean/Condition
        self.output = nn.Sequential(convbnrelu(in_planes, intermed_planes, 1),
                                    convbnrelu(intermed_planes, intermed_planes, 3, stride=stride, groups=intermed_planes),
                                    convbnrelu(intermed_planes, out_planes, 1, act=False))

    def forward(self, x):
        #residual = x
        out = self.output(x)
        if self.residual:
            return (out + x)#+residual
        else:
            return out

class MobileNetv2(nn.Module):
    """
    Building the Encoder — A MobileNetv2
    """
    def __init__(self, return_idx=[6]):
        super().__init__()
        # expansion rate, output channels, number of repeats, stride
        self.mobilenet_config = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
        ]
        self.in_channels = 32  # number of input channels
        self.num_layers = len(self.mobilenet_config)
        self.layer1 = convbnrelu(3, self.in_channels, kernel_size=3, stride=2)

        self.return_idx = [1, 2, 3, 4, 5, 6]
        #self.return_idx = make_list(return_idx)

        c_layer = 2
        for t, c, n, s in self.mobilenet_config:
            layers = []
            for idx in range(n):
                layers.append(InvertedResidualBlock(self.in_channels,c,expansion_factor=t,stride=s if idx == 0 else 1,))
                self.in_channels = c
            setattr(self, "layer{}".format(c_layer), nn.Sequential(*layers))
            c_layer += 1

        self._out_c = [self.mobilenet_config[idx][1] for idx in self.return_idx] # Output: [24, 32, 64, 96, 160, 320]

    def forward(self, x):
        outs = []
        x = self.layer1(x)
        outs.append(self.layer2(x))         # 16, x / 2
        outs.append(self.layer3(outs[-1]))  # 24, x / 4
        outs.append(self.layer4(outs[-1]))  # 32, x / 8
        outs.append(self.layer5(outs[-1]))  # 64, x / 16
        outs.append(self.layer6(outs[-1]))  # 96, x / 16
        outs.append(self.layer7(outs[-1]))  # 160, x / 32
        outs.append(self.layer8(outs[-1]))  # 320, x / 32
        return [outs[idx] for idx in self.return_idx]

def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]

class CRPBlock(nn.Module):
    """CRP are implemented as Skip-Connection Operations"""
    def __init__(self, in_planes, out_planes, n_stages, groups=False):
        super().__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False, groups=in_planes if groups else 1))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x

class MTLWRefineNet(nn.Module):
    """
    Building the Decoder - A Multi-Task Lighweight RefineNet
    Paper: https://arxiv.org/pdf/1810.03272.pdf
    """
    def __init__(self, input_sizes, num_classes, agg_size=256, n_crp=4):
        super().__init__()

        stem_convs = nn.ModuleList()
        crp_blocks = nn.ModuleList()
        adapt_convs = nn.ModuleList()
        heads = nn.ModuleList()

        # Reverse since we recover information from the end
        input_sizes = list(reversed((input_sizes)))

        # No reverse for collapse indices is needed
        self.collapse_ind = [[0, 1], [2, 3], 4, 5]

        groups = [False] * len(self.collapse_ind)
        groups[-1] = True

        for size in input_sizes:
            stem_convs.append(conv1x1(size, agg_size, bias=False))

        for group in groups:
            crp_blocks.append(self._make_crp(agg_size, agg_size, n_crp, group))
            adapt_convs.append(conv1x1(agg_size, agg_size, bias=False))

        self.stem_convs = stem_convs
        self.crp_blocks = crp_blocks
        self.adapt_convs = adapt_convs[:-1]

        num_classes = list(num_classes)
        for n_out in num_classes:
            heads.append(
                nn.Sequential(
                    conv1x1(agg_size, agg_size, groups=agg_size, bias=False),
                    nn.ReLU6(inplace=False),
                    conv3x3(agg_size, n_out, bias=True),
                )
            )

        self.heads = heads
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, xs):
        xs = list(reversed(xs))
        for idx, (conv, x) in enumerate(zip(self.stem_convs, xs)):
            xs[idx] = conv(x)

        # Collapse layers
        c_xs = [sum([xs[idx] for idx in make_list(c_idx)]) for c_idx in self.collapse_ind ]

        for idx, (crp, x) in enumerate(zip(self.crp_blocks, c_xs)):
            if idx == 0:
                y = self.relu(x)
            else:
                y = self.relu(x + y)
            y = crp(y)
            if idx < (len(c_xs) - 1):
                y = self.adapt_convs[idx](y)
                y = F.interpolate(y, size=c_xs[idx + 1].size()[2:],
                    mode="bilinear", align_corners=True,)

        outs = []
        for head in self.heads:
            outs.append(head(y))
        return outs

    @staticmethod
    def _make_crp(in_planes, out_planes, stages, groups):
        # Same as previous, but showing the use of a @staticmethod
        layers = [CRPBlock(in_planes, out_planes, stages, groups)]
        return nn.Sequential(*layers)


def train(model, opts, crits, dataloader, loss_coeffs=(1.0,), grad_norm=0.0):
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_meter = AverageMeter()
    pbar = tqdm(dataloader)

    for i, sample in enumerate(dataloader):
        loss = 0.0
        input = sample['image'].float().to(device)  #TODO: Get the Input
        targets = [sample[name].to(device) for name in dataloader.dataset.masks_names]  #TODO: Get the Targets

        #FORWARD
        outputs = model(input)  #TODO: Run a Forward pass

        for out, target, crit, loss_coeff in zip(outputs, targets, crits, loss_coeffs):
            # TODO: Increment the Loss
            loss += loss_coeff * crit(F.interpolate(out, size=target.size()[1:], mode="bilinear", align_corners=False).squeeze(dim=1), target.squeeze(dim=1))
            # loss += loss_coeff * crit(out, target)

        # BACKWARD
        #TODO: Zero Out the Gradients
        for opt in opts:
            opt.zero_grad()
        #TODO: Call Loss.Backward
        loss.backward()

        if grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        #TODO: Run one step
        for opt in opts:
            opt.step()

        loss_meter.update(loss.item())
        pbar.set_description(
            "Loss {:.3f} | Avg. Loss {:.3f}".format(loss.item(), loss_meter.avg)
        )

def validate(model, metrics, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    for metric in metrics:
        metric.reset()

    pbar = tqdm(dataloader)

    def get_val(metrics):
        results = [(m.name, m.val()) for m in metrics]
        names, vals = list(zip(*results))
        out = ["{} : {:4f}".format(name, val) for name, val in results]
        return vals, " | ".join(out)

    with torch.no_grad():
        for sample in pbar:
            # Get the Data
            input = sample["image"].float().to(device)
            targets = [sample[k].to(device) for k in dataloader.dataset.masks_names]

            #input, targets = get_input_and_targets(sample=sample, dataloader=dataloader, device=device)
            targets = [target.squeeze(dim=1).cpu().numpy() for target in targets]

            # Forward
            outputs = model(input)
            #outputs = make_list(outputs)

            # Backward
            for out, target, metric in zip(outputs, targets, metrics):
                metric.update(
                    F.interpolate(out, size=target.shape[1:], mode="bilinear", align_corners=False)
                    .squeeze(dim=1)
                    .cpu()
                    .numpy(),
                    target,
                )
            pbar.set_description(get_val(metrics)[1])
    vals, _ = get_val(metrics)
    print("----" * 5)
    return vals

def training():
    """
    training the model
    """
    # TODO: Load images and labels
    depth = glob.glob("nyud/depth/*.png")
    seg = glob.glob("nyud/masks/*.png")
    images = glob.glob("nyud/rgb/*.png")

    # Load color Map for Semantic Segmentation
    CMAP = np.load('nyud/cmap_nyud.npy')

    # Normalization
    img_scale = 1.0 / 255
    depth_scale = 5000.0    # for NYU dataset

    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])

    normalise_params = [img_scale, img_mean.reshape((1, 1, 3)), img_std.reshape((1, 1, 3)), depth_scale,]
    transform_common = [Normalise(*normalise_params), ToTensor()]

    # Transforms
    crop_size = 400
    transform_train = transforms.Compose([RandomMirror(), RandomCrop(crop_size)] + transform_common)
    transform_val = transforms.Compose(transform_common)

    # DataLoader
    train_batch_size = 4
    val_batch_size = 4
    train_file = "nyud/train_list_depth.txt"
    val_file = "nyud/val_list_depth.txt"

    # TODO: TRAIN DATALOADER
    trainloader = DataLoader(HydranetDataset(train_file, transform=transform_train), batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    # TODO: VALIDATION DATALOADER
    valloader = DataLoader(HydranetDataset(val_file, transform=transform_val), batch_size=val_batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    # Load pre-trained Encoder weights
    encoder = MobileNetv2()
    encoder.load_state_dict(torch.load("checkpoints/mobilenetv2-e6e8dd43.pth"))

    # Load Decoder, train it from scratch
    num_classes = (40, 1)
    decoder = MTLWRefineNet(encoder._out_c, num_classes)
    # hydranet = nn.DataParallel(nn.Sequential(encoder, decoder).cuda())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hydranet = nn.Sequential(encoder, decoder).to(device)
    print("Model has {} parameters".format(sum([p.numel() for p in hydranet.parameters()])))

    # Define Loss Function
    # The Segmentation Loss is the Cross Entropy Loss, working as a per-pixel classification function with 15 or so classes.
    # The Depth Loss will be the Inverse Huber Loss.
    # TODO: Set ignore_index
    ignore_index = 255
    ignore_depth = 0
    #TODO: Define the Loss for Segmentation
    # nn.CrossEntropyLoss applies F.log_softmax and nn.NLLLoss internally on your input, so you should pass the raw logits to it.
    crit_segm = nn.CrossEntropyLoss(ignore_index=ignore_index)
    #TODO: Define the Loss for Depth
    crit_depth = InvHuberLoss(ignore_index=ignore_depth)

    # Optimizer - Using the Stochastic Gradient Descent, add weight decay and momentum
    lr_encoder = 1e-2
    lr_decoder = 1e-3
    momentum_encoder = 0.9
    momentum_decoder = 0.9
    weight_decay_encoder = 1e-5
    weight_decay_decoder = 1e-5
    #TODO: Create a List of 2 Optimizers: One for the encoder, one for the decoder
    optims = [torch.optim.SGD(encoder.parameters(), lr=lr_encoder, momentum=momentum_encoder, weight_decay=weight_decay_encoder),
              torch.optim.SGD(decoder.parameters(), lr=lr_decoder, momentum=momentum_decoder, weight_decay=weight_decay_decoder)]

    # Model Definition & State Loading
    n_epochs = 1000
    init_vals = (0.0, 10000.0)
    comp_fns = [operator.gt, operator.lt]
    ckpt_dir = "./checkpoints"
    ckpt_path = "./checkpoints/checkpoint.pth.tar"

    saver = Saver(
        args=locals(),
        ckpt_dir=ckpt_dir,
        best_val=init_vals,
        condition=comp_fns,
        save_several_mode=all,
    )
    start_epoch, _, state_dict = saver.maybe_load(ckpt_path=ckpt_path, keys_to_load=["epoch", "best_val", "state_dict"],)
    load_state_dict(hydranet, state_dict)

    if start_epoch is None:
        start_epoch = 0

    # Set Learning Rate Scheduler
    opt_scheds = []
    for opt in optims:
        opt_scheds.append(torch.optim.lr_scheduler.MultiStepLR(opt, np.arange(start_epoch + 1, n_epochs, 100), gamma=0.1))

    #TODO: Define a Training Loop!
    val_every = 5
    loss_coeffs = (0.5, 0.5)
    for i in range(start_epoch, n_epochs):
        print("Epoch {:d}".format(i))
        train(hydranet, optims, [crit_segm, crit_depth], trainloader, loss_coeffs=loss_coeffs)
        for opt in opt_scheds:
            opt.step()
        if i % val_every == 0:
            metrics = [MeanIoU(num_classes[0]), RMSE(ignore_val=ignore_depth)]
            vals = validate(hydranet, metrics, valloader)
            saver.maybe_save(new_val = vals, dict_to_save={"state_dict": hydranet.state_dict(), "epoch": i})

if __name__ == '__main__':
    # training the model
    training()
