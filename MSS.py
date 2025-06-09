from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import os
from tqdm.auto import tqdm

def read_mask(path):
    mask = np.array(Image.open(path))

    # Masks stored in RGB channels or as class ids
    if mask.ndim == 3:
        mask = mask.astype(np.float32) / 255.0
    else:
        mask = np.stack([mask==0, mask==1, mask==2], axis=-1).astype(np.float32)

    return mask

class MaSTr1325Dataset(torch.utils.data.Dataset):
    """
    Custom MaSTr1325 dataset loader with hardcoded image/mask directories.
    """

    def __init__(self, transform=None, normalize_t=None, preload=False, include_original=False):
        self.image_dir = Path("/kaggle/input/maritimeds/MaSTr1325_images_512x384")
        self.mask_dir = Path("/kaggle/input/maritimeds/MaSTr1325_masks_512x384")
        self.include_original = include_original
        self.transform = transform
        self.normalize_t = normalize_t
        self.cache = None

        # Get list of image base names (without extension)
        self.images = sorted([p.stem for p in self.image_dir.glob("*.jpg")])

        if preload:
            self.preload_into_memory()

    def preload_into_memory(self):
        self.cache = []
        for idx in tqdm(range(len(self)), desc="Preloading dataset into memory"):
            self.cache.append(self._read_sample(idx))

    def _read_sample(self, idx):
        img_name = self.images[idx]
        img_path = self.image_dir / f"{img_name}.jpg"
        mask_path = self.mask_dir / f"{img_name}m.png"

        img = np.array(Image.open(img_path))
        mask = read_mask(mask_path)

        data = {
            'image': img,
            'mask': mask,
            'img_name': img_name
        }

        return data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.cache is not None:
            data = self.cache[idx]
        else:
            data = self._read_sample(idx)

        img = data['image']
        mask = data['mask']

        if self.transform:
            img = Image.fromarray(img)
            mask = Image.fromarray((mask * 255).astype(np.uint8))  # Assuming 3-channel one-hot RGB

            img = self.transform(img)
            mask = self.transform(mask)

        if self.normalize_t:
            img = self.normalize_t(img)

        if self.include_original:
            return img, mask, data['image']  # include unprocessed original
        return img, mask


from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.ToTensor(),
])

dataset = MaSTr1325Dataset(transform=transform, preload=True)
img, mask = dataset[0]


import matplotlib.pyplot as plt
import random
import numpy as np

def plot_random_images_with_masks(dataset, num_samples=10):
    """
    Plot num_samples random images with their corresponding masks from the dataset.
    
    Args:
        dataset: an instance of MaSTr1325Dataset
        num_samples: number of samples to plot (default=10)
    """
    indices = random.sample(range(len(dataset)), num_samples)

    plt.figure(figsize=(15, num_samples * 2))
    for i, idx in enumerate(indices):
        image, mask = dataset[idx]

        # If image/mask is a tensor, convert to numpy
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).numpy()
        if torch.is_tensor(mask):
            mask = mask.permute(1, 2, 0).numpy()

        # Clamp and convert to displayable format
        image = np.clip(image, 0, 1)
        mask = np.clip(mask, 0, 1)

        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(image)
        plt.title(f"Image {idx}")
        plt.axis('off')

        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(mask)
        plt.title(f"Mask {idx}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Assuming dataset is already loaded
plot_random_images_with_masks(dataset)

import torch
from torch import nn

def time_independent_forward(module, x):
    # Squash samples and timesteps into a single axis
    B,T,*S = x.shape
    x_reshape = x.contiguous().view(B*T, *S)  # (batch * timesteps, input_size)

    y = module(x_reshape)

    # Reshape back into batch and timesteps
    _,*S = y.shape
    y = y.contiguous().view(B, T, *S)  # (batch, timesteps, input_size)

    return y

class TemporalContextModule(nn.Module):
    """Stores a running memory of previoius features."""
    def __init__(self, in_features, hist_len=5, sequential=False):
        super(TemporalContextModule, self).__init__()

        self.conv_in = nn.Conv2d(in_features, in_features//2, 1)
        self.conv_agg = nn.Conv3d(in_features//2, in_features//2, (hist_len+1, 3, 3), padding=(0,1,1))

        self.hist_len = hist_len
        self._is_sequential = sequential
        self._sequential_mem = None


    def forward(self, feat, feat_mem=None):
        if self._is_sequential:
            return self.forward_sequential(feat)
        else:
            return self.forward_unrolled(feat, feat_mem)

    def clear_state(self):
        """Clears feature memory. Should be called before inference on a new sequence."""

        self._sequential_mem = None

    def sequential(self):
        """Switch to sequential mode."""

        self._is_sequential = True
        self._sequential_mem = None
        return self

    def unrolled(self):
        """Switch to unrolled mode."""

        self._is_sequential = False
        self._sequential_mem = None
        return self

    def _aggregate(self, hist_volume):
        # Avg pool aggregation
        agg = self.conv_agg(hist_volume.permute(0,2,1,3,4)).squeeze(2)

        out = torch.cat([agg, hist_volume[:,-1]], 1)
        return out

    def forward_sequential(self, feat):
        assert feat.size(0) == 1, "Batch size should be 1 for sequential inference."
        feat_in = self.conv_in(feat)
        if self._sequential_mem is None:
            self._sequential_mem = feat_in.unsqueeze(1).repeat(1,self.hist_len,1,1,1)

        hist_volume = torch.cat([self._sequential_mem, feat_in.unsqueeze(1)], dim=1)

        # Discard oldest frame from memory
        self._sequential_mem = hist_volume[:, 1:]

        return self._aggregate(hist_volume)

    def forward_unrolled(self, feat, feat_mem):
        hist_volume = torch.cat([feat_mem, feat.unsqueeze(1)], dim=1)
        hist_volume = time_independent_forward(self.conv_in, hist_volume)

        return self._aggregate(hist_volume)

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, last_arm=False):
        super(AttentionRefinementModule, self).__init__()

        self.last_arm = last_arm

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x

        x = self.global_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        weights = self.sigmoid(x)

        out = weights * input

        if self.last_arm:
            weights = self.global_pool(out)
            out = weights * out

        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, bg_channels, sm_channels, num_features):
        super(FeatureFusionModule, self).__init__()

        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(bg_channels + sm_channels, num_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(num_features, num_features, 1)
        self.conv3 = nn.Conv2d(num_features, num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_big, x_small):
        if x_big.size(2) > x_small.size(2):
            x_small = self.upsampling(x_small)

        x = torch.cat((x_big, x_small), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        conv1_out = self.relu(x)

        x = self.global_pool(conv1_out)
        x = self.conv2(x)
        x = self.conv3(x)
        weights = self.sigmoid(x)

        mul = weights * conv1_out
        out = conv1_out + mul

        return out

class ASPPv2Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, bias=False, bn=False, relu=False):
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=bias))

        if bn:
            modules.append(nn.BatchNorm2d(out_channels))

        if relu:
            modules.append(nn.ReLU())

        super(ASPPv2Conv, self).__init__(*modules)

class ASPPv2(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256, relu=False, biased=True):
        super(ASPPv2, self).__init__()
        modules = []

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPv2Conv(in_channels, out_channels, rate, bias=biased, relu=relu))

        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # Sum convolution results
        res = torch.stack(res).sum(0)
        return res

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import PIL

def water_obstacle_separation_loss(features, gt_mask, clipping_value=None, include_sky=False):
    """Computes the water-obstacle separation loss from intermediate features.

    Args:
        features (torch.tensor): Features tensor
        gt_mask (torch.tensor): Ground truth tensor
        clipping_value (float): Clip loss at clipping_value * sigma.
        include_sky (bool): Include sky into separation loss
    """
    epsilon_watercost = 0.01
    min_samples = 5

    # Resize gt mask to match the extracted features shape (x,y)
    feature_size = (features.size(2), features.size(3))
    gt_mask = F.interpolate(gt_mask, size=feature_size, mode='area')

    # Create water and obstacles masks.
    # The masks should be of type float so we can multiply it later in order to mask the elements
    # (1 = water, 2 = sky, 0 = obstacles)
    if include_sky:
        mask_water = (gt_mask[:,1] + gt_mask[:,2]).unsqueeze(1)
    else:
        mask_water = gt_mask[:,1].unsqueeze(1)

    mask_obstacles = gt_mask[:,0].unsqueeze(1)

    # Count number of water and obstacle pixels, clamp to at least 1 (for numerical stability)
    elements_water = mask_water.sum((0,2,3), keepdim=True).clamp(min=1.)
    elements_obstacles = mask_obstacles.sum((0,2,3), keepdim=True)

    # Zero loss if number of samples for any class is smaller than min_samples
    if elements_obstacles.squeeze() < min_samples or elements_water.squeeze() < min_samples:
        return torch.tensor(0.)

    # Only keep water and obstacle pixels. Set the rest to 0.
    water_pixels = mask_water * features
    obstacle_pixels = mask_obstacles * features

    # Mean value of water pixels per feature (batch average)
    mean_water = water_pixels.sum((0,2,3), keepdim=True) / elements_water

    # Mean water value matrices for water and obstacle pixels
    mean_water_wat = mean_water * mask_water
    mean_water_obs = mean_water * mask_obstacles

    # Variance of water pixels (per channel, batch average)
    var_water = (water_pixels - mean_water_wat).pow(2).sum((0,2,3), keepdim=True) / elements_water

    # Average quare difference of obstacle pixels and mean water values (per channel)
    difference_obs_wat = (obstacle_pixels - mean_water_obs).pow(2).sum((0,2,3), keepdim=True)

    # Compute the separation
    loss_c = elements_obstacles * var_water / (difference_obs_wat + epsilon_watercost)

    # Clip loss
    if clipping_value is not None:
        loss_c = loss_c.clip(min=1./clipping_value**2)

    var_cost = loss_c.mean()

    return var_cost

def focal_loss(logits, labels, gamma=2.0, alpha=4.0, target_scale='labels'):
    """Focal loss of the segmentation output `logits` and ground truth `labels`."""

    epsilon = 1.e-9

    if target_scale == 'logits':
        # Resize one-hot labels to match the logits scale
        logits_size = (logits.size(2), logits.size(3))
        labels = F.interpolate(labels, size=logits_size, mode='area')
    elif target_scale == 'labels':
        # Resize network output to match the label size
        labels_size = (labels.size(2), labels.size(3))
        logits = TF.resize(logits, labels_size, interpolation=PIL.Image.BILINEAR)
    else:
        raise ValueError('Invalid value for target_scale: %s' % target_scale)

    logits_sm = torch.softmax(logits, 1)

    # Focal loss
    fl = -labels * torch.log(logits_sm + epsilon) * (1. - logits_sm) ** gamma
    fl = fl.sum(1) # Sum focal loss along channel dimension

    # Return mean of the focal loss along spatial and batch dimensions
    return fl.mean()

import torch
from torchmetrics import Metric

class PixelAccuracy(Metric):
    def __init__(self, num_classes: int, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, "Predictions and target shapes must match"

        valid_mask = target < self.num_classes
        self.correct += torch.sum((preds == target) & valid_mask)
        self.total += torch.sum(valid_mask)

    def compute(self):
        return self.correct / self.total.clamp(min=1)


class ClassIoU(Metric):
    def __init__(self, class_i: int, num_classes: int, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.class_i = class_i
        self.num_classes = num_classes

        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, "Predictions and target shapes must match"

        valid_mask = target < self.num_classes
        preds_mask = (preds == self.class_i) & valid_mask
        target_mask = (target == self.class_i) & valid_mask

        self.intersection += torch.sum(preds_mask & target_mask)
        self.union += torch.sum(preds_mask | target_mask)

    def compute(self):
        return self.intersection / self.union.clamp(min=1)


from PIL import Image

import torch
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

# from .loss import focal_loss, water_obstacle_separation_loss
# from .metrics import PixelAccuracy, ClassIoU

NUM_EPOCHS = 100
LEARNING_RATE = 1e-6
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-6
LR_DECAY_POW = 0.9
FOCAL_LOSS_SCALE = 'labels'
SL_METHOD = 'wasr'
SL_LAMBDA_DEFAULTS = {
    'wasr': 0.01,
    'none': 1.0
}


class LitModel(pl.LightningModule):
    """PyTorch Lightning wrapper for semantic segmentation model."""

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
        parser.add_argument("--momentum", type=float, default=MOMENTUM)
        parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
        parser.add_argument("--lr-decay-pow", type=float, default=LR_DECAY_POW)
        parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
        parser.add_argument("--focal-loss-scale", type=str, default=FOCAL_LOSS_SCALE, choices=['logits', 'labels'])
        parser.add_argument("--separation-loss", default=SL_METHOD, type=str, choices=['wasr', 'none'])
        parser.add_argument("--separation-loss-lambda", default=None, type=float)
        parser.add_argument("--separation-loss-clipping", default=None, type=float)
        parser.add_argument("--separation-loss-sky", action='store_true')
        return parser

    @staticmethod
    def parse_args(args):
        assert args.separation_loss in SL_LAMBDA_DEFAULTS
        if args.separation_loss_lambda is None:
            args.separation_loss_lambda = SL_LAMBDA_DEFAULTS[args.separation_loss]
        return args

    def __init__(self, model, num_classes, args):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.lr_decay_pow = args.lr_decay_pow
        self.focal_loss_scale = args.focal_loss_scale
        self.separation_loss = args.separation_loss
        self.separation_loss_clipping = args.separation_loss_clipping
        self.separation_loss_lambda = args.separation_loss_lambda
        self.separation_loss_sky = args.separation_loss_sky

        self.val_accuracy = PixelAccuracy(num_classes)
        self.val_iou_0 = ClassIoU(0, num_classes)
        self.val_iou_1 = ClassIoU(1, num_classes)
        self.val_iou_2 = ClassIoU(2, num_classes)

    def forward(self, x):
        return self.model(x)['out']

    def training_step(self, batch, batch_idx):
        features, labels = batch
        out = self.model(features)

        fl = focal_loss(out['out'], labels['segmentation'], target_scale=self.focal_loss_scale)

        separation_loss = torch.tensor(0.0, device=self.device)
        if self.separation_loss == 'wasr':
            separation_loss = water_obstacle_separation_loss(
                out['aux'], labels['segmentation'], self.separation_loss_clipping, include_sky=self.separation_loss_sky)

        loss = fl + self.separation_loss_lambda * separation_loss

        self.log('train/loss', loss.item())
        self.log('train/focal_loss', fl.item())
        self.log('train/separation_loss', separation_loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        out = self.model(features)

        loss = focal_loss(out['out'], labels['segmentation'], target_scale=self.focal_loss_scale)
        self.log('val/loss', loss.item(), prog_bar=True)

        labels_size = (labels['segmentation'].size(2), labels['segmentation'].size(3))
        logits = TF.resize(out['out'], labels_size, interpolation=Image.BILINEAR)
        preds = logits.argmax(1)

        labels_hard = labels['segmentation'].argmax(1)
        ignore_mask = labels['segmentation'].sum(1) < 0.9
        labels_hard = labels_hard * ~ignore_mask + 4 * ignore_mask

        self.val_accuracy.update(preds, labels_hard)
        self.val_iou_0.update(preds, labels_hard)
        self.val_iou_1.update(preds, labels_hard)
        self.val_iou_2.update(preds, labels_hard)

    def validation_epoch_end(self, outputs):
        self.log('val/accuracy', self.val_accuracy.compute(), prog_bar=True)
        self.log('val/iou/obstacle', self.val_iou_0.compute())
        self.log('val/iou/water', self.val_iou_1.compute())
        self.log('val/iou/sky', self.val_iou_2.compute())

        # Reset metrics
        self.val_accuracy.reset()
        self.val_iou_0.reset()
        self.val_iou_1.reset()
        self.val_iou_2.reset()

    def configure_optimizers(self):
        encoder_parameters = []
        decoder_w_parameters = []
        decoder_b_parameters = []
        for name, param in self.model.named_parameters():
            if name.startswith('backbone'):
                encoder_parameters.append(param)
            elif 'weight' in name:
                decoder_w_parameters.append(param)
            else:
                decoder_b_parameters.append(param)

        optimizer = torch.optim.RMSprop([
            {'params': encoder_parameters, 'lr': self.learning_rate},
            {'params': decoder_w_parameters, 'lr': self.learning_rate * 10},
            {'params': decoder_b_parameters, 'lr': self.learning_rate * 20},
        ], momentum=self.momentum, alpha=0.9, weight_decay=self.weight_decay)

        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / self.epochs) ** self.lr_decay_pow)

        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model'] = self.model.state_dict()


import argparse
import os
import math
import json
import torch
from torch.utils.data import DataLoader, ConcatDataset, BatchSampler
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from wasr_t.wasr_t import wasr_temporal_resnet101
from wasr_t.mobile_wasr_t import wasr_temporal_lraspp_mobilenetv3
from wasr_t.train import LitModel
from wasr_t.utils import MainLoggerCollection, Option
from wasr_t.callbacks import ModelExport
from wasr_t.data.mastr import MaSTr1325Dataset
from wasr_t.data.transforms import get_augmentation_transform, PytorchHubNormalization
from wasr_t.data.sampling import DatasetRandomSampler, DistributedSamplerWrapper, DatasetBatchSampler

# Configuration Defaults
WANDB_LOGGING = False
DEVICE_BATCH_SIZE = 3
TRAIN_CONFIG = 'configs/mastr1325_train.yaml'
VAL_CONFIG = 'configs/mastr1325_val.yaml'
NUM_CLASSES = 3
PATIENCE = None
LOG_STEPS = 20
NUM_WORKERS = 1
NUM_GPUS = -1
NUM_NODES = 1
RANDOM_SEED = None
OUTPUT_DIR = 'output'
PRETRAINED_DEEPLAB = True
PRECISION = 32
MONITOR_VAR = 'val/iou/obstacle'
MONITOR_VAR_MODE = 'max'
ADDITONAL_SAMPLES_RATIO = 0.5
HIST_LEN = 5
BACKBONE_GRAD_STEPS = 2
SIZE = (512, 384)

def get_arguments(input_args=None):
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Add arguments
    parser.add_argument("--batch-size", type=int, default=DEVICE_BATCH_SIZE)
    parser.add_argument("--train-config", type=str, default=TRAIN_CONFIG)
    parser.add_argument("--additional-train-config", type=str, default=None)
    parser.add_argument("--additional-samples-ratio", type=float, default=ADDITONAL_SAMPLES_RATIO)
    parser.add_argument("--val-config", type=str, default=VAL_CONFIG)
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--patience", type=Option(int), default=PATIENCE)
    parser.add_argument("--log-steps", type=int, default=LOG_STEPS)
    parser.add_argument("--visualization-steps", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=NUM_NODES)
    parser.add_argument("--gpus", default=NUM_GPUS)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--pretrained", type=bool, default=PRETRAINED_DEEPLAB)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--pretrained-weights", type=str, default=None)
    parser.add_argument("--monitor-metric", type=str, default=MONITOR_VAR)
    parser.add_argument("--monitor-metric-mode", type=str, default=MONITOR_VAR_MODE, choices=['min', 'max'])
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--precision", default=PRECISION, type=int, choices=[16, 32])
    parser.add_argument("--hist-len", default=HIST_LEN, type=int)
    parser.add_argument("--backbone-grad-steps", default=BACKBONE_GRAD_STEPS, type=int)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--mobile", action='store_true')
    parser.add_argument("--size", type=int, nargs=2, default=SIZE)

    parser = LitModel.add_argparse_args(parser)
    args = parser.parse_args(input_args)
    args = LitModel.parse_args(args)
    return args

class DataModule(pl.LightningDataModule):
    def __init__(self, args, normalize_t):
        super().__init__()
        self.args = args
        self.normalize_t = normalize_t

    def train_dataloader(self):
        transform = get_augmentation_transform() if not self.args.no_augmentation else None
        alternative_mask_subdir = self.args.mask_dir

        train_ds = MaSTr1325Dataset(self.args.train_config, transform=transform,
                                    normalize_t=self.normalize_t, masks_subdir=alternative_mask_subdir)

        if self.args.additional_train_config is not None:
            orig_ds = train_ds
            add_ds = MaSTr1325Dataset(self.args.additional_train_config, transform=transform,
                                      normalize_t=self.normalize_t, masks_subdir=alternative_mask_subdir)
            train_ds = ConcatDataset([orig_ds, add_ds])

            sample_ratio = self.args.additional_samples_ratio
            bs = self.args.batch_size

            if sample_ratio < 1:
                orig_ratio = 1 - sample_ratio
                n_samples = int((1 / orig_ratio) * len(orig_ds))
                sampler = DatasetRandomSampler(train_ds, [orig_ratio, sample_ratio], n_samples, shuffle=True)
                b_sampler = BatchSampler(sampler, batch_size=bs, drop_last=True)
                return DataLoader(train_ds, batch_sampler=b_sampler, num_workers=self.args.workers)
            else:
                b_sampler = DatasetBatchSampler(orig_ds, add_ds, bs, int(sample_ratio), shuffle=True)
                return DataLoader(train_ds, batch_sampler=b_sampler, num_workers=self.args.workers)

        return DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers)

    def val_dataloader(self):
        val_ds = MaSTr1325Dataset(self.args.val_config,
                                  transform=None,
                                  normalize_t=self.normalize_t,
                                  masks_subdir=self.args.mask_dir)
        return DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.workers)








#WaSR-T 

from collections import OrderedDict
import contextlib

import torch
from torch import nn
from torchvision.models.resnet import resnet101
from torch.hub import load_state_dict_from_url

from wasr_t.utils import IntermediateLayerGetter
import wasr_t.layers as L

model_urls = {
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'
}

def wasr_temporal_resnet101(num_classes=3, pretrained=True, sequential=False, backbone_grad_steps=2, hist_len=5):
    # Pretrained ResNet101 backbone
    backbone = resnet101(pretrained=True, replace_stride_with_dilation=[False, True, True])
    return_layers = {
        'layer4': 'out',
        'layer1': 'skip1',
        'layer2': 'skip2',
        'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    decoder = WaSRTDecoder(num_classes, hist_len=hist_len, sequential=sequential)

    model = WaSRT(backbone, decoder, backbone_grad_steps=backbone_grad_steps, sequential=sequential)

    # Load pretrained DeeplabV3 weights (COCO)
    if pretrained:
        model_url = model_urls['deeplabv3_resnet101_coco']
        state_dict = load_state_dict_from_url(model_url, progress=True)

        # Only load backbone weights, since decoder is entirely different
        keys_to_remove = [key for key in state_dict.keys() if not key.startswith('backbone.')]
        for key in keys_to_remove: del state_dict[key]

        model.load_state_dict(state_dict, strict=False)

    return model

class WaSRT(nn.Module):
    """WaSR-T model"""
    def __init__(self, backbone, decoder, backbone_grad_steps=2, sequential=False):
        super(WaSRT, self).__init__()

        self.backbone = backbone
        self.decoder = decoder
        self.backbone_grad_steps = backbone_grad_steps

        self._is_sequential = sequential

    def forward(self, x):
        if self._is_sequential:
            return self.forward_sequential(x)
        else:
            return self.forward_unrolled(x)

    def forward_sequential(self, x):
        features = self.backbone(x['image'])

        x = self.decoder(features)

        output = OrderedDict([
            ('out', x),
            ('aux', features['aux'])
        ])

        return output

    def forward_unrolled(self, x):
        features = self.backbone(x['image'])

        extract_feats = ['out','skip1','skip2']
        feats_hist = {f:[] for f in extract_feats}
        hist_len = x['hist_images'].shape[1]
        for i in range(hist_len):
            # Compute gradients only in last backbone_grad_steps - 1 steps
            use_grad = i >= hist_len - self.backbone_grad_steps + 1
            ctx = contextlib.nullcontext() if use_grad else torch.no_grad()
            with ctx:
                feats = self.backbone(x['hist_images'][:,i])
                for f in extract_feats:
                   feats_hist[f].append(feats[f])

        # Stack tensors
        for f in extract_feats:
            feats_hist[f] = torch.stack(feats_hist[f], 1)

        x = self.decoder(features, feats_hist)

        # Return segmentation map and aux feature map
        output = OrderedDict([
            ('out', x),
            ('aux', features['aux'])
        ])

        return output

    def sequential(self):
        """Switch network to sequential mode."""

        self._is_sequential = True
        self.decoder.sequential()

        return self

    def unrolled(self):
        """Switch network to unrolled mode."""

        self._is_sequential = False
        self.decoder.unrolled()

        return self

    def clear_state(self):
        """Clears state of the network. Used to reset model between sequences in sequential mode."""

        self.decoder.clear_state()


class WaSRTDecoder(nn.Module):
    def __init__(self, num_classes, hist_len=5, sequential=False):
        super(WaSRTDecoder, self).__init__()

        self.arm1 = L.AttentionRefinementModule(2048)
        self.arm2 = nn.Sequential(
            L.AttentionRefinementModule(512, last_arm=True),
            nn.Conv2d(512, 2048, 1) # Equalize number of features with ARM1
        )

        # Temporal Context Module
        self.tcm = L.TemporalContextModule(2048, hist_len=hist_len, sequential=sequential)

        self.ffm = L.FeatureFusionModule(256, 2048, 1024)
        self.aspp = L.ASPPv2(1024, [6, 12, 18, 24], num_classes)

    def forward(self, x, x_hist=None):
        if x_hist is None: x_hist={'skip1':None, 'skip2': None, 'out': None}
        feats_out = self.tcm(x['out'], x_hist['out'])

        arm1 = self.arm1(feats_out)
        arm2 = self.arm2(x['skip2'])
        arm_combined = arm1 + arm2

        x = self.ffm(x['skip1'], arm_combined)

        output = self.aspp(x)

        return output

    def clear_state(self):
        self.tcm.clear_state()

    def sequential(self):
        """Switch to sequential mode."""

        self.tcm.sequential()

        return self

    def unrolled(self):
        """Switch to unrolled mode."""

        self.tcm.unrolled()

        return self