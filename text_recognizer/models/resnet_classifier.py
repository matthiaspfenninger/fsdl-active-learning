import argparse
from typing import Any, Dict
import torch
import torch.nn as nn
import torchvision.transforms as tt
import torchvision

PRETRAINED = True
NUM_CLASSES = 4
NUM_CHANNELS = 11
DROPOUT = False
DROPUT_PROB = 0.5
DROPOUT_HIDDEN_DIM = 512

class ResnetClassifier(nn.Module):
    """Classify an image of arbitrary size through a (pretrained) ResNet network"""

    def __init__(self, data_config: Dict[str, Any] = None, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        n_channels = self.args.get("n_channels", NUM_CHANNELS)
        n_classes = self.args.get("n_classes", NUM_CLASSES)
        pretrained = self.args.get("pretrained", PRETRAINED)
        dropout = self.args.get("dropout", DROPOUT)

        # base ResNet model
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)

        # preprocessing steps to resize images (adapted from https://pytorch.org/hub/pytorch_vision_resnet/)
        new_channel_mean = 0.485 # from existing channel 0
        new_channel_std = 0.229 # from existing channel 0
        self.preprocess = tt.Compose([
            tt.Resize(224),
            tt.Normalize(
              mean=[0.485, 0.456, 0.406, 
                new_channel_mean, new_channel_mean, new_channel_mean, new_channel_mean, new_channel_mean, new_channel_mean, new_channel_mean, new_channel_mean], 
              std=[0.229, 0.224, 0.225, 
                new_channel_std, new_channel_std, new_channel_std, new_channel_std, new_channel_std, new_channel_std, new_channel_std, new_channel_std]),
        ])

        # changing the architecture of the laster layers
        # if dropout is activated, add an additional fully connected layer with dropout before the last layer
        if dropout:
            self.resnet.fc = nn.Sequential(
                nn.Linear(self.resnet.fc.in_features, self.resnet.fc.in_features), # additional fc layer
                nn.BatchNorm1d(self.resnet.fc.in_features), # adding batchnorm
                nn.ReLU(), # additional nonlinearity
                nn.Dropout(DROPUT_PROB), # additional dropout layer
                nn.Linear(self.resnet.fc.in_features, DROPOUT_HIDDEN_DIM), # additional fc layer
                nn.BatchNorm1d(DROPOUT_HIDDEN_DIM), # adding batchnorm
                nn.ReLU(), # additional nonlinearity
                nn.Dropout(DROPUT_PROB), # additional dropout layer
                nn.Linear(DROPOUT_HIDDEN_DIM, n_classes) # same fc layer as we had before
        )
        # otherwise just adapt no. of classes in last fully-connected layer
        else:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)

        # adapting the no. of input channels to the first conv layer 
        # (adapted from https://discuss.pytorch.org/t/how-to-modify-the-input-channels-of-a-resnet-model/2623/10)
        existing_layer = self.resnet.conv1

        new_layer = nn.Conv2d(in_channels=n_channels, 
                        out_channels=existing_layer.out_channels, 
                        kernel_size=existing_layer.kernel_size, 
                        stride=existing_layer.stride, 
                        padding=existing_layer.padding,
                        bias=existing_layer.bias)


        new_layer.weight[:, :existing_layer.in_channels, :, :] = existing_layer.weight.clone() # copying the weights from the old to the new layer
        
        copy_weights = 0 # take channel 0 weights to initialize new ones
        for i in range(n_channels - existing_layer.in_channels): # copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
            channel = existing_layer.in_channels + i
            new_layer.weight[:, channel:channel+1, :, :] = existing_layer.weight[:, copy_weights:copy_weights+1, :, :].clone()

        new_layer.weight = nn.Parameter(new_layer.weight)

        self.resnet.conv1 = new_layer


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor (H, W can be arbitrary, will be reshaped by reprocessing)

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        
        x = x.float()
        x = self.preprocess(x)
        x = self.resnet(x)

        return x

    def add_to_argparse(parser):
        parser.add_argument("--pretrained", type=bool, default=PRETRAINED)
        parser.add_argument("--n_classes", type=int, default=NUM_CLASSES)
        parser.add_argument("--n_channels", type=int, default=NUM_CHANNELS)
        parser.add_argument("--dropout", type=bool, default=DROPOUT)
        return parser
