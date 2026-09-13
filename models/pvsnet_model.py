import sys
import os
import torch
import torch.nn as nn
import torchvision
import rff.layers as rff

# assuming we import parameters from parent dir if needed, or we just rely on arguments
import parameters as params

def getLinearLayer(in_feat, out_feat, activation=nn.ReLU(True)):
    return nn.Sequential(
        nn.Linear(in_features=in_feat, out_features=out_feat, bias=True),
        activation
    )

def getConvLayer(in_channel,out_channel,stride=1,padding=1,activation=nn.ReLU()):
    return nn.Sequential(nn.Conv2d(in_channel, 
                    out_channel,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    padding_mode='reflect'),
                    activation)

def getConvTransposeLayer(in_channel, out_channel,kernel=3,stride=1,padding=1,activation=nn.ReLU()):
    return nn.Sequential(nn.ConvTranspose2d(in_channel,
                                            out_channel,
                                            kernel_size = kernel,
                                            stride=stride,
                                            padding=padding),
                                            activation)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
    def forward(self, input):
        return input.view(input.size(0), 1, self.height//8, self.width//8)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + self.shortcut(residual)
        out = self.relu(out)
        return out

class MLPEncoder(nn.Module):
    def __init__(self, pose_dims=3, height=256, width=256):
        super().__init__()
        self.m = params.params_m if hasattr(params, 'params_m') else 32
        self.height = height
        self.width = width
        self.positional_encoding = rff.PositionalEncoding(sigma=1, m=self.m)
        self.layer1 = getLinearLayer(2*pose_dims*self.m, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = getLinearLayer(1024, 2048)
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = getLinearLayer(2048, (self.height//8)*(self.width//8))
        self.unflat = UnFlatten(self.height, self.width)

        self.up_layer1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_layer2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_layer3 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.unflat(x)
        x = self.up_layer1(x)
        x = self.up_layer2(x)
        x = self.up_layer3(x)
        return x
    
class UpperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet152(pretrained=False)
        layers = list(model.children())
        self.ResNetEncoder = torch.nn.Sequential(*layers[:5].copy())
        del model

    def forward(self, x):
        x1 = x[:, 0:3, :, :]
        x1 = self.ResNetEncoder(x1)
        return x1

    def apply_resnet_encoder(self, x):
        x1 = x[:, 0:3, :, :]
        x1 = self.ResNetEncoder(x1)
        return x1

class LowerEncoder(nn.Module):
    def __init__(self, total_image_input=1, is_lite=False):
        super().__init__()
        self.is_lite = is_lite
        self.encoder_pre = ResidualBlock((total_image_input*3)+1, 20)
        self.encoder_layer1 = ResidualBlock(20, 30)
        self.encoder_layer2 = ResidualBlock(30, 50)

        self.encoder_layer3 = nn.Sequential(
            ResidualBlock(50, 100),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder_layer4 = ResidualBlock(100, 200)
        
        if self.is_lite:
            self.encoder_layer5 = nn.Sequential(
                ResidualBlock(200, 200),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.encoder_layer6 = ResidualBlock(200, 200)
            self.encoder_layer7 = nn.Sequential(
                ResidualBlock(200, 200),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.encoder_layer8 = ResidualBlock(200, 500)
            self.encoder_layer9 = nn.Sequential(
                ResidualBlock(500, 500),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.encoder_layer10 = ResidualBlock(500, 500)
            self.encoder_layer11 = ResidualBlock(500, 500)
        else:
            self.encoder_layer5 = nn.Sequential(
                ResidualBlock(200, 400),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.encoder_layer6 = ResidualBlock(400, 600)
            self.encoder_layer7 = nn.Sequential(
                ResidualBlock(600, 800),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.encoder_layer8 = ResidualBlock(800, 1000)
            self.encoder_layer9 = nn.Sequential(
                ResidualBlock(1000, 1200),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.encoder_layer10 = ResidualBlock(1200, 1400)
            self.encoder_layer11 = ResidualBlock(1400, 1600)
            
    def forward(self, x):
        x = self.encoder_pre(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        skip1 = self.encoder_layer3(x)
        x = self.encoder_layer4(skip1)
        skip2 = self.encoder_layer5(x)
        x = self.encoder_layer6(skip2)
        skip3 = self.encoder_layer7(x)
        x = self.encoder_layer8(skip3)
        skip4 = self.encoder_layer9(x)
        x = self.encoder_layer10(skip4)
        x = self.encoder_layer11(x)
        return x, [skip1, skip2, skip3, skip4]

class MergeDecoder(nn.Module):
    def __init__(self, is_lite=False):
        super().__init__()
        self.is_lite = is_lite

        if self.is_lite:
            self.decoder_layer1 = ResidualBlock(500, 500)
            self.decoder_layer2 = ResidualBlock(500, 500)
            self.decoder_layer3 = ResidualBlock(500, 500)
            self.decoder_layer4 = nn.Sequential(
                nn.ConvTranspose2d(500, 200, 2, stride=2, padding=0), 
                nn.ReLU(True)
            )
            self.decoder_layer5 = ResidualBlock(200, 200)
            self.decoder_layer6 = nn.Sequential(
                nn.ConvTranspose2d(200, 200, 2, stride=2, padding=0), 
                nn.ReLU(True)
            )
            self.decoder_layer7 = ResidualBlock(200, 200)
        else:
            self.decoder_layer1 = ResidualBlock(1600, 1400)
            self.decoder_layer2 = ResidualBlock(1400, 1200)
            self.decoder_layer3 = ResidualBlock(1200, 1000)
            self.decoder_layer4 = nn.Sequential(
                nn.ConvTranspose2d(1000, 800, 2, stride=2, padding=0), 
                nn.ReLU(True)
            )
            self.decoder_layer5 = ResidualBlock(800, 600)
            self.decoder_layer6 = nn.Sequential(
                nn.ConvTranspose2d(600, 400, 2, stride=2, padding=0), 
                nn.ReLU(True)
            )
            self.decoder_layer7 = ResidualBlock(400, 200)

        self.decoder_layer8 = nn.Sequential(
            nn.ConvTranspose2d(200, 100, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer9 = ResidualBlock(100, 100)

        self.decoder_layer10 = nn.Sequential(
            nn.ConvTranspose2d(100, 100, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer11 = ResidualBlock(100, 100)
        self.decoder_layer12 = ResidualBlock(100, 50)
        self.decoder_layer13 = ResidualBlock(50, 40)
        self.decoder_layer14 = ResidualBlock(40, 20)
        self.decoder_layer15 = nn.Sequential(
            nn.Conv2d(20, 8, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.decoder_layer16 = nn.Sequential(
            nn.Conv2d(8, 3, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, lower_skip_list, upper_skip_list):
        x = self.decoder_layer1(x)
        x = self.decoder_layer2(x)
        x = x + lower_skip_list[3] + upper_skip_list[1]

        x = self.decoder_layer3(x)
        x = self.decoder_layer4(x)
        x = x + lower_skip_list[2] + upper_skip_list[0]
        
        x = self.decoder_layer5(x)
        x = self.decoder_layer6(x)
        x = x + lower_skip_list[1]

        x = self.decoder_layer7(x)
        x = self.decoder_layer8(x)
        x = x + lower_skip_list[0]

        x = self.decoder_layer9(x)
        x = self.decoder_layer10(x)
        x = self.decoder_layer11(x)
        x = self.decoder_layer12(x)
        x = self.decoder_layer13(x)
        x = self.decoder_layer14(x)
        x = self.decoder_layer15(x)
        x = self.decoder_layer16(x)        
        return x

class PVSNet(nn.Module):
    def __init__(self, total_image_input=1, pose_dims=3, height=256, width=256, is_lite=False):
        super().__init__()
        self.target_positional_embedding = MLPEncoder(pose_dims=pose_dims, height=height, width=width)
        self.upper_encoder = UpperEncoder()
        self.lower_encoder = LowerEncoder(total_image_input=total_image_input, is_lite=is_lite)
        self.merge_decoder = MergeDecoder(is_lite=is_lite)

        if is_lite:
            self.upper_encoder_extra_1 = nn.Sequential(
                ResidualBlock(256, 200),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.upper_encoder_extra_2 = nn.Sequential(
                ResidualBlock(200, 500),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.upper_encoder_extra_1 = nn.Sequential(
                ResidualBlock(256, 800),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.upper_encoder_extra_2 = nn.Sequential(
                ResidualBlock(800, 1200),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
    def forward(self, x, pos):
        target_position_feature = self.target_positional_embedding(pos)

        # First Encoder Branch
        upper_features_1 = self.upper_encoder.apply_resnet_encoder(x)
        upper_features_1 = self.upper_encoder_extra_1(upper_features_1)
        upper_features_2 = self.upper_encoder_extra_2(upper_features_1)

        # Second Encoder Branch
        stacked_tensor = torch.cat((x,target_position_feature),dim=1)
        lower_feature, skip_list = self.lower_encoder(stacked_tensor)

        # Decoder
        merged_feature = self.merge_decoder(lower_feature, skip_list, [upper_features_1, upper_features_2])
        return merged_feature
