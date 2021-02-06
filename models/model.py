from time import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_encoding import *
from .transformer import TransformerEncoder, TransformerEncoderLayer, _get_clones
from .conv_lstm import ConvLSTM

from utils.utils import print_


class JointModel(nn.Module):
    def __init__(
        self,
        in_channels=2048,
        out_channels=512,
        stride=1,
        num_layers=1,
        num_encoder_layers=1,
        dropout=0.2,
        normalize_before=True,
        skip_conn=False,
        mask_dim=112,
        vocab_size=8801,
    ):
        super(JointModel, self).__init__()

        self.skip_conn = skip_conn
        self.mask_dim = mask_dim

        ################ W/O GLoVE ###################
        ## self.embedding = nn.Embedding(vocab_size, 300)
        ##############################################

        self.text_encoder = TextEncoder(num_layers=num_layers)

        self.pool = nn.AdaptiveMaxPool2d((28, 28))
        self.conv_3x3 = nn.ModuleDict(
            {
                "layer2": nn.Sequential(
                    nn.Conv2d(
                        512, out_channels, kernel_size=3, stride=stride, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                ),
                "layer3": nn.Sequential(
                    nn.Conv2d(
                        1024, out_channels, kernel_size=3, stride=stride, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                ),
                "layer4": nn.Sequential(
                    nn.Conv2d(
                        2048, out_channels, kernel_size=3, stride=stride, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                ),
            }
        )

        num_level = len(self.conv_3x3)
        
        ############### JRM ################
        encoder_layer = TransformerEncoderLayer(
            out_channels,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout,
            normalize_before=normalize_before,
        )
        encoder_norm = nn.LayerNorm(out_channels) if normalize_before else None
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        ######################################

        ########################## W/O CMMLF #############################
        ## self.conv_fuse = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channels))
        ################################################################

        ############################### Conv3D #########################
        ## self.conv_3d = nn.Sequential(nn.Conv3d(out_channels, out_channels, 3, padding=(0, 1, 1)), nn.BatchNorm3d(out_channels))
        ################################################################

        ############### CMMLF ######################
        self.cmmlf = CMMLF(out_channels, 196, dropout)
        ############################################

        ################ CONV LSTM ##############
        ## self.conv_lstm = ConvLSTM(
        ##     input_dim=512,
        ##     hidden_dim=[512],
        ##     kernel_size=(3, 3),
        ##     num_layers=1,
        ##     batch_first=True,
        ##     bias=True,
        ##     return_all_layers=False,
        ## )
        #########################################

        ### with Baseline in_channels = 512 * 3
        self.aspp_decoder = ASPP(
            in_channels=out_channels*3, atrous_rates=[6, 12, 24], out_channels=512
        )

        self.conv_upsample = ConvUpsample(
            in_channels=512,
            out_channels=1,
            channels=[256, 256],
            upsample=[True, True],
            drop=dropout,
        )

        self.upsample = nn.Sequential(
            nn.Upsample(
                size=(self.mask_dim, self.mask_dim), mode="bilinear", align_corners=True
            ),
            ## nn.Hardsigmoid(),
            nn.Sigmoid(),
	    ## nn.Tanh(),
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def forward(self, image, phrase, img_mask, phrase_mask):

        # import pdb; pdb.set_trace()

        image_features = []
        for key in self.conv_3x3:
            layer_output = self.activation(self.conv_3x3[key](self.pool(image[key])))
            image_features.append(layer_output)

        ######### W/O GLoVE ###########
        ## phrase = self.embedding(phrase)
        ###############################

        f_text = self.text_encoder(phrase)
        f_text = f_text.permute(0, 2, 1)
        _, E, L = f_text.shape

        joint_features = []
        for i in range(len(image_features)):
            f_img = image_features[i]
            B, C, H, W = f_img.shape

            f_img = f_img.flatten(2)

            f_joint = torch.cat([f_img, f_text], dim=2)
            src = f_joint.flatten(2).permute(2, 0, 1)

            ###################### JRM #########################
            pos_embed_img = positionalencoding2d(B, d_model=C, height=H, width=W)
            pos_embed_img = pos_embed_img.flatten(2).permute(2, 0, 1)

            pos_embed_txt = positionalencoding1d(B, max_len=phrase_mask.shape[1])
            pos_embed_txt = pos_embed_txt.permute(1, 0, 2)

            pos_embed = torch.cat([pos_embed_img, pos_embed_txt], dim=0)
            ## pos_embed = None

            src_key_padding_mask = ~torch.cat([img_mask, phrase_mask], dim=1).bool()

            enc_out = self.transformer_encoder(
                src, pos=pos_embed, src_key_padding_mask=src_key_padding_mask
            )
            enc_out = enc_out.permute(1, 2, 0)
            #####################################################

            ################## Only CMMLF ######################
            ## enc_out = src.permute(1, 2, 0)
            ###################################################

            ########### W/O JRM ##########
            ## enc_out = src.permute(1, 2, 0)
            ##############################

            ####################### W/O CMMLF #####################
            ## f_img_out = enc_out[:, :, : H * W].view(B, C, H, W)

            ## f_txt_out = enc_out[:, :, H * W :].transpose(1, 2)  # B, L, E
            ## masked_sum = f_txt_out * phrase_mask[:, :, None]
            ## f_txt_out = masked_sum.sum(dim=1) / phrase_mask.sum(dim=-1, keepdim=True)

            ## f_out = torch.cat(
            ##     [f_img_out, f_txt_out[:, :, None, None].expand(B, -1, H, W)], dim=1
            ## )
            ## 
            ## enc_out = self.activation(self.conv_fuse(f_out))
            #######################################################

            joint_features.append(enc_out)

        ###################### CMMLF ########################
        level_features = torch.stack(joint_features, dim=1)
        fused_feature = self.activation(self.cmmlf(level_features, H, W, phrase_mask))
        #####################################################

        ####################### ConvLSTM ###################
        ## level_features = torch.stack(joint_features[::-1], dim=1)
        ## fused_feature = self.conv_lstm(level_features)[0][:, -1]
        ####################################################

        ##################### Conv3D ######################
        ## level_features = torch.stack(joint_features, dim=2)
        ## fused_feature = self.activation(self.conv_3d(level_features)).permute(0, 2, 1, 3, 4)
        ## fused_feature = fused_feature.squeeze(1)
        ###################################################

        ################## Baseline ####################
        ## fused_feature = torch.cat(joint_features, dim=1)
        ################################################

        x = self.aspp_decoder(fused_feature)
        x = self.upsample(self.conv_upsample(x)).squeeze(1)

        ## x = (1 + x)/2
       
        return x


class CMMLF(nn.Module):
    def __init__(self, channel_dim, num_regions, dropout):
        super().__init__()

        self.num_regions = num_regions

        self.conv2d = nn.Sequential(
            nn.Conv2d(channel_dim * 2, channel_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_dim),
            nn.Dropout2d(dropout),
        )

        self.conv_3d = nn.Sequential(
            nn.Conv3d(channel_dim, 512, 3, padding=(0, 1, 1)), nn.BatchNorm3d(512)
        )

    def forward(self, level_features, H, W, phrase_mask):

        B, N, C, _ = level_features.shape

        visual_features = level_features[:, :, :, : self.num_regions]
        textual_features = level_features[:, :, :, self.num_regions :]

        masked_sum = textual_features * phrase_mask[:, None, None, :]
        textual_features = masked_sum.sum(dim=-1) / phrase_mask[:, None, None, :].sum(
            dim=-1
        )

        v1, v2, v3 = visual_features.view(B, N, C, H, W).unbind(dim=1)
        l1, l2, l3 = textual_features.unbind(dim=1)

        v12 = self.conv2d(
            torch.cat([v1, l2[:, :, None, None].expand(B, C, H, W)], dim=1)
        ).sigmoid()
        v13 = self.conv2d(
            torch.cat([v1, l3[:, :, None, None].expand(B, C, H, W)], dim=1)
        ).sigmoid()

        v21 = self.conv2d(
            torch.cat([v2, l1[:, :, None, None].expand(B, C, H, W)], dim=1)
        ).sigmoid()
        v23 = self.conv2d(
            torch.cat([v2, l3[:, :, None, None].expand(B, C, H, W)], dim=1)
        ).sigmoid()

        v31 = self.conv2d(
            torch.cat([v3, l1[:, :, None, None].expand(B, C, H, W)], dim=1)
        ).sigmoid()
        v32 = self.conv2d(
            torch.cat([v3, l2[:, :, None, None].expand(B, C, H, W)], dim=1)
        ).sigmoid()

        v1 = v1 + v12 * v1 + v13 * v1
        v2 = v2 + v21 * v2 + v23 * v2
        v3 = v3 + v31 * v3 + v32 * v3

        v = torch.stack([v1, v2, v3], dim=2)

        fused_feature = self.conv_3d(v).squeeze(2)

        return fused_feature


class TextEncoder(nn.Module):
    def __init__(
        self,
        input_size=300,
        hidden_size=512,
        num_layers=1,
        batch_first=True,
        dropout=0.0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=batch_first,
            dropout=dropout,
        )

    def forward(self, input):
        self.lstm.flatten_parameters()
        output, (_, _) = self.lstm(input)
        return output


class ConvUpsample(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        out_channels=1,
        channels=[512, 256, 128, 64],
        upsample=[True, True, False, False],
        scale_factor=2,
        drop=0.2,
    ):
        super().__init__()

        linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        assert len(channels) == len(upsample)

        modules = []

        for i in range(len(channels)):

            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels[i]),
                    nn.ReLU(),
                    nn.Dropout2d(drop),
                )
            )

            if upsample[i]:
                modules.append(linear_upsampling)

            in_channels = channels[i]

        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
                ),
                ## nn.LeakyReLU(),
            )
        )

        self.deconv = nn.Sequential(*modules)

    def forward(self, x):
        return self.deconv(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
