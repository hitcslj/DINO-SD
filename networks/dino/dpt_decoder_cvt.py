import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import FeatureFusionBlock, _make_scratch
from networks.transformer import CVT, SimpleCVT
def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class DPT_decoder_CVT(nn.Module):
    def __init__(self, encoder='vitl', nclass=1, features=256, out_channels=[256, 512, 1024, 1024], use_bn=True, use_clstoken=False, skip=True, patchsize=14):
        super(DPT_decoder_CVT, self).__init__()
        assert encoder in ['vits', 'vitb', 'vitl']
        dim_dict = {'vits': 384, 'vitb': 768, 'vitl':1024}
        in_channels = dim_dict[encoder]
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.psize = patchsize
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels = in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0 
            ),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0
            ),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            )
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
                
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                # nn.ReLU(True),
                nn.Identity(),
            )
        
        self.cross = {}
        # for i in range(len(out_channels[:2])):
        #     self.cross[i] = CVT(input_channel=features, downsample_ratio=2 ** (2-i), iter_num=4)
        for i in range(len(out_channels[2:])):
            self.cross[2 + i] = SimpleCVT(input_channel=features, iter_num=4)

        self.decoder_cross = nn.ModuleList(list(self.cross.values()))
        self.skip = skip

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)

            x = self.resize_layers[i](x)
            
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1) # 6, 256, 4h, 4w
        layer_2_rn = self.scratch.layer2_rn(layer_2) # 6, 256. 2h, 2w
        layer_3_rn = self.scratch.layer3_rn(layer_3) # 6, 256, h, w
        layer_4_rn = self.scratch.layer4_rn(layer_4) # 6, 256, h/2, w/2
        
        if self.skip:
            # B, C, H, W = layer_1_rn.shape
            # layer_1_rn = layer_1_rn + self.cross[0](layer_1_rn.reshape(-1, 6, C, H, W)).reshape(B, C, H, W)
            # B, C, H, W = layer_2_rn.shape
            # layer_2_rn = layer_2_rn + self.cross[1](layer_2_rn.reshape(-1, 6, C, H, W)).reshape(B, C, H, W)
            B, C, H, W = layer_3_rn.shape
            layer_3_rn = layer_3_rn + self.cross[2](layer_3_rn.reshape(-1, 6, C, H, W)).reshape(B, C, H, W)
            B, C, H, W = layer_4_rn.shape
            layer_4_rn = layer_4_rn + self.cross[3](layer_4_rn.reshape(-1, 6, C, H, W)).reshape(B, C, H, W)
        else:
            # B, C, H, W = layer_1_rn.shape
            # layer_1_rn = self.cross[0](layer_1_rn.reshape(-1, 6, C, H, W)).reshape(B, C, H, W)
            # B, C, H, W = layer_2_rn.shape
            # layer_2_rn = self.cross[1](layer_2_rn.reshape(-1, 6, C, H, W)).reshape(B, C, H, W)
            B, C, H, W = layer_3_rn.shape
            layer_3_rn = self.cross[2](layer_3_rn.reshape(-1, 6, C, H, W)).reshape(B, C, H, W)
            B, C, H, W = layer_4_rn.shape
            layer_4_rn = self.cross[3](layer_4_rn.reshape(-1, 6, C, H, W)).reshape(B, C, H, W)


        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * self.psize), int(patch_w * self.psize)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        out = F.sigmoid(out)
        outputs = {}
        outputs[("disp", 0)] = out

        return outputs