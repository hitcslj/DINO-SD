import torch
import torch.nn as nn
import torch.nn.functional as F

class DINO_Encoder(nn.Module):
    def __init__(self, encoder='vitl', pretrained=True, patch_size=14):
        super(DINO_Encoder, self).__init__()
        assert encoder in ['vits', 'vitb', 'vitl']
        self.pretrained = torch.hub.load('./torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False, patch_size=patch_size)
        if pretrained:
            self.pretrained.load_state_dict(torch.load(f'./pretrain_models/dinov2_{encoder}14_pretrain.pth'))


    def forward(self, x):
        h, w = x.shape[-2:] # 224 = 14 * 16  
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        patch_h, patch_w = h // 14, w // 14
        return features, patch_h, patch_w