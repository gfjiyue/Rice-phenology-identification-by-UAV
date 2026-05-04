# model.py
import torch
import torch.nn as nn
import torchvision.models as tv


def _create_backbone(name: str):
    import torch.nn.functional as F
    name = name.lower().strip()

    # Return a unified structure: cnn is a features + pool + flatten sequence; feat_dim is detected automatically
    def _wrap_and_probe(feat_module: nn.Module) -> (nn.Sequential, int):
        cnn = nn.Sequential(
            feat_module,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = cnn(dummy)              # (1, C)
            feat_dim = out.shape[1]
        return cnn, feat_dim

    if name in {"mobilenet", "mobilenet_v2", "mv2"}:
        try:
            m = tv.mobilenet_v2(weights=tv.MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception:
            m = tv.mobilenet_v2(pretrained=True)
        return _wrap_and_probe(m.features)         # Usually C=1280

    elif name in {"mobilenetv3", "mobilenetv3l", "mobilenet_v3_large", "mobilenetv3-large", "mv3l"}:
        try:
            m = tv.mobilenet_v3_large(weights=tv.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        except Exception:
            m = tv.mobilenet_v3_large(pretrained=True)
        return _wrap_and_probe(m.features)         # Usually C=960

    elif name in {"mobilenetv3s", "mobilenet_v3_small", "mobilenetv3-small", "mv3s"}:
        try:
            m = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except Exception:
            m = tv.mobilenet_v3_small(pretrained=True)
        return _wrap_and_probe(m.features)         # Usually C=576

    elif name in {"resnet", "resnet18"}:
        try:
            m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            m = tv.resnet18(pretrained=True)
        feats = nn.Sequential(*list(m.children())[:-1])  # Remove fc
        # Here feats outputs (B, C, 1, 1); _wrap_and_probe will pool/flatten again later, which does not affect the result
        return _wrap_and_probe(feats)             # Usually C=512

    elif name in {"resnet50"}:
        try:
            m = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            m = tv.resnet50(pretrained=True)
        feats = nn.Sequential(*list(m.children())[:-1])
        return _wrap_and_probe(feats)             # Usually C=2048

    else:
        raise ValueError(
            f"Unsupported backbone: {name}. "
            f"Use one of: mobilenet/mv2, mobilenetv3l, mobilenetv3s, resnet18, resnet50."
        )


class DualTemporalGated(nn.Module):
    """
    Dual CNN branches (20m/4m) -> project to the same dimension -> missing-aware gated fusion -> LSTM -> classification
    z_fuse_t = z20_t + m_t * sigma(W [z20_t; z4_t; m_t]) * z4_t
    """
    def __init__(self,
                 backbone_20m: str = "mobilenet",
                 backbone_4m: str  = "mobilenet",
                 proj_dim: int     = 512,
                 hidden_size: int  = 128,
                 num_layers: int   = 1,
                 num_classes: int  = 6,
                 dropout: float    = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.cnn20, fd20 = _create_backbone(backbone_20m)
        self.cnn4 , fd4  = _create_backbone(backbone_4m)

        self.proj20 = nn.Linear(fd20, proj_dim)
        self.proj4  = nn.Linear(fd4 , proj_dim)
        self.gate   = nn.Linear(proj_dim*2 + 1, 1)   # [z20, z4, m] -> alpha

        self.lstm = nn.LSTM(input_size=proj_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x20, x4, lengths, mask4=None):
        """
        x20, x4: (B,T,3,H,W); lengths:(B,); mask4:(B,T) in {0,1}
        return: (B,T,num_classes)
        """
        B, T, C, H, W = x20.shape
        x20 = x20.view(B*T, C, H, W)
        x4  = x4 .view(B*T, C, H, W)

        f20 = self.cnn20(x20)     # (B*T, fd20)
        f4  = self.cnn4 (x4 )     # (B*T, fd4)
        z20 = self.proj20(f20)    # (B*T, P)
        z4  = self.proj4 (f4 )    # (B*T, P)

        # Reshape back to (B,T,P)
        P = z20.size(-1)
        z20 = z20.view(B, T, P)
        z4  = z4 .view(B, T, P)

        if mask4 is None:
            m = torch.ones(B, T, 1, device=z20.device, dtype=z20.dtype)
        else:
            m = mask4.unsqueeze(-1).to(z20.dtype)  # (B,T,1)

        alpha = torch.sigmoid(self.gate(torch.cat([z20, z4, m], dim=-1)))  # (B,T,1)
        z_fuse = z20 + m * alpha * z4                                      # If 4m is missing, it degrades to pure 20m

        # Do not use pack here because padding has already been done by repeating the last frame
        lstm_out, _ = self.lstm(z_fuse)     # (B,T,H*)
        out = self.classifier(lstm_out)     # (B,T,C)
        return out

def get_model_dual(backbone_20m="mobilenet", backbone_4m="mobilenet",
                   hidden_size=128, num_layers=1, num_classes=6,
                   proj_dim=512, dropout=0.0, bidirectional=False):
    return DualTemporalGated(backbone_20m, backbone_4m,
                             proj_dim, hidden_size, num_layers,
                             num_classes, dropout, bidirectional)
