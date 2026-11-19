import torch
from torch import nn
from torch.nn import functional as F


def _block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, num_classes, width=24):
        super().__init__()
        if not 4 <= width <= 128 or width % 4:
            raise ValueError("width cần là bội số của 4 trong khoảng 4–128.")
        channels = [width * 2**level for level in range(4)]
        self.encoder = nn.ModuleList(
            _block(source, target)
            for source, target in zip([3, *channels[:2]], channels[:3])
        )
        self.bridge = _block(channels[2], channels[3])
        self.decoder = nn.ModuleList(
            _block(channels[level + 1] + channels[level], channels[level])
            for level in (2, 1, 0)
        )
        self.head = nn.Conv2d(width, num_classes, kernel_size=1)

    def forward(self, image):
        skips = []
        x = image
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bridge(x)
        for decoder, skip in zip(self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)


class ResidualBlock(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.body = _block(source, target)
        self.shortcut = (
            nn.Identity() if source == target
            else nn.Conv2d(source, target, 1, bias=False)
        )

    def forward(self, x):
        return F.silu(self.body(x) + self.shortcut(x))


class AttentionGate(nn.Module):
    """Additive spatial gating of encoder features using decoder context."""

    def __init__(self, skip_channels, gate_channels):
        super().__init__()
        hidden = max(4, skip_channels // 2)
        self.skip = nn.Conv2d(skip_channels, hidden, 1, bias=False)
        self.gate = nn.Conv2d(gate_channels, hidden, 1, bias=False)
        self.score = nn.Conv2d(hidden, 1, 1)

    def forward(self, skip, gate):
        score = self.score(F.silu(self.skip(skip) + self.gate(gate))).sigmoid()
        return skip * score


class DilatedContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        branch = channels // 4
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, branch, 3, padding=rate, dilation=rate, bias=False),
                nn.GroupNorm(4, branch), nn.SiLU(inplace=True),
            ) for rate in (1, 2, 4, 8)
        ])
        self.fuse = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        return F.silu(x + self.fuse(torch.cat([branch(x) for branch in self.branches], 1)))


class AttentionUNet(nn.Module):
    """Lightweight residual U-Net with gated skips and multi-dilation context.

    This is a local architecture inspired by attention/residual building blocks,
    not a reproduction of paper weights and not pretrained on public geology.
    """

    def __init__(self, num_classes, width=24):
        super().__init__()
        if not 4 <= width <= 128 or width % 4:
            raise ValueError("width cần là bội số của 4 trong khoảng 4–128.")
        channels = [width * 2**level for level in range(4)]
        self.encoder = nn.ModuleList(
            ResidualBlock(source, target)
            for source, target in zip([3, *channels[:2]], channels[:3])
        )
        self.bridge = nn.Sequential(
            ResidualBlock(channels[2], channels[3]), DilatedContext(channels[3]),
        )
        self.gates = nn.ModuleList(
            AttentionGate(channels[level], channels[level + 1]) for level in (2, 1, 0)
        )
        self.decoder = nn.ModuleList(
            ResidualBlock(channels[level + 1] + channels[level], channels[level])
            for level in (2, 1, 0)
        )
        self.head = nn.Conv2d(width, num_classes, 1)

    def forward(self, image):
        skips, x = [], image
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bridge(x)
        for decoder, gate, skip in zip(self.decoder, self.gates, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, gate(skip, x)], dim=1))
        return self.head(x)


def build_model(arch, num_classes, width=24):
    if not 2 <= num_classes <= 255:
        raise ValueError("Số lớp cần nằm trong khoảng 2–255.")
    if arch == "unet":
        return UNet(num_classes, width)
    if arch == "attention_unet":
        return AttentionUNet(num_classes, width)
    raise ValueError(f"Kiến trúc không hỗ trợ: {arch}. Dùng unet hoặc attention_unet.")
