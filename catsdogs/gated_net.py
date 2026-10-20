"""
GatedScaleNet — a from-scratch architecture combining three existing ideas
into one network, fixed to actually be efficient and ONNX-exportable.

Where each piece comes from, stated honestly (citing sources is normal,
expected practice — not a weakness):

  1. Multi-scale branches (3x3 / 5x5 / 7x7 run in parallel, then combined)
     -> the core idea of Inception / GoogLeNet (Szegedy et al., 2014)

  2. Per-channel softmax gating to combine those branches, instead of just
     concatenating them
     -> Selective Kernel Networks / SKNet (Li et al., CVPR 2019)

  3. "Look at multiple depths, not just the end" — implemented here as
     DEEP SUPERVISION (small auxiliary classifier heads attached mid-network
     DURING TRAINING ONLY), instead of the earlier draft's runtime
     confidence-based early exit
     -> this is literally how GoogLeNet itself trained (auxiliary
        classifiers improve gradient flow to early layers), repurposed here
     -> this fixes the ONNX problem: at inference the network has ONE fixed
        path, no data-dependent branching, so it exports cleanly

What's actually new here: this specific combination — multi-scale gated
blocks + proper stage-wise downsampling + deep supervision + a cheap
hybrid local/global head placed ONLY at the final low-resolution stage —
isn't lifted from one single paper. The pieces are known; the assembly,
and the fixes that make it practical for this project, are not copied
from anywhere.

Three concrete engineering fixes vs. the earlier flawed draft:
  FIX 1: Every stage actually downsamples (112 -> 56 -> 28 -> 14), instead
         of staying at 112x112 the whole network. This is the single
         biggest fix — it's what keeps compute and parameter count sane.
  FIX 2: Self-attention, if used at all, only runs at the FINAL 14x14
         stage (196 tokens), not on a 112x112 (12,544 token) map. That is
         roughly a 4000x reduction in attention cost.
  FIX 3: No data-dependent control flow anywhere in the inference path, so
         torch.onnx.export traces one clean, fixed graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultiScaleBlock(nn.Module):
    """One block: three parallel conv branches (different receptive fields),
    combined with SKNet-style PER-CHANNEL softmax gating (not a single
    global gate), then a residual connection.

    Dilated convolutions are used for the "5x5" and "7x7" branches instead
    of literal 5x5/7x7 kernels: a dilated 3x3 sees the same area with far
    fewer parameters (3x3 dilation=2 covers a 5x5 area using only 9 weights
    per channel, not 25). Same receptive-field idea, cheaper to compute —
    an explicit engineering choice, not an accident.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.branch_a = self._conv_bn_relu(channels, kernel_size=3, dilation=1)  # 3x3 field
        self.branch_b = self._conv_bn_relu(channels, kernel_size=3, dilation=2)  # ~5x5 field
        self.branch_c = self._conv_bn_relu(channels, kernel_size=3, dilation=3)  # ~7x7 field

        # The "gate": squeeze each branch to one number per channel (global
        # average pool), then a small FC bottleneck decides how much each
        # of the 3 branches should count, PER CHANNEL — matching what
        # SKNet's Select operator actually does (confirmed against its
        # paper directly, not assumed).
        hidden = max(channels // reduction, 8)
        self.gate_fc1 = nn.Linear(channels, hidden)
        self.gate_fc2 = nn.Linear(hidden, channels * 3)  # 3 logits per channel

        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(channels)

    @staticmethod
    def _conv_bn_relu(channels, kernel_size, dilation):
        padding = dilation * (kernel_size - 1) // 2  # keeps H, W unchanged
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)

        # Squeeze: one descriptor per branch, per channel
        summary = a.mean(dim=(2, 3)) + b.mean(dim=(2, 3)) + c.mean(dim=(2, 3))  # (B, C)
        gate_hidden = F.relu(self.gate_fc1(summary))
        gate_logits = self.gate_fc2(gate_hidden)                                 # (B, 3C)
        gate_logits = gate_logits.view(-1, 3, a.shape[1])                        # (B, 3, C)
        gates = torch.softmax(gate_logits, dim=1)                                # softmax OVER the 3 branches, per channel

        g_a = gates[:, 0, :].unsqueeze(-1).unsqueeze(-1)
        g_b = gates[:, 1, :].unsqueeze(-1).unsqueeze(-1)
        g_c = gates[:, 2, :].unsqueeze(-1).unsqueeze(-1)

        fused = g_a * a + g_b * b + g_c * c
        fused = self.project_bn(self.project(fused))
        return F.relu(x + fused)  # residual connection (He et al., 2015)


class DownsampleStage(nn.Module):
    """Halves H and W, doubles the channel count. This is the piece that
    was MISSING from the original draft — without it, every block runs at
    full 112x112 resolution forever, which is why that design's compute
    and parameter estimates were wrong."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class AuxHead(nn.Module):
    """A small classifier head attached mid-network, used ONLY during
    training (deep supervision). Encourages useful features early in the
    network, the same reasoning GoogLeNet used its auxiliary classifiers
    for. Never runs during eval/export, so it adds zero cost at inference
    and never appears in the exported ONNX graph."""

    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class HybridHead(nn.Module):
    """Combines a simple pooled ('local') summary with a cheap self-attention
    ('global') summary, fused by a learned gate. Deliberately placed ONLY
    here, at the final 14x14 (196-token) stage — the same idea as the
    original draft's attention head, but ~4000x cheaper because it never
    touches the 112x112 map."""

    def __init__(self, channels: int, num_classes: int, num_heads: int = 4):
        super().__init__()
        self.local_pool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.gate_fc = nn.Linear(channels * 2, channels)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        local = self.local_pool(x).flatten(1)                 # (B, C) — "what's in the image"

        tokens = x.flatten(2).transpose(1, 2)                  # (B, H*W, C) — 196 tokens, cheap
        attended, _ = self.attn(tokens, tokens, tokens)
        attended = self.norm(attended)
        glob = attended.mean(dim=1)                             # (B, C) — "how parts relate spatially"

        gate = torch.sigmoid(self.gate_fc(torch.cat([local, glob], dim=1)))
        fused = gate * local + (1 - gate) * glob
        return self.classifier(fused)


class GatedScaleNet(nn.Module):
    """
    Input (3, 224, 224)
      -> multi-scale stem (3x3/5x5/7x7, stride 2)      ->  32 x 112x112
      -> GatedMultiScaleBlock                           ->  32 x 112x112
      -> DownsampleStage (32 -> 64)                      ->  64 x 56x56
      -> GatedMultiScaleBlock  [+ aux head 1]            ->  64 x 56x56
      -> DownsampleStage (64 -> 128)                     -> 128 x 28x28
      -> GatedMultiScaleBlock  [+ aux head 2]            -> 128 x 28x28
      -> DownsampleStage (128 -> 256)                    -> 256 x 14x14
      -> GatedMultiScaleBlock                            -> 256 x 14x14
      -> HybridHead (local pool + cheap self-attention)  -> num_classes

    forward() returns:
      - during training: (main_logits, [aux_logits_1, aux_logits_2])
      - during eval:       main_logits only  (the clean path ONNX exports)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Multi-scale stem (Inception-style), stride 2 each branch
        self.stem_a = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.stem_b = nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False)
        self.stem_c = nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(96)
        self.stem_project = nn.Conv2d(96, 32, kernel_size=1, bias=False)
        self.stem_project_bn = nn.BatchNorm2d(32)

        self.block1 = GatedMultiScaleBlock(32)
        self.down1 = DownsampleStage(32, 64)

        self.block2 = GatedMultiScaleBlock(64)
        self.aux1 = AuxHead(64, num_classes)
        self.down2 = DownsampleStage(64, 128)

        self.block3 = GatedMultiScaleBlock(128)
        self.aux2 = AuxHead(128, num_classes)
        self.down3 = DownsampleStage(128, 256)

        self.block4 = GatedMultiScaleBlock(256)
        self.head = HybridHead(256, num_classes)

    def forward(self, x: torch.Tensor):
        s = torch.cat([self.stem_a(x), self.stem_b(x), self.stem_c(x)], dim=1)
        s = F.relu(self.stem_bn(s))
        s = F.relu(self.stem_project_bn(self.stem_project(s)))

        x = self.block1(s)
        x = self.down1(x)

        x = self.block2(x)
        aux1_out = self.aux1(x) if self.training else None
        x = self.down2(x)

        x = self.block3(x)
        aux2_out = self.aux2(x) if self.training else None
        x = self.down3(x)

        x = self.block4(x)
        main_out = self.head(x)

        if self.training:
            return main_out, [aux1_out, aux2_out]
        return main_out  # <-- the ONLY path torch.onnx.export ever traces
