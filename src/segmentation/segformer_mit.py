"""Pure-PyTorch SegFormer (MixVisionTransformer + SegformerHead) implementation.

This mirrors the MMSegmentation v1.2.2 architecture so that the original
SegFormer-B5 building checkpoint (saved as an mmengine checkpoint) can be
loaded and run without installing mmcv / mmsegmentation / mmengine on
Windows.  Behavior is verified against the exported ONNX graph.

Reference implementations (Apache-2.0):
  - mmsegmentation v1.2.2 `mmseg/models/backbones/mit.py`
  - mmsegmentation v1.2.2 `mmseg/models/decode_heads/segformer_head.py`
"""

from __future__ import annotations

import math
import sys
import types
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# mmengine stub so `torch.load` can unpickle mmengine-format checkpoints
# ---------------------------------------------------------------------------
class _DummyMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


class _Dummy(metaclass=_DummyMeta):
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        setattr(self, name, _Dummy)
        return _Dummy


class _MmengineStubFinder(MetaPathFinder, Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mmengine" or fullname.startswith("mmengine."):
            return ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        sys.modules[spec.name] = m
        return m

    def exec_module(self, module):
        pass


def load_mmseg_checkpoint(path: str) -> dict:
    """Load an mmengine-format checkpoint without mmengine installed."""
    if not any(isinstance(f, _MmengineStubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MmengineStubFinder())
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Building blocks (mirrors mmsegmentation v1.2.2)
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    """Stochastic depth. Identity at inference time."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x / keep * mask


class PatchEmbed(nn.Module):
    """Overlapping patch embedding: conv + layer norm."""

    def __init__(self, in_channels: int, embed_dims: int, kernel_size: int,
                 stride: int, padding: int, eps: float = 1e-6):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=True,
        )
        self.norm = nn.LayerNorm(embed_dims, eps=eps)

    def forward(self, x: torch.Tensor):
        x = self.projection(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (h, w)


class MixFFN(nn.Module):
    """SegFormer MixFFN: 1x1 conv -> 3x3 depthwise conv -> 1x1 conv."""

    def __init__(self, embed_dims: int, feedforward_channels: int,
                 ffn_drop: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        fc1 = nn.Conv2d(embed_dims, feedforward_channels, 1, 1, bias=True)
        pe_conv = nn.Conv2d(
            feedforward_channels, feedforward_channels, 3, 1,
            padding=(3 - 1) // 2, bias=True, groups=feedforward_channels,
        )
        fc2 = nn.Conv2d(feedforward_channels, embed_dims, 1, 1, bias=True)
        drop = nn.Dropout(ffn_drop)
        self.layers = nn.Sequential(fc1, pe_conv, nn.GELU(), drop, fc2, drop)
        self.dropout_layer = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor, hw_shape, identity=None):
        h, w = hw_shape
        b, n, c = x.shape
        out = x.transpose(1, 2).reshape(b, c, h, w)
        out = self.layers(out)
        out = out.flatten(2).transpose(1, 2)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(nn.Module):
    """SegFormer spatial-reduction attention."""

    def __init__(self, embed_dims: int, num_heads: int, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, drop_path_rate: float = 0.0,
                 qkv_bias: bool = True, sr_ratio: int = 1, eps: float = 1e-6):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, attn_drop, bias=qkv_bias
        )  # batch_first=False (matches mmcv wrapper behavior)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(drop_path_rate)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                embed_dims, embed_dims, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.norm = nn.LayerNorm(embed_dims, eps=eps)

    def forward(self, x: torch.Tensor, hw_shape, identity=None):
        if identity is None:
            identity = x
        if self.sr_ratio > 1:
            h, w = hw_shape
            b, n, c = x.shape
            x_kv = x.transpose(1, 2).reshape(b, c, h, w)
            x_kv = self.sr(x_kv)
            x_kv = x_kv.flatten(2).transpose(1, 2)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
        # batch_first=True in mmcv wrapper -> transpose to (N, B, C)
        out = self.attn(query=x.transpose(0, 1), key=x_kv.transpose(0, 1),
                        value=x_kv.transpose(0, 1))[0]
        out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dims: int, num_heads: int,
                 feedforward_channels: int, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0,
                 qkv_bias: bool = True, sr_ratio: int = 1, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims, eps=eps)
        self.attn = EfficientMultiheadAttention(
            embed_dims, num_heads, attn_drop=attn_drop_rate,
            proj_drop=drop_rate, drop_path_rate=drop_path_rate,
            qkv_bias=qkv_bias, sr_ratio=sr_ratio, eps=eps,
        )
        self.norm2 = nn.LayerNorm(embed_dims, eps=eps)
        self.ffn = MixFFN(
            embed_dims, feedforward_channels, ffn_drop=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x: torch.Tensor, hw_shape):
        x = self.attn(self.norm1(x), hw_shape, identity=x)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        return x


class MixVisionTransformer(nn.Module):
    """SegFormer backbone (B5 defaults, override with num_layers/num_heads)."""

    def __init__(self, in_channels: int = 3, embed_dims: int = 64,
                 num_layers=(3, 6, 40, 3), num_heads=(1, 2, 5, 8),
                 patch_sizes=(7, 3, 3, 3), strides=(4, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1), out_indices=(0, 1, 2, 3),
                 mlp_ratio: int = 4, qkv_bias: bool = True,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1, eps: float = 1e-6):
        super().__init__()
        assert len(num_layers) == len(num_heads) == len(patch_sizes) == len(strides) == len(sr_ratios)
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.out_indices = out_indices
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]

        self.layers = nn.ModuleList()
        cur = 0
        in_ch = in_channels
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_ch, embed_dims_i, patch_sizes[i], strides[i],
                patch_sizes[i] // 2, eps=eps,
            )
            blocks = nn.ModuleList([
                TransformerEncoderLayer(
                    embed_dims_i, num_heads[i], mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx], qkv_bias=qkv_bias,
                    sr_ratio=sr_ratios[i], eps=eps,
                )
                for idx in range(num_layer)
            ])
            norm = nn.LayerNorm(embed_dims_i, eps=eps)
            self.layers.append(nn.ModuleList([patch_embed, blocks, norm]))
            in_ch = embed_dims_i
            cur += num_layer

    def forward(self, x: torch.Tensor):
        outs = []
        for i, (patch_embed, blocks, norm) in enumerate(self.layers):
            x, hw_shape = patch_embed(x)
            for block in blocks:
                x = block(x, hw_shape)
            x = norm(x)
            h, w = hw_shape
            b, n, c = x.shape
            x = x.transpose(1, 2).reshape(b, c, h, w)
            if i in self.out_indices:
                outs.append(x)
        return outs


class SegformerHead(nn.Module):
    def __init__(self, in_channels=(64, 128, 320, 512), channels: int = 256,
                 num_classes: int = 2, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = channels
        self.num_classes = num_classes
        self.interpolate_mode = "bilinear"
        self.align_corners = False

        class _ConvBNReLU(nn.Module):
            def __init__(self, cin, cout, eps, momentum, with_relu=True):
                super().__init__()
                self.conv = nn.Conv2d(cin, cout, 1, 1, bias=False)
                self.bn = nn.BatchNorm2d(cout, eps=eps, momentum=momentum)
                self.activate = nn.ReLU(inplace=True) if with_relu else nn.Identity()

            def forward(self, x):
                return self.activate(self.bn(self.conv(x)))

        self.convs = nn.ModuleList()
        for c in self.in_channels:
            self.convs.append(_ConvBNReLU(c, channels, eps, momentum))
        self.fusion_conv = _ConvBNReLU(
            channels * len(self.in_channels), channels, eps, momentum,
            with_relu=True,
        )
        self.conv_seg = nn.Conv2d(channels, num_classes, 1, 1, bias=True)

    def forward(self, inputs):
        outs = []
        h, w = inputs[0].shape[2:]
        for x, conv in zip(inputs, self.convs):
            x = conv(x)
            x = nn.functional.interpolate(
                x, size=(h, w), mode=self.interpolate_mode,
                align_corners=self.align_corners,
            )
            outs.append(x)
        out = self.fusion_conv(torch.cat(outs, dim=1))
        return self.conv_seg(out)


class SegFormerB5(nn.Module):
    """Full SegFormer-B5 building model. Expects normalized BGR float32 input."""

    def __init__(self, num_classes: int = 2, drop_path_rate: float = 0.1):
        super().__init__()
        self.backbone = MixVisionTransformer(
            embed_dims=64, num_layers=(3, 6, 40, 3), num_heads=(1, 2, 5, 8),
            drop_path_rate=drop_path_rate,
        )
        self.decode_head = SegformerHead(
            in_channels=(64, 128, 320, 512), channels=256, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        return nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization used by the exported ONNX graph."""
        mean = torch.tensor([123.675, 116.28, 103.53], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


def build_segformer_b5_from_checkpoint(path: str, num_classes: int = 2) -> SegFormerB5:
    ckpt = load_mmseg_checkpoint(path)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = SegFormerB5(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[segformer] missing: {missing[:5]} ... unexpected: {unexpected[:5]} ...")
    model.eval()
    return model
