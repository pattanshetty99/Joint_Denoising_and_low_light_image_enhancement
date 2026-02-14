import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class WindowSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, window_size=8):
        super().__init__()
        self.heads = heads
        self.ws = window_size
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h = (self.ws - H % self.ws) % self.ws
        pad_w = (self.ws - W % self.ws) % self.ws
        x = F.pad(x, (0, pad_w, 0, pad_h))

        H_pad, W_pad = x.shape[2], x.shape[3]

        x = rearrange(
            x,
            "b c (nh wh) (nw ww) -> (b nh nw) c wh ww",
            wh=self.ws, ww=self.ws
        )

        qkv = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=self.heads),
            qkv,
        )

        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)

        out = rearrange(
            out,
            "b h (x y) d -> b (h d) x y",
            x=self.ws, y=self.ws,
        )

        out = self.to_out(out)

        out = rearrange(
            out,
            "(b nh nw) c wh ww -> b c (nh wh) (nw ww)",
            nh=H_pad // self.ws,
            nw=W_pad // self.ws
        )

        return out[:, :, :H, :W]


class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.attn = WindowSelfAttention(dim)
        self.norm2 = nn.GroupNorm(8, dim)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
