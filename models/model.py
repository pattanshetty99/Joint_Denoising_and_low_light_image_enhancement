import torch
import torch.nn as nn
from .blocks import Encoder, StructureDecoder, ColorBranch, DenoiseBlock, IlluminationBranch


class LowLightEnhancer_Color(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.structure_decoder = StructureDecoder()
        self.color_branch = ColorBranch()
        self.denoiser = DenoiseBlock()
        self.illum_branch = IlluminationBranch()

    def forward(self, x):
        illum = self.illum_branch(x)
        enhanced_input = x / (illum + 1e-6)

        feat = self.encoder(x)
        feat = self.denoiser(feat)

        struct_out = self.structure_decoder(feat)
        color_residual = self.color_branch(x)

        out = struct_out + color_residual + x
        return torch.clamp(out, 0, 1)
