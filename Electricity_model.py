import math
import copy

import torch
from torch import nn

from model_helpers import LayerNorm, PositionalEmbedding, TransformerBlock


class TransformerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.original_len = args.window_size
        self.latent_len = int(self.original_len / 2)
        self.dropout_rate = args.drop_out

        self.hidden = args.hidden
        self.heads = args.heads
        self.n_layers = args.n_layers
        self.output_size = args.output_size

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            padding_mode="replicate",
        )
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.position = PositionalEmbedding(
            max_len=self.latent_len, d_model=self.hidden
        )
        self.layer_norm = LayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.hidden, self.heads, self.hidden * 4, self.dropout_rate
                )
                for _ in range(self.n_layers)
            ]
        )

        self.deconv = nn.ConvTranspose1d(
            in_channels=self.hidden,
            out_channels=self.hidden,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.linear1 = nn.Linear(self.hidden, 128)
        self.linear2 = nn.Linear(128, self.output_size)

        self.truncated_normal_init()

    def _truncated_normal_init(
        self,
        mean: float = 0.0,
        std: float = 0.02,
        lower: float = -0.04,
        upper: float = 0.04,
    ) -> None:
        """
        Initialize model parameters with a truncated normal distribution.

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.02.
            lower (float, optional): Lower bound of the truncated normal distribution. Defaults to -0.04.
            upper (float, optional): Upper bound of the truncated normal distribution. Defaults to 0.04.
        """
        for name, param in self.named_parameters():
            if "layer_norm" in name:
                continue
            else:
                with torch.no_grad():
                    lower_bound = (
                        1.0 + math.erf(((lower - mean) / std) / math.sqrt(2.0))
                    ) / 2.0
                    upper_bound = (
                        1.0 + math.erf(((upper - mean) / std) / math.sqrt(2.0))
                    ) / 2.0
                    param.uniform_(2 * lower_bound - 1, 2 * upper_bound - 1)
                    param.erfinv_()
                    param.mul_(std * math.sqrt(2.0))
                    param.add_(mean)

    def forward(self, sequence):
        x_token = self.pool(self.conv(sequence)).permute(0, 2, 1)

        embedding = x_token + self.position(sequence)
        x = self.dropout(self.layer_norm(embedding))

        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        x = self.deconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x).permute(0, 2, 1)
        return x


class ELECTRICITY(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.Discriminator = TransformerModel(args)
        self.pretrain = args.pretrain
        args_gen = copy.copy(args)
        args_gen.hidden = 64
        self.Generator = TransformerModel(args_gen)

    def forward(self, sequence, mask=None):
        if self.pretrain:
            gen_out = self.Generator(sequence)
            disc_in = sequence
            disc_in[mask] = gen_out[mask]
        else:
            disc_in = sequence
        disc_out = self.Discriminator(disc_in)
        if self.pretrain:
            return disc_out, gen_out
        else:
            return disc_out, None
