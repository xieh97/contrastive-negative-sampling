import torch
import torch.nn as nn
import torch.nn.functional as F

from models import init_weights


class CRNNEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(CRNNEncoder, self).__init__()
        self.args = args
        self.kwargs = kwargs

        self.cnn = nn.Sequential(
            # Conv2D block
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # LPPool
            nn.LPPool2d(4, (2, 4)),

            # Conv2D block
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # Conv2D block
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # LPPool
            nn.LPPool2d(4, (2, 4)),

            # Conv2D block
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # Conv2D block
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # LPPool
            nn.LPPool2d(4, (1, 4)),

            nn.Dropout(0.3)
        )
        self.cnn.apply(init_weights)

        with torch.no_grad():
            rnn_in_dim = self.cnn(torch.randn(1, 1, 500, kwargs["in_dim"])).shape
            rnn_in_dim = rnn_in_dim[1] * rnn_in_dim[-1]
        self.gru = nn.GRU(rnn_in_dim, kwargs["out_dim"] // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        """
        :param x: tensor, (batch_size, time_steps, Mel_bands).
        :return: tensor, (batch_size, time_steps, embed_dim).
        """
        batch, time, mel = x.shape

        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)  # (N, T/4, 128*M/64)

        x, _ = self.gru(x)  # (N, T/4, E)

        if self.kwargs.get("up_sampling", None) is not None:
            x = F.interpolate(x.transpose(1, 2), time, mode=self.kwargs["up_sampling"],
                              align_corners=False).transpose(1, 2)

        return x
