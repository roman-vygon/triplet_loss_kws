import torch
from nemo.backends.pytorch import TrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs
from torch.nn import functional as F


class Att_rnn(TrainableNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
        }

    def __init__(self):
        super().__init__()
        self.zero_pad1 = torch.nn.ZeroPad2d((0, 0, 2, 2))
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=(5, 1),
                                     dilation=(1, 1),
                                     stride=(1, 1),
                                     )
        # self.bn1 = torch.nn.BatchNorm2d(10)
        self.zero_pad2 = torch.nn.ZeroPad2d((0, 0, 2, 2))
        self.conv2 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=1,
                                     kernel_size=(5, 1),
                                     dilation=(1, 1),
                                     stride=(1, 1),
                                     )

        # self.bn2 = torch.nn.BatchNorm2d(1)

        self.gru1 = torch.nn.GRU(input_size=64,
                                 hidden_size=128,
                                 bidirectional=True, batch_first=True)

        self.gru2 = torch.nn.GRU(input_size=256,
                                 hidden_size=16,
                                 bidirectional=True, batch_first=True)

        self.multiheads = []
        for _ in range(4):
            self.multiheads.append(torch.nn.Linear(32, 32).cuda())

    def forward(self, audio_signal):

        audio_signal = audio_signal.permute(0, 2, 1)
        audio_signal = audio_signal.unsqueeze(1)
        x = self.zero_pad1(audio_signal)
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.bn1(x)

        x = self.zero_pad2(x)
        x = self.conv2(x)
        x = F.relu(x)
        # x = self.bn2(x)

        x = x.squeeze()
        x, h_n = self.gru1(x)

        x, h_n = self.gru2(x)
        middle = x.shape[1] // 2
        mid_feature = x[:, middle, :]
        multiheads = []
        for head in self.multiheads:
            y = head(mid_feature)
            att_weigths = torch.einsum('bt,bft->bf', y, x)
            att_weigths = F.softmax(att_weigths)
            multiheads.append(torch.einsum('bf, bft->bt', att_weigths, x))
        return torch.cat(multiheads, 1).unsqueeze(-2)


"""BiRNN model with multihead attention."""
