import torch
import torch.nn as nn

from nemo.backends.pytorch.nm import TrainableNM
from nemo.collections.asr.parts.jasper import init_weights
from nemo.core.neural_types import *


class LinearLayer(TrainableNM):

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        encoder_output:
            0: AxisType(BatchTag)

            1: AxisType(EncodedRepresentationTag)

            2: AxisType(ProcessedTimeTag)
        """

        return {"encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.        
                
        embs: 
            0: AxisType(BatchTag)
            1: AxisType(EncodedRepresentationTah) 
        """
        return {
            "embs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
        }

    def __init__(self, feat_in=128 * 8, emb_size=128, init_mode="xavier_uniform"):
        super().__init__()
        self.linear = nn.Linear(feat_in, emb_size)
        self.apply(lambda x: init_weights(x, mode=init_mode))
        self.to(self._device)
        self.emb_size= emb_size

    def forward(self, encoder_output):
        encoder_output = encoder_output.flatten(start_dim=1).float()
        out = torch.reshape(self.linear(encoder_output), (-1, 1, self.emb_size))
        
        return out
