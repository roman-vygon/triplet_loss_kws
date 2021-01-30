import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs


class ClassificationNet(TrainableNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "embeddings": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),

        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {"logits": NeuralType(('B', 'D'), LogitsType())}

    def __init__(self, n_classes, n_embed):
        super().__init__()

        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(n_embed, n_classes)

    def forward(self, embeddings):
        embeddings = torch.flatten(embeddings, start_dim = -2)
        output = self.nonlinear(embeddings)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores
