import torch.nn.functional as F
from nemo import logging
from nemo.backends.pytorch import NonTrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs


class L2Regularizer(NonTrainableNM):
    """Performs L2 regularization over samples. 
        i.e. makes the sum of squares of embeddings equal to 1    
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "embeds": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        return {
            "l2_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),

        }

    def __init__(self):
        super().__init__()
        logging.info("{}: L2 regularization".format(self.__class__))

    def regularize(self, embeds):
        l2_embeds = F.normalize(embeds, p=2, dim=-1)
        # l2_embeds.requires_grad = True      <==== uncomment this if you run into var has no gradient problem

        return l2_embeds

    def forward(self, embeds):
        l2_signal = self.regularize(embeds)
        return l2_signal


if __name__ == "__main__":
    pass
